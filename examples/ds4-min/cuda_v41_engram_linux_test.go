//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func cudaEngramRows(t *testing.T) *dsv41.EngramRows {
	t.Helper()
	spec := dsv41engram.LayoutSpec{TokenMap: []uint32{0, 1, 2}, CompressedVocabSize: 3, Rows: [2]uint32{72, 72}}
	var tables [2]*dsv41engram.Table
	for i := range tables {
		spec.Multipliers[i] = [4]uint64{1, 3, 5, 7}
		for j := range spec.Primes[i] {
			spec.Primes[i][j] = 3
		}
		raw := make([]byte, 72*264)
		for row := 0; row < 72; row++ {
			for j := 0; j < 256; j++ {
				raw[row*264+j] = byte(0x20 + (row+j+i)%16)
			}
			for j := 256; j < 264; j++ {
				raw[row*264+j] = 127
			}
		}
		var err error
		tables[i], err = dsv41engram.NewTable(bytes.NewReader(raw), int64(len(raw)), 0, 72)
		if err != nil {
			t.Fatal(err)
		}
	}
	l, err := dsv41engram.NewLayout(spec)
	if err != nil {
		t.Fatal(err)
	}
	s, err := dsv41.NewEngramRows(l, tables, 8)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestCUDAV41EngramCleanup(t *testing.T) {
	for _, fault := range []string{"drain", "destroy", "free", "unload", "none"} {
		t.Run(fault, func(t *testing.T) {
			rows := cudaEngramRows(t)
			if err := rows.Prepare(1, true, 0); err != nil {
				t.Fatal(err)
			}
			s := &cudaV41Engram{module: 1, blas: 1, rows: rows, valid: true, prepared: true}
			cause := errors.New("injected cleanup")
			fired := false
			counts := map[string]int{}
			call := func(name string) error {
				counts[name]++
				if fault == name && !fired {
					fired = true
					return cause
				}
				return nil
			}
			s.cleanup = &cudaV41EngramCleanup{drain: func() error { return call("drain") }, destroy: func(cublas.Handle) error { return call("destroy") }, unload: func(driver.Module) error { return call("unload") }, freeFloat: func(cudamem.Slice[float32]) error { return call("free") }, freeHalf: func(cudamem.Slice[uint16]) error { return call("free") }}
			err := s.Close()
			if fault != "none" {
				if !errors.Is(err, cause) || !s.closing || s.valid || s.rows == nil {
					t.Fatal("cleanup owner not retained", err)
				}
				if rows.Position() != 0 || rows.Commit() == nil {
					t.Fatal("cleanup published staged token")
				}
				if err := s.Commit(); err == nil {
					t.Fatal("terminal owner committed")
				}
				if fault == "drain" && counts["destroy"] != 0 {
					t.Fatal("destroy before successful drain")
				}
				if fault == "destroy" && (s.blas != 1 || counts["destroy"] != 1 || counts["unload"] != 0) {
					t.Fatal("failed destroy lost retry owner", s.blas, counts)
				}
				if fault == "unload" && (s.module != 1 || counts["destroy"] != 1 || counts["unload"] != 1) {
					t.Fatal("failed unload lost retry owner", s.module, counts)
				}
				if err := s.Close(); err != nil {
					t.Fatal("retry", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if s.module != 0 || s.blas != 0 || s.rows != nil {
				t.Fatal("resources retained after successful cleanup")
			}
			want := map[string]int{"drain": 2, "destroy": 1, "free": 4, "unload": 1}
			switch fault {
			case "none":
				want["drain"] = 1
			case "destroy":
				want["destroy"] = 2
			case "free":
				want["free"] = 5
			case "unload":
				want["free"] = 8
				want["unload"] = 2
			}
			if !reflect.DeepEqual(counts, want) {
				t.Fatal("cleanup retry callbacks", counts, "want", want)
			}
			before := fmt.Sprint(counts)
			if err := s.Close(); err != nil || fmt.Sprint(counts) != before {
				t.Fatal("closed owner reacquired resources", err)
			}
		})
	}
}

func TestCUDAV41EngramCommitAdmission(t *testing.T) {
	rows := cudaEngramRows(t)
	defer rows.Close()
	if err := rows.Prepare(1, true, 0); err != nil {
		t.Fatal(err)
	}
	s := &cudaV41Engram{module: 1, rows: rows, prepared: true, valid: true}
	for _, applied := range []uint8{0, 1, 2} {
		s.applied = applied
		if err := s.Commit(); err == nil || rows.Position() != 0 {
			t.Fatal("partial token published")
		}
	}
	s.applied = 3
	s.valid = false
	if err := s.Commit(); err == nil || rows.Position() != 0 {
		t.Fatal("failed device token published")
	}
	s.valid = true
	if err := s.Commit(); err != nil || rows.Position() != 1 {
		t.Fatal("full token commit", err)
	}
	if err := s.Commit(); err == nil || rows.Position() != 1 {
		t.Fatal("double commit")
	}
	if err := rows.Prepare(2, true, 1); err != nil {
		t.Fatal(err)
	}
	s.prepared = true
	s.Abort()
	if err := s.Commit(); err == nil || rows.Position() != 1 {
		t.Fatal("later graph failure published staged history")
	}
}

func TestCUDAV41EngramScalarReference(t *testing.T) {
	residual, kv, q, k := make([]float32, 20480), make([]float32, 25600), make([]float32, 20480), make([]float32, 20480)
	for i := range residual {
		residual[i] = .25
		kv[i] = .5
	}
	for i := 20480; i < len(kv); i++ {
		kv[i] = .125
	}
	got := engramReferenceGate(residual, kv, q, k)
	for i, v := range got {
		if v != .3125 {
			t.Fatalf("zero-dot shared value coordinate%d got%g", i, v)
		}
	}
	for i := range q {
		q[i] = 1
		k[i] = 1
		if i/5120%2 != 0 {
			k[i] = -1
		}
	}
	got = engramReferenceGate(residual, kv, q, k)
	if got[0] <= got[5120] || got[10240] <= got[15360] {
		t.Fatal("signed-root gate lost sign")
	}
	if residual[0] != .25 || kv[20480] != .125 {
		t.Fatal("reference mutated inputs")
	}
}

func TestCUDAV41EngramHalfValues(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_ENGRAM_GATE") != "1" {
		t.Skip("opt-in V4.1 Engram device gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	r := &cudaFoundation{device: dev}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	p, err := compileCUDAV41Device(cudaV41EngramSource, nvrtcx.WithName("cuda_v41_engram.cu"), nvrtcx.WithDeviceArch(dev), nvrtcx.WithFastMath())
	if err != nil {
		t.Fatal(err)
	}
	module, _, err := p.LoadOutputWithLog(dev, nvrtcx.OutputCUBIN)
	if module != 0 {
		t.Cleanup(func() {
			if err := dev.WithContext(func() error {
				if err := syncStream(r.stream()); err != nil {
					return err
				}
				return module.Close()
			}); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	var fn driver.Function
	if err := dev.WithContext(func() error { var e error; fn, e = module.Function("ds4_v41_engram_half"); return e }); err != nil {
		t.Fatal(err)
	}
	input := make([]float32, 0, 65536*2)
	for h := 0; h < 65536; h++ {
		if h&0x7c00 == 0x7c00 && h&1023 != 0 {
			continue
		}
		v := quant.Float16ToFloat32(uint16(h))
		input = append(input, v)
		if !math.IsInf(float64(v), 0) {
			input = append(input, math.Nextafter32(v, float32(math.Inf(1))))
		}
	}
	input = append(input, 65519, 65520, -65519, -65520, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32)
	x := cudaSynthUpload(t, r, input)
	out, err := cudamem.Alloc[uint16](dev, len(input))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := out.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := dev.WithContext(func() error {
		if err := cudalaunch.Kernel(fn).Grid(uint32((len(input)+255)/256), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(out, x, uint32(len(input))); err != nil {
			return err
		}
		return syncStream(r.stream())
	}); err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, len(input))
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range input {
		if want := cudaQ81HalfReference(v); got[i] != want {
			t.Fatalf("F16 input%08x got%04x want%04x", math.Float32bits(v), got[i], want)
		}
	}
	t.Logf("F16 conversion values=%d", len(input))
}

// This gate executes real F16xF16 cuBLAS and all 25600 projection / 20480
// residual values, not a CPU implementation disguised as a device path.
// A named mutant changes the actual compiled source and must FAIL this test.
func TestCUDAV41EngramValues(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_ENGRAM_GATE") != "1" {
		t.Skip("opt-in V4.1 Engram device gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	r := &cudaFoundation{device: dev}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	r.major, r.minor, err = dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	source := cudaV41EngramSource
	switch os.Getenv("DS4_CUDA_V41_ENGRAM_MUTANT") {
	case "":
	case "gate":
		source = engramMutate(t, source, "const float gate=1.f/", "const float gate=.125f/")
	case "value":
		source = engramMutate(t, source, "value=4u*width", "value=3u*width")
	case "norm":
		source = engramMutate(t, source, "h2+=h*h;k2+=k*k;", "h2+=h*h*4.f;k2+=k*k;")
	case "half":
		source = engramMutate(t, source, "out[i]=h;", "out[i]=0;")
	default:
		t.Fatal("unknown Engram mutant")
	}
	weights := make([]uint16, 6144*25600)
	for row := 0; row < 25600; row++ {
		for j, h := range []uint16{0x3400, 0xb000, 0x2800, 0x3000} {
			weights[row*6144+(row*17+j*131)%6144] = h
		}
	}
	var s *cudaV41Engram
	var borrowed []cudamem.Slice[float32]
	var secondProjection cudamem.Slice[uint16]
	dw, err := cudamem.Alloc[uint16](dev, len(weights))
	// Close the component before releasing any borrowed input/weight, even
	// when a launch or drain failed. A failed close leaves all borrows retained.
	t.Cleanup(func() {
		if s != nil {
			if err := s.Close(); err != nil {
				t.Error(err)
				return
			}
		} else if err := dev.WithContext(func() error { return syncStream(r.stream()) }); err != nil {
			t.Error(err)
			return
		}
		for _, x := range borrowed {
			if err := x.Free(); err != nil {
				t.Error(err)
			}
		}
		if err := dw.Free(); err != nil {
			t.Error(err)
		}
		if err := secondProjection.Free(); err != nil {
			t.Error(err)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	upload := func(values []float32) cudamem.Slice[float32] {
		x, e := cudamem.Alloc[float32](dev, len(values))
		borrowed = append(borrowed, x)
		if e != nil {
			t.Fatal(e)
		}
		if e := x.CopyFromHost(values); e != nil {
			t.Fatal(e)
		}
		return x
	}
	if err := dw.CopyFromHost(weights); err != nil {
		t.Fatal(err)
	}
	secondProjection, err = cudamem.Alloc[uint16](dev, len(weights))
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 25600; row++ {
		for j, h := range []uint16{0x3000, 0x2800, 0xac00, 0x3400} {
			weights[row*6144+(row*17+j*131)%6144] = h
		}
	}
	if err := secondProjection.CopyFromHost(weights); err != nil {
		t.Fatal(err)
	}
	q, k := make([]float32, 20480), make([]float32, 20480)
	for i := range q {
		q[i] = float32(1+i%3) / 4
		k[i] = float32(1+i%5) / 8
		if i/5120%2 != 0 {
			k[i] = -k[i]
		}
	}
	dq, dk := upload(q), upload(k)
	var binding [2]cudaV41EngramWeight
	desc := func(layer int, suffix string, encoding uint32, n int, dims ...uint64) dsv41.TensorDescriptor {
		size := int64(n * 4)
		if encoding == 1 {
			size = int64(n * 2)
		}
		return dsv41.TensorDescriptor{Name: fmt.Sprintf("blk.%d.engram_%s.weight", layer, suffix), Encoding: encoding, Dimensions: dims, Bytes: size}
	}
	for i, layer := range []int{1, 14} {
		binding[i] = cudaV41EngramWeight{Projection: desc(layer, "kv", 1, len(weights), 6144, 25600), Query: desc(layer, "q_norm", 0, 20480, 5120, 4), Key: desc(layer, "k_norm", 0, 20480, 5120, 4), F16: dw, Q: dq, K: dk}
	}
	binding[1].F16 = secondProjection
	for _, fault := range []string{"name", "encoding", "shape", "bytes", "offset", "extent"} {
		bad := binding
		bad[0].Projection.Dimensions = append([]uint64(nil), bad[0].Projection.Dimensions...)
		switch fault {
		case "name":
			bad[0].Projection.Name = "blk.14.engram_kv.weight"
		case "encoding":
			bad[0].Projection.Encoding = 0
		case "shape":
			bad[0].Projection.Dimensions[0]--
		case "bytes":
			bad[0].Projection.Bytes--
		case "offset":
			bad[0].Projection.Offset = math.MaxInt64
		case "extent":
			bad[0].F16 = dw.Slice(0, dw.Len()-1)
		}
		if err := checkCUDAV41EngramWeights(bad); err == nil {
			t.Fatal("malformed weight admitted", fault)
		}
	}
	rows := cudaEngramRows(t)
	if err := rows.Prepare(1, true, 0); err != nil {
		t.Fatal(err)
	}
	var input [2][]float32
	for i, layer := range []int{1, 14} {
		input[i], _, err = rows.Rows(layer)
		if err != nil {
			t.Fatal(err)
		}
	}
	rows.Abort()
	s, err = compileCUDAV41Engram(r, binding, rows, source)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Commit(); err == nil {
		t.Fatal("unprepared native token committed")
	}
	if err := s.Prepare(1, true, 0); err != nil {
		t.Fatal(err)
	}
	initial := make([]float32, 20480)
	for i := range initial {
		initial[i] = float32(i%9-4) / 16
	}
	if err := s.Apply(14, dq); err == nil {
		t.Fatal("unordered site admitted")
	}
	if err := s.Apply(1, dq); err == nil {
		t.Fatal("weight/residual alias admitted")
	}
	if err := s.Apply(1, dq.Slice(0, 20479)); err == nil {
		t.Fatal("short residual admitted")
	}
	checkText := func() {
		frontier := rows.Position()
		for index, layer := range []int{1, 14} {
			residual := upload(initial)
			wantKV := make([]float32, 25600)
			// The sparse fixture has exactly four nonzero F16 coefficients per
			// row. Independent float64 accumulation visits all of them explicitly.
			for row := range wantKV {
				var sum float64
				coefficients := []uint16{0x3400, 0xb000, 0x2800, 0x3000}
				if index == 1 {
					coefficients = []uint16{0x3000, 0x2800, 0xac00, 0x3400}
				}
				for j, h := range coefficients {
					col := (row*17 + j*131) % 6144
					sum += float64(quant.Float16ToFloat32(h)) * float64(quant.Float16ToFloat32(cudaQ81HalfReference(input[index][col])))
				}
				wantKV[row] = engramReferenceBF16(float32(sum))
			}
			want := engramReferenceGate(initial, wantKV, q, k)
			if err := s.Apply(layer, residual); err != nil {
				t.Fatal(err)
			}
			gotKV, got := make([]float32, 25600), make([]float32, 20480)
			if err := s.kv.CopyToHost(gotKV); err != nil {
				t.Fatal(err)
			}
			if err := residual.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i := range gotKV {
				if math.Float32bits(gotKV[i]) != math.Float32bits(wantKV[i]) {
					t.Fatalf("projection layer%d coordinate%d got%g want%g", layer, i, gotKV[i], wantKV[i])
				}
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("gate layer%d coordinate%d got%g want%g", layer, i, got[i], want[i])
				}
			}
			t.Logf("layer=%d projection_values=%d residual_values=%d projection_sha256=%x residual_sha256=%x", layer, len(gotKV), len(got), engramFloatSHA(gotKV), engramFloatSHA(got))
			if rows.Position() != frontier {
				t.Fatal("insertion published whole token")
			}
			if err := s.Apply(layer, residual); err == nil {
				t.Fatal("duplicate insertion admitted")
			}
		}
	}
	checkText()
	if err := s.Commit(); err != nil || rows.Position() != 1 {
		t.Fatal("native full token commit", err)
	}
	// Masked token must advance DEAD history but leave the residual untouched.
	if err := s.Prepare(2, false, 1); err != nil {
		t.Fatal(err)
	}
	for _, layer := range []int{1, 14} {
		residual := upload(initial)
		if err := s.Apply(layer, residual); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(initial))
		if err := residual.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, initial) {
			t.Fatal("image mask changed residual")
		}
	}
	if err := s.Commit(); err != nil || rows.Position() != 2 {
		t.Fatal("masked token commit", err)
	}
	// Read text after the committed image, so DEAD history is consumed by
	// the real device path rather than only by the host hash tests.
	if err := s.Prepare(0, true, 2); err != nil {
		t.Fatal(err)
	}
	for i, layer := range []int{1, 14} {
		input[i], _, err = rows.Rows(layer)
		if err != nil {
			t.Fatal(err)
		}
	}
	checkText()
	if err := s.Commit(); err != nil || rows.Position() != 3 {
		t.Fatal("post-image text commit", err)
	}
	// Execute real queued work, then inject a host return failure; separately
	// refuse the first drain. Neither path may publish or release its borrow.
	for _, fault := range []string{"after-enqueue", "drain"} {
		if err := s.Prepare(1, true, 3); err != nil {
			t.Fatal(err)
		}
		ops := s.operations()
		normal := ops
		cause := errors.New("injected " + fault)
		if fault == "after-enqueue" {
			ops.run = func(fn func() error) error { return errors.Join(normal.run(fn), cause) }
		} else {
			once := true
			ops.drain = func() error {
				if once {
					once = false
					return cause
				}
				return normal.drain()
			}
		}
		s.cleanup = &ops
		residual := upload(initial)
		if err := s.Apply(1, residual); !errors.Is(err, cause) {
			t.Fatal("runtime cause lost", fault, err)
		}
		if rows.Position() != 3 || s.valid || s.prepared || s.Commit() == nil {
			t.Fatal("failed device token published", fault)
		}
		if fault == "drain" && s.pending.Ptr() != residual.Ptr() {
			t.Fatal("failed drain lost borrowed residual")
		}
		s.cleanup = &normal
		if err := s.Reset(); err != nil || rows.Position() != 0 || s.pending.Ptr() != 0 {
			t.Fatal("failed-token reset", fault, err)
		}
		// Restore a three-token committed frontier without any projection
		// dependency so the second failure arm repeats exactly that contract.
		for position := 0; position < 3; position++ {
			if err := s.Prepare(0, false, position); err != nil {
				t.Fatal(err)
			}
			for _, layer := range []int{1, 14} {
				if err := s.Apply(layer, residual); err != nil {
					t.Fatal(err)
				}
			}
			if err := s.Commit(); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := s.Reset(); err != nil || rows.Position() != 0 {
		t.Fatal("native reset", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Prepare(0, true, 0); err == nil {
		t.Fatal("closed native owner reused")
	}
}

func engramMutate(t *testing.T, source, old, next string) string {
	t.Helper()
	if strings.Count(source, old) != 1 {
		t.Fatal("Engram mutant site not unique")
	}
	return strings.Replace(source, old, next, 1)
}

// Reference uses float64 norm/dot/signed-sqrt sigmoid, not the device's four
// warp accumulators or dsv41.AddEngram. Exact final BF16 bits are the gate for
// this fixture; any disagreement is retained, never waived as tolerance.
func engramReferenceGate(residual, kv, q, k []float32) []float32 {
	out := append([]float32(nil), residual...)
	for head := 0; head < 4; head++ {
		var h2, k2, dot float64
		for i := 0; i < 5120; i++ {
			j := head*5120 + i
			h, key := float64(residual[j]), float64(kv[j])
			h2 += h * h
			k2 += key * key
			dot += h * float64(q[j]) * float64(k[j]) * key
		}
		d := dot / math.Sqrt(h2/5120+1e-20) / math.Sqrt(k2/5120+1e-20) / math.Sqrt(5120)
		z := math.Copysign(math.Sqrt(math.Max(math.Abs(d), 1e-6)), d)
		gate := 1 / (1 + math.Exp(-z))
		for i := 0; i < 5120; i++ {
			j := head*5120 + i
			out[j] = engramReferenceBF16(float32(float64(residual[j]) + gate*float64(kv[20480+i])))
		}
	}
	return out
}

func engramReferenceBF16(v float32) float32 {
	bits := math.Float32bits(v)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + (bits >> 16 & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}
func engramFloatSHA(values []float32) [32]byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	return sha256.Sum256(raw)
}
