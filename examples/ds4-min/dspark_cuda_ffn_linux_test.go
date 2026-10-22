//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/lm/quant"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestDSparkFFNConstructorCleanupOwner(t *testing.T) {
	constructionErr := errors.New("construction failed")
	releaseErr := errors.New("release failed")
	t.Run("synchronize", func(t *testing.T) {
		syncs, frees := 0, 0
		s := &cudaDSparkFFN{synchronize: func() error {
			syncs++
			if syncs < 3 {
				return releaseErr
			}
			return nil
		}}
		if err := s.acquire(func() (func() error, error) { return func() error { frees++; return nil }, nil }); err != nil {
			t.Fatal(err)
		}
		err := s.finishConstruction(constructionErr)
		var owner io.Closer
		if !errors.Is(err, constructionErr) || !errors.Is(err, releaseErr) || !errors.As(err, &owner) {
			t.Fatal("sync cleanup owner lost")
		}
		if frees != 0 || !errors.Is(owner.Close(), releaseErr) || frees != 0 {
			t.Fatal("sync failure released resources")
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if err := owner.Close(); err != nil {
			t.Fatal(err)
		}
		if frees != 1 {
			t.Fatalf("free count %d", frees)
		}
	})
	for failedStep, stage := range []string{"alloc", "compile", "copy"} {
		t.Run(stage, func(t *testing.T) {
			calls := make([]int, failedStep+1)
			s := &cudaDSparkFFN{synchronize: func() error { return nil }}
			for i := range calls {
				index := i
				err := s.acquire(func() (func() error, error) {
					release := func() error {
						calls[index]++
						if index == 0 && calls[index] < 3 {
							return releaseErr
						}
						return nil
					}
					// A partial acquisition must retain its release callback,
					// not only callbacks belonging to prior successful steps.
					if i == failedStep {
						return release, constructionErr
					}
					return release, nil
				})
				if i < failedStep && err != nil {
					t.Fatal(err)
				}
				if i == failedStep && !errors.Is(err, constructionErr) {
					t.Fatal("cause lost")
				}
			}
			err := s.finishConstruction(constructionErr)
			if !errors.Is(err, constructionErr) || !errors.Is(err, releaseErr) {
				t.Fatal("construction/cleanup causes lost")
			}
			var owner io.Closer
			if !errors.As(fmt.Errorf("outer: %w", err), &owner) {
				t.Fatal("constructor cleanup owner unreachable")
			}
			if !errors.Is(owner.Close(), releaseErr) {
				t.Fatal("first retry must fail")
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			want := make([]int, len(calls))
			for i := range want {
				want[i] = 1
			}
			want[0] = 3
			if !slices.Equal(calls, want) {
				t.Fatalf("release calls %v", calls)
			}
		})
	}
	// Delete-retention mutant: it attempts cleanup but joins only the error,
	// exactly the previous constructor bug. The ownership gate must reject it.
	s := &cudaDSparkFFN{synchronize: func() error { return nil }}
	if err := s.acquire(func() (func() error, error) { return func() error { return releaseErr }, nil }); err != nil {
		t.Fatal(err)
	}
	mutant := errors.Join(constructionErr, s.Close())
	var owner io.Closer
	if errors.As(mutant, &owner) {
		t.Fatal("delete-retention mutant unexpectedly exposed owner")
	}
	if len(backendErrorOwners(mutant)) != 0 {
		t.Fatal("delete-retention mutant gate not discriminating")
	}
}

func TestDSparkFFNModuleOwnership(t *testing.T) {
	contextErr, unloadErr := errors.New("context restore"), errors.New("unload")
	t.Run("restore after successful unload", func(t *testing.T) {
		s := &cudaDSparkFFN{module: driver.Module(17), synchronize: func() error { return nil }}
		calls, contexts := 0, 0
		release := func() error {
			return s.releaseModule(func(f func() error) error {
				contexts++
				if err := f(); err != nil {
					return err
				}
				return contextErr
			}, func(m driver.Module) error {
				calls++
				if m != 17 {
					t.Fatal("wrong handle")
				}
				return nil
			})
		}
		if err := s.acquire(func() (func() error, error) { return release, nil }); err != nil {
			t.Fatal(err)
		}
		if !errors.Is(s.Close(), contextErr) {
			t.Fatal("restore error hidden")
		}
		if s.module != 0 || calls != 1 {
			t.Fatal("successful unload retained handle")
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if calls != 1 || contexts != 1 {
			t.Fatalf("unload/context repeated %d/%d", calls, contexts)
		}
	})
	t.Run("unload failure retains handle", func(t *testing.T) {
		s := &cudaDSparkFFN{module: driver.Module(19)}
		calls := 0
		withContext := func(f func() error) error { return f() }
		unload := func(m driver.Module) error {
			calls++
			if m != 19 {
				t.Fatal("wrong handle")
			}
			if calls < 3 {
				return unloadErr
			}
			return nil
		}
		for i := 0; i < 2; i++ {
			if !errors.Is(s.releaseModule(withContext, unload), unloadErr) || s.module != 19 {
				t.Fatal("failed unload ownership lost")
			}
		}
		if err := s.releaseModule(withContext, unload); err != nil {
			t.Fatal(err)
		}
		if s.module != 0 {
			t.Fatal("completed unload retained handle")
		}
		if err := s.releaseModule(withContext, unload); err != nil {
			t.Fatal(err)
		}
		if calls != 3 {
			t.Fatal("unload repeated")
		}
	})
	t.Run("context admission failure", func(t *testing.T) {
		s := &cudaDSparkFFN{module: driver.Module(23)}
		if !errors.Is(s.releaseModule(func(func() error) error { return contextErr }, func(driver.Module) error { t.Fatal("unload outside context"); return nil }), contextErr) || s.module != 23 {
			t.Fatal("context failure ownership lost")
		}
	})
}

// Independent producer reference: D2S6 has two half scales per 128 values,
// six half sums of ORIGINAL inputs, and a shared 128-code payload.
func dsparkD2S6Reference(input []float32, cols, tokens int) []byte {
	out := make([]byte, cols/128*tokens*144)
	for token := 0; token < tokens; token++ {
		for chunk := 0; chunk < cols/128; chunk++ {
			p := out[(chunk*tokens+token)*144:]
			x := input[token*cols+chunk*128:][:128]
			for block := 0; block < 2; block++ {
				var maximum float32
				for _, v := range x[block*64 : (block+1)*64] {
					maximum = max(maximum, float32(math.Abs(float64(v))))
				}
				var inverse, scale float32
				if maximum != 0 {
					inverse = 127 / maximum
					scale = 1 / inverse
				}
				binary.LittleEndian.PutUint16(p[block*2:], cudaQ81HalfReference(scale))
				for i := block * 64; i < (block+1)*64; i++ {
					p[16+i] = byte(int8(math.Round(float64(x[i] * inverse))))
				}
			}
			for group := 0; group < 6; group++ {
				var lane [4]float32
				for i := range lane {
					v := x[group*16+i*4:]
					lane[i] = ((v[0] + v[1]) + v[2]) + v[3]
				}
				sum := (lane[0] + lane[2]) + (lane[1] + lane[3])
				binary.LittleEndian.PutUint16(p[4+group*2:], cudaQ81HalfReference(sum))
			}
		}
	}
	return out
}

func TestDSparkFFND2S6Layout(t *testing.T) {
	input := make([]float32, 5*256)
	for token := 0; token < 5; token++ {
		for i := 0; i < 256; i++ {
			input[token*256+i] = float32(token + 1)
		}
	}
	p := dsparkD2S6Reference(input, 256, 5)
	for chunk := 0; chunk < 2; chunk++ {
		for token := 0; token < 5; token++ {
			b := p[(chunk*5+token)*144:][:144]
			wantScale := cudaQ81HalfReference(float32(token+1) / 127)
			for i := 0; i < 2; i++ {
				if binary.LittleEndian.Uint16(b[i*2:]) != wantScale {
					t.Fatal("scale layout")
				}
			}
			for i := 0; i < 6; i++ {
				if binary.LittleEndian.Uint16(b[4+i*2:]) != cudaQ81HalfReference(float32((token+1)*16)) {
					t.Fatal("sum layout")
				}
			}
			for _, code := range b[16:] {
				if code != 127 {
					t.Fatal("code payload")
				}
			}
		}
	}
}

func dsparkQ2MMQReference(weights, activation []byte, cols, pairs, pair int) float64 {
	var value float64
	for group := 0; group < cols/16; group++ {
		w := weights[group/16*84:][:84]
		sub := group % 16
		a := activation[((group/8)*pairs+pair)*144:][:144]
		qbase, shift := 32*(sub/8)+16*(sub&1), ((sub/2)&3)*2
		var dot, codesum int
		for i := 0; i < 16; i++ {
			code := int(int8(a[16+(group%8)*16+i]))
			dot += int((w[16+qbase+i]>>shift)&3) * code
			codesum += code
		}
		d := quant.Float16ToFloat32(binary.LittleEndian.Uint16(w[80:]))
		m := quant.Float16ToFloat32(binary.LittleEndian.Uint16(w[82:]))
		wd := quant.Float16ToFloat32(cudaQ81HalfReference(d * float32(w[sub]&15)))
		wm := quant.Float16ToFloat32(cudaQ81HalfReference(m * float32(w[sub]>>4)))
		scale := float64(quant.Float16ToFloat32(binary.LittleEndian.Uint16(a[(group%8)/4*2:])))
		sum := scale * float64(codesum)
		if group%8 < 6 {
			sum = float64(quant.Float16ToFloat32(binary.LittleEndian.Uint16(a[4+(group%8)*2:])))
		}
		value += float64(wd)*float64(dot)*scale - float64(wm)*sum
	}
	return value
}

func TestDSparkFFND2S6ConsumerMutations(t *testing.T) {
	x := make([]float32, 256)
	for i := range x {
		x[i] = 127
	}
	a := dsparkD2S6Reference(x, 256, 1)
	w := make([]byte, 84)
	for i := 0; i < 16; i++ {
		w[i] = 0x21
	}
	for i := 16; i < 80; i++ {
		w[i] = 255
	}
	binary.LittleEndian.PutUint16(w[80:], 0x3c00)
	binary.LittleEndian.PutUint16(w[82:], 0x3800)
	const want = 65024
	if got := dsparkQ2MMQReference(w, a, 256, 1, 0); got != want {
		t.Fatalf("got %g want %g", got, float64(want))
	}
	for _, name := range []string{"scale", "original sum", "codes", "layout"} {
		mutant := append([]byte(nil), a...)
		switch name {
		case "scale":
			binary.LittleEndian.PutUint16(mutant, 0x4000)
		case "original sum":
			binary.LittleEndian.PutUint16(mutant[4:], 0)
		case "codes":
			mutant[16] = 0
		case "layout":
			copy(mutant[:16], mutant[16:32])
		}
		if got := dsparkQ2MMQReference(w, mutant, 256, 1, 0); got == want {
			t.Fatalf("%s mutation missed", name)
		}
	}
}

func dsparkIQ2D4Reference(t *testing.T, weights, activation []byte, ids []int32, cols, rows, tokens, slots int) []float64 {
	t.Helper()
	out := make([]float64, tokens*slots*rows)
	for pair, id := range ids {
		for row := 0; row < rows; row++ {
			at := (int(id)*rows + row) * (cols / 256) * 66
			w, err := quant.DequantizeIQ2_XXS(weights[at:at+(cols/256)*66], cols)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range w {
				p := activation[((i/128)*tokens+pair/slots)*144:][:144]
				d := math.Float32frombits(binary.LittleEndian.Uint32(p[(i%128)/32*4:]))
				out[pair*rows+row] += float64(v) * float64(d) * float64(int8(p[16+i%128]))
			}
		}
	}
	return out
}

func dsparkFFNValueGate(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatal("value count")
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[i]) > 1e-4+1e-4*math.Abs(want[i]) {
			t.Fatalf("value %d got %g want %g", i, v, want[i])
		}
	}
}

func dsparkFFNModule(t *testing.T, r *cudaFoundation, source string) func(string) driver.Function {
	t.Helper()
	opts := append(cudaMathHeaderOptions(), nvrtcx.WithName("dspark-ffn-native-gate.cu"), nvrtcx.WithFlag("--gpu-architecture=sm_121"))
	program, err := nvrtcx.Compile(cudaMathCompilationSource()+source, opts...)
	if err != nil {
		t.Fatal(err)
	}
	image := program.CUBIN()
	if len(image) == 0 {
		t.Fatal("empty CUBIN")
	}
	module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.device.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	})
	t.Logf("source SHA256=%x headers SHA256=%x CUBIN SHA256=%x", sha256.Sum256([]byte(source)), sha256.Sum256([]byte(cudaMathSource)), sha256.Sum256(image))
	return func(name string) driver.Function {
		var fn driver.Function
		if err := r.device.WithContext(func() error { var e error; fn, e = module.Function(name); return e }); err != nil {
			t.Fatal(err)
		}
		return fn
	}
}

// Five independent rows use the NEW D4 IQ2/D2S6 Q2 production entries. The
// older selected-Q8_1 test below is a distinct alternate, never this gate.
func TestCUDADSparkFFNRoutedMMQValueAndMutants(t *testing.T) {
	r := cudaDSparkFFNGateFoundation(t)
	const cols, rows, outRows, tokens, slots, experts = 256, 256, 17, 5, 6, 256
	bind := dsparkFFNModule(t, r, cudaDSparkFFNSource)
	x := make([]float32, cols*tokens)
	for i := range x {
		x[i] = float32((i*37)%251-125) / 128
	}
	in := cudaSynthUpload(t, r, x)
	q := cudaExpertAllocBytes(t, r, cols/128*tokens*144)
	cudaQualificationMeasure(t, r, func() error { return r.encodeQuantMMQ(in, q, cols, cols, tokens) })
	codes := make([]byte, q.Len())
	if err := q.CopyToHost(codes); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(codes, dsparkMMQProducerReference(x, cols, tokens)) {
		t.Fatal("D4 producer bytes")
	}
	if bytes.Equal(codes[:144], codes[144:288]) {
		t.Fatal("constant D4 fixture")
	}
	logits := make([]float32, tokens*experts)
	bias := make([]float32, experts)
	for i := range logits {
		logits[i] = float32((i*13)%101-50) / 20
	}
	for rank, id := range []int{17, 229, 7, 128, 0, 255} {
		bias[id] = 1000 - float32(rank*20)
	}
	biasBytes := make([]byte, experts*4)
	for i, v := range bias {
		binary.LittleEndian.PutUint32(biasBytes[i*4:], math.Float32bits(v))
	}
	dlogits := cudaSynthUpload(t, r, logits)
	dbias := cudaSynthTensor(t, r, "router bias", 0, []uint64{experts}, biasBytes)
	ids, err := cudamem.Alloc[int32](r.device, tokens*slots)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ids.Free(); err != nil {
			t.Error(err)
		}
	})
	weights := cudaTestAlloc(t, r, tokens*slots)
	scores := cudaTestAlloc(t, r, experts)
	wantIDs := make([]int32, tokens*slots)
	wantW := make([]float32, tokens*slots)
	for token := 0; token < tokens; token++ {
		selected, w, err := dsparkSelect(logits[token*experts:(token+1)*experts], bias, slots)
		if err != nil {
			t.Fatal(err)
		}
		for i, id := range selected {
			wantIDs[token*slots+i] = int32(id)
		}
		copy(wantW[token*slots:], w)
		cudaQualificationMeasure(t, r, func() error {
			return r.encodeRouterBiased(dlogits.Slice(token*experts, (token+1)*experts), dbias, scores, ids.Slice(token*slots, (token+1)*slots), weights.Slice(token*slots, (token+1)*slots))
		})
	}
	actualIDs := make([]int32, len(wantIDs))
	if err := ids.CopyToHost(actualIDs); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(actualIDs, wantIDs) {
		t.Fatalf("IDs %v want %v", actualIDs, wantIDs)
	}
	if err := cudaQualificationClose(cudaTestRead(t, weights), wantW, 1e-5, 1e-5); err != nil {
		t.Fatal(err)
	}
	g, u, d := cudaExpertFixture(16, cols, rows, experts, 17), cudaExpertFixture(16, cols, rows, experts, 29), cudaExpertFixture(10, rows, outRows, experts, 43)
	gate := cudaSynthTensor(t, r, "native IQ2 gate", 16, []uint64{cols, rows, experts}, g)
	up := cudaSynthTensor(t, r, "native IQ2 up", 16, []uint64{cols, rows, experts}, u)
	down := cudaSynthTensor(t, r, "native Q2 down", 10, []uint64{rows, outRows, experts}, d)
	goOut, upOut, mid := cudaTestAlloc(t, r, tokens*slots*rows), cudaTestAlloc(t, r, tokens*slots*rows), cudaTestAlloc(t, r, tokens*slots*rows)
	launchIQ := func(fn driver.Function) error {
		return cudalaunch.Kernel(fn).Grid(tokens*slots*rows, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(gate.device, up.device, ids, q, goOut, upOut, uint32(cols), uint32(rows), uint32(tokens), uint32(slots), uint32(experts))
	}
	cudaQualificationMeasure(t, r, func() error { return launchIQ(bind("ds4_dspark_ffn_iq2_mmq")) })
	wantG, wantU := dsparkIQ2D4Reference(t, g, codes, actualIDs, cols, rows, tokens, slots), dsparkIQ2D4Reference(t, u, codes, actualIDs, cols, rows, tokens, slots)
	dsparkFFNValueGate(t, cudaTestRead(t, goOut), wantG)
	dsparkFFNValueGate(t, cudaTestRead(t, upOut), wantU)
	actualG, actualU, actualW := cudaTestRead(t, goOut), cudaTestRead(t, upOut), cudaTestRead(t, weights)
	cudaQualificationMeasure(t, r, func() error {
		return cudalaunch.Kernel(bind("ds4_dspark_ffn_weighted_mid")).Grid(tokens*slots*rows/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(goOut, upOut, weights, mid, uint32(rows), uint32(tokens*slots))
	})
	wantMid := make([]float32, mid.Len())
	for pair := 0; pair < tokens*slots; pair++ {
		v, err := dsparkSwiGLU(actualG[pair*rows:(pair+1)*rows], actualU[pair*rows:(pair+1)*rows], actualW[pair])
		if err != nil {
			t.Fatal(err)
		}
		copy(wantMid[pair*rows:], v)
	}
	if err := cudaQualificationClose(cudaTestRead(t, mid), wantMid, 1e-4, 1e-4); err != nil {
		t.Fatal(err)
	}
	qm := cudaExpertAllocBytes(t, r, rows/128*tokens*slots*144)
	cudaQualificationMeasure(t, r, func() error {
		return cudalaunch.Kernel(bind("ds4_dspark_ffn_quant_d2s6")).Grid(rows/128*tokens*slots, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(mid, qm, uint32(rows), uint32(tokens*slots))
	})
	mcodes := make([]byte, qm.Len())
	if err := qm.CopyToHost(mcodes); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mcodes, dsparkD2S6Reference(cudaTestRead(t, mid), rows, tokens*slots)) {
		t.Fatal("D2S6 producer bytes")
	}
	for _, mutation := range []struct{ old, new string }{
		{"q.ds[2+lane/4] = __float2half_rn(sum);", "q.ds[2+lane/4] = __float2half_rn(sum + 1.0f);"},
		{"q.ds[lane/16] = __float2half_rn(maximum == 0.0f ? 0.0f : 1.0f / inverse);", "q.ds[lane/16] = __float2half_rn(1.0f);"},
		{"q.qs[lane*4] = roundf(v.x * inverse);", "q.qs[lane*4] = roundf(v.x * inverse) + 1;"},
	} {
		source := strings.Replace(cudaDSparkFFNSource, mutation.old, mutation.new, 1)
		if source == cudaDSparkFFNSource {
			t.Fatal("producer mutant missing")
		}
		mutant := dsparkFFNModule(t, r, source)
		changed := cudaExpertAllocBytes(t, r, qm.Len())
		cudaQualificationMeasure(t, r, func() error {
			return cudalaunch.Kernel(mutant("ds4_dspark_ffn_quant_d2s6")).Grid(rows/128*tokens*slots, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(mid, changed, uint32(rows), uint32(tokens*slots))
		})
		got := make([]byte, changed.Len())
		if err := changed.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if bytes.Equal(got, mcodes) {
			t.Fatal("producer byte gate missed same-site mutant")
		}
	}
	dout := cudaTestAlloc(t, r, tokens*slots*outRows)
	launchQ2 := func(fn driver.Function) error {
		return cudalaunch.Kernel(fn).Grid(tokens*slots*outRows, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(down.device, ids, qm, dout, uint32(rows), uint32(outRows), uint32(tokens*slots), uint32(experts))
	}
	cudaQualificationMeasure(t, r, func() error { return launchQ2(bind("ds4_dspark_ffn_q2_mmq")) })
	wantDown := make([]float64, dout.Len())
	for pair, id := range actualIDs {
		for row := 0; row < outRows; row++ {
			at := (int(id)*outRows + row) * (rows / 256) * 84
			wantDown[pair*outRows+row] = dsparkQ2MMQReference(d[at:], mcodes, rows, tokens*slots, pair)
		}
	}
	dsparkFFNValueGate(t, cudaTestRead(t, dout), wantDown)
	// Independent extremes distinguish upper-only gate clamp from a symmetric
	// clamp, and cover all five rows and all actual selected slot weights.
	gs, us := make([]float32, tokens*slots*rows), make([]float32, tokens*slots*rows)
	for i := range gs {
		gs[i] = []float32{-100, -20, -10, -1, 0, 10, 20, 100}[i%8]
		us[i] = []float32{-100, -20, -10, -1, 0, 10, 20, 100}[(i/8)%8]
	}
	dg, du, clamped := cudaSynthUpload(t, r, gs), cudaSynthUpload(t, r, us), cudaTestAlloc(t, r, len(gs))
	cudaQualificationMeasure(t, r, func() error {
		return cudalaunch.Kernel(bind("ds4_dspark_ffn_weighted_mid")).Grid(tokens*slots*rows/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(dg, du, weights, clamped, uint32(rows), uint32(tokens*slots))
	})
	wantClamp := make([]float32, len(gs))
	for pair := 0; pair < tokens*slots; pair++ {
		v, err := dsparkSwiGLU(gs[pair*rows:(pair+1)*rows], us[pair*rows:(pair+1)*rows], actualW[pair])
		if err != nil {
			t.Fatal(err)
		}
		copy(wantClamp[pair*rows:], v)
	}
	if err := cudaQualificationClose(cudaTestRead(t, clamped), wantClamp, 1e-4, 1e-4); err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []struct {
		old, new string
		iq       bool
	}{{"gate_out[blockIdx.x] = g;", "gate_out[blockIdx.x] = g + 1.0f;", true}, {"out[blockIdx.x] = value;", "out[blockIdx.x] = value + 1.0f;", false}} {
		source := strings.Replace(cudaDSparkFFNSource, mutation.old, mutation.new, 1)
		if source == cudaDSparkFFNSource {
			t.Fatal("same-site mutant missing")
		}
		mutant := dsparkFFNModule(t, r, source)
		var got []float32
		var want []float64
		if mutation.iq {
			cudaQualificationMeasure(t, r, func() error { return launchIQ(mutant("ds4_dspark_ffn_iq2_mmq")) })
			got, want = cudaTestRead(t, goOut), wantG
			dsparkFFNValueGate(t, cudaTestRead(t, upOut), wantU)
		} else {
			cudaQualificationMeasure(t, r, func() error { return launchQ2(mutant("ds4_dspark_ffn_q2_mmq")) })
			got, want = cudaTestRead(t, dout), wantDown
		}
		bad := 0
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("mutant nonfinite")
			}
			if math.Abs(float64(v)-want[i]) > 1e-4+1e-4*math.Abs(want[i]) {
				bad++
			}
		}
		if bad != len(want) {
			t.Fatalf("same-site mutant %d/%d", bad, len(want))
		}
	}
	t.Logf("D4 SHA256=%x D2S6 SHA256=%x IDs=%v", sha256.Sum256(codes), sha256.Sum256(mcodes), actualIDs)
}

// This exercises the selected branch on nontrivial actual router IDs. The
// source reference uses the independently golden-tested quant decoders and
// actual emitted Q8_1 codes, not the original floating activation vector.
func TestCUDADSparkFFNSelectedValueAndMutant(t *testing.T) {
	r := cudaDSparkFFNGateFoundation(t)
	const cols, midRows, outRows, experts, slots = 256, 256, 17, 256, 6
	logits, bias := make([]float32, experts), make([]float32, experts)
	for i := range logits {
		logits[i] = float32(i%19-9) / 2
	}
	for slot, id := range []int{17, 229, 7, 128, 0, 255} {
		bias[id] = float32(100 - slot*2)
	}
	wantIDs, wantW, err := dsparkSelect(logits, bias, slots)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := cudamem.Alloc[int32](r.device, slots)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ids.Free(); err != nil {
			t.Error(err)
		}
	})
	bs := make([]byte, experts*4)
	for i, v := range bias {
		binary.LittleEndian.PutUint32(bs[i*4:], math.Float32bits(v))
	}
	dBias := cudaSynthTensor(t, r, "support biased router", 0, []uint64{experts}, bs)
	dLogits, dScores, dW := cudaSynthUpload(t, r, logits), cudaTestAlloc(t, r, experts), cudaTestAlloc(t, r, slots)
	cudaQualificationMeasure(t, r, func() error { return r.encodeRouterBiased(dLogits, dBias, dScores, ids, dW) })
	actualIDs := make([]int32, slots)
	if err := ids.CopyToHost(actualIDs); err != nil {
		t.Fatal(err)
	}
	expectedIDs := make([]int32, slots)
	for i, id := range wantIDs {
		expectedIDs[i] = int32(id)
	}
	if !slices.Equal(actualIDs, expectedIDs) {
		t.Fatalf("biased selection=%v want%v", actualIDs, expectedIDs)
	}
	if err := cudaQualificationClose(cudaTestRead(t, dW), wantW, 1e-5, 1e-5); err != nil {
		t.Fatalf("unbiased normalized weights: %v", err)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i*37)%251-125) / 128
	}
	q := cudaExpertQ81Upload(t, r, x, cols)
	actualQ := make([]byte, q.Len())
	if err := q.CopyToHost(actualQ); err != nil {
		t.Fatal(err)
	}
	g, u, d := cudaExpertFixture(16, cols, midRows, experts, 17), cudaExpertFixture(16, cols, midRows, experts, 29), cudaExpertFixture(10, midRows, outRows, experts, 43)
	gate := cudaSynthTensor(t, r, "support IQ2 gate", 16, []uint64{cols, midRows, experts}, g)
	up := cudaSynthTensor(t, r, "support IQ2 up", 16, []uint64{cols, midRows, experts}, u)
	down := cudaSynthTensor(t, r, "support Q2 down", 10, []uint64{midRows, outRows, experts}, d)
	gout, uout := cudaTestAlloc(t, r, slots*midRows), cudaTestAlloc(t, r, slots*midRows)
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecIQ2SelectedQ81(gate, up, ids, q, gout, uout) })
	for _, o := range []struct {
		name string
		raw  []byte
		out  cudamem.Slice[float32]
	}{{"gate", g, gout}, {"up", u, uout}} {
		want := cudaExpertIQ2Q81Reference(t, o.raw, actualQ, cols, midRows, actualIDs)
		if err := cudaQualificationClose(cudaTestRead(t, o.out), want, 1e-4, 1e-4); err != nil {
			t.Fatalf("%s: %v", o.name, err)
		}
	}
	// Independent clamp extremes also make negative-gate versus symmetric
	// clamping distinguishable and exercise all six nonconstant weights.
	gs, us := make([]float32, slots*midRows), make([]float32, slots*midRows)
	for i := range gs {
		gs[i] = []float32{-20, -10, -1, 0, 10, 20, 100}[i%7]
		us[i] = []float32{-100, -20, -10, -1, 0, 10, 20}[i%7]
	}
	dg, du, mid := cudaSynthUpload(t, r, gs), cudaSynthUpload(t, r, us), cudaTestAlloc(t, r, len(gs))
	cudaQualificationMeasure(t, r, func() error { return r.encodeSwiGLUSlots(dg, du, dW, mid, slots, 1) })
	actualMid := cudaTestRead(t, mid)
	actualW := cudaTestRead(t, dW)
	wantMid := make([]float32, len(gs))
	for slot := 0; slot < slots; slot++ {
		v, e := dsparkSwiGLU(gs[slot*midRows:(slot+1)*midRows], us[slot*midRows:(slot+1)*midRows], actualW[slot])
		if e != nil {
			t.Fatal(e)
		}
		copy(wantMid[slot*midRows:], v)
	}
	if err := cudaQualificationClose(actualMid, wantMid, 1e-4, 1e-4); err != nil {
		t.Fatalf("clamped weighted SwiGLU: %v", err)
	}
	qMid := cudaExpertAllocBytes(t, r, len(gs)/32*36)
	cudaQualificationMeasure(t, r, func() error { return r.encodeSwiGLUQuantQ81(dg, du, dW, qMid, midRows, slots) })
	actualQM := make([]byte, qMid.Len())
	if err := qMid.CopyToHost(actualQM); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(actualQM, cudaQ81ActivationReference(actualMid)) {
		t.Fatal("fused weighted quantizer differs from independently gated materialized activation")
	}
	dout := cudaTestAlloc(t, r, slots*outRows)
	cudaQualificationMeasure(t, r, func() error { return r.encodeMatvecQ2KSelectedQ81(down, ids, qMid, dout) })
	wantDown := cudaExpertReference(t, d, 10, midRows, outRows, actualIDs, cudaExpertQ81Decode(actualQM), true)
	if err := cudaQualificationClose(cudaTestRead(t, dout), wantDown, 1e-4, 1e-4); err != nil {
		t.Fatalf("selected Q2 down: %v", err)
	}
	t.Logf("selected IDs=%v weights=%v Q8_1 SHA256=%x weighted Q8_1 SHA256=%x", actualIDs, actualW, sha256.Sum256(actualQ), sha256.Sum256(actualQM))
	// Compile the same actual production source with one selected IQ2 write
	// perturbed. Do not use a different neighboring kernel as the control.
	source := cudaExpertQ81Source
	mutant := strings.Replace(source, "if (lane == 0) { gate_out[output] = g; up_out[output] = u; }", "if (lane == 0) { gate_out[output] = g + 1.0f; up_out[output] = u; }", 1)
	if mutant == source {
		t.Fatal("selected same-site mutation did not apply")
	}
	opts := append(cudaMathCompileOptions(12, 1), nvrtcx.WithHeader(cudaSourceRoot+"cuda_experts_q81.cu", mutant))
	program, err := nvrtcx.Compile(cudaMathCompilationSource(), opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(program.CUBIN()) == 0 {
		t.Fatal("selected mutant emitted no CUBIN")
	}
	t.Logf("selected production headers SHA256=%x selected header SHA256=%x mutant header SHA256=%x mutant CUBIN SHA256=%x", sha256.Sum256([]byte(cudaMathSource)), sha256.Sum256([]byte(source)), sha256.Sum256([]byte(mutant)), sha256.Sum256(program.CUBIN()))
	module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.device.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	}()
	var fn driver.Function
	if err := r.device.WithContext(func() error { var e error; fn, e = module.Function("ds4_iq2_xxs_pair_selected_q8_1"); return e }); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error {
		return cudalaunch.Kernel(fn).Grid(slots*midRows/2, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(gate.device, up.device, ids, q, gout, uout, uint32(cols), uint32(midRows), uint32(slots), uint32(experts))
	})
	wantGate := cudaExpertIQ2Q81Reference(t, g, actualQ, cols, midRows, actualIDs)
	changed := cudaTestRead(t, gout)
	bad := 0
	for i, v := range changed {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-float64(wantGate[i])) > 1e-4+1e-4*math.Abs(float64(wantGate[i])) {
			bad++
		}
	}
	if bad != len(wantGate) {
		t.Fatalf("selected IQ2 mutant violations=%d/%d", bad, len(wantGate))
	}
	if err := cudaQualificationClose(cudaTestRead(t, uout), cudaExpertIQ2Q81Reference(t, u, actualQ, cols, midRows, actualIDs), 1e-4, 1e-4); err != nil {
		t.Fatalf("IQ2 adjacent up output changed: %v", err)
	}
	q2mutant := strings.Replace(source, "if (lane == 0) out[output] = sum;", "if (lane == 0) out[output] = sum + 1.0f;", 1)
	if q2mutant == source {
		t.Fatal("Q2 same-site mutation did not apply")
	}
	q2opts := append(cudaMathCompileOptions(12, 1), nvrtcx.WithHeader(cudaSourceRoot+"cuda_experts_q81.cu", q2mutant))
	q2program, err := nvrtcx.Compile(cudaMathCompilationSource(), q2opts...)
	if err != nil {
		t.Fatal(err)
	}
	if len(q2program.CUBIN()) == 0 {
		t.Fatal("Q2 mutant emitted no CUBIN")
	}
	t.Logf("Q2 mutant header SHA256=%x CUBIN SHA256=%x", sha256.Sum256([]byte(q2mutant)), sha256.Sum256(q2program.CUBIN()))
	q2module, _, err := q2program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.device.WithContext(q2module.Close); err != nil {
			t.Error(err)
		}
	}()
	var q2fn driver.Function
	if err := r.device.WithContext(func() error { var e error; q2fn, e = q2module.Function("ds4_q2_k_selected_q8_1"); return e }); err != nil {
		t.Fatal(err)
	}
	cudaQualificationMeasure(t, r, func() error {
		return cudalaunch.Kernel(q2fn).Grid((slots*outRows+1)/2, 1, 1).Block(64, 1, 1).On(r.stream()).Launch(down.device, ids, qMid, dout, uint32(midRows), uint32(outRows), uint32(slots), uint32(experts))
	})
	bad = 0
	for i, v := range cudaTestRead(t, dout) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-float64(wantDown[i])) > 1e-4+1e-4*math.Abs(float64(wantDown[i])) {
			bad++
		}
	}
	if bad != len(wantDown) {
		t.Fatalf("selected Q2 mutant violations=%d/%d", bad, len(wantDown))
	}
	t.Logf("selected same-site mutant violations IQ2=%d/%d Q2=%d/%d", len(wantGate), len(wantGate), bad, len(wantDown))
}

// The reference decodes actual device-emitted float32 MMQ scales/codes,
// rather than pretending the input float vector survives quantization.
func TestCUDADSparkFFNSharedMMQValueAndMutant(t *testing.T) {
	r := cudaDSparkFFNGateFoundation(t)
	const cols, rows, tokens = 256, 3, 5
	x := make([]float32, cols*tokens)
	for i := range x {
		x[i] = float32(i%73-36) / 13
	}
	weights := make([]byte, rows*(cols/32)*34)
	for row := 0; row < rows; row++ {
		for b := 0; b < cols/32; b++ {
			p := weights[(row*(cols/32)+b)*34:]
			binary.LittleEndian.PutUint16(p, uint16(0x2400+row*37+b*19))
			for i := 0; i < 32; i++ {
				p[2+i] = byte(int8((row*7+b*11+i*3)%31 - 15))
			}
		}
	}
	in, wt := cudaSynthUpload(t, r, x), cudaExpertAllocBytes(t, r, len(weights))
	if err := wt.CopyFromHost(weights); err != nil {
		t.Fatal(err)
	}
	q := cudaExpertAllocBytes(t, r, (cols/128)*tokens*144)
	cudaQualificationMeasure(t, r, func() error { return r.encodeQuantMMQ(in, q, cols, cols, tokens) })
	codes := make([]byte, q.Len())
	if err := q.CopyToHost(codes); err != nil {
		t.Fatal(err)
	}
	producer := dsparkMMQProducerReference(x, cols, tokens)
	if !bytes.Equal(codes, producer) {
		t.Fatal("MMQ producer differs from independent scalar source reference")
	}
	nonzero := 0
	for _, b := range codes {
		if b != 0 {
			nonzero++
		}
	}
	if nonzero == 0 || bytes.Equal(codes[:144], codes[144:288]) {
		t.Fatal("fixture emitted zero or constant MMQ rows")
	}
	producerSource, err := cudaSourceFS.ReadFile("cuda_moe_d2r.cu")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("MMQ producer source SHA256=%x emitted SHA256=%x", sha256.Sum256(producerSource), sha256.Sum256(codes))
	want := make([]float64, rows*tokens)
	for token := 0; token < tokens; token++ {
		for row := 0; row < rows; row++ {
			for b := 0; b < cols/32; b++ {
				p := weights[(row*(cols/32)+b)*34:]
				a := codes[((b/4)*tokens+token)*144:]
				d := math.Float32frombits(binary.LittleEndian.Uint32(a[(b%4)*4:]))
				if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
					t.Fatal("nonfinite emitted MMQ scale")
				}
				var dot int32
				for i := 0; i < 32; i++ {
					dot += int32(int8(p[2+i])) * int32(int8(a[16+(b%4)*32+i]))
				}
				want[token*rows+row] += float64(quant.Float16ToFloat32(binary.LittleEndian.Uint16(p))) * float64(d) * float64(dot)
			}
		}
	}
	out := cudaSynthUpload(t, r, make([]float32, len(want)))
	for _, mutant := range []bool{false, true} {
		source := cudaDSparkFFNSource
		if mutant {
			source = strings.Replace(source, "out[token * rows + row] = value;", "out[token * rows + row] = value + 1.0f;", 1)
			if source == cudaDSparkFFNSource {
				t.Fatal("mutant did not apply")
			}
		}
		program, err := nvrtcx.Compile(cudaMathCompilationSource()+source, append(cudaMathHeaderOptions(), nvrtcx.WithName("dspark_ffn_mmq_gate.cu"), nvrtcx.WithDeviceArch(r.device))...)
		if err != nil {
			t.Fatal(err)
		}
		if len(program.CUBIN()) == 0 {
			t.Fatal("sm_121 compiler emitted no CUBIN")
		}
		t.Logf("mutant=%v source SHA256=%x CUBIN SHA256=%x", mutant, sha256.Sum256([]byte(source)), sha256.Sum256(program.CUBIN()))
		module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
		if err != nil {
			t.Fatal(err)
		}
		var fn driver.Function
		if err := r.device.WithContext(func() error { var e error; fn, e = module.Function("ds4_dspark_ffn_q8_mmq"); return e }); err != nil {
			t.Fatal(err)
		}
		cudaQualificationMeasure(t, r, func() error {
			return cudalaunch.Kernel(fn).Grid(rows*tokens, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(wt, q, out, uint32(cols), uint32(rows), uint32(tokens))
		})
		got := make([]float32, len(want))
		if err := out.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[i]) > 1e-4+1e-4*math.Abs(want[i]) {
				bad++
			}
		}
		if (!mutant && bad != 0) || (mutant && bad != len(want)) {
			t.Fatalf("mutant=%v mismatches=%d/%d", mutant, bad, len(want))
		}
		if err := r.device.WithContext(module.Close); err != nil {
			t.Fatal(err)
		}
	}
}

// The source producer uses a maximum over each 32-value group, float32
// reciprocal scale, half-away code rounding, and float32 reciprocal storage.
// This reference does not call a device quantizer or decode its output.
func dsparkMMQProducerReference(x []float32, cols, tokens int) []byte {
	out := make([]byte, cols/128*tokens*144)
	for token := 0; token < tokens; token++ {
		for b := 0; b < cols/32; b++ {
			var amax float32
			for i := 0; i < 32; i++ {
				amax = max(amax, float32(math.Abs(float64(x[token*cols+b*32+i]))))
			}
			var inverse, scale float32
			if amax != 0 {
				inverse = 127 / amax
				scale = 1 / inverse
			}
			p := out[((b/4)*tokens+token)*144:]
			binary.LittleEndian.PutUint32(p[(b%4)*4:], math.Float32bits(scale))
			for i := 0; i < 32; i++ {
				v := float32(x[token*cols+b*32+i] * inverse)
				p[16+(b%4)*32+i] = byte(int8(math.Round(float64(v))))
			}
		}
	}
	return out
}

func TestDSparkFFNOwnershipFailures(t *testing.T) {
	failure := errors.New("injected resource failure")
	for fail := 0; fail < 4; fail++ {
		s := &cudaDSparkFFN{synchronize: func() error { return nil }}
		acquired, freed := 0, 0
		for n := 0; n < 3; n++ {
			err := s.acquire(func() (func() error, error) {
				if n == fail {
					return nil, failure
				}
				acquired++
				return func() error { freed++; return nil }, nil
			})
			if err != nil {
				if !errors.Is(err, failure) {
					t.Fatal(err)
				}
				break
			}
		}
		if err := s.close(); err != nil {
			t.Fatal(err)
		}
		if freed != acquired {
			t.Fatalf("failure index %d leaked: acquired=%d freed=%d", fail, acquired, freed)
		}
		if err := s.close(); err != nil || freed != acquired {
			t.Fatalf("double close retried successful frees: %v", err)
		}
	}
	freeA, freeB, syncs := 0, 0, 0
	s := &cudaDSparkFFN{synchronize: func() error {
		syncs++
		if syncs == 1 {
			return failure
		}
		return nil
	}}
	if err := s.acquire(func() (func() error, error) { return func() error { freeA++; return nil }, nil }); err != nil {
		t.Fatal(err)
	}
	if err := s.acquire(func() (func() error, error) {
		return func() error {
			freeB++
			if freeB == 1 {
				return failure
			}
			return nil
		}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := s.close(); !errors.Is(err, failure) || freeA != 0 || freeB != 0 {
		t.Fatal("synchronization failure freed live memory")
	}
	if err := s.close(); !errors.Is(err, failure) || freeA != 1 || freeB != 1 {
		t.Fatal("partial release lost ownership")
	}
	if err := s.close(); err != nil || freeA != 1 || freeB != 2 {
		t.Fatal("partial release retried successful free or lost failed free")
	}
	if err := s.acquire(func() (func() error, error) { t.Fatal("closed owner attempted acquisition"); return nil, nil }); err == nil {
		t.Fatal("closed owner admitted acquisition")
	}
}

func TestDSparkFFNPartialEnqueue(t *testing.T) {
	failure := errors.New("injected enqueue failure")
	s := &cudaDSparkFFN{}
	written := 0
	if err := s.enqueue(func() error { written++; return failure }); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if err := s.enqueue(func() error { written++; return nil }); err == nil || written != 1 {
		t.Fatal("partial output was reusable after failed enqueue")
	}
	if err := (&cudaDSparkFFN{closed: true}).enqueue(func() error { t.Fatal("closed owner enqueued"); return nil }); err == nil {
		t.Fatal("closed owner admitted enqueue")
	}
}

func TestDSparkFFNDeviceOverlap(t *testing.T) {
	for _, tt := range []struct {
		name string
		a, b driver.DevicePtr
		size uint64
		want bool
	}{
		{"same", 1024, 1024, 64, true},
		{"partial", 1024, 1056, 64, true},
		{"reverse", 1056, 1024, 64, true},
		{"adjacent", 1024, 1088, 64, false},
		{"disjoint", 1024, 4096, 64, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := dsparkDeviceOverlap(tt.a, tt.b, tt.size); got != tt.want {
				t.Fatalf("overlap = %v, want %v", got, tt.want)
			}
		})
	}
	if !dsparkFFNRangeOverlap(1024, 64, 1056, 1024) || !dsparkFFNRangeOverlap(1056, 1024, 1024, 64) || dsparkFFNRangeOverlap(1024, 0, 1024, 64) || dsparkFFNRangeOverlap(1024, 64, 1088, 1024) {
		t.Fatal("unequal scratch extent overlap classification")
	}
}

func TestDSparkFFNMMQProducerReference(t *testing.T) {
	const cols, tokens = 128, 5
	x := make([]float32, cols*tokens)
	for token := 0; token < tokens; token++ {
		for block := 0; block < 4; block++ {
			p := x[token*cols+block*32:]
			p[0], p[1], p[2], p[3], p[4], p[5] = 127, -127, -1.5, -0.5, 0.5, 1.5
			p[6] = float32(token + block)
		}
	}
	raw := dsparkMMQProducerReference(x, cols, tokens)
	for token := 0; token < tokens; token++ {
		for block := 0; block < 4; block++ {
			p := raw[token*144:]
			if binary.LittleEndian.Uint32(p[block*4:]) != math.Float32bits(1) {
				t.Fatal("scale fixture did not produce exact 1")
			}
			want := []byte{127, 129, 254, 255, 1, 2, byte(token + block)}
			if !bytes.Equal(p[16+block*32:16+block*32+7], want) {
				t.Fatalf("rounding/layout token=%d block=%d", token, block)
			}
		}
	}
}

// This gate covers the new ordered sum site only. Existing selected-Q8_1
// projection qualification tests remain separately required; this is not a
// complete FFN or actual support-model qualification.
func TestCUDADSparkFFNSumValueAndMutant(t *testing.T) {
	r := cudaDSparkFFNGateFoundation(t)
	const width = 257 // Exercise the last-block predicate as well as six slots.
	x, shared := make([]float32, 6*width), make([]float32, width)
	want := make([]float64, width)
	for i := range shared {
		shared[i] = float32(i%19-9) / 16
		want[i] = float64(shared[i])
		for slot := 0; slot < 6; slot++ {
			v := float32((i*13+slot*7)%31-15) / 8
			x[slot*width+i] = v
			want[i] += float64(v)
		}
	}
	in, sh := cudaSynthUpload(t, r, x), cudaSynthUpload(t, r, shared)
	out := cudaSynthUpload(t, r, make([]float32, width))
	for _, mutant := range []bool{false, true} {
		source := cudaDSparkFFNSource
		if mutant {
			source = strings.Replace(source, "out[i] = v + shared[i];", "out[i] = v + shared[i] + 1.0f;", 1)
			if source == cudaDSparkFFNSource {
				t.Fatal("mutant did not apply")
			}
		}
		program, err := nvrtcx.Compile(cudaMathCompilationSource()+source, append(cudaMathHeaderOptions(), nvrtcx.WithName("dspark_ffn_sum_gate.cu"), nvrtcx.WithDeviceArch(r.device))...)
		if err != nil {
			t.Fatal(err)
		}
		module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
		if err != nil {
			t.Fatal(err)
		}
		var fn driver.Function
		err = r.device.WithContext(func() error { var e error; fn, e = module.Function("ds4_dspark_ffn_sum"); return e })
		if err != nil {
			t.Fatal(err)
		}
		cudaQualificationMeasure(t, r, func() error {
			return cudalaunch.Kernel(fn).Grid(2, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(in, sh, out, uint32(width), uint32(6))
		})
		got := make([]float32, width)
		if err := out.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		bad := 0
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-want[i]) > 1e-6 {
				bad++
			}
		}
		if (!mutant && bad != 0) || (mutant && bad != width) {
			t.Fatalf("mutant=%v mismatches=%d/%d", mutant, bad, width)
		}
		if err := r.device.WithContext(module.Close); err != nil {
			t.Fatal(err)
		}
	}
}

// Literal device gates force the reviewed sm_121 CUBIN, including producers
// reused from the production math module. A PTX fallback is not a gate pass.
func cudaDSparkFFNGateFoundation(t *testing.T) *cudaFoundation {
	t.Helper()
	r := cudaSynthFoundation(t)
	if r.major != 12 || r.minor != 1 {
		t.Fatalf("literal gate requires sm_121, got sm_%d%d", r.major, r.minor)
	}
	source := cudaMathCompilationSource()
	program, err := nvrtcx.Compile(source, cudaMathCompileOptions(12, 1)...)
	if err != nil {
		t.Fatal(err)
	}
	if len(program.CUBIN()) == 0 {
		t.Fatal("foundation emitted no sm_121 CUBIN")
	}
	module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	err = r.device.WithContext(func() error {
		for _, k := range r.cudaKernels() {
			fn, e := module.Function(k.name)
			if e != nil {
				return e
			}
			*k.dst = fn
		}
		return nil
	})
	if err != nil {
		_ = r.device.WithContext(module.Close)
		t.Fatal(err)
	}
	r.matrixMod, r.mathCompiled = module, program
	t.Logf("foundation sm_121 translation unit SHA256=%x headers SHA256=%x CUBIN SHA256=%x", sha256.Sum256([]byte(source)), sha256.Sum256([]byte(cudaMathSource)), sha256.Sum256(program.CUBIN()))
	return r
}
