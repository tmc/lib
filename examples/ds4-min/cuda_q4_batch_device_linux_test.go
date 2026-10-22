//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4imatrix"
	"github.com/tmc/lib/examples/ds4-min/dsv4tools"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAQ4BatchPreflight(t *testing.T) {
	fresh := func() *cudaRuntime {
		r := &cudaRuntime{cudaFoundation: &cudaFoundation{device: new(cuda.Device)}, model: new(cudaModel), batch: &cudaBatchResources{grouping: new(cudaMoEGrouping)}}
		r.model.layers[37].expertGate.typeID = typeQ4K
		r.model.layers[37].expertUp.typeID = typeQ4K
		r.model.layers[37].expertDown.typeID = typeQ4K
		return r
	}
	for _, tc := range []struct {
		name          string
		change        func(*cudaRuntime) *cudaRuntime
		index, tokens int
		want          string
	}{
		{"nil", func(*cudaRuntime) *cudaRuntime { return nil }, 37, 2, "closed or missing runtime"},
		{"foundation", func(r *cudaRuntime) *cudaRuntime { r.cudaFoundation = nil; return r }, 37, 2, "closed or missing runtime"},
		{"model", func(r *cudaRuntime) *cudaRuntime { r.model = nil; return r }, 37, 2, "closed or missing runtime"},
		{"batch", func(r *cudaRuntime) *cudaRuntime { r.batch = nil; return r }, 37, 2, "closed or missing runtime"},
		{"grouping", func(r *cudaRuntime) *cudaRuntime { r.batch.grouping = nil; return r }, 37, 2, "closed or missing runtime"},
		{"closing", func(r *cudaRuntime) *cudaRuntime { r.batch.closing = true; return r }, 37, 2, "closed or missing runtime"},
		{"foundation-closing", func(r *cudaRuntime) *cudaRuntime { r.cudaFoundation.closing = true; return r }, 37, 2, "closed or missing runtime"},
		{"layer-negative", func(r *cudaRuntime) *cudaRuntime { return r }, -1, 2, "closed or missing runtime"},
		{"layer-high", func(r *cudaRuntime) *cudaRuntime { return r }, 43, 2, "closed or missing runtime"},
		{"empty", func(r *cudaRuntime) *cudaRuntime { return r }, 37, 0, "invalid token count"},
		{"extent", func(r *cudaRuntime) *cudaRuntime { return r }, 37, 2, "norm scratch truncated"},
		{"mixed-triplet", func(r *cudaRuntime) *cudaRuntime { r.model.layers[37].expertUp.typeID = typeIQ2XXS; return r }, 37, 2, "gate/up types"},
		{"q2-layer", func(r *cudaRuntime) *cudaRuntime {
			l := &r.model.layers[37]
			l.expertGate.typeID = typeIQ2XXS
			l.expertUp.typeID = typeIQ2XXS
			l.expertDown.typeID = typeQ2K
			return r
		}, 37, 2, "has format IQ2_XXS/Q2_K"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.change(fresh())
			if err := r.evalBatchQ4Experts(make([]cudaTokenState, tc.tokens), tc.index); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("preflight: %v", err)
			}
		})
	}
	var f *cudaFoundation
	var tensor cudaModelTensor
	var bytes cudamem.Slice[byte]
	var values cudamem.Slice[float32]
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"nil-pair-group", func() error { return f.encodeQ4KPairGather(tensor, tensor, nil, bytes, values, values) }},
		{"nil-down-group", func() error { return f.encodeQ4KDownGather(tensor, nil, bytes, values) }},
		{"nil-shape-group", func() error { return q4GatherShape(tensor, nil, 6, 0, 0) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.call(); err == nil || !strings.Contains(err.Error(), "missing grouping") {
				t.Fatalf("grouping preflight: %v", err)
			}
		})
	}
}

// TestCUDAQ4BatchDevice exercises the production batch orchestration with
// two synthetic experts at full Flash row widths. It is not a model gate.
func TestCUDAQ4BatchDevice(t *testing.T) {
	if os.Getenv("DS4_Q4_BATCH_DEVICE") != "1" {
		t.Skip("DS4_Q4_BATCH_DEVICE not set")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	f := &cudaFoundation{device: dev}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	f.major, f.minor = major, minor
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%q sm_%d%d full_width_fixture_bytes=%d", name, major, minor, 3*2*4096*2048/256*144)
	// Mutants run the ordinary assertions below and must fail there. They
	// are opt-in test controls, never production dispatch configuration.
	if mutation := os.Getenv("DS4_Q4_BATCH_MUTATION"); mutation != "" {
		var site, replacement, symbol string
		switch mutation {
		case "token-address":
			site, replacement, symbol = "unsigned(pair) / slots", "0", "ds4_q4_k_pair_selected_q8_k_gather"
		case "gate-up-swap":
			site, replacement, symbol = "ds4_q4_k_gather_body<true>(gate, up,", "ds4_q4_k_gather_body<true>(up, gate,", "ds4_q4_k_pair_selected_q8_k_gather"
		case "down-slots":
			site, replacement, symbol = "cols, rows, experts, 1, npairs", "cols, rows, experts, 6, npairs", "ds4_q4_k_selected_q8_k_gather"
		default:
			t.Fatalf("unknown source mutation %q", mutation)
		}
		if err := f.withContext(f.loadMathKernels); err != nil {
			t.Fatal(err)
		}
		source := cudaMathSourceFromFS()
		if strings.Count(source, site) != 1 {
			t.Fatal("mutation site not unique")
		}
		mutant := strings.Replace(source, site, replacement, 1)
		t.Logf("mutation=%s source_sha256=%x mutant_sha256=%x", mutation, sha256.Sum256([]byte(source)), sha256.Sum256([]byte(mutant)))
		program, err := nvrtcx.Compile(mutant, cudaMathCompileOptions(major, minor)...)
		if err != nil {
			t.Fatal(err)
		}
		module, err := program.Load(dev)
		if module != 0 {
			t.Cleanup(func() {
				if err := syncStream(f.stream()); err != nil {
					t.Error(err)
				}
				if err := dev.WithContext(module.Close); err != nil {
					t.Error(err)
				}
			})
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := dev.WithContext(func() error {
			fn, err := module.Function(symbol)
			if err == nil {
				if mutation == "token-address" || mutation == "gate-up-swap" {
					f.q4KPairGatherFn = fn
				} else {
					f.q4KDownGatherFn = fn
				}
			}
			return err
		}); err != nil {
			t.Fatal(err)
		}
	}
	allocFloat := func(n int) cudamem.Slice[float32] { return cudaTestAlloc(t, f, n) }
	allocByte := func(n int) cudamem.Slice[byte] { return cudaExpertAllocBytes(t, f, n) }
	allocIDs := func(n int) cudamem.Slice[int32] {
		s, err := cudamem.Alloc[int32](dev, n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Free(); err != nil {
				t.Error(err)
			}
		})
		return s
	}
	t.Run("selected-255", func(t *testing.T) {
		const cols, outRows, experts = 512, 3, 256
		raw := make([]byte, experts*outRows*cols/256*144)
		for b := 0; b < len(raw)/144; b++ {
			p := raw[b*144 : (b+1)*144]
			binary.LittleEndian.PutUint16(p, 0x3800)
			binary.LittleEndian.PutUint16(p[2:], 0xb400)
			for i := 4; i < 144; i++ {
				p[i] = byte(b*19 + i*37 + 43)
			}
		}
		// Negate both Q4_K scale terms, not the packed codes. The up
		// matrix is exactly the negative of gate after dequantization.
		upRaw := append([]byte(nil), raw...)
		for b := 0; b < len(upRaw); b += 144 {
			for _, off := range []int{0, 2} {
				v := binary.LittleEndian.Uint16(upRaw[b+off:])
				binary.LittleEndian.PutUint16(upRaw[b+off:], v^0x8000)
			}
		}
		tensorFor := func(raw []byte) cudaModelTensor {
			data := allocByte(len(raw))
			if err := data.CopyFromHost(raw); err != nil {
				t.Fatal(err)
			}
			return cudaModelTensor{typeID: typeQ4K, dimensions: []uint64{cols, outRows, experts}, device: data.Ptr(), bytes: uint64(len(raw))}
		}
		tensor, upTensor := tensorFor(raw), tensorFor(upRaw)
		group, err := newCUDAMoEGrouping(dev, experts, 12)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := group.Close(); err != nil {
				t.Error(err)
			}
		})
		ids := allocIDs(12)
		selected := []int32{255, 2, 0, 2, 17, 255, 17, 255, 2, 0, 17, -1}
		if err := ids.CopyFromHost(selected); err != nil {
			t.Fatal(err)
		}
		x := make([]float32, 12*cols)
		for i := range x {
			x[i] = float32((i*13+i/cols*31)%251-125) / 73
		}
		for b := 0; b < len(x)/256; b++ {
			x[b*256+7] = 3.125
			if b%2 != 0 {
				x[b*256+7] = -3.125
			}
		}
		q := allocByte(len(x) / 256 * 292)
		if err := f.encodeQuantQ8K(cudaSynthUpload(t, f, x), q, cols); err != nil {
			t.Fatal(err)
		}
		gate, up, down := allocFloat(12*outRows), allocFloat(12*outRows), allocFloat(12*outRows)
		if err := f.encodeMoEGroupCSR(ids, group); err != nil {
			t.Fatal(err)
		}
		if err := f.encodeQ4KPairGather(tensor, upTensor, group, q.Slice(0, 2*cols/256*292), gate, up); err != nil {
			t.Fatal(err)
		}
		if err := f.encodeQ4KDownGather(tensor, group, q, down); err != nil {
			t.Fatal(err)
		}
		if err := syncStream(f.stream()); err != nil {
			t.Fatal(err)
		}
		cudaExpertCheckBytes(t, q, cudaExpertQ8Reference(x))
		decoded := cudaExpertQ8Decode(cudaExpertQ8Reference(x))
		var gateReference []float32
		for _, stage := range []struct {
			tensor  cudaModelTensor
			raw     []byte
			name    string
			output  cudamem.Slice[float32]
			slotted bool
		}{{tensor, raw, "gate", gate, false}, {upTensor, upRaw, "up", up, false}, {tensor, raw, "down", down, true}} {
			got := cudaTestRead(t, stage.output)
			want := allocFloat(12 * outRows)
			if stage.slotted {
				if err := f.encodeMatvecQ4KSelectedQ8K(stage.tensor, ids, q, want, true); err != nil {
					t.Fatal(err)
				}
			} else {
				for token := range 2 {
					if err := f.encodeMatvecQ4KSelectedQ8K(stage.tensor, ids.Slice(token*6, (token+1)*6), q.Slice(token*cols/256*292, (token+1)*cols/256*292), want.Slice(token*6*outRows, (token+1)*6*outRows), false); err != nil {
						t.Fatal(err)
					}
				}
			}
			if err := syncStream(f.stream()); err != nil {
				t.Fatal(err)
			}
			baseline := cudaTestRead(t, want)
			q4BatchDump(t, "selected-255-"+stage.name, got)
			// Decode Q4_K and reconstructed Q8_K on the host, independently
			// of both the gather and scalar CUDA projections.
			var finiteGot, reference []float32
			for slot, id := range selected {
				if id < 0 {
					continue
				}
				inputRow := slot / 6
				if stage.slotted {
					inputRow = slot
				}
				for row := range outRows {
					var sum float64
					for block := 0; block < cols/256; block++ {
						off := ((int(id)*outRows+row)*(cols/256) + block) * 144
						weights, err := dsv4tools.DequantizeQ4K(stage.raw[off : off+144])
						if err != nil {
							t.Fatal(err)
						}
						for j, v := range weights {
							sum += float64(v) * float64(decoded[inputRow*cols+block*256+j])
						}
					}
					reference = append(reference, float32(sum))
					finiteGot = append(finiteGot, got[slot*outRows+row])
				}
			}
			if err := cudaQualificationClose(finiteGot, reference, 1e-4, 1e-4); err != nil {
				t.Fatalf("%s independent host reference: %v", stage.name, err)
			}
			if stage.name == "gate" {
				gateReference = append([]float32(nil), reference...)
			}
			if stage.name == "up" {
				different := 0
				for i, v := range reference {
					if v != -gateReference[i] {
						t.Fatalf("up reference%d is not negative gate: %g vs %g", i, v, gateReference[i])
					}
					if v != gateReference[i] {
						different++
					}
				}
				if different == 0 {
					t.Fatal("gate/up reference outputs do not discriminate")
				}
				if err := cudaQualificationClose(gateReference, reference, 1e-4, 1e-4); err == nil {
					t.Fatal("swapped gate/up references pass the output tolerance")
				}
				t.Logf("gate_up_distinct_reference_outputs=%d/%d", different, len(reference))
			}
			for i := range got {
				if selected[i/outRows] < 0 {
					if !math.IsNaN(float64(got[i])) {
						t.Fatalf("%s invalid slot%d finite", stage.name, i)
					}
					continue
				}
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Float32bits(got[i]) != math.Float32bits(baseline[i]) {
					t.Fatalf("%s selected value%d: gathered=%g decode=%g", stage.name, i, got[i], baseline[i])
				}
			}
		}
		t.Log("finite_outputs=99 invalid_outputs=9 exact_q8k_bytes=7008")
	})
	weights := func(cols, rows, seed int) cudaModelTensor {
		raw := make([]byte, 2*rows*cols/256*144)
		for b := 0; b < len(raw)/144; b++ {
			p := raw[b*144 : (b+1)*144]
			binary.LittleEndian.PutUint16(p, 0x1000)
			binary.LittleEndian.PutUint16(p[2:], 0x9400)
			for i := 4; i < 144; i++ {
				p[i] = byte(b*seed + i*37 + 43)
			}
		}
		d := allocByte(len(raw))
		if err := d.CopyFromHost(raw); err != nil {
			t.Fatal(err)
		}
		return cudaModelTensor{typeID: typeQ4K, dimensions: []uint64{uint64(cols), uint64(rows), 2}, device: d.Ptr(), bytes: uint64(len(raw))}
	}
	r := &cudaRuntime{cudaFoundation: f, model: new(cudaModel), batch: new(cudaBatchResources)}
	l := &r.model.layers[37]
	l.expertGate, l.expertUp, l.expertDown = weights(4096, 2048, 19), weights(4096, 2048, 31), weights(2048, 4096, 47)
	g, err := newCUDAMoEGrouping(dev, 2, 12)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := g.Close(); err != nil {
			t.Error(err)
		}
	})
	r.batch.grouping = g
	d := &r.batch.dense
	d.attnNorm = allocFloat(2 * 4096)
	// The non-Q4 dispatch evaluates this view before its real IQ2 format
	// validator. Give that view valid storage so a wrong dispatch predicate
	// reaches the format refusal, not an unrelated slice-bounds panic.
	d.q81FFN = allocByte(2 * 4096 / 32 * 36)
	d.q4Input, d.q4Experts = allocByte(2*4096/256*292), allocByte(12*2048/256*292)
	d.q4Mid = allocFloat(12 * 2048)
	d.expertGate, d.expertUp, d.expertDown = allocFloat(12*2048), allocFloat(12*2048), allocFloat(12*4096)
	d.expertIDs, d.expertW = allocIDs(12), allocFloat(12)
	ids := []int32{1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1}
	if err := d.expertIDs.CopyFromHost(ids); err != nil {
		t.Fatal(err)
	}
	w := make([]float32, 12)
	for i := range w {
		w[i] = float32(i+1) / 91
	}
	if err := d.expertW.CopyFromHost(w); err != nil {
		t.Fatal(err)
	}
	var rows [2]cudaTokenState
	for token := range rows {
		s := &rows[token]
		s.ffnPre = allocFloat(8192)
		s.ffnNorm = s.ffnPre.Slice(4096, 8192)
		input := make([]float32, 4096)
		for i := range input {
			input[i] = float32((i*13+token*31)%251-125) / 73
		}
		if err := s.ffnNorm.CopyFromHost(input); err != nil {
			t.Fatal(err)
		}
		s.expertIDs = d.expertIDs.Slice(token*6, (token+1)*6)
		s.expertW = d.expertW.Slice(token*6, (token+1)*6)
		s.q8KInput, s.q8KExperts = allocByte(4096/256*292), allocByte(6*2048/256*292)
		s.expertGate, s.expertUp, s.expertMid, s.expertDown = allocFloat(6*2048), allocFloat(6*2048), allocFloat(6*2048), allocFloat(6*4096)
		if err := r.evalQ4Experts(s, 37); err != nil {
			t.Fatal(err)
		}
	}
	if err := syncStream(f.stream()); err != nil {
		t.Fatal(err)
	}
	check := func() error {
		outputs := []cudamem.Slice[float32]{d.expertGate, d.expertUp, d.q4Mid, d.expertDown}
		if d.profileMid.Len() != 0 {
			outputs = append(outputs, d.profileMid)
		}
		for _, out := range outputs {
			poison := make([]float32, out.Len())
			for i := range poison {
				poison[i] = math.Float32frombits(0x7fc12345)
			}
			if err := out.CopyFromHost(poison); err != nil {
				return err
			}
		}
		return r.evalBatchExperts(rows[:], 37)
	}
	if err := check(); err != nil {
		t.Fatalf("grouped dispatcher: %v", err)
	}
	if err := syncStream(f.stream()); err != nil {
		t.Fatal(err)
	}
	want := make([][]float32, 4)
	for token := range rows {
		s := &rows[token]
		for stage, pair := range [][2]cudamem.Slice[float32]{{s.expertGate, d.expertGate}, {s.expertUp, d.expertUp}, {s.expertMid, d.q4Mid}, {s.expertDown, d.expertDown}} {
			a := cudaTestRead(t, q4BatchTokenSlice(pair[1], token))
			b := cudaTestRead(t, pair[0])
			q4BatchDump(t, fmt.Sprintf("batch-stage%d-token%d", stage, token), a)
			for i := range a {
				if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("stage%d token%d value%d: batch=%g decode=%g", stage, token, i, a[i], b[i])
				}
			}
			want[stage] = append(want[stage], b...)
		}
		for _, pair := range [][2]cudamem.Slice[byte]{{s.q8KInput, d.q4Input}, {s.q8KExperts, d.q4Experts}} {
			b := make([]byte, pair[0].Len())
			if err := pair[0].CopyToHost(b); err != nil {
				t.Fatal(err)
			}
			cudaExpertCheckBytes(t, pair[1].Slice(token*len(b), (token+1)*len(b)), b)
		}
	}
	// Collector-on must copy the actual weighted mid before quantization.
	r.collectors = &profileCollectors{imatrix: new(dsv4imatrix.Collector)}
	d.profileMid = allocFloat(12 * 2048)
	if err := check(); err != nil {
		t.Fatalf("collector grouped dispatcher: %v", err)
	}
	if err := syncStream(f.stream()); err != nil {
		t.Fatal(err)
	}
	// Recheck every result after collector-on; none may survive from the
	// first call, and collector-only dispatch mutations must be observable.
	for stage, out := range []cudamem.Slice[float32]{d.expertGate, d.expertUp, d.q4Mid, d.expertDown} {
		got := cudaTestRead(t, out)
		q4BatchDump(t, fmt.Sprintf("collector-stage%d", stage), got)
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(want[stage][i]) {
				t.Fatalf("collector stage%d value%d: batch=%g decode=%g", stage, i, v, want[stage][i])
			}
		}
	}
	profileMid := cudaTestRead(t, d.profileMid)
	q4BatchDump(t, "collector-mid", profileMid)
	for i := range profileMid {
		if math.Float32bits(profileMid[i]) != math.Float32bits(want[2][i]) {
			t.Fatalf("collector mid%d differs", i)
		}
	}
	t.Logf("cases=3 ran=3 skipped=0 stages=4 exact_float_values=%d exact_q8k_bytes=%d collector_mid_values=%d", 12*(2048*3+4096), 2*4096/256*292+12*2048/256*292, len(profileMid))
}

func q4BatchTokenSlice(s cudamem.Slice[float32], token int) cudamem.Slice[float32] {
	return s.Slice(token*(s.Len()/2), (token+1)*(s.Len()/2))
}

// q4BatchDump writes v as little-endian float32 to DS4_Q4_BATCH_OUT/name.f32
// when that directory is set, so repeat runs can be compared bit for bit.
func q4BatchDump(t *testing.T, name string, v []float32) {
	dir := os.Getenv("DS4_Q4_BATCH_OUT")
	if dir == "" {
		return
	}
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	if err := os.WriteFile(filepath.Join(dir, name+".f32"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}
