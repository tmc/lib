//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAQ4Resources(t *testing.T) {
	t.Setenv("DS4_CUDA_DISABLE_MOE_D2R", "0")
	t.Setenv("DS4_CUDA_MOE_D2R", "0")
	m := new(cudaModel)
	for i := range m.layers {
		m.layers[i].expertGate.typeID = typeIQ2XXS
	}
	if cudaModelNeedsQ4(m) {
		t.Fatal("Q2 model allocated Q8_K resources")
	}
	m.layers[37].expertGate.typeID = typeQ4K
	if !cudaModelNeedsQ4(m) {
		t.Fatal("Q4 splice omitted Q8_K resources")
	}
	r := &cudaRuntime{cudaFoundation: new(cudaFoundation), model: m}
	if !r.needsMoED2RScratch() {
		t.Fatal("unbuilt IQ2 layer omitted D2R scratch")
	}
	for i := range m.layers {
		m.layers[i].expertGate.typeID = typeQ4K
	}
	if r.needsMoED2RScratch() {
		t.Fatal("Q4-only layers allocated IQ2 D2R scratch")
	}
}

func TestCUDAQ4KSelectedQ8K(t *testing.T) {
	const cols, rows, experts = 512, 5, 256
	ids := []int32{255, 2, 7, 0, 2, 17}
	raw := make([]byte, experts*rows*(cols/256)*144)
	for b := 0; b < len(raw); b += 144 {
		p := raw[b : b+144]
		binary.LittleEndian.PutUint16(p, 0x3800)
		binary.LittleEndian.PutUint16(p[2:], 0xb400)
		for i := 4; i < 144; i++ {
			p[i] = byte((b/144*19 + i*37) % 256)
		}
	}
	x := make([]float32, cols*len(ids))
	for i := range x {
		x[i] = float32((i*13+i/cols*31)%251-125) / 73
	}
	var baseline [2][]float32
	for _, mutated := range []bool{false, true} {
		r := cudaSynthFoundation(t)
		if mutated {
			if err := r.device.WithContext(r.loadMathKernels); err != nil {
				t.Fatal(err)
			}
			source := cudaQ4TestMutation(t)
			compiled, err := nvrtcx.Compile(source, cudaMathCompileOptions(r.major, r.minor)...)
			if err != nil {
				t.Fatal(err)
			}
			mod, err := compiled.Load(r.device)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = r.device.WithContext(mod.Close) })
			err = r.device.WithContext(func() error {
				r.q4KSelectedQ8KFn, err = mod.Function("ds4_q4_k_selected_q8_k")
				return err
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		weights := cudaSynthTensor(t, r, "q4-experts", typeQ4K, []uint64{cols, rows, experts}, raw)
		dIDs, err := cudamem.Alloc[int32](r.device, len(ids))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = dIDs.Free() })
		if err := dIDs.CopyFromHost(ids); err != nil {
			t.Fatal(err)
		}
		for route, slotted := range []bool{false, true} {
			input := x[:cols]
			if slotted {
				input = x
			}
			dInput := cudaSynthUpload(t, r, input)
			q, err := cudamem.Alloc[byte](r.device, len(input)/256*292)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = q.Free() })
			out := cudaSynthUpload(t, r, make([]float32, rows*len(ids)))
			err = r.encodeQuantQ8K(dInput, q, cols)
			if err != nil {
				t.Fatal(err)
			}
			if err := r.encodeMatvecQ4KSelectedQ8K(weights, dIDs, q, out, slotted); err != nil {
				t.Fatal(err)
			}
			if err := syncStream(r.stream()); err != nil {
				t.Fatal(err)
			}
			got := cudaTestRead(t, out)
			if mutated {
				if err := cudaQualificationClose(got, baseline[route], 1e-4, 1e-4); err == nil {
					t.Fatal("device scale-high-bit mutant passed")
				}
				continue
			}
			decoded := cudaExpertQ8Decode(cudaExpertQ8Reference(input))
			want := make([]float32, len(got))
			for slot, id := range ids {
				for row := range rows {
					lo := (int(id)*rows + row) * cols / 256 * 144
					in := decoded[:cols]
					if slotted {
						in = decoded[slot*cols : (slot+1)*cols]
					}
					var sum float64
					for block := 0; block < cols/256; block++ {
						w, err := dsv4tools.DequantizeQ4K(raw[lo+block*144 : lo+(block+1)*144])
						if err != nil {
							t.Fatal(err)
						}
						for j, v := range w {
							sum += float64(v) * float64(in[block*256+j])
						}
					}
					want[slot*rows+row] = float32(sum)
				}
			}
			if err := cudaQualificationClose(got, want, 1e-4, 1e-4); err != nil {
				t.Fatal(err)
			}
			var maxAbs float64
			for i := range got {
				maxAbs = math.Max(maxAbs, math.Abs(float64(got[i]-want[i])))
			}
			t.Logf("slotted=%v: %d selected outputs max_abs=%g", slotted, len(got), maxAbs)
			baseline[route] = append([]float32(nil), got...)
			wrongIDs := append([]int32(nil), ids...)
			wrongIDs[0] = 1
			if err := dIDs.CopyFromHost(wrongIDs); err != nil {
				t.Fatal(err)
			}
			if err := r.encodeMatvecQ4KSelectedQ8K(weights, dIDs, q, out, slotted); err != nil {
				t.Fatal(err)
			}
			if err := syncStream(r.stream()); err != nil {
				t.Fatal(err)
			}
			if err := cudaQualificationClose(cudaTestRead(t, out), want, 1e-4, 1e-4); err == nil {
				t.Fatal("wrong selected expert passed")
			}
			if err := dIDs.CopyFromHost(ids); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func cudaQ4TestMutation(t *testing.T) string {
	t.Helper()
	source := cudaMathSourceFromFS()
	const target = "((scales[g - 4] >> 6) << 4)"
	if strings.Count(source, target) != 1 {
		t.Fatal("scale-high-bit mutation target is not unique")
	}
	mutant := strings.Replace(source, target, "0", 1)
	a, b := sha256.Sum256([]byte(source)), sha256.Sum256([]byte(mutant))
	if a == b {
		t.Fatal("compilation source did not change")
	}
	t.Logf("compiled candidate source=%x mutant source=%x", a, b)
	return mutant
}

func TestCUDAQ4MutationCompilationInput(t *testing.T) {
	mutant := cudaQ4TestMutation(t)
	if strings.Contains(mutant, "((scales[g - 4] >> 6) << 4)") {
		t.Fatal("mutation absent from actual compilation input")
	}
}
