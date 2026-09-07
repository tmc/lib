//go:build !tpu

package main

import "testing"

// TestKernelNamesMatchSource ties the Kernels struct to the device source.
//
// TestForwardSourceDefinesEveryWeightVariant checks the source against a
// hand-written list; this checks that the list is the one the binding code
// actually asks for. Together they mean a kernel cannot be added to Kernels, or
// have its weights tag changed, without the device entry point existing.
func TestKernelNamesMatchSource(t *testing.T) {
	names := kernelSourceNames(t)
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		// Kernels binds every kernel the engine dispatches. weightKernelBases
		// additionally lists WeightConvertProbe, which the native tests reach
		// through the module directly and which production never binds.
		got := weightKernelNames(d)
		want := len(weightKernelBases) + len(plainKernelNames) - 1
		if len(got) != want {
			t.Errorf("%s: %d kernel names %v, want %d", d, len(got), got, want)
		}
		for _, n := range got {
			if n == "WeightConvertProbe"+d.Suffix() {
				t.Errorf("%s: production binds the test-only conversion probe", d)
			}
		}
		for _, n := range got {
			if !names[n] {
				t.Errorf("%s binds %q, which forward.cu does not define", d, n)
			}
		}
	}

	// The f32 names must be exactly the unsuffixed ones, so the reference path
	// does not depend on the dispatch scheme existing at all.
	for _, n := range weightKernelNames(WeightF32) {
		if len(n) > 3 && (n[len(n)-3:] == "F16" || n[len(n)-4:] == "BF16") {
			t.Errorf("f32 mode binds suffixed kernel %q", n)
		}
	}
}

// TestWeightBytesCountsTwoBytesPerHalfWeight checks the allocation accounting.
//
// It builds the accounting structure directly rather than uploading, so it runs
// without a GPU. What it establishes is that the reported figure is elems times
// the stored width, and that a tied LM head is counted once. The measured
// device figure on real hardware comes from the cudamem.Tracker and is reported
// separately; this test only fixes what WeightBytes means.
func TestWeightBytesCountsTwoBytesPerHalfWeight(t *testing.T) {
	const (
		vocab, hidden, inter, layers = 4096, 512, 1024, 3
	)
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		t.Run(d.String(), func(t *testing.T) {
			w := &GPUModelWeights{dtype: d}
			w.EmbedTokens = WeightSlice{dtype: d, elems: vocab * hidden}
			w.Norm = WeightSlice{dtype: d, elems: hidden}
			w.LMHead = w.EmbedTokens // tied, and so must not be counted twice
			var elems int
			elems += vocab*hidden + hidden
			for i := 0; i < layers; i++ {
				l := GPULayerWeights{
					InputNorm: WeightSlice{dtype: d, elems: hidden},
					PostNorm:  WeightSlice{dtype: d, elems: hidden},
					WQ:        WeightSlice{dtype: d, elems: hidden * hidden},
					WK:        WeightSlice{dtype: d, elems: hidden * hidden},
					WV:        WeightSlice{dtype: d, elems: hidden * hidden},
					WO:        WeightSlice{dtype: d, elems: hidden * hidden},
					WGate:     WeightSlice{dtype: d, elems: hidden * inter},
					WUp:       WeightSlice{dtype: d, elems: hidden * inter},
					WDown:     WeightSlice{dtype: d, elems: hidden * inter},
					// Biases absent: the zero value must contribute nothing.
				}
				w.Layers = append(w.Layers, l)
				elems += 2*hidden + 4*hidden*hidden + 3*hidden*inter
			}

			want := elems * d.Bytes()
			if got := w.WeightBytes(); got != want {
				t.Fatalf("WeightBytes() = %d, want %d (%d weights x %d bytes)", got, want, elems, d.Bytes())
			}
			if got := w.DType(); got != d {
				t.Fatalf("DType() = %v, want %v", got, d)
			}
			t.Logf("%s: %d weights, %d bytes (%.2f MiB)", d, elems, want, float64(want)/(1<<20))
		})
	}

	// The half modes must report exactly half of the float32 figure. This is
	// the storage claim, stated as a ratio so it cannot be satisfied by an
	// accounting change that happens to move both numbers.
	build := func(d WeightDType) *GPUModelWeights {
		w := &GPUModelWeights{dtype: d}
		w.EmbedTokens = WeightSlice{dtype: d, elems: vocab * hidden}
		w.Norm = WeightSlice{dtype: d, elems: hidden}
		w.LMHead = w.EmbedTokens
		w.Layers = []GPULayerWeights{{WQ: WeightSlice{dtype: d, elems: hidden * hidden}}}
		return w
	}
	ref := build(WeightF32).WeightBytes()
	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		if got := build(d).WeightBytes(); got*2 != ref {
			t.Errorf("%s weights use %d bytes, want exactly half of f32's %d", d, got, ref)
		}
	}
}

// TestZeroWeightSliceIsAbsent checks that an unset optional tensor reads as
// absent rather than as a zero-length allocation at some address.
//
// The kernels test the pointer to decide whether a bias or norm weight is
// present, so this is the behavior an omitted tensor depends on.
func TestZeroWeightSliceIsAbsent(t *testing.T) {
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		var w WeightSlice
		w.dtype = d
		if w.Len() != 0 {
			t.Errorf("%s: zero slice Len() = %d", d, w.Len())
		}
		if w.Bytes() != 0 {
			t.Errorf("%s: zero slice Bytes() = %d", d, w.Bytes())
		}
		if w.Ptr() != 0 {
			t.Errorf("%s: zero slice Ptr() = %v, want the null device pointer", d, w.Ptr())
		}
	}
}

// TestInvalidDTypeRejectedAtDeviceBoundaries checks the guards on the
// allocation and compilation entry points, not just the command-line parser.
//
// A WeightDType can reach these functions from a caller that never went through
// ParseWeightDType, so validating only at the CLI would leave the hazard open
// to every other caller. These fail before allocating or compiling anything,
// so no device state exists to clean up.
func TestInvalidDTypeRejectedAtDeviceBoundaries(t *testing.T) {
	for _, d := range []WeightDType{-1, 3, 7} {
		if _, err := UploadModelToGPUAs(nil, nil, nil, nil, d); err == nil {
			t.Errorf("UploadModelToGPUAs accepted dtype %d", int(d))
		}
		if _, err := UploadRetainedModelToGPUAs(nil, nil, nil, nil, d); err == nil {
			t.Errorf("UploadRetainedModelToGPUAs accepted dtype %d", int(d))
		}
		if _, err := CompileGPUModulesAs(nil, d); err == nil {
			t.Errorf("CompileGPUModulesAs accepted dtype %d", int(d))
		}
		if _, err := uploadWeights(nil, d, []float32{1, 2, 3}); err == nil {
			t.Errorf("uploadWeights accepted dtype %d", int(d))
		}
		if err := bindWeightKernels(nil, d, &Kernels{}); err == nil {
			t.Errorf("bindWeightKernels accepted dtype %d", int(d))
		}
	}
}
