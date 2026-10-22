//go:build linux

package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestCUDAQualificationAttnQKVD2R checks that the tensor-core GEMM the query
// and key/value projections now run through agrees with the matvec it replaced.
// The two disagree in the last bits by construction -- one quantizes the
// activations into 32-element Q8_1 blocks and the other into the 128-element
// chunks the MMQ layout wants -- so the gate is a relative bound, and a control
// below shows that bound rejects the products of a different weight matrix.
func TestCUDAQualificationAttnQKVD2R(t *testing.T) {
	r := cudaSynthFoundation(t)
	for _, tc := range []struct{ m, k, batch int }{
		{cudaQLoraWidth, cudaWidth, 64},         // attnQA
		{cudaAttentionHeads, cudaQLoraWidth, 7}, // attnQB, and groupM 1
		{cudaKVWidth, cudaWidth, 64},            // attnKV
		{256, 4096, 129},                        // a batch past the 128-column tile
	} {
		t.Run(fmt.Sprintf("m%d-k%d-b%d", tc.m, tc.k, tc.batch), func(t *testing.T) {
			rnd := rand.New(rand.NewSource(int64(tc.m*31 + tc.k)))
			weights := cudaSynthQ80(rnd, tc.m, tc.k, -8)
			tensor := cudaSynthTensor(t, r, "qkv fixture", 8, []uint64{uint64(tc.k), uint64(tc.m)}, weights)
			other := cudaSynthQ80(rand.New(rand.NewSource(1)), tc.m, tc.k, -8)

			x := make([]float32, tc.batch*tc.k)
			for i := range x {
				x[i] = float32((i*37+i/17+i/tc.k*29)%257-128) / 128
			}
			in := cudaSynthUpload(t, r, x)

			// The matvec arm, through the Q8_1 blocks it reads.
			q81 := cudaExpertAllocBytes(t, r, tc.batch*tc.k/32*36)
			if err := r.encodeQuantQ81Batch(in, q81, tc.k, tc.k, tc.batch); err != nil {
				t.Fatal(err)
			}
			wantDev := cudaTestAlloc(t, r, tc.batch*tc.m)
			if err := r.encodeQ80MatvecQ81Batch(tensor, q81, wantDev, tc.batch); err != nil {
				t.Fatal(err)
			}
			want := cudaTestRead(t, wantDev)

			// The GEMM arm, through the repacked planes and the MMQ blocks.
			mmq := cudaExpertAllocBytes(t, r, ((tc.k/128)*tc.batch+128)*144)
			if err := r.encodeQuantMMQ(in, mmq, tc.k, tc.k, tc.batch); err != nil {
				t.Fatal(err)
			}
			gotDev := cudaTestAlloc(t, r, tc.batch*tc.m)
			if err := gotDev.CopyFromHost(makeNaNs(tc.batch * tc.m)); err != nil {
				t.Fatal(err)
			}
			packed := cudaExpertAllocBytes(t, r, denseQ8AlignedBytes(tc.m, tc.k))
			if err := r.repackQ80(packed, tensor.device, uint64(tc.m)*uint64(tc.k/32)); err != nil {
				t.Fatal(err)
			}
			if err := r.encodeDenseQ8D2R(packed, mmq, gotDev, tc.m, tc.batch, tc.k, tc.m); err != nil {
				t.Fatal(err)
			}
			got := cudaTestRead(t, gotDev)

			// The bound is on the deviation relative to the scale of the whole
			// product, not to each element: products near zero are the
			// difference of large terms and carry no relative accuracy.
			const tol = 0.005
			scale := 0.0
			for _, v := range want {
				scale = math.Max(scale, math.Abs(float64(v)))
			}
			if scale == 0 {
				t.Fatal("reference products are all zero")
			}
			deviation := func(a []float32) float64 {
				worst := 0.0
				for i, v := range a {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("non-finite product at %d", i)
					}
					worst = math.Max(worst, math.Abs(float64(v-want[i])))
				}
				return worst / scale
			}
			if d := deviation(got); d > tol {
				t.Fatalf("GEMM deviates from the matvec by %g of the product scale, want at most %g", d, tol)
			} else {
				t.Logf("deviation %g of scale %g", d, scale)
			}

			// The control: the same bound applied to the products of a
			// different weight matrix has to reject them.
			otherTensor := cudaSynthTensor(t, r, "qkv control", 8, []uint64{uint64(tc.k), uint64(tc.m)}, other)
			if err := r.repackQ80(packed, otherTensor.device, uint64(tc.m)*uint64(tc.k/32)); err != nil {
				t.Fatal(err)
			}
			if err := r.encodeDenseQ8D2R(packed, mmq, gotDev, tc.m, tc.batch, tc.k, tc.m); err != nil {
				t.Fatal(err)
			}
			if d := deviation(cudaTestRead(t, gotDev)); d <= tol {
				t.Fatalf("a different weight matrix passed the bound at %g: the bound cannot fail", d)
			}
		})
	}
}
