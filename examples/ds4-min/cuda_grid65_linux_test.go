//go:build linux

package main

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestCUDAAttentionMixedPrepareGrid65VsGrid1(t *testing.T) {
	r := cudaSynthFoundation(t)
	if err := r.loadMathKernels(); err != nil {
		t.Fatal(err)
	}

	const (
		numHeads = 64
		headDim  = 512
		qLen     = numHeads * headDim
		kvLen    = headDim
		rawCap   = 4096
		outLen   = 32768 + 512 + 32768
		nComp    = 0
	)

	dq, err := cudamem.Alloc[float32](r.device, qLen)
	if err != nil {
		t.Fatal(err)
	}
	defer dq.Free()

	dkv, err := cudamem.Alloc[float32](r.device, kvLen)
	if err != nil {
		t.Fatal(err)
	}
	defer dkv.Free()

	raw1, err := cudamem.Alloc[float32](r.device, rawCap*headDim)
	if err != nil {
		t.Fatal(err)
	}
	defer raw1.Free()

	raw2, err := cudamem.Alloc[float32](r.device, rawCap*headDim)
	if err != nil {
		t.Fatal(err)
	}
	defer raw2.Free()

	out1, err := cudamem.Alloc[float32](r.device, outLen)
	if err != nil {
		t.Fatal(err)
	}
	defer out1.Free()

	out2, err := cudamem.Alloc[float32](r.device, outLen)
	if err != nil {
		t.Fatal(err)
	}
	defer out2.Free()

	sinks, err := cudamem.Alloc[float32](r.device, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer sinks.Free()

	comp, err := cudamem.Alloc[float32](r.device, headDim)
	if err != nil {
		t.Fatal(err)
	}
	defer comp.Free()

	allowed, err := cudamem.Alloc[uint32](r.device, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer allowed.Free()

	scores, err := cudamem.Alloc[float32](r.device, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer scores.Free()

	denoms, err := cudamem.Alloc[float32](r.device, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer denoms.Free()

	for _, pos := range []int{0, 1, 2, 5, 12, 129} {
		t.Run(fmt.Sprintf("pos_%d", pos), func(t *testing.T) {
			hq := make([]float32, qLen)
			for i := range hq {
				hq[i] = float32((i*17+pos*31)%1009-504) / 256.0
			}
			hkv := make([]float32, kvLen)
			for i := range hkv {
				hkv[i] = float32((i*23+pos*47)%503-251) / 128.0
			}

			if err := dq.CopyFromHost(hq); err != nil {
				t.Fatal(err)
			}
			if err := dkv.CopyFromHost(hkv); err != nil {
				t.Fatal(err)
			}

			// Run Grid(1)
			if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(1).Block(64).On(r.stream()).Launch(
				dq, dkv, sinks, raw1, comp, allowed, scores, denoms, out1,
				uint32(pos), uint32(2), uint32(rawCap), uint32(nComp)); err != nil {
				t.Fatal(err)
			}

			// Run Grid(65)
			if err := cudalaunch.Kernel(r.mixedAttentionPrepareFn).Grid(65).Block(64).On(r.stream()).Launch(
				dq, dkv, sinks, raw2, comp, allowed, scores, denoms, out2,
				uint32(pos), uint32(2), uint32(rawCap), uint32(nComp)); err != nil {
				t.Fatal(err)
			}

			if err := r.stream().Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}

			hOut1 := make([]float32, outLen)
			hOut2 := make([]float32, outLen)
			if err := out1.CopyToHost(hOut1); err != nil {
				t.Fatal(err)
			}
			if err := out2.CopyToHost(hOut2); err != nil {
				t.Fatal(err)
			}

			diffCount := 0
			for i := 0; i < outLen; i++ {
				u1 := math.Float32bits(hOut1[i])
				u2 := math.Float32bits(hOut2[i])
				if u1 != u2 {
					if diffCount < 10 {
						t.Errorf("pos %d out[%d] bit diff: grid1=%v (0x%08x) vs grid65=%v (0x%08x)",
							pos, i, hOut1[i], u1, hOut2[i], u2)
					}
					diffCount++
				}
			}
			if diffCount > 0 {
				t.Fatalf("pos %d: %d out diffs", pos, diffCount)
			}

			hRaw1 := make([]float32, 512)
			hRaw2 := make([]float32, 512)
			rowOffset := (pos % rawCap) * 512
			if err := raw1.Slice(rowOffset, rowOffset+512).CopyToHost(hRaw1); err != nil {
				t.Fatal(err)
			}
			if err := raw2.Slice(rowOffset, rowOffset+512).CopyToHost(hRaw2); err != nil {
				t.Fatal(err)
			}

			rawDiffCount := 0
			for i := 0; i < 512; i++ {
				u1 := math.Float32bits(hRaw1[i])
				u2 := math.Float32bits(hRaw2[i])
				if u1 != u2 {
					if rawDiffCount < 10 {
						t.Errorf("pos %d raw[%d] bit diff: grid1=%v (0x%08x) vs grid65=%v (0x%08x)",
							pos, i, hRaw1[i], u1, hRaw2[i], u2)
					}
					rawDiffCount++
				}
			}
			if rawDiffCount > 0 {
				t.Fatalf("pos %d: %d raw diffs", pos, rawDiffCount)
			}
		})
	}
}
