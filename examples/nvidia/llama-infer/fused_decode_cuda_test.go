package main

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

const (
	bf16FusedQ0LanesSource      = "bf16_to_f32(uq0.x), bf16_to_f32(uq0.y)"
	bf16FusedQ0LanesSwapped     = "bf16_to_f32(uq0.y), bf16_to_f32(uq0.x)"
	bf16FusedOutProjLanesSource = "bf16_to_f32(uo.x), bf16_to_f32(uo.y)"
	bf16FusedOutProjSwapped     = "bf16_to_f32(uo.y), bf16_to_f32(uo.x)"
	bf16FusedGateLanesSource    = "bf16_to_f32(ugate.x), bf16_to_f32(ugate.y)"
	bf16FusedGateLanesSwapped   = "bf16_to_f32(ugate.y), bf16_to_f32(ugate.x)"
	bf16FusedDownLanesSource    = "bf16_to_f32(udown.x), bf16_to_f32(udown.y)"
	bf16FusedDownLanesSwapped   = "bf16_to_f32(udown.y), bf16_to_f32(udown.x)"
)

// TestFusedDecodeBF16MutationsApply checks without a device that each BF16
// lane-swap fragment in fused_decode.cu occurs exactly once, so every mutant
// in TestFusedDecodeBF16Controls modifies its intended kernel.
func TestFusedDecodeBF16MutationsApply(t *testing.T) {
	for _, frag := range []string{
		bf16FusedQ0LanesSource,
		bf16FusedOutProjLanesSource,
		bf16FusedGateLanesSource,
		bf16FusedDownLanesSource,
	} {
		if n := strings.Count(fusedDecodeCUDA, frag); n != 1 {
			t.Errorf("%q occurs %d times in fused_decode.cu, want 1", frag, n)
		}
	}
}

func mutateFusedDecode(d *parityDevice, bf16Old, bf16New, fusedOld, fusedNew string) *GPUModules {
	d.t.Helper()
	bf16Src := forwardBF16CUDA
	if bf16Old != "" {
		if n := strings.Count(bf16Src, bf16Old); n != 1 {
			d.t.Fatalf("mutation %q matches %d times in forward_bf16.cu, want 1", bf16Old, n)
		}
		bf16Src = strings.Replace(bf16Src, bf16Old, bf16New, 1)
	}
	fusedSrc := fusedDecodeCUDA
	if fusedOld != "" {
		if n := strings.Count(fusedSrc, fusedOld); n != 1 {
			d.t.Fatalf("mutation %q matches %d times in fused_decode.cu, want 1", fusedOld, n)
		}
		fusedSrc = strings.Replace(fusedSrc, fusedOld, fusedNew, 1)
	}
	gm, err := compileGPUModules(d.dev, ForwardKernelsCUDA+"\n"+bf16Src+"\n"+fusedSrc)
	if gm != nil {
		d.t.Cleanup(func() {
			if err := gm.Close(); err != nil {
				d.t.Error(err)
			}
		})
	}
	if err != nil {
		d.t.Fatalf("compile mutant: %v", err)
	}
	return gm
}

func TestFusedRMSNormQKVRoPEStoreParity(t *testing.T) {
	d := newParityDevice(t)
	const maxSeq = 9
	positions := []int{0, 1, 7}

	for _, shape := range fusedDecodeTestShapes {
		for _, bias := range []bool{false, true} {
			for _, pos := range positions {
				name := fmt.Sprintf("%s/bias=%v/pos=%d", shape.name, bias, pos)
				t.Run(name, func(t *testing.T) {
					d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
					p, err := NewFusedDecodeLayerPlan(
						shape.hidden, shape.inter, shape.qHeads, shape.kvHeads, shape.headDim,
						maxSeq, 1e-5, 1e4,
					)
					if err != nil {
						t.Fatal(err)
					}
					b, baseCache := makeFusedDecodeLayerFixture(p, bias, false, pos)
					dx := d.upload(b.x)
					dg := d.upload(b.attnNorm)
					dwq := d.upload(b.wq)
					dwk := d.upload(b.wk)
					dwv := d.upload(b.wv)
					optUpload := func(s []float32) any {
						if s == nil {
							return nullDevicePtr
						}
						return d.upload(s)
					}
					dbq := optUpload(b.qBias)
					dbk := optUpload(b.kBias)
					dbv := optUpload(b.vBias)

					// Fused launch.
					dq := d.upload(make([]float32, p.qDim))
					dkCache := d.upload(baseCache.k)
					dvCache := d.upload(baseCache.v)
					grid := p.QKVRoPEStoreGrid()
					if err := d.dev.WithContext(func() error {
						return cudalaunch.Kernel(d.gm.fusedQKVRoPEStoreFn).Grid(grid, 1, 1).Block(128, 1, 1).
							Launch(
								dx, dg, dwq, dwk, dwv, dbq, dbk, dbv,
								dq, dkCache, dvCache,
								int32(p.hidden), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), int32(pos),
								p.epsilon, p.theta,
								float32(0), float32(0), float32(0), int32(0),
							)
					}); err != nil {
						t.Fatal(err)
					}

					// Unfused GPU sequence: RMSNorm -> 3x LinearGEMV -> ApplyRoPE -> StoreKVCache.
					dNorm := d.upload(make([]float32, p.hidden))
					uQ := d.upload(make([]float32, p.qDim))
					uK := d.upload(make([]float32, p.kvDim))
					uV := d.upload(make([]float32, p.kvDim))
					uKCache := d.upload(baseCache.k)
					uVCache := d.upload(baseCache.v)
					if err := d.dev.WithContext(func() error {
						if err := cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(dx, dg, dNorm, int32(p.hidden), p.epsilon); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.qDim), 1, 1).Block(128, 1, 1).
							Launch(dNorm, dwq, dbq, uQ, int32(p.hidden), int32(p.qDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.kvDim), 1, 1).Block(128, 1, 1).
							Launch(dNorm, dwk, dbk, uK, int32(p.hidden), int32(p.kvDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.kvDim), 1, 1).Block(128, 1, 1).
							Launch(dNorm, dwv, dbv, uV, int32(p.hidden), int32(p.kvDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.ropeFn).Grid(uint32(p.qHeads), 1, 1).Block(uint32(p.headDim/2), 1, 1).
							Launch(uQ, uK, int32(pos), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), p.theta); err != nil {
							return err
						}
						return cudalaunch.Kernel(d.gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1).
							Launch(uK, uV, uKCache, uVCache, int32(pos), int32(p.kvHeads), int32(p.headDim))
					}); err != nil {
						t.Fatal(err)
					}

					// CPU reference.
					wantQ := make([]float32, p.qDim)
					cpuCache := cloneFusedCache(baseCache)
					fusedRMSNormQKVRoPEStoreCPU(
						b.x, b.attnNorm, b.wq, b.wk, b.wv, b.qBias, b.kBias, b.vBias,
						wantQ, cpuCache.k, cpuCache.v,
						p.hidden, p.qHeads, p.kvHeads, p.headDim, pos, p.epsilon, p.theta,
					)

					gotQ := d.download(dq)
					gotKCache := d.download(dkCache)
					gotVCache := d.download(dvCache)
					unfusedQ := d.download(uQ)
					unfusedKCache := d.download(uKCache)
					unfusedVCache := d.download(uVCache)

					row := func(s []float32, r int) []float32 {
						return s[r*p.kvDim : (r+1)*p.kvDim]
					}
					if i := parityMismatch(gotQ, wantQ, kernelTolerance); i >= 0 {
						t.Errorf("fused q[%d]: got %g want %g", i, gotQ[i], wantQ[i])
					}
					if i := parityMismatch(gotQ, unfusedQ, kernelTolerance); i >= 0 {
						t.Errorf("fused vs unfused q[%d]: got %g want %g", i, gotQ[i], unfusedQ[i])
					}
					if i := parityMismatch(row(gotKCache, pos), row(cpuCache.k, pos), kernelTolerance); i >= 0 {
						t.Errorf("fused k_cache[%d]: got %g want %g", i, row(gotKCache, pos)[i], row(cpuCache.k, pos)[i])
					}
					if i := parityMismatch(row(gotKCache, pos), row(unfusedKCache, pos), kernelTolerance); i >= 0 {
						t.Errorf("fused vs unfused k_cache[%d]: got %g want %g", i, row(gotKCache, pos)[i], row(unfusedKCache, pos)[i])
					}
					if i := parityMismatch(row(gotVCache, pos), row(cpuCache.v, pos), kernelTolerance); i >= 0 {
						t.Errorf("fused v_cache[%d]: got %g want %g", i, row(gotVCache, pos)[i], row(cpuCache.v, pos)[i])
					}
					if i := parityMismatch(row(gotVCache, pos), row(unfusedVCache, pos), kernelTolerance); i >= 0 {
						t.Errorf("fused vs unfused v_cache[%d]: got %g want %g", i, row(gotVCache, pos)[i], row(unfusedVCache, pos)[i])
					}
					for r := 0; r < maxSeq; r++ {
						if r == pos {
							continue
						}
						if !sameBits(row(gotKCache, r), row(baseCache.k, r)) || !sameBits(row(gotVCache, r), row(baseCache.v, r)) {
							t.Errorf("pos %d mutated cache row %d", pos, r)
						}
					}

					// Reject QKVRoPEStore defects.
					for _, defect := range []fusedDecodeDefect{
						fusedDefectNoRsqrt,
						fusedDefectVTerm,
						fusedDefectRoPEPrev,
						fusedDefectRowNext,
					} {
						badCache := cloneFusedCache(baseCache)
						bad := fusedDecodeDefectiveOracle(p, b, badCache, pos, defect)
						qDiff := parityMismatch(gotQ, bad.Q, kernelTolerance) >= 0
						kDiff := parityMismatch(row(gotKCache, pos), row(badCache.k, pos), kernelTolerance) >= 0
						vDiff := parityMismatch(row(gotVCache, pos), row(badCache.v, pos), kernelTolerance) >= 0
						if !qDiff && !kDiff && !vDiff {
							t.Errorf("defect %v not rejected", defect)
						}
					}
				})
			}
		}
	}
}

func TestFusedOutProjResidualParity(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			qDim := shape.qHeads * shape.headDim
			hidden := shape.hidden
			attnOut := parityValues(qDim, 3)
			residual := parityValues(hidden, 7)
			for i := range residual {
				if residual[i] == 0 {
					residual[i] = 0.25
				}
			}
			wo := parityValues(hidden*qDim, 11)

			dAttn := d.upload(attnOut)
			dRes := d.upload(residual)
			dWO := d.upload(wo)
			dOut := d.upload(make([]float32, hidden))
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.fusedOutProjResFn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1).
					Launch(dAttn, dRes, dWO, dOut, int32(qDim), int32(hidden))
			}); err != nil {
				t.Fatal(err)
			}

			uProj := d.upload(make([]float32, hidden))
			uOut := d.upload(residual)
			if err := d.dev.WithContext(func() error {
				if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1).
					Launch(dAttn, dWO, nullDevicePtr, uProj, int32(qDim), int32(hidden)); err != nil {
					return err
				}
				return cudalaunch.Kernel(d.gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(uOut, uProj, int32(hidden))
			}); err != nil {
				t.Fatal(err)
			}

			want := make([]float32, hidden)
			fusedOutProjResidualCPU(attnOut, residual, wo, want, qDim, hidden)
			got := d.download(dOut)
			unfused := d.download(uOut)

			if i := parityMismatch(got, want, kernelTolerance); i >= 0 {
				t.Errorf("fused out[%d]: got %g want %g", i, got[i], want[i])
			}
			if i := parityMismatch(got, unfused, kernelTolerance); i >= 0 {
				t.Errorf("fused vs unfused out[%d]: got %g want %g", i, got[i], unfused[i])
			}
			noRes := matmulVec(attnOut, wo, qDim, hidden)
			if parityMismatch(got, noRes, kernelTolerance) < 0 {
				t.Error("missing residual defect not rejected")
			}
		})
	}
}

func TestFusedRMSNormGateUpSwiGLUParity(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			hidden, inter := shape.hidden, shape.inter
			const eps = float32(1e-5)
			x := parityValues(hidden, 2)
			normW := parityValues(hidden, 5)
			wGate := parityValues(inter*hidden, 9)
			wUp := parityValues(inter*hidden, 15)

			dx := d.upload(x)
			dNormW := d.upload(normW)
			dGateW := d.upload(wGate)
			dUpW := d.upload(wUp)
			dHB := d.upload(make([]float32, inter))
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.fusedGateUpSwiGLUFn).Grid(uint32(inter), 1, 1).Block(128, 1, 1).
					Launch(dx, dNormW, dGateW, dUpW, dHB, int32(hidden), int32(inter), eps)
			}); err != nil {
				t.Fatal(err)
			}

			uNorm := d.upload(make([]float32, hidden))
			uGate := d.upload(make([]float32, inter))
			uUp := d.upload(make([]float32, inter))
			uHB := d.upload(make([]float32, inter))
			if err := d.dev.WithContext(func() error {
				if err := cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(dx, dNormW, uNorm, int32(hidden), eps); err != nil {
					return err
				}
				if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(inter), 1, 1).Block(128, 1, 1).
					Launch(uNorm, dGateW, nullDevicePtr, uGate, int32(hidden), int32(inter)); err != nil {
					return err
				}
				if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(inter), 1, 1).Block(128, 1, 1).
					Launch(uNorm, dUpW, nullDevicePtr, uUp, int32(hidden), int32(inter)); err != nil {
					return err
				}
				return cudalaunch.Kernel(d.gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(uGate, uUp, uHB, int32(inter))
			}); err != nil {
				t.Fatal(err)
			}

			want := make([]float32, inter)
			fusedRMSNormGateUpSwiGLUCPU(x, normW, wGate, wUp, want, hidden, inter, eps)
			got := d.download(dHB)
			unfused := d.download(uHB)

			if i := parityMismatch(got, want, kernelTolerance); i >= 0 {
				t.Errorf("fused hb[%d]: got %g want %g", i, got[i], want[i])
			}
			if i := parityMismatch(got, unfused, kernelTolerance); i >= 0 {
				t.Errorf("fused vs unfused hb[%d]: got %g want %g", i, got[i], unfused[i])
			}

			// Swapped gate/up control.
			swapped := make([]float32, inter)
			fusedRMSNormGateUpSwiGLUCPU(x, normW, wUp, wGate, swapped, hidden, inter, eps)
			if parityMismatch(got, swapped, kernelTolerance) < 0 {
				t.Error("swapped gate/up defect not rejected")
			}
		})
	}
}

func TestFusedDownProjResidualParity(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			hidden, inter := shape.hidden, shape.inter
			hb := parityValues(inter, 4)
			residual := parityValues(hidden, 8)
			for i := range residual {
				if residual[i] == 0 {
					residual[i] = 0.25
				}
			}
			wDown := parityValues(hidden*inter, 12)

			dHB := d.upload(hb)
			dRes := d.upload(residual)
			dWDown := d.upload(wDown)
			dOut := d.upload(make([]float32, hidden))
			if err := d.dev.WithContext(func() error {
				return cudalaunch.Kernel(d.gm.fusedDownProjResFn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1).
					Launch(dHB, dRes, dWDown, dOut, int32(inter), int32(hidden))
			}); err != nil {
				t.Fatal(err)
			}

			uProj := d.upload(make([]float32, hidden))
			uOut := d.upload(residual)
			if err := d.dev.WithContext(func() error {
				if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1).
					Launch(dHB, dWDown, nullDevicePtr, uProj, int32(inter), int32(hidden)); err != nil {
					return err
				}
				return cudalaunch.Kernel(d.gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
					Launch(uOut, uProj, int32(hidden))
			}); err != nil {
				t.Fatal(err)
			}

			want := make([]float32, hidden)
			fusedDownProjResidualCPU(hb, residual, wDown, want, inter, hidden)
			got := d.download(dOut)
			unfused := d.download(uOut)

			if i := parityMismatch(got, want, kernelTolerance); i >= 0 {
				t.Errorf("fused out[%d]: got %g want %g", i, got[i], want[i])
			}
			if i := parityMismatch(got, unfused, kernelTolerance); i >= 0 {
				t.Errorf("fused vs unfused out[%d]: got %g want %g", i, got[i], unfused[i])
			}
			noRes := matmulVec(hb, wDown, inter, hidden)
			if parityMismatch(got, noRes, kernelTolerance) < 0 {
				t.Error("missing down residual defect not rejected")
			}
		})
	}
}

func fusedQKVRoPEStoreBF16Mismatch(
	d *parityDevice, gm, ref *GPUModules,
	in, qHeads, kvHeads, headDim, pos, maxSeq int,
	bias bool,
) (which int, idx int) {
	d.t.Helper()
	qDim := qHeads * headDim
	kvDim := kvHeads * headDim
	r := rand.New(rand.NewPCG(uint64(in*100+pos), uint64(qDim*100+kvDim)))
	x := bf16Activations(r, in)
	g := bf16Activations(r, in)
	wq16 := bf16Weights(r, qDim*in)
	wk16 := bf16Weights(r, kvDim*in)
	wv16 := bf16Weights(r, kvDim*in)

	var bq, bk, bv any = nullDevicePtr, nullDevicePtr, nullDevicePtr
	if bias {
		bq = d.upload(bf16Activations(r, qDim))
		bk = d.upload(bf16Activations(r, kvDim))
		bv = d.upload(bf16Activations(r, kvDim))
	}
	dx := d.upload(x)
	dg := d.upload(g)
	dwq16, dwk16, dwv16 := uploadBF16(d, wq16), uploadBF16(d, wk16), uploadBF16(d, wv16)
	dwq32, dwk32, dwv32 := d.upload(widenBF16(wq16)), d.upload(widenBF16(wk16)), d.upload(widenBF16(wv16))

	grid := uint32((qHeads + kvHeads) * (headDim / 2))
	run := func(fn driver.Function, wq, wk, wv any) [3][]float32 {
		oq := d.upload(make([]float32, qDim))
		kc := d.upload(make([]float32, maxSeq*kvDim))
		vc := d.upload(make([]float32, maxSeq*kvDim))
		if err := d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(grid, 1, 1).Block(128, 1, 1).
				Launch(dx, dg, wq, wk, wv, bq, bk, bv, oq, kc, vc,
					int32(in), int32(qHeads), int32(kvHeads), int32(headDim), int32(pos),
					float32(1e-5), float32(1e4),
					float32(0), float32(0), float32(0), int32(0))
		}); err != nil {
			d.t.Fatal(err)
		}
		return [3][]float32{d.download(oq), d.download(kc), d.download(vc)}
	}
	got := run(gm.fusedQKVRoPEStoreBF16Fn, dwq16, dwk16, dwv16)
	want := run(ref.fusedQKVRoPEStoreFn, dwq32, dwk32, dwv32)
	for p := range 3 {
		if i := bitMismatch(got[p], want[p]); i >= 0 {
			return p, i
		}
	}
	return -1, -1
}

func fusedOutProjResBF16Mismatch(d *parityDevice, gm, ref *GPUModules, inDim, outDim int) int {
	d.t.Helper()
	r := rand.New(rand.NewPCG(uint64(inDim), uint64(outDim+1000)))
	attnOut := bf16Activations(r, inDim)
	res := bf16Activations(r, outDim)
	w16 := bf16Weights(r, outDim*inDim)
	dAttn := d.upload(attnOut)
	dRes := d.upload(res)
	dw16 := uploadBF16(d, w16)
	dw32 := d.upload(widenBF16(w16))

	run := func(fn driver.Function, w any) []float32 {
		o := d.upload(make([]float32, outDim))
		if err := d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(uint32(outDim), 1, 1).Block(128, 1, 1).
				Launch(dAttn, dRes, w, o, int32(inDim), int32(outDim))
		}); err != nil {
			d.t.Fatal(err)
		}
		return d.download(o)
	}
	return bitMismatch(run(gm.fusedOutProjResBF16Fn, dw16), run(ref.fusedOutProjResFn, dw32))
}

func fusedGateUpSwiGLUBF16Mismatch(d *parityDevice, gm, ref *GPUModules, hidden, inter int) int {
	d.t.Helper()
	r := rand.New(rand.NewPCG(uint64(hidden), uint64(inter+2000)))
	x := bf16Activations(r, hidden)
	normW := bf16Activations(r, hidden)
	wg16 := bf16Weights(r, inter*hidden)
	wu16 := bf16Weights(r, inter*hidden)
	dx := d.upload(x)
	dNormW := d.upload(normW)
	dwg16, dwu16 := uploadBF16(d, wg16), uploadBF16(d, wu16)
	dwg32, dwu32 := d.upload(widenBF16(wg16)), d.upload(widenBF16(wu16))

	run := func(fn driver.Function, wg, wu any) []float32 {
		o := d.upload(make([]float32, inter))
		if err := d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(uint32(inter), 1, 1).Block(128, 1, 1).
				Launch(dx, dNormW, wg, wu, o, int32(hidden), int32(inter), float32(1e-5))
		}); err != nil {
			d.t.Fatal(err)
		}
		return d.download(o)
	}
	return bitMismatch(run(gm.fusedGateUpSwiGLUBF16Fn, dwg16, dwu16), run(ref.fusedGateUpSwiGLUFn, dwg32, dwu32))
}

func fusedDownProjResBF16Mismatch(d *parityDevice, gm, ref *GPUModules, inter, hidden int) int {
	d.t.Helper()
	r := rand.New(rand.NewPCG(uint64(inter), uint64(hidden+3000)))
	hb := bf16Activations(r, inter)
	res := bf16Activations(r, hidden)
	w16 := bf16Weights(r, hidden*inter)
	dHB := d.upload(hb)
	dRes := d.upload(res)
	dw16 := uploadBF16(d, w16)
	dw32 := d.upload(widenBF16(w16))

	run := func(fn driver.Function, w any) []float32 {
		o := d.upload(make([]float32, hidden))
		if err := d.dev.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(uint32(hidden), 1, 1).Block(128, 1, 1).
				Launch(dHB, dRes, w, o, int32(inter), int32(hidden))
		}); err != nil {
			d.t.Fatal(err)
		}
		return d.download(o)
	}
	return bitMismatch(run(gm.fusedDownProjResBF16Fn, dw16), run(ref.fusedDownProjResFn, dw32))
}

func TestFusedRMSNormQKVRoPEStoreBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		for _, bias := range []bool{false, true} {
			for _, pos := range []int{0, 1, 7} {
				name := fmt.Sprintf("%s/bias=%v/pos=%d", shape.name, bias, pos)
				t.Run(name, func(t *testing.T) {
					d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
					if which, idx := fusedQKVRoPEStoreBF16Mismatch(d, d.gm, d.gm, shape.hidden, shape.qHeads, shape.kvHeads, shape.headDim, pos, 9, bias); which >= 0 {
						t.Errorf("output %d[%d] differs in BF16 vs widened FP32", which, idx)
					}
				})
			}
		}
	}
}

func TestFusedOutProjResidualBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			qDim := shape.qHeads * shape.headDim
			if i := fusedOutProjResBF16Mismatch(d, d.gm, d.gm, qDim, shape.hidden); i >= 0 {
				t.Errorf("out[%d] differs in BF16 vs widened FP32", i)
			}
		})
	}
}

func TestFusedRMSNormGateUpSwiGLUBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			if i := fusedGateUpSwiGLUBF16Mismatch(d, d.gm, d.gm, shape.hidden, shape.inter); i >= 0 {
				t.Errorf("out_hb[%d] differs in BF16 vs widened FP32", i)
			}
		})
	}
}

func TestFusedDownProjResidualBF16BitExact(t *testing.T) {
	d := newParityDevice(t)
	for _, shape := range fusedDecodeTestShapes {
		t.Run(shape.name, func(t *testing.T) {
			d := &parityDevice{t: t, dev: d.dev, gm: d.gm}
			if i := fusedDownProjResBF16Mismatch(d, d.gm, d.gm, shape.inter, shape.hidden); i >= 0 {
				t.Errorf("out[%d] differs in BF16 vs widened FP32", i)
			}
		})
	}
}

func TestFusedDecodeBF16Controls(t *testing.T) {
	d := newParityDevice(t)

	// 1. Shift-by-15 in bf16_to_f32 must be caught across all 4 BF16 kernels on
	// both aligned and odd shapes.
	gmShift := mutateFusedDecode(d, bf16WidenSource, "((unsigned int)b) << 15", "", "")
	for _, shape := range fusedDecodeTestShapes {
		qDim := shape.qHeads * shape.headDim
		if which, _ := fusedQKVRoPEStoreBF16Mismatch(d, gmShift, d.gm, shape.hidden, shape.qHeads, shape.kvHeads, shape.headDim, 1, 9, true); which < 0 {
			t.Errorf("shift-by-15 not detected by FusedRMSNormQKVRoPEStoreBF16 on %s", shape.name)
		}
		if fusedOutProjResBF16Mismatch(d, gmShift, d.gm, qDim, shape.hidden) < 0 {
			t.Errorf("shift-by-15 not detected by FusedOutProjResidualBF16 on %s", shape.name)
		}
		if fusedGateUpSwiGLUBF16Mismatch(d, gmShift, d.gm, shape.hidden, shape.inter) < 0 {
			t.Errorf("shift-by-15 not detected by FusedRMSNormGateUpSwiGLUBF16 on %s", shape.name)
		}
		if fusedDownProjResBF16Mismatch(d, gmShift, d.gm, shape.inter, shape.hidden) < 0 {
			t.Errorf("shift-by-15 not detected by FusedDownProjResidualBF16 on %s", shape.name)
		}
	}

	// 2. Lane swaps on ushort4 loads must fail on aligned vector-path shapes
	// and stay inactive on odd scalar-path shapes.
	gmQKVSwap := mutateFusedDecode(d, "", "", bf16FusedQ0LanesSource, bf16FusedQ0LanesSwapped)
	if which, _ := fusedQKVRoPEStoreBF16Mismatch(d, gmQKVSwap, d.gm, 64, 4, 2, 16, 1, 9, true); which < 0 {
		t.Error("permuted uq0 ushort4 lanes not detected on aligned shape")
	}
	if which, _ := fusedQKVRoPEStoreBF16Mismatch(d, gmQKVSwap, d.gm, 7, 2, 1, 6, 1, 9, true); which >= 0 {
		t.Error("permuted uq0 ushort4 lanes fired on odd in_dim scalar path")
	}

	gmOutSwap := mutateFusedDecode(d, "", "", bf16FusedOutProjLanesSource, bf16FusedOutProjSwapped)
	if fusedOutProjResBF16Mismatch(d, gmOutSwap, d.gm, 64, 64) < 0 {
		t.Error("permuted uo ushort4 lanes not detected on aligned shape")
	}
	if fusedOutProjResBF16Mismatch(d, gmOutSwap, d.gm, 7, 8) >= 0 {
		t.Error("permuted uo ushort4 lanes fired on odd in_dim scalar path")
	}

	gmGateSwap := mutateFusedDecode(d, "", "", bf16FusedGateLanesSource, bf16FusedGateLanesSwapped)
	if fusedGateUpSwiGLUBF16Mismatch(d, gmGateSwap, d.gm, 64, 128) < 0 {
		t.Error("permuted ugate ushort4 lanes not detected on aligned shape")
	}
	if fusedGateUpSwiGLUBF16Mismatch(d, gmGateSwap, d.gm, 7, 11) >= 0 {
		t.Error("permuted ugate ushort4 lanes fired on odd hidden_dim scalar path")
	}

	gmDownSwap := mutateFusedDecode(d, "", "", bf16FusedDownLanesSource, bf16FusedDownLanesSwapped)
	if fusedDownProjResBF16Mismatch(d, gmDownSwap, d.gm, 128, 64) < 0 {
		t.Error("permuted udown ushort4 lanes not detected on aligned shape")
	}
	if fusedDownProjResBF16Mismatch(d, gmDownSwap, d.gm, 11, 7) >= 0 {
		t.Error("permuted udown ushort4 lanes fired on odd inter_dim scalar path")
	}
}

func TestFusedDecodeLayerGPUParity(t *testing.T) {
	k := newForwardKit(t)
	const maxSeq = 9
	positions := []int{0, 1, 7}

	attnShared, err := cudaattn.DecodeSharedMemoryBytes(maxSeq, attnThreads)
	if err != nil {
		t.Fatal(err)
	}

	for _, shape := range fusedDecodeTestShapes {
		for _, bias := range []bool{false, true} {
			for _, pos := range positions {
				name := fmt.Sprintf("%s/bias=%v/pos=%d", shape.name, bias, pos)
				t.Run(name, func(t *testing.T) {
					d := &parityDevice{t: t, dev: k.d.dev, gm: k.d.gm}
					p, err := NewFusedDecodeLayerPlan(
						shape.hidden, shape.inter, shape.qHeads, shape.kvHeads, shape.headDim,
						maxSeq, 1e-5, 1e4,
					)
					if err != nil {
						t.Fatal(err)
					}
					// Use BF16-widened weights so we can simultaneously verify:
					// 1) 5-kernel FP32 GPU vs executeFusedDecodeLayerCPU (<= 1e-5)
					// 2) 5-kernel FP32 GPU vs 15-kernel unfused GPU (<= 1e-5)
					// 3) 5-kernel BF16 GPU vs 5-kernel FP32 GPU (bit-for-bit exact!)
					b, baseCache := makeFusedDecodeLayerFixture(p, bias, true, pos)

					cpuCache := cloneFusedCache(baseCache)
					want, err := executeFusedDecodeLayerCPU(p, b, cpuCache, pos)
					if err != nil {
						t.Fatal(err)
					}

					dx := d.upload(b.x)
					dAttnNorm := d.upload(b.attnNorm)
					dFFNNorm := d.upload(b.ffnNorm)
					optUpload := func(s []float32) any {
						if s == nil {
							return nullDevicePtr
						}
						return d.upload(s)
					}
					dbq := optUpload(b.qBias)
					dbk := optUpload(b.kBias)
					dbv := optUpload(b.vBias)

					dwq32, dwk32, dwv32 := d.upload(b.wq), d.upload(b.wk), d.upload(b.wv)
					dwo32 := d.upload(b.wo)
					dwGate32, dwUp32, dwDown32 := d.upload(b.wGate), d.upload(b.wUp), d.upload(b.wDown)

					dwq16, dwk16, dwv16 := uploadBF16(d, roundBF16(b.wq)), uploadBF16(d, roundBF16(b.wk)), uploadBF16(d, roundBF16(b.wv))
					dwo16 := uploadBF16(d, roundBF16(b.wo))
					dwGate16, dwUp16, dwDown16 := uploadBF16(d, roundBF16(b.wGate)), uploadBF16(d, roundBF16(b.wUp)), uploadBF16(d, roundBF16(b.wDown))

					attnScale := cudaattn.DefaultConfig(p.qHeads, p.kvHeads, p.headDim).Scale
					qkvGrid := p.QKVRoPEStoreGrid()

					run5KernelLayer := func(
						qkvFn, outFn, gateUpFn, downFn driver.Function,
						wq, wk, wv, wo, wGate, wUp, wDown any,
					) (fusedDecodeLayerOutputs, attnCache) {
						dQ := d.upload(make([]float32, p.qDim))
						dKC := d.upload(baseCache.k)
						dVC := d.upload(baseCache.v)
						dAttnOut := d.upload(make([]float32, p.qDim))
						dAttnRes := d.upload(make([]float32, p.hidden))
						dHB := d.upload(make([]float32, p.inter))
						dOut := d.upload(make([]float32, p.hidden))

						if err := d.dev.WithContext(func() error {
							if err := cudalaunch.Kernel(qkvFn).Grid(qkvGrid, 1, 1).Block(128, 1, 1).
								Launch(
									dx, dAttnNorm, wq, wk, wv, dbq, dbk, dbv,
									dQ, dKC, dVC,
									int32(p.hidden), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), int32(pos),
									p.epsilon, p.theta,
									float32(0), float32(0), float32(0), int32(0),
								); err != nil {
								return err
							}
							if err := cudalaunch.Kernel(k.attnFn).Grid(uint32(p.qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(uint32(attnShared)).
								Launch(dQ, dKC, dVC, dAttnOut, int32(pos), int32(p.maxSeq), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), attnScale); err != nil {
								return err
							}
							if err := cudalaunch.Kernel(outFn).Grid(uint32(p.hidden), 1, 1).Block(128, 1, 1).
								Launch(dAttnOut, dx, wo, dAttnRes, int32(p.qDim), int32(p.hidden)); err != nil {
								return err
							}
							if err := cudalaunch.Kernel(gateUpFn).Grid(uint32(p.inter), 1, 1).Block(128, 1, 1).
								Launch(dAttnRes, dFFNNorm, wGate, wUp, dHB, int32(p.hidden), int32(p.inter), p.epsilon); err != nil {
								return err
							}
							return cudalaunch.Kernel(downFn).Grid(uint32(p.hidden), 1, 1).Block(128, 1, 1).
								Launch(dHB, dAttnRes, wDown, dOut, int32(p.inter), int32(p.hidden))
						}); err != nil {
							t.Fatal(err)
						}
						return fusedDecodeLayerOutputs{
							Q:            d.download(dQ),
							AttnOut:      d.download(dAttnOut),
							AttnResidual: d.download(dAttnRes),
							HB:           d.download(dHB),
							Out:          d.download(dOut),
						}, attnCache{
							k: d.download(dKC),
							v: d.download(dVC),
						}
					}

					gotF32, cacheF32 := run5KernelLayer(
						d.gm.fusedQKVRoPEStoreFn, d.gm.fusedOutProjResFn, d.gm.fusedGateUpSwiGLUFn, d.gm.fusedDownProjResFn,
						dwq32, dwk32, dwv32, dwo32, dwGate32, dwUp32, dwDown32,
					)
					gotBF16, cacheBF16 := run5KernelLayer(
						d.gm.fusedQKVRoPEStoreBF16Fn, d.gm.fusedOutProjResBF16Fn, d.gm.fusedGateUpSwiGLUBF16Fn, d.gm.fusedDownProjResBF16Fn,
						dwq16, dwk16, dwv16, dwo16, dwGate16, dwUp16, dwDown16,
					)

					// Unfused 15-kernel GPU layer execution.
					uX := d.upload(b.x)
					uNorm := d.upload(make([]float32, p.hidden))
					uQ := d.upload(make([]float32, p.qDim))
					uK := d.upload(make([]float32, p.kvDim))
					uV := d.upload(make([]float32, p.kvDim))
					uKC := d.upload(baseCache.k)
					uVC := d.upload(baseCache.v)
					uAttnOut := d.upload(make([]float32, p.qDim))
					uGate := d.upload(make([]float32, p.inter))
					uUp := d.upload(make([]float32, p.inter))
					uHB := d.upload(make([]float32, p.inter))
					uFFNOut := d.upload(make([]float32, p.hidden))
					if err := d.dev.WithContext(func() error {
						if err := cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(uX, dAttnNorm, uNorm, int32(p.hidden), p.epsilon); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.qDim), 1, 1).Block(128, 1, 1).
							Launch(uNorm, dwq32, dbq, uQ, int32(p.hidden), int32(p.qDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.kvDim), 1, 1).Block(128, 1, 1).
							Launch(uNorm, dwk32, dbk, uK, int32(p.hidden), int32(p.kvDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.kvDim), 1, 1).Block(128, 1, 1).
							Launch(uNorm, dwv32, dbv, uV, int32(p.hidden), int32(p.kvDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.ropeFn).Grid(uint32(p.qHeads), 1, 1).Block(uint32(p.headDim/2), 1, 1).
							Launch(uQ, uK, int32(pos), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), p.theta); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1).
							Launch(uK, uV, uKC, uVC, int32(pos), int32(p.kvHeads), int32(p.headDim)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(k.attnFn).Grid(uint32(p.qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(uint32(attnShared)).
							Launch(uQ, uKC, uVC, uAttnOut, int32(pos), int32(p.maxSeq), int32(p.qHeads), int32(p.kvHeads), int32(p.headDim), attnScale); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.hidden), 1, 1).Block(128, 1, 1).
							Launch(uAttnOut, dwo32, nullDevicePtr, uNorm, int32(p.qDim), int32(p.hidden)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(uX, uNorm, int32(p.hidden)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.normFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(uX, dFFNNorm, uNorm, int32(p.hidden), p.epsilon); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.inter), 1, 1).Block(128, 1, 1).
							Launch(uNorm, dwGate32, nullDevicePtr, uGate, int32(p.hidden), int32(p.inter)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.inter), 1, 1).Block(128, 1, 1).
							Launch(uNorm, dwUp32, nullDevicePtr, uUp, int32(p.hidden), int32(p.inter)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.swigluFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(uGate, uUp, uHB, int32(p.inter)); err != nil {
							return err
						}
						if err := cudalaunch.Kernel(d.gm.gemvFn).Grid(uint32(p.hidden), 1, 1).Block(128, 1, 1).
							Launch(uHB, dwDown32, nullDevicePtr, uFFNOut, int32(p.inter), int32(p.hidden)); err != nil {
							return err
						}
						return cudalaunch.Kernel(d.gm.addResFn).Grid(1, 1, 1).Block(256, 1, 1).
							Launch(uX, uFFNOut, int32(p.hidden))
					}); err != nil {
						t.Fatal(err)
					}
					unfusedOut := d.download(uX)

					row := func(s []float32, r int) []float32 {
						return s[r*p.kvDim : (r+1)*p.kvDim]
					}
					// 1. GPU FP32 vs CPU reference across all intermediates.
					if i := parityMismatch(gotF32.Q, want.Q, kernelTolerance); i >= 0 {
						t.Errorf("Q[%d]: got %g want %g", i, gotF32.Q[i], want.Q[i])
					}
					if i := parityMismatch(row(cacheF32.k, pos), row(cpuCache.k, pos), kernelTolerance); i >= 0 {
						t.Errorf("KCache[%d]: got %g want %g", i, row(cacheF32.k, pos)[i], row(cpuCache.k, pos)[i])
					}
					if i := parityMismatch(row(cacheF32.v, pos), row(cpuCache.v, pos), kernelTolerance); i >= 0 {
						t.Errorf("VCache[%d]: got %g want %g", i, row(cacheF32.v, pos)[i], row(cpuCache.v, pos)[i])
					}
					if i := parityMismatch(gotF32.AttnOut, want.AttnOut, kernelTolerance); i >= 0 {
						t.Errorf("AttnOut[%d]: got %g want %g", i, gotF32.AttnOut[i], want.AttnOut[i])
					}
					if i := parityMismatch(gotF32.AttnResidual, want.AttnResidual, kernelTolerance); i >= 0 {
						t.Errorf("AttnResidual[%d]: got %g want %g", i, gotF32.AttnResidual[i], want.AttnResidual[i])
					}
					if i := parityMismatch(gotF32.HB, want.HB, kernelTolerance); i >= 0 {
						t.Errorf("HB[%d]: got %g want %g", i, gotF32.HB[i], want.HB[i])
					}
					if i := parityMismatch(gotF32.Out, want.Out, kernelTolerance); i >= 0 {
						t.Errorf("Out[%d]: got %g want %g", i, gotF32.Out[i], want.Out[i])
					}
					// 2. GPU 5-kernel vs GPU 15-kernel unfused.
					if i := parityMismatch(gotF32.Out, unfusedOut, kernelTolerance); i >= 0 {
						t.Errorf("5-kernel vs 15-kernel Out[%d]: got %g want %g", i, gotF32.Out[i], unfusedOut[i])
					}
					// 3. GPU 5-kernel BF16 vs GPU 5-kernel FP32 bit-for-bit equality.
					if i := bitMismatch(gotBF16.Q, gotF32.Q); i >= 0 {
						t.Errorf("BF16 vs FP32 Q[%d] bit mismatch", i)
					}
					if i := bitMismatch(row(cacheBF16.k, pos), row(cacheF32.k, pos)); i >= 0 {
						t.Errorf("BF16 vs FP32 KCache[%d] bit mismatch", i)
					}
					if i := bitMismatch(row(cacheBF16.v, pos), row(cacheF32.v, pos)); i >= 0 {
						t.Errorf("BF16 vs FP32 VCache[%d] bit mismatch", i)
					}
					if i := bitMismatch(gotBF16.AttnResidual, gotF32.AttnResidual); i >= 0 {
						t.Errorf("BF16 vs FP32 AttnResidual[%d] bit mismatch", i)
					}
					if i := bitMismatch(gotBF16.HB, gotF32.HB); i >= 0 {
						t.Errorf("BF16 vs FP32 HB[%d] bit mismatch", i)
					}
					if i := bitMismatch(gotBF16.Out, gotF32.Out); i >= 0 {
						t.Errorf("BF16 vs FP32 Out[%d] bit mismatch", i)
					}

					// 4. Reject all 7 defect controls.
					for _, dCtrl := range allFusedDecodeDefects {
						if dCtrl.defect == fusedDefectHeadSwap && (shape.kvHeads == 1 || shape.kvHeads == shape.qHeads) {
							continue
						}
						badCache := cloneFusedCache(baseCache)
						bad := fusedDecodeDefectiveOracle(p, b, badCache, pos, dCtrl.defect)
						rejected := parityMismatch(gotF32.Out, bad.Out, kernelTolerance) >= 0 ||
							parityMismatch(gotF32.AttnResidual, bad.AttnResidual, kernelTolerance) >= 0 ||
							parityMismatch(gotF32.HB, bad.HB, kernelTolerance) >= 0 ||
							parityMismatch(gotF32.Q, bad.Q, kernelTolerance) >= 0 ||
							parityMismatch(row(cacheF32.k, pos), row(badCache.k, pos), kernelTolerance) >= 0 ||
							parityMismatch(row(cacheF32.v, pos), row(badCache.v, pos), kernelTolerance) >= 0
						if !rejected {
							t.Errorf("defect %q not rejected on GPU", dCtrl.name)
						}
					}
				})
			}
		}
	}
}
