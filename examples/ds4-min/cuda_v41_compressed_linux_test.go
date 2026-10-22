//go:build linux

package main

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/examples/ds4-min/dsv4cpu"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func vcUpload[T any](t *testing.T, r *cudaFoundation, x []T) cudamem.Slice[T] {
	t.Helper()
	if len(x) == 0 {
		return cudamem.Slice[T]{}
	}
	a, e := cudamem.Alloc[T](r.device, len(x))
	if a.Ptr() != 0 {
		t.Cleanup(func() {
			if e := a.Free(); e != nil {
				t.Error(e)
			}
		})
	}
	if e != nil {
		t.Fatal(e)
	}
	if e = a.CopyFromHost(x); e != nil {
		t.Fatal(e)
	}
	return a
}
func vcCheck(t *testing.T, a cudamem.Slice[float32], want []float32) int {
	t.Helper()
	got := make([]float32, a.Len())
	if e := a.CopyToHost(got); e != nil {
		t.Fatal(e)
	}
	if len(got) != len(want) {
		t.Fatal("readback shape")
	}
	diff := 0
	for i, v := range got {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			diff++
		}
	}
	return diff
}
func vcBF(x []float32) {
	for i := range x {
		x[i] = dsv41.BF16(x[i])
	}
}
func vcRope(x []float32, pos int) {
	f := cudaV41Frequencies(true)
	width := 512
	if len(x) == 128 || len(x) == 4096 {
		width = 128
	}
	for h := 0; h < len(x)/width; h++ {
		for lane, v := range f {
			theta := float32(float32(pos) * v)
			c, s := float32(math.Cos(float64(theta))), float32(math.Sin(float64(theta)))
			i := h*width + width - 64 + lane*2
			a, b := x[i], x[i+1]
			x[i] = dsv41.BF16(float32(a*c) - float32(b*s))
			x[i+1] = dsv41.BF16(float32(a*s) + float32(b*c))
		}
	}
}
func vcScores(q, w, keys []float32) []float32 {
	out := make([]float32, len(keys)/128)
	for k := range out {
		var total float32
		for h := 0; h < 32; h++ {
			var lanes [32]float32
			for l := range lanes {
				qi, ki := h*128+l, k*128+l
				a := float32(float32(q[qi]*keys[ki]) + float32(q[qi+64]*keys[ki+64]))
				b := float32(float32(q[qi+32]*keys[ki+32]) + float32(q[qi+96]*keys[ki+96]))
				lanes[l] = a + b
			}
			for d := 16; d > 0; d /= 2 {
				for l := 0; l < d; l++ {
					lanes[l] += lanes[l+d]
				}
			}
			total += float32(max(lanes[0], 0) * w[h])
		}
		out[k] = total / 64
	}
	return out
}

func TestCUDAV41CompressedProducerReader(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	for _, variant := range []struct{ name, old, new string }{{"baseline", "", ""}, {"score_scale", "1.f/64", "1.f/32"}} {
		t.Run(variant.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			var w [4]cudaV41CompressedWeights
			norm := make([]float32, 512)
			kn := make([]float32, 128)
			for i := range norm {
				norm[i] = 1 + float32(i%5)/32
			}
			for i := range kn {
				kn[i] = 1 + float32(i%3)/16
			}
			iw := make([]uint16, 128*512)
			for row := 0; row < 128; row++ {
				for i := 0; i < 512; i++ {
					iw[row*512+i] = 0x2000
					if (row*3+i)%7 < 3 {
						iw[row*512+i] = 0xa000
					}
				}
			}
			for i := range w {
				w[i] = cudaV41CompressedWeights{vcUpload(t, r, norm), vcUpload(t, r, kn), vcUpload(t, r, iw)}
			}
			source := cudaV41CompressedSource
			if variant.old != "" {
				if strings.Count(source, variant.old) != 1 {
					t.Fatal("mutation site")
				}
				source = strings.Replace(source, variant.old, variant.new, 1)
			}
			op, e := compileCUDAV41Compressed(r, w, 5, source)
			if op != nil {
				t.Cleanup(func() {
					if e := op.Close(); e != nil {
						t.Error(e)
					}
				})
			}
			if e != nil {
				t.Fatal(e)
			}
			var caches, keys [4][]float32
			var pools [3]dsv41.Pool
			var candidates dsv41.Candidates
			changed := 0
			for repeat := 0; repeat < 2; repeat++ {
				if e := op.Reset(); e != nil {
					t.Fatal(e)
				}
				candidates.Reset()
				for i := range caches {
					caches[i] = nil
					keys[i] = nil
				}
				for i := range pools {
					pools[i].Reset()
				}
				for pos := 0; pos < 5; pos++ {
					for layer := 2; layer < 40; layer++ {
						schedule, _ := dsv41.Schedule(layer)
						owner := schedule.Owner
						n := (pos + 1) / schedule.Ratio
						top := min(n, 512)
						var in cudaV41CompressedInput
						var latent []float32
						if schedule.PublishKV {
							pk, ps := make([]float32, 512), make([]float32, 512)
							for i := range pk {
								pk[i] = float32(math.Sin(float64(i*3+pos*17+owner*11) * .021))
								ps[i] = float32(i%13+pos*3-owner) / 16
							}
							in.PoolKV = vcUpload(t, r, pk)
							if schedule.Ratio == 2 {
								in.PoolScore = vcUpload(t, r, ps)
								if (pos+1)%2 == 0 {
									latent = make([]float32, 512)
								}
								if e := pools[owner].Append(latent, pk, ps, 512, pos); e != nil {
									t.Fatal(e)
								}
							} else {
								latent = append([]float32(nil), pk...)
								vcBF(latent)
							}
							if len(latent) > 0 {
								if e := dsv4cpu.RMSNorm(latent, latent, norm, 1e-20); e != nil {
									t.Fatal(e)
								}
								vcBF(latent)
								key := make([]float32, 128)
								for row := range key {
									for i, v := range latent {
										sign := float32(1)
										if iw[row*512+i]&32768 != 0 {
											sign = -1
										}
										key[row] += float32(v * (sign / 128))
									}
								}
								vcBF(key)
								if e := dsv4cpu.RMSNorm(key, key, kn, 1e-20); e != nil {
									t.Fatal(e)
								}
								vcBF(key)
								vcRope(key, pos+1-schedule.Ratio)
								if e := dsv41.Quantize(key, 128, dsv41.FP4E8M0); e != nil {
									t.Fatal(e)
								}
								vcRope(latent, pos+1-schedule.Ratio)
								if e := dsv41.Quantize(latent, 512, dsv41.FP4E4M3); e != nil {
									t.Fatal(e)
								}
								keys[owner] = append(keys[owner], key...)
								caches[owner] = append(caches[owner], latent...)
							}
						}
						var scores []float32
						if schedule.RefreshIndex && n > 0 {
							q, weight := make([]float32, 4096), make([]float32, 32)
							for i := range q {
								q[i] = dsv41.BF16(float32(math.Cos(float64(i*7+pos+layer) * .013)))
							}
							for i := range weight {
								weight[i] = dsv41.BF16(float32(i%5-2) / 8)
							}
							in.Query = vcUpload(t, r, q)
							in.Weight = vcUpload(t, r, weight)
							vcRope(q, pos)
							if e := dsv41.Quantize(q, 128, dsv41.FP4E8M0); e != nil {
								t.Fatal(e)
							}
							scores = vcScores(q, weight, keys[owner])
						}
						ids, e := candidates.Pick(layer, pos, scores)
						if e != nil {
							t.Fatal(e)
						}
						want := make([]float32, top*512)
						for i, id := range ids {
							copy(want[i*512:], caches[owner][id*512:(id+1)*512])
						}
						out, dids := vcUpload(t, r, make([]float32, len(want))), vcUpload(t, r, make([]int32, top))
						inputs := []cudamem.Slice[float32]{in.PoolKV, in.PoolScore, in.Query, in.Weight}
						snapshots := make([][]float32, len(inputs))
						for i, a := range inputs {
							snapshots[i] = make([]float32, a.Len())
							if e := a.CopyToHost(snapshots[i]); e != nil {
								t.Fatal(e)
							}
						}
						t.Cleanup(func() {
							if e := op.Close(); e != nil {
								t.Error(e)
							}
						})
						if e := op.Step(out, dids, in, uint32(layer), uint32(pos)); e != nil {
							t.Fatal(e)
						}
						for i, a := range inputs {
							if vcCheck(t, a, snapshots[i]) != 0 {
								t.Fatal("borrowed input changed")
							}
						}
						outputDiff := vcCheck(t, out, want)
						if variant.name == "baseline" && outputDiff != 0 {
							t.Fatalf("gather position=%d layer=%d differences=%d", pos, layer, outputDiff)
						}
						gotIDs := make([]int32, top)
						if top > 0 {
							if e := dids.CopyToHost(gotIDs); e != nil {
								t.Fatal(e)
							}
						}
						for i, id := range ids {
							if int(gotIDs[i]) != id {
								t.Fatalf("selection position=%d layer=%d", pos, layer)
							}
						}
						for o := range caches {
							full, fullKeys := make([]float32, op.cache[o].Len()), make([]float32, op.keys[o].Len())
							copy(full, caches[o])
							copy(fullKeys, keys[o])
							if vcCheck(t, op.cache[o], full) != 0 || vcCheck(t, op.keys[o], fullKeys) != 0 {
								t.Fatalf("persistent cache owner=%d position=%d layer=%d", o, pos, layer)
							}
						}
						if schedule.RefreshIndex && n > 0 {
							diff := vcCheck(t, op.scores.Slice(0, n), scores)
							if variant.name == "baseline" && diff != 0 {
								t.Fatalf("index scores position=%d layer=%d differences=%d", pos, layer, diff)
							}
							changed += diff
						}
					}
				}
			}
			if variant.name != "baseline" && changed == 0 {
				t.Fatal("actual score-scale mutant survived")
			}
			for _, weight := range w {
				if vcCheck(t, weight.Norm, norm) != 0 || vcCheck(t, weight.IndexNorm, kn) != 0 {
					t.Fatal("norm weights changed")
				}
				got := make([]uint16, len(iw))
				if e := weight.Index.CopyToHost(got); e != nil {
					t.Fatal(e)
				}
				for i, v := range got {
					if v != iw[i] {
						t.Fatal("index projection weights changed")
					}
				}
			}
			if variant.name == "baseline" {
				testVCContextExhaustion(t, r)
				if !op.valid || op.position != 5 {
					t.Fatal("admission changed frontier")
				}
				if e := op.Reset(); e != nil {
					t.Fatal(e)
				}
				if e := op.Step(cudamem.Slice[float32]{}, cudamem.Slice[int32]{}, cudaV41CompressedInput{}, 3, 0); e == nil {
					t.Fatal("layer gap admitted")
				}
				testVCFailure(t, r, op)
			}
			if e := op.Close(); e != nil {
				t.Fatal(e)
			}
			if e := op.Reset(); e == nil {
				t.Fatal("closed reset admitted")
			}
			if e := op.Close(); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func TestCUDAV41CompressedContextExhaustion(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	testVCContextExhaustion(t, cudaSynthFoundation(t))
}

func testVCContextExhaustion(t *testing.T, r *cudaFoundation) {
	t.Helper()
	var weights [4]cudaV41CompressedWeights
	norm, indexNorm := make([]float32, 512), make([]float32, 128)
	for i := range norm {
		norm[i] = 1
	}
	for i := range indexNorm {
		indexNorm[i] = 1
	}
	for i := range weights {
		weights[i] = cudaV41CompressedWeights{vcUpload(t, r, norm), vcUpload(t, r, indexNorm), vcUpload(t, r, make([]uint16, 512*128))}
	}
	// An even exhausted context is important: at position2 ratio2 appends
	// carry only, so removing the guard cannot run past cache capacity and
	// masquerade as detection through a bounds panic or invalid launch.
	op, err := compileCUDAV41Compressed(r, weights, 2, cudaV41CompressedSource)
	if op != nil {
		t.Cleanup(func() {
			if err := op.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	poolKV, poolScore := vcUpload(t, r, make([]float32, 512)), vcUpload(t, r, make([]float32, 512))
	query, weight := vcUpload(t, r, make([]float32, 4096)), vcUpload(t, r, make([]float32, 32))
	out, ids := vcUpload(t, r, make([]float32, 2*512)), vcUpload(t, r, make([]int32, 2))
	t.Cleanup(func() {
		if err := op.Close(); err != nil {
			t.Error(err)
		}
	})
	for position := 0; position < 2; position++ {
		for layer := 2; layer < 40; layer++ {
			l, err := dsv41.Schedule(layer)
			if err != nil {
				t.Fatal(err)
			}
			n := (position + 1) / l.Ratio
			var in cudaV41CompressedInput
			if l.PublishKV {
				in.PoolKV = poolKV
				if l.Ratio == 2 {
					in.PoolScore = poolScore
				}
			}
			if l.RefreshIndex && n > 0 {
				in.Query, in.Weight = query, weight
			}
			if err := op.Step(out.Slice(0, n*512), ids.Slice(0, n), in, uint32(layer), uint32(position)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if !op.valid || op.position != 2 || op.layer != 2 || op.pool[0].position != 2 {
		t.Fatal("context fixture did not reach live exhausted frontier")
	}
	in := cudaV41CompressedInput{PoolKV: poolKV, PoolScore: poolScore, Query: query, Weight: weight}
	err = op.Step(out.Slice(0, 512), ids.Slice(0, 1), in, 2, 2)
	if err == nil {
		t.Fatal("context admitted")
	}
	if !strings.Contains(err.Error(), "invalid frontier") {
		t.Fatalf("context rejection reached wrong boundary: %v", err)
	}
	if !op.valid || op.position != 2 || op.layer != 2 || op.pool[0].position != 2 {
		t.Fatal("context rejection changed parent or carry frontier")
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDAV41CompressedMaskTieAndCausal(t *testing.T) {
	if os.Getenv("DS4_CUDA_V41_OPS_GATE") != "1" {
		t.Skip("opt-in V4.1 CUDA gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	for _, v := range []struct{ name, old, new string }{{"baseline", "", ""}, {"latest", "b+1==nb?inf:v", "v"}, {"mask", "if(layer>20)", "if(layer>200)"}} {
		t.Run(v.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			source := cudaV41CompressedSource
			if v.old != "" {
				if strings.Count(source, v.old) != 1 {
					t.Fatal("mask mutation site")
				}
				source = strings.Replace(source, v.old, v.new, 1)
			}
			// Constructor supplies all actual functions; directly test the device
			// selector with >2048 blocks without executing 16401 model token steps.
			var w [4]cudaV41CompressedWeights
			for i := range w {
				w[i] = cudaV41CompressedWeights{vcUpload(t, r, make([]float32, 512)), vcUpload(t, r, make([]float32, 128)), vcUpload(t, r, make([]uint16, 65536))}
			}
			const n = 16401
			op, e := compileCUDAV41Compressed(r, w, n, source)
			if op != nil {
				t.Cleanup(func() {
					if e := op.Close(); e != nil {
						t.Error(e)
					}
				})
			}
			if e != nil {
				t.Fatal(e)
			}
			scores := make([]float32, n)
			if e := op.scores.CopyFromHost(scores); e != nil {
				t.Fatal(e)
			}
			launch := func(layer uint32) {
				t.Helper()
				if e := r.device.WithContext(func() error {
					return cudalaunch.Kernel(op.pick).Grid(1, 1, 1).Block(1, 1, 1).On(r.stream()).Launch(op.ids, op.scores, op.mask, op.blocks, op.used, uint32(n), layer)
				}); e != nil {
					t.Fatal(e)
				}
				if e := op.drain(); e != nil {
					t.Fatal(e)
				}
			}
			launch(20)
			firstIDs := make([]int32, 512)
			if e := op.ids.CopyToHost(firstIDs); e != nil {
				t.Fatal(e)
			}
			for i, id := range firstIDs {
				if int(id) != i {
					t.Fatal("layer20 equal-score lower-ID tie order")
				}
			}
			mask := make([]uint32, op.mask.Len())
			if e := op.mask.CopyToHost(mask); e != nil {
				t.Fatal(e)
			}
			maskDiff := 0
			for i, x := range mask {
				want := uint32(0)
				if i < 2047 || i == len(mask)-1 {
					want = 1
				}
				if x != want {
					maskDiff++
				}
			}
			if v.name == "latest" {
				if maskDiff == 0 {
					t.Fatal("latest-block mutant survived")
				}
				return
			}
			if maskDiff != 0 {
				t.Fatal("stable/latest block mask mismatch")
			}
			// Give excluded blocks higher scores than every allowed block. Layer24
			// must reuse layer20 mask while rebuilding top512, not recompute the mask.
			for i := range scores {
				if mask[i/8] == 0 {
					scores[i] = 100
				} else {
					scores[i] = 0
				}
			}
			if e := op.scores.CopyFromHost(scores); e != nil {
				t.Fatal(e)
			}
			launch(24)
			ids := make([]int32, 512)
			if e := op.ids.CopyToHost(ids); e != nil {
				t.Fatal(e)
			}
			diff := 0
			for i, id := range ids {
				if int(id) != i {
					diff++
				}
			}
			if v.name == "mask" {
				if diff == 0 {
					t.Fatal("candidate mask mutant survived")
				}
				return
			}
			if diff != 0 {
				t.Fatal("masked stable top512 mismatch")
			}
			// Actual scores kernel's causal guard: only one visible key at position0.
			q, weight, key := vcUpload(t, r, make([]float32, 4096)), vcUpload(t, r, make([]float32, 32)), vcUpload(t, r, make([]float32, 3*128))
			if e := r.device.WithContext(func() error {
				return cudalaunch.Kernel(op.score).Grid(1, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(op.scores, q, weight, key, uint32(3), uint32(0), uint32(1))
			}); e != nil {
				t.Fatal(e)
			}
			if e := op.drain(); e != nil {
				t.Fatal(e)
			}
			if vcCheck(t, op.scores.Slice(0, 3), []float32{0, float32(math.Inf(-1)), float32(math.Inf(-1))}) != 0 {
				t.Fatal("causal score leak")
			}
		})
	}
}

func testVCFailure(t *testing.T, r *cudaFoundation, op *cudaV41Compressed) {
	t.Helper()
	pk, ps, q, w := vcUpload(t, r, make([]float32, 512)), vcUpload(t, r, make([]float32, 512)), vcUpload(t, r, make([]float32, 4096)), vcUpload(t, r, make([]float32, 32))
	out, ids := vcUpload(t, r, make([]float32, 512)), vcUpload(t, r, make([]int32, 1))
	t.Cleanup(func() {
		if e := op.Close(); e != nil {
			t.Error(e)
		}
	})
	if e := op.Step(cudamem.Slice[float32]{}, cudamem.Slice[int32]{}, cudaV41CompressedInput{PoolKV: pk, PoolScore: pk}, 2, 0); e == nil {
		t.Fatal("input alias admitted")
	}
	if !op.valid || op.position != 0 || op.layer != 2 {
		t.Fatal("alias admission mutated frontier")
	}
	for layer := 2; layer < 40; layer++ {
		l, _ := dsv41.Schedule(layer)
		var in cudaV41CompressedInput
		var dst cudamem.Slice[float32]
		var selected cudamem.Slice[int32]
		if l.PublishKV {
			in.PoolKV = pk
			if l.Ratio == 2 {
				in.PoolScore = ps
			}
		}
		if l.Ratio == 1 {
			dst = out
			selected = ids
			if l.RefreshIndex {
				in.Query = q
				in.Weight = w
			}
		}
		if e := op.Step(dst, selected, in, uint32(layer), 0); e != nil {
			t.Fatal(e)
		}
	}
	fn := op.index
	op.index = 0
	e := op.Step(out, ids, cudaV41CompressedInput{PoolKV: pk, PoolScore: ps, Query: q, Weight: w}, 2, 1)
	op.index = fn
	if e == nil || op.valid || op.position != 1 || op.layer != 2 || op.pool[0].position != 2 {
		t.Fatal("post-pool failure did not invalidate parent with uncommitted frontier")
	}
	if len(op.pending) != 0 || op.pendingIDs.Ptr() != 0 {
		t.Fatal("successful drain retained borrows")
	}
	if e := op.Step(out, ids, cudaV41CompressedInput{PoolKV: pk, PoolScore: ps, Query: q, Weight: w}, 2, 1); e == nil {
		t.Fatal("partial token retried without reset")
	}
	if e := op.Reset(); e != nil {
		t.Fatal(e)
	}
	if op.position != 0 || op.layer != 2 || op.pool[0].position != 0 {
		t.Fatal("reset parent/child frontier drift")
	}
	if e := op.Step(cudamem.Slice[float32]{}, cudamem.Slice[int32]{}, cudaV41CompressedInput{PoolKV: pk, PoolScore: ps}, 2, 0); e != nil {
		t.Fatal(e)
	}
	if vcCheck(t, op.cache[0], make([]float32, op.cache[0].Len())) != 0 {
		t.Fatal("reset/replay retained published cache")
	}
}
