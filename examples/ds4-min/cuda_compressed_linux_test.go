//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Frozen before hardware qualification. Quantization-bin failures require
// diagnosis; this gate is not a production-wide error budget.
const cudaCompressedATol = 1e-4
const cudaCompressedRTol = 1e-4

func cudaCompressedCheck(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if err := cudaQualificationClose(got, want, cudaCompressedATol, cudaCompressedRTol); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}
func cudaCompressedUpload(t *testing.T, d cudamem.Slice[float32], v []float32) {
	t.Helper()
	if err := d.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
}
func cudaCompressedTensor(t *testing.T, r *cudaFoundation, name string, v []float32) cudaModelTensor {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return cudaSynthTensor(t, r, name, 0, []uint64{uint64(len(v))}, b)
}
func cudaCompressedMask(t *testing.T, r *cudaFoundation, n int) cudamem.Slice[uint32] {
	d, err := cudamem.Alloc[uint32](r.device, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Free(); err != nil {
			t.Error(err)
		}
	})
	return d
}

func TestCUDACompressedCompressor(t *testing.T) {
	for _, tc := range []struct{ ratio, head int }{{4, 512}, {128, 512}, {4, 128}} {
		t.Run(fmt.Sprintf("ratio%d_head%d", tc.ratio, tc.head), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			width, rows := tc.head, tc.ratio
			if tc.ratio == 4 {
				width *= 2
				rows *= 2
			}
			state := cudaCompressedState{headDim: tc.head, width: width, ratio: tc.ratio, cap: 3,
				cache: cudaTestAlloc(t, r, 3*tc.head), stateKV: cudaTestAlloc(t, r, rows*width), stateScore: cudaTestAlloc(t, r, rows*width), pooled: cudaTestAlloc(t, r, tc.head)}
			sk, sg := make([]float32, rows*width), make([]float32, rows*width)
			for i := range sg {
				sg[i] = float32(math.Inf(-1))
			}
			cudaCompressedUpload(t, state.stateKV, sk)
			cudaCompressedUpload(t, state.stateScore, sg)
			poison := make([]float32, 3*tc.head)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			cudaCompressedUpload(t, state.cache, poison)
			dk, dg := cudaTestAlloc(t, r, width), cudaTestAlloc(t, r, width)
			apeBytes := make([]byte, 2*width*tc.ratio)
			apeValues := make([]float32, width*tc.ratio)
			// Exact half values vary by temporal row and channel to expose APE indexing.
			halfs := [...]uint16{0x0000, 0x3400, 0xb400, 0x3800, 0xb800}
			for i := range apeValues {
				h := halfs[(i*3+i/width)%len(halfs)]
				binary.LittleEndian.PutUint16(apeBytes[i*2:], h)
				apeValues[i] = cudaTestFloat16(h)
			}
			ape := cudaSynthTensor(t, r, "ape", 1, []uint64{uint64(width), uint64(tc.ratio)}, apeBytes)
			weights := make([]float32, tc.head)
			for i := range weights {
				weights[i] = 1 + float32(i%5-2)/16
			}
			norm := cudaCompressedTensor(t, r, "norm", weights)
			var tokens, gates [][]float32
			for position := 0; position < 2*tc.ratio; position++ {
				kv, gate := make([]float32, width), make([]float32, width)
				for i := range kv {
					kv[i] = float32((i*13+position*11)%71-35) / 32
					gate[i] = float32((i*7+position*3)%29-14) / 8
				}
				tokens = append(tokens, kv)
				biased := append([]float32(nil), gate...)
				for i := range biased {
					biased[i] += apeValues[(position%tc.ratio)*width+i]
				}
				gates = append(gates, biased)
				cudaCompressedUpload(t, dk, kv)
				cudaCompressedUpload(t, dg, gate)
				row := position % tc.ratio
				if tc.ratio == 4 {
					row += tc.ratio
				}
				copy(sk[row*width:], kv)
				copy(sg[row*width:], biased)
				emit := false
				launch := func() error {
					var err error
					emit, err = r.encodeCompressor(dk, dg, ape, norm, &state, position, tc.head == 128)
					return err
				}
				boundary := (position+1)%tc.ratio == 0
				if boundary {
					cudaQualificationMeasure(t, r, launch)
				} else {
					if err := launch(); err != nil {
						t.Fatal(err)
					}
					if err := r.device.WithContext(func() error { return r.device.DefaultStream().Synchronize(context.Background()) }); err != nil {
						t.Fatal(err)
					}
				}
				if emit != boundary {
					t.Fatalf("position%d emit=%v", position, emit)
				}
				if !emit {
					continue
				}
				pooled := cudaCompressedPoolReference(tokens, gates, tc.ratio, tc.head)
				want := cudaCompressedFinishReference(pooled, weights, position+1-tc.ratio, tc.head == 128)
				got := cudaTestRead(t, state.cache)
				cudaCompressedCheck(t, "emitted row", got[state.n*tc.head:(state.n+1)*tc.head], want)
				for i := (state.n + 1) * tc.head; i < len(got); i++ {
					if !math.IsNaN(float64(got[i])) {
						t.Fatalf("wrote future cache element%d", i)
					}
				}
				if tc.ratio == 4 {
					copy(sk[:tc.ratio*width], sk[tc.ratio*width:])
					copy(sg[:tc.ratio*width], sg[tc.ratio*width:])
				}
				cudaCompressedCheck(t, "state kv", cudaTestRead(t, state.stateKV), sk)
				cudaCompressedCheck(t, "state scores", cudaTestRead(t, state.stateScore), sg)
				state.n++
			}
			if state.n != 2 {
				t.Fatalf("emitted%d rows", state.n)
			}
			state.n = state.cap
			if _, err := r.encodeCompressor(dk, dg, ape, norm, &state, 2*tc.ratio-1, tc.head == 128); err == nil {
				t.Fatal("accepted cache overflow")
			}
		})
	}
}

func TestCUDACompressedCompressorQueued(t *testing.T) {
	for _, ratio := range []int{4, 128} {
		t.Run(fmt.Sprintf("ratio%d", ratio), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			const head = 512
			width, rows := head, ratio
			if ratio == 4 {
				width *= 2
				rows *= 2
			}
			ape := cudaSynthTensor(t, r, "ape", 1, []uint64{uint64(width), uint64(ratio)}, make([]byte, 2*width*ratio))
			normValues := make([]float32, head)
			for i := range normValues {
				normValues[i] = 1
			}
			norm := cudaCompressedTensor(t, r, "norm", normValues)
			var inputs [][2]cudamem.Slice[float32]
			for position := 0; position <= ratio; position++ {
				kv, gate := make([]float32, width), make([]float32, width)
				for i := range kv {
					kv[i] = float32((position*17+i*5)%47-23) / 16
					gate[i] = float32((position*11+i*3)%31-15) / 8
				}
				dk, dg := cudaTestAlloc(t, r, width), cudaTestAlloc(t, r, width)
				cudaCompressedUpload(t, dk, kv)
				cudaCompressedUpload(t, dg, gate)
				inputs = append(inputs, [2]cudamem.Slice[float32]{dk, dg})
			}
			run := func(syncEach bool) (cache, stateKV []float32) {
				state := cudaCompressedState{
					headDim:    head,
					width:      width,
					ratio:      ratio,
					cap:        2,
					cache:      cudaTestAlloc(t, r, 2*head),
					stateKV:    cudaTestAlloc(t, r, rows*width),
					stateScore: cudaTestAlloc(t, r, rows*width),
					pooled:     cudaTestAlloc(t, r, head),
				}
				cudaCompressedUpload(t, state.stateKV, make([]float32, rows*width))
				scores := make([]float32, rows*width)
				for i := range scores {
					scores[i] = float32(math.Inf(-1))
				}
				cudaCompressedUpload(t, state.stateScore, scores)
				for position, input := range inputs {
					emit, err := r.encodeCompressor(input[0], input[1], ape, norm, &state, position, false)
					if err != nil {
						t.Fatal(err)
					}
					if emit {
						state.n++
					}
					if syncEach {
						if err := r.device.WithContext(func() error {
							return r.device.DefaultStream().Synchronize(context.Background())
						}); err != nil {
							t.Fatal(err)
						}
					}
				}
				if err := r.device.WithContext(func() error {
					return r.device.DefaultStream().Synchronize(context.Background())
				}); err != nil {
					t.Fatal(err)
				}
				if state.n != 1 {
					t.Fatalf("emitted %d rows, want 1", state.n)
				}
				return cudaTestRead(t, state.cache), cudaTestRead(t, state.stateKV)
			}
			wantCache, wantStateKV := run(true)
			gotCache, gotStateKV := run(false)
			if err := exactFloat32Bits(gotCache, wantCache); err != nil {
				t.Fatalf("queued cache differs from synchronized cache: %v", err)
			}
			if err := exactFloat32Bits(gotStateKV, wantStateKV); err != nil {
				t.Fatalf("queued following-token state differs from synchronized state: %v", err)
			}
		})
	}
}

func TestCUDACompressedFillU32(t *testing.T) {
	r := cudaSynthFoundation(t)
	dst := cudaCompressedMask(t, r, 513)
	values := make([]uint32, dst.Len())
	for i := range values {
		values[i] = 7
	}
	if err := dst.CopyFromHost(values); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeFillU32(dst, 512, 1); err != nil {
		t.Fatal(err)
	}
	if err := r.device.WithContext(func() error {
		return r.device.DefaultStream().Synchronize(context.Background())
	}); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, dst.Len())
	if err := dst.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		want := uint32(1)
		if i == len(got)-1 {
			want = 7
		}
		if value != want {
			t.Fatalf("value %d = %d, want %d", i, value, want)
		}
	}
	for _, count := range []int{-1, dst.Len() + 1} {
		if err := r.encodeFillU32(dst, count, 1); err == nil {
			t.Fatalf("accepted count %d", count)
		}
	}
}

// Pool the original chronological token list, selecting the first lane of the
// prior chunk and the second lane of the current chunk for ratio4.
func cudaCompressedPoolReference(tokens, gates [][]float32, ratio, head int) []float32 {
	out := make([]float32, head)
	end := len(tokens)
	start := max(0, end-ratio)
	if ratio == 4 {
		start = max(0, end-2*ratio)
	}
	for j := range out {
		maximum := math.Inf(-1)
		index := func(t int) int {
			if ratio == 4 && t >= end-ratio {
				return head + j
			}
			return j
		}
		for t := start; t < end; t++ {
			maximum = math.Max(maximum, float64(gates[t][index(t)]))
		}
		sum, denom := 0.0, 0.0
		for t := start; t < end; t++ {
			i := index(t)
			w := math.Exp(float64(gates[t][i]) - maximum)
			sum += w * float64(tokens[t][i])
			denom += w
		}
		out[j] = float32(sum / denom)
	}
	return out
}
func cudaCompressedRotateReference(x []float32, position int) {
	low := math.Max(0, math.Floor(64*math.Log(65536/(32*2*math.Pi))/(2*math.Log(160000))))
	high := math.Min(63, math.Ceil(64*math.Log(65536/(2*math.Pi))/(2*math.Log(160000))))
	for r := 0; r < 64; r += 2 {
		ramp := 1 - min(1, max(0, (float64(r/2)-low)/math.Max(.001, high-low)))
		theta := float64(position) * math.Pow(160000, -float64(r)/64) * (.0625*(1-ramp) + ramp)
		s, c := math.Sincos(theta)
		i := len(x) - 64 + r
		a, b := float64(x[i]), float64(x[i+1])
		x[i], x[i+1] = float32(a*c-b*s), float32(a*s+b*c)
	}
}
func cudaCompressedQATReference(x []float32) []float32 {
	y := make([]float64, 128)
	for i := range y {
		for j, v := range x {
			if bits.OnesCount(uint(i&j))%2 == 0 {
				y[i] += float64(v)
			} else {
				y[i] -= float64(v)
			}
		}
		y[i] /= math.Sqrt(128)
	}
	out := make([]float32, 128)
	levels := [...]float64{0, .5, 1, 1.5, 2, 3, 4, 6}
	for off := 0; off < 128; off += 32 {
		amax := 7.052966104933725e-38
		for _, v := range y[off : off+32] {
			amax = math.Max(amax, math.Abs(v))
		}
		scale := math.Exp2(math.Ceil(math.Log2(amax / 6)))
		for i := off; i < off+32; i++ {
			v := math.Min(math.Abs(y[i])/scale, 6)
			best := 0
			for j := 1; j < 8; j++ {
				d, b := math.Abs(v-levels[j]), math.Abs(v-levels[best])
				if d < b || d == b && j%2 == 0 && best%2 != 0 {
					best = j
				}
			}
			out[i] = float32(math.Copysign(levels[best]*scale, y[i]))
		}
	}
	return out
}
func cudaCompressedFinishReference(pooled, weights []float32, position int, indexer bool) []float32 {
	out := make([]float32, len(pooled))
	ss := 0.0
	for _, v := range pooled {
		ss += float64(v) * float64(v)
	}
	for i, v := range pooled {
		out[i] = float32(float64(v) / math.Sqrt(ss/float64(len(out))+1e-6) * float64(weights[i]))
	}
	cudaCompressedRotateReference(out, position)
	if indexer {
		out = cudaCompressedQATReference(out)
	} else {
		cudaCompressedFP8Reference(out)
	}
	for i := range out {
		out[i] = cudaAttentionQualificationRound(out[i], 10, -14)
	}
	return out
}
func cudaCompressedFP8Reference(row []float32) {
	for off := 0; off < 448; off += 64 {
		amax := 1e-4
		for _, v := range row[off : off+64] {
			amax = math.Max(amax, math.Abs(float64(v)))
		}
		scale := math.Exp2(math.Ceil(math.Log2(amax / 448)))
		for i := off; i < off+64; i++ {
			row[i] = float32(float64(cudaAttentionQualificationRound(float32(float64(row[i])/scale), 3, -6)) * scale)
		}
	}
}

func TestCUDACompressedIndexer(t *testing.T) {
	r := cudaSynthFoundation(t)
	q, weights := make([]float32, 64*128), make([]float32, 64)
	for i := range q {
		q[i] = float32((i*13)%61-30) / 32
	}
	for h := range weights {
		weights[h] = float32(h%5+1) / 8
	}
	transformed := make([]float32, len(q))
	for h := range 64 {
		row := append([]float32(nil), q[h*128:(h+1)*128]...)
		cudaCompressedRotateReference(row, 2051)
		copy(transformed[h*128:], cudaCompressedQATReference(row))
	}
	// Each cache row is a positive scalar multiple of one quantized query.
	// Scores have a strict ordering: row0 is rejected at 513 rows.
	cache := make([]float32, 513*128)
	for c := range 513 {
		for i := range 128 {
			cache[c*128+i] = transformed[i] * float32(c+1) / 512
		}
	}
	dq, dw, dc := cudaTestAlloc(t, r, len(q)), cudaTestAlloc(t, r, len(weights)), cudaTestAlloc(t, r, len(cache))
	ds, dout := cudaTestAlloc(t, r, 513), cudaTestAlloc(t, r, len(q))
	allowed := cudaCompressedMask(t, r, 513)
	cudaCompressedUpload(t, dq, q)
	cudaCompressedUpload(t, dw, weights)
	cudaCompressedUpload(t, dc, cache)
	for _, count := range []int{511, 512, 513} {
		t.Run(fmt.Sprintf("rows%d", count), func(t *testing.T) {
			mask := make([]uint32, 513)
			for i := range mask {
				mask[i] = 99
			}
			if err := allowed.CopyFromHost(mask); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error { return r.encodeIndexer(dq, dw, dc, count, dout, ds, allowed, 2051) })
			if err := allowed.CopyToHost(mask); err != nil {
				t.Fatal(err)
			}
			for i, v := range mask {
				want := uint32(1)
				if i >= count {
					want = 99
				}
				if count == 513 && i == 0 {
					want = 0
				}
				if v != want {
					t.Fatalf("allowed[%d]=%d want%d", i, v, want)
				}
			}
			if count > 512 {
				cudaCompressedCheck(t, "indexer query", cudaTestRead(t, dout), transformed)
				scores := cudaCompressedIndexerScoresReference(transformed, weights, cache)
				cudaCompressedCheck(t, "indexer scores", cudaTestRead(t, ds), scores)
				order := make([]int, count)
				for i := range order {
					order[i] = i
				}
				sort.SliceStable(order, func(i, j int) bool { return scores[order[i]] > scores[order[j]] })
				if order[len(order)-1] != 0 {
					t.Fatal("fixture has no strict excluded row")
				}
			}
		})
	}
	// Equal scores exercise stable lower-index tie breaking independently.
	cudaCompressedUpload(t, dw, make([]float32, 64))
	cudaQualificationMeasure(t, r, func() error { return r.encodeIndexer(dq, dw, dc, 513, dout, ds, allowed, 2051) })
	mask := make([]uint32, 513)
	if err := allowed.CopyToHost(mask); err != nil {
		t.Fatal(err)
	}
	for i, v := range mask {
		want := uint32(1)
		if i == 512 {
			want = 0
		}
		if v != want {
			t.Fatalf("tie allowed[%d]=%d want%d", i, v, want)
		}
	}
}
func cudaCompressedIndexerScoresReference(q, weights, cache []float32) []float32 {
	scores := make([]float32, len(cache)/128)
	for c := range scores {
		sum := 0.0
		for h, w := range weights {
			dot := 0.0
			for i := range 128 {
				dot += float64(q[h*128+i]) * float64(cache[c*128+i])
			}
			sum += math.Max(0, dot) * float64(w) / math.Sqrt(64*128)
		}
		scores[c] = float32(sum)
	}
	return scores
}

func TestCUDACompressedMixedAttention(t *testing.T) {
	r := cudaSynthFoundation(t)
	const capacity = 128
	q, kv := cudaAttentionQualificationInputs(129)
	rotated := append([]float32(nil), q...)
	for h := range 64 {
		cudaCompressedRotateReference(rotated[h*512:(h+1)*512], 129)
	}
	cache := make([]float32, capacity*512)
	var history [][]float32
	for position := 2; position <= 129; position++ {
		_, row := cudaAttentionQualificationInputs(position)
		cudaCompressedRotateReference(row, position)
		cudaCompressedFP8Reference(row)
		for i := range row {
			row[i] = cudaAttentionQualificationRound(row[i], 10, -14)
		}
		history = append(history, row)
		copy(cache[(position%capacity)*512:], row)
	}
	// The device begins with prior history only. Its current slot contains a
	// stale sentinel, so omitting the kernel's current-row store cannot pass.
	seed := append([]float32(nil), cache...)
	for i := range 512 {
		seed[(129%capacity)*512+i] = 13
	}
	comp := make([]float32, 3*512)
	for i := range comp {
		comp[i] = float32((i*7)%43-21) / 16
	}
	sinks := make([]float32, 64)
	for h := range sinks {
		sinks[h] = float32(h%7-3) / 4
	}
	sink := cudaCompressedTensor(t, r, "sinks", sinks)
	dq, dk, dr, dc := cudaTestAlloc(t, r, len(q)), cudaTestAlloc(t, r, 512), cudaTestAlloc(t, r, len(cache)), cudaTestAlloc(t, r, len(comp))
	ds, do := cudaTestAlloc(t, r, 64*(capacity+3)), cudaTestAlloc(t, r, 32768+512+32768)
	allow := cudaCompressedMask(t, r, 3)
	cudaCompressedUpload(t, dq, q)
	cudaCompressedUpload(t, dk, kv)
	cudaCompressedUpload(t, dc, comp)
	for _, tc := range []struct {
		name string
		n    int
		mask []uint32
	}{{"no compressed", 0, []uint32{1, 0, 0}}, {"one compressed", 1, []uint32{1, 0, 0}}, {"masked middle", 3, []uint32{1, 0, 1}}, {"all masked", 3, []uint32{0, 0, 0}}} {
		t.Run(tc.name, func(t *testing.T) {
			cudaCompressedUpload(t, dr, seed)
			if err := allow.CopyFromHost(tc.mask); err != nil {
				t.Fatal(err)
			}
			poison := make([]float32, do.Len())
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			cudaCompressedUpload(t, do, poison)
			cudaQualificationMeasure(t, r, func() error { return r.encodeMixedAttention(dq, dk, sink, dr, dc, tc.n, allow, ds, do, 129, 2) })
			rows := append([][]float32(nil), history...)
			for c := 0; c < tc.n; c++ {
				if tc.mask[c] != 0 {
					rows = append(rows, comp[c*512:(c+1)*512])
				}
			}
			want := cudaCompressedAttentionReference(rotated, rows, sinks, 129)
			got := cudaTestRead(t, do)
			cudaCompressedCheck(t, "rotated q", got[:32768], rotated)
			cudaCompressedCheck(t, "attention", got[33280:], want)
			cudaCompressedCheck(t, "cache", cudaTestRead(t, dr), cache)
			if tc.name == "no compressed" {
				// Execute the old missing-store defect, not merely a corrupted oracle.
				begin := strings.Index(cudaMathSource, "extern \"C\" __global__ void ds4_attention_mixed(")
				if begin < 0 {
					t.Fatal("mixed kernel mutation anchor missing")
				}
				end := strings.Index(cudaMathSource[begin:], "\n}\n")
				if end < 0 {
					t.Fatal("mixed kernel mutation end missing")
				}
				fn := cudaMathSource[begin : begin+end+3]
				store := "raw[row*512+i]=ds4_half_round(out[32768+i]);"
				if strings.Count(fn, store) != 1 {
					t.Fatal("mixed cache-store mutation anchor changed")
				}
				source := cudaMathSource[:begin] + strings.Replace(fn, store, "(void)row;", 1) + cudaMathSource[begin+end+3:]
				program, err := nvrtcx.Compile(source, nvrtcx.WithName("missing_cache_store.cu"), nvrtcx.WithArch(r.major, r.minor))
				if err != nil {
					t.Fatal(err)
				}
				module, err := program.Load(r.device)
				if err != nil {
					t.Fatal(err)
				}
				defer module.Unload()
				mutant, err := module.GetFunction("ds4_attention_mixed")
				if err != nil {
					t.Fatal(err)
				}
				original := r.mixedAttentionFn
				r.mixedAttentionFn = mutant
				defer func() { r.mixedAttentionFn = original }()
				cudaCompressedUpload(t, dr, seed)
				cudaQualificationMeasure(t, r, func() error { return r.encodeMixedAttention(dq, dk, sink, dr, dc, tc.n, allow, ds, do, 129, 2) })
				if cudaQualificationClose(cudaTestRead(t, dr), cache, cudaCompressedATol, cudaCompressedRTol) == nil {
					t.Fatal("missing-store mutant passed cache gate")
				}
				if cudaQualificationClose(cudaTestRead(t, do)[33280:], want, cudaCompressedATol, cudaCompressedRTol) == nil {
					t.Fatal("missing-store mutant passed attention gate")
				}
				r.mixedAttentionFn = original
			}
			if tc.name == "masked middle" {
				badRows := append(append([][]float32(nil), rows...), comp[512:1024])
				bad := cudaCompressedAttentionReference(rotated, badRows, sinks, 129)
				if cudaQualificationClose(bad, want, cudaCompressedATol, cudaCompressedRTol) == nil {
					t.Fatal("fixture cannot detect ignored mask")
				}
			}
		})
	}
}
func cudaCompressedAttentionReference(q []float32, rows [][]float32, sinks []float32, position int) []float32 {
	out := make([]float32, len(q))
	for h := range 64 {
		weights := make([]float64, len(rows))
		maximum := float64(sinks[h])
		for j, row := range rows {
			dot := 0.0
			for i, v := range row {
				dot += float64(q[h*512+i]) * float64(v)
			}
			weights[j] = dot / math.Sqrt(512)
			maximum = math.Max(maximum, weights[j])
		}
		denom := math.Exp(float64(sinks[h]) - maximum)
		for j := range weights {
			weights[j] = math.Exp(weights[j] - maximum)
			denom += weights[j]
		}
		for i := range 512 {
			sum := 0.0
			for j, w := range weights {
				sum += w * float64(rows[j][i])
			}
			out[h*512+i] = float32(sum / denom)
		}
		cudaCompressedRotateReference(out[h*512:(h+1)*512], -position)
	}
	return out
}
