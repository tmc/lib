//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// Registered before device execution. This bounded fixture allows float32
// accumulation and transcendental rounding; failures must not widen the gate.
const cudaAttentionQualificationATol = 2e-5
const cudaAttentionQualificationRTol = 2e-4

func TestCUDAQualificationRawAttention(t *testing.T) {
	for _, capacity := range []int{4, 128} {
		t.Run(fmt.Sprintf("capacity_%d", capacity), func(t *testing.T) {
			cudaAttentionQualificationRun(t, capacity)
		})
	}
}

func cudaAttentionQualificationRun(t *testing.T, capacity int) {
	r := cudaSynthFoundation(t)
	sinks := make([]float32, 64)
	data := make([]byte, 4*len(sinks))
	for i := range sinks {
		sinks[i] = float32(i%9-4) / 4
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(sinks[i]))
	}
	sink := cudaSynthTensor(t, r, "qualification.sinks", 0, []uint64{64}, data)
	dq, dk := cudaTestAlloc(t, r, 64*512), cudaTestAlloc(t, r, 512)
	dc, ds := cudaTestAlloc(t, r, capacity*512), cudaTestAlloc(t, r, 64*capacity)
	dout := cudaTestAlloc(t, r, 32768+512+32768)
	badSink := sink
	badSink.dimensions = []uint64{63}
	badType := sink
	badType.typeID = 1
	for name, call := range map[string]func() error{
		"sink type":         func() error { return r.encodeRawAttention(dq, dk, badType, dc, ds, dout, 0) },
		"query length":      func() error { return r.encodeRawAttention(dq.Slice(0, dq.Len()-1), dk, sink, dc, ds, dout, 0) },
		"kv length":         func() error { return r.encodeRawAttention(dq, dk.Slice(0, 511), sink, dc, ds, dout, 0) },
		"sink dimensions":   func() error { return r.encodeRawAttention(dq, dk, badSink, dc, ds, dout, 0) },
		"cache length":      func() error { return r.encodeRawAttention(dq, dk, sink, dc.Slice(0, dc.Len()-1), ds, dout, 0) },
		"score stride":      func() error { return r.encodeRawAttention(dq, dk, sink, dc, ds.Slice(0, ds.Len()-1), dout, 0) },
		"output length":     func() error { return r.encodeRawAttention(dq, dk, sink, dc, ds, dout.Slice(0, dout.Len()-1), 0) },
		"negative position": func() error { return r.encodeRawAttention(dq, dk, sink, dc, ds, dout, -1) },
	} {
		t.Run(name, func(t *testing.T) {
			if err := call(); err == nil {
				t.Fatal("accepted invalid attention shape")
			}
		})
	}
	// Unused slots must not influence attention before the ring is full.
	cache := make([]float32, capacity*512)
	for i := range cache {
		cache[i] = 13
	}
	var history [][]float32
	first, last := 0, 10
	if capacity == 128 {
		// Seed genuine prior tokens to test the production-size boundary without
		// evaluating every intermediate attention output.
		first, last = 127, 129
		for position := 0; position < first; position++ {
			q, kv := cudaAttentionQualificationInputs(position)
			_, row := cudaAttentionQualificationPrepare(q, kv, position)
			history = append(history, row)
			copy(cache[position*512:], row)
		}
	}
	if err := dc.CopyFromHost(cache); err != nil {
		t.Fatal(err)
	}
	for position := first; position <= last; position++ {
		t.Run(fmt.Sprintf("position_%d", position), func(t *testing.T) {
			q, kv := cudaAttentionQualificationInputs(position)
			rotated, row := cudaAttentionQualificationPrepare(q, kv, position)
			history = append(history, row)
			if len(history) > capacity {
				history = history[1:]
			}
			copy(cache[(position%capacity)*512:], row)
			want, wantScores := cudaAttentionQualificationReference(rotated, history, sinks, position, false)
			if err := dq.CopyFromHost(q); err != nil {
				t.Fatal(err)
			}
			if err := dk.CopyFromHost(kv); err != nil {
				t.Fatal(err)
			}
			poison := make([]float32, dout.Len())
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := dout.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			if err := ds.CopyFromHost(poison[:ds.Len()]); err != nil {
				t.Fatal(err)
			}
			cudaQualificationMeasure(t, r, func() error {
				return r.encodeRawAttention(dq, dk, sink, dc, ds, dout, position)
			})
			got := cudaTestRead(t, dout)
			check := func(name string, got, want []float32) {
				t.Helper()
				if err := cudaQualificationClose(got, want, cudaAttentionQualificationATol, cudaAttentionQualificationRTol); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
			}
			check("rotated query", got[:32768], rotated)
			// The output KV row precedes binary16 storage; validate its rounded form
			// separately from the cache that the attention operation actually reads.
			stored := append([]float32(nil), got[32768:33280]...)
			for i := range stored {
				stored[i] = cudaAttentionQualificationRound(stored[i], 10, -14)
			}
			check("stored row", stored, row)
			check("attention", got[33280:], want)
			check("cache", cudaTestRead(t, dc), cache)
			scores := cudaTestRead(t, ds)
			for h := range 64 {
				check("chronological scores", scores[h*capacity:h*capacity+len(history)], wantScores[h*len(history):(h+1)*len(history)])
			}
			if len(history) > 1 {
				dropped, _ := cudaAttentionQualificationReference(rotated, history[len(history)-1:], sinks, position, false)
				mismatched, _ := cudaAttentionQualificationReference(rotated, history, sinks, position, true)
				for name, bad := range map[string][]float32{"history omitted": dropped, "score/value rows mispaired": mismatched} {
					if cudaQualificationClose(bad, want, cudaAttentionQualificationATol, cudaAttentionQualificationRTol) == nil {
						t.Fatalf("fixture cannot reject %s", name)
					}
				}
			}
		})
	}
}

func cudaAttentionQualificationInputs(position int) ([]float32, []float32) {
	q, kv := make([]float32, 64*512), make([]float32, 512)
	for i := range q {
		q[i] = float32((i*17+position*13)%61-30) / 32
	}
	for i := range kv {
		kv[i] = float32((i*11+position*23)%59-29) / 32
	}
	return q, kv
}

// RoPE uses an analytic frequency per pair instead of the kernel's recurrent
// float32 theta. The reference retains tokens in chronological order and never
// calculates a cache slot, independently checking device ring indexing.
func cudaAttentionQualificationRotate(v []float32, position int) {
	for i := 0; i < 64; i += 2 {
		theta := float64(position) * math.Pow(10000, -float64(i)/64)
		s, c := math.Sincos(theta)
		a, b := float64(v[448+i]), float64(v[449+i])
		v[448+i], v[449+i] = float32(a*c-b*s), float32(a*s+b*c)
	}
}

// Round on a power-of-two lattice with ties to even. mantissa and minExponent
// describe the destination format, independently of the kernel's bit routines.
func cudaAttentionQualificationRound(x float32, mantissa, minExponent int) float32 {
	_, e := math.Frexp(math.Abs(float64(x)))
	step := math.Ldexp(1, max(e-1, minExponent)-mantissa)
	return float32(math.RoundToEven(float64(x)/step) * step)
}

func cudaAttentionQualificationPrepare(q, kv []float32, position int) ([]float32, []float32) {
	rotated, row := append([]float32(nil), q...), append([]float32(nil), kv...)
	for h := range 64 {
		cudaAttentionQualificationRotate(rotated[h*512:(h+1)*512], position)
	}
	cudaAttentionQualificationRotate(row, position)
	for start := 0; start < 448; start += 64 {
		largest := 1e-4
		for _, v := range row[start : start+64] {
			largest = math.Max(largest, math.Abs(float64(v)))
		}
		scale := math.Exp2(math.Ceil(math.Log2(largest / 448)))
		for i := start; i < start+64; i++ {
			row[i] = float32(float64(cudaAttentionQualificationRound(float32(float64(row[i])/scale), 3, -6)) * scale)
		}
	}
	for i := range row {
		row[i] = cudaAttentionQualificationRound(row[i], 10, -14)
	}
	return rotated, row
}

func cudaAttentionQualificationReference(q []float32, history [][]float32, sinks []float32, position int, mispair bool) ([]float32, []float32) {
	out, scores := make([]float32, 64*512), make([]float32, 64*len(history))
	for h := range 64 {
		weights := make([]float64, len(history))
		maximum := float64(sinks[h])
		for j, row := range history {
			dot := 0.0
			for i, v := range row {
				dot += float64(q[h*512+i]) * float64(v)
			}
			weights[j] = dot / math.Sqrt(512)
			scores[h*len(history)+j] = float32(weights[j])
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
				index := j
				if mispair {
					index = (j + 1) % len(history)
				}
				sum += w * float64(history[index][i])
			}
			out[h*512+i] = float32(sum / denom)
		}
		cudaAttentionQualificationRotate(out[h*512:(h+1)*512], -position)
	}
	return out, scores
}
