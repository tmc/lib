package main

import (
	"fmt"
	"math"
)

// dsparkRope applies only the adjacent rotary tail pairs. DSpark uses the
// dense base, scale=1/ext_factor=0 (ds4.c:34137; CUDA rope_tail_kernel:6940).
func dsparkRope(x []float32, rot, position int, inverse bool) error {
	if rot < 2 || rot%2 != 0 || rot > len(x) || position < 0 || !allFinite(x) {
		return errShape
	}
	tail := x[len(x)-rot:]
	for i := 0; i < rot; i += 2 {
		theta := float64(position) * math.Pow(float64(Flash.RopeBase), -float64(i)/float64(rot))
		c, s := float32(math.Cos(theta)), float32(math.Sin(theta))
		if inverse {
			s = -s
		}
		a, b := tail[i], tail[i+1]
		tail[i], tail[i+1] = a*c-b*s, a*s+b*c
	}
	return nil
}

func dsparkE4M3(code int) float32 {
	e, m := code>>3&15, code&7
	if e == 0 {
		return float32(m) / 512
	}
	return float32(math.Ldexp(1+float64(m)/8, e-7))
}
func dsparkE4M3Round(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign = -1
		x = -x
	}
	x = min(x, 448)
	lo, hi := 0, 126
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if dsparkE4M3(mid) <= x {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if lo < 126 {
		a, b := x-dsparkE4M3(lo), dsparkE4M3(lo+1)-x
		if b < a || b == a && (lo+1)%2 == 0 {
			lo++
		}
	}
	return sign * dsparkE4M3(lo)
}
func dsparkHalfRound(x float32) float32 {
	sign := float32(1)
	if math.Signbit(float64(x)) {
		sign = -1
		x = -x
	}
	if x >= 65520 {
		return sign * float32(math.Inf(1))
	}
	lo, hi := 0, 0x7bff
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if dsparkHalf(uint16(mid)) <= x {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	if lo < 0x7bff {
		a, b := x-dsparkHalf(uint16(lo)), dsparkHalf(uint16(lo+1))-x
		if b < a || b == a && (lo+1)%2 == 0 {
			lo++
		}
	}
	return sign * dsparkHalf(uint16(lo))
}

// dsparkQuantizeKV clones the post-RoPE KV row and performs the per-64
// nonrotary E4M3 round trip followed by half storage on all coordinates
// (ds4_cuda.cu:7146-7169 and store_raw_kv_batch_kernel).
func dsparkQuantizeKV(x []float32, rot int) ([]float32, error) {
	if rot < 0 || rot > len(x) || !allFinite(x) {
		return nil, errShape
	}
	out := append([]float32(nil), x...)
	for start := 0; start < len(x)-rot; start += 64 {
		end := min(start+64, len(x)-rot)
		amax := float32(1e-4)
		for _, v := range x[start:end] {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		scale := float32(math.Exp2(math.Ceil(math.Log2(float64(amax / 448)))))
		for i := start; i < end; i++ {
			out[i] = dsparkE4M3Round(x[i]/scale) * scale
		}
	}
	for i, v := range out {
		out[i] = dsparkHalfRound(v)
	}
	if !allFinite(out) {
		return nil, fmt.Errorf("dspark KV: nonfinite half storage")
	}
	return out, nil
}

// attend reads every visible ring row, including future draft rows, and a
// zero-valued sink. This is noncausal support attention, not target attention.
func (s *dsparkState) attend(stage, start, count int, query []float32, sink float32) ([]float32, error) {
	if s == nil || stage < 0 || stage >= len(s.raw) || start < 0 || start >= s.capacity || count < 1 || count > s.capacity || len(query) != Flash.HeadDim || !allFinite(query, []float32{sink}) {
		return nil, errShape
	}
	scores := make([]float64, count)
	maximum := float64(sink)
	for row := range scores {
		kv := s.raw[stage][((start+row)%s.capacity)*Flash.HeadDim:]
		var dot float64
		for d, q := range query {
			dot += float64(q) * float64(kv[d])
		}
		scores[row] = dot / math.Sqrt(float64(Flash.HeadDim))
		maximum = max(maximum, scores[row])
	}
	denominator := math.Exp(float64(sink) - maximum)
	for i, v := range scores {
		scores[i] = math.Exp(v - maximum)
		denominator += scores[i]
	}
	out := make([]float32, Flash.HeadDim)
	for d := range out {
		var sum float64
		for row, weight := range scores {
			sum += float64(s.raw[stage][((start+row)%s.capacity)*Flash.HeadDim+d]) * weight
		}
		out[d] = float32(sum / denominator)
	}
	return out, nil
}
