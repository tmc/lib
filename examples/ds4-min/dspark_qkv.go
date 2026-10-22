package main

import "fmt"

// stageQKV executes actual support HC and Q/KV projections for a feature row
// followed by draft rows (ds4.c:34162-34317). The feature row produces KV but
// not Q. Query rows use dense RoPE, not target-layer compressed RoPE.
func (s *dsparkState) stageQKV(stage int, input []float32, positions []int) (queries, kv, post, combine [][]float32, err error) {
	if s == nil || stage < 0 || stage >= s.model.stages || len(positions) != s.model.block+1 || len(input) != len(positions)*Flash.Embed*Flash.HC {
		return nil, nil, nil, nil, errShape
	}
	for row, position := range positions {
		want := s.position
		if row > 0 {
			want += row - 1
		}
		if position != want {
			return nil, nil, nil, nil, fmt.Errorf("dspark QKV: position mismatch")
		}
	}
	name := func(suffix string) string { return fmt.Sprintf("mtp.%d.%s", stage, suffix) }
	kvNorm, err := s.model.denseRow(name("attn_kv_a_norm.weight"), 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	qNorm, err := s.model.denseRow(name("attn_q_a_norm.weight"), 0)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	ones := make([]float32, Flash.HeadDim)
	for i := range ones {
		ones[i] = 1
	}
	for row, position := range positions {
		width := Flash.Embed * Flash.HC
		normalized, p, c, err := s.model.hcPre(stage, input[row*width:(row+1)*width], false)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		raw, err := s.model.denseMatvec(name("attn_kv.weight"), normalized)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		k, err := rmsNorm(raw, kvNorm, Flash.RMSEps)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		if err := dsparkRope(k, Flash.RopeDim, position, false); err != nil {
			return nil, nil, nil, nil, err
		}
		stored, err := dsparkQuantizeKV(k, Flash.RopeDim)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		kv = append(kv, stored)
		if row == 0 {
			continue
		}
		low, err := s.model.denseMatvec(name("attn_q_a.weight"), normalized)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		low, err = rmsNorm(low, qNorm, Flash.RMSEps)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		q, err := s.model.denseMatvec(name("attn_q_b.weight"), low)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		for head := 0; head < Flash.Heads; head++ {
			values := q[head*Flash.HeadDim : (head+1)*Flash.HeadDim]
			n, err := rmsNorm(values, ones, Flash.RMSEps)
			if err != nil {
				return nil, nil, nil, nil, err
			}
			if err := dsparkRope(n, Flash.RopeDim, position, false); err != nil {
				return nil, nil, nil, nil, err
			}
			copy(values, n)
		}
		queries = append(queries, q)
		post = append(post, p)
		combine = append(combine, c)
	}
	return queries, kv, post, combine, nil
}
