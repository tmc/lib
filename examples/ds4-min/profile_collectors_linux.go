//go:build linux

package main

import "fmt"

func (r *cudaRuntime) collectCUDAProfiles(rows []cudaTokenState, layer, position int) error {
	if r.collectors == nil {
		return nil
	}
	if err := syncStream(r.stream()); err != nil {
		return fmt.Errorf("profile sync: %w", err)
	}
	ids := make([]int32, len(rows)*6)
	weights := make([]float32, len(ids))
	var input, mid []float32
	if r.collectors.imatrix != nil {
		input = make([]float32, len(rows)*cudaWidth)
		mid = make([]float32, len(rows)*6*2048)
	}
	for i, row := range rows {
		if err := row.expertIDs.CopyToHost(ids[i*6 : (i+1)*6]); err != nil {
			return err
		}
		if err := row.expertW.CopyToHost(weights[i*6 : (i+1)*6]); err != nil {
			return err
		}
		if r.collectors.imatrix != nil {
			if err := row.ffnNorm.CopyToHost(input[i*cudaWidth : (i+1)*cudaWidth]); err != nil {
				return err
			}
			if err := row.expertMid.CopyToHost(mid[i*6*2048 : (i+1)*6*2048]); err != nil {
				return err
			}
		}
	}
	return r.collectors.observe(layer, position, len(rows), ids, weights, input, mid)
}

func (s *cudaRequestState) setProfileCollectors(p *profileCollectors) error {
	if s == nil || s.closed || s.runtime == nil || s.runtime.pos != 0 || s.runtime.batch != nil {
		return fmt.Errorf("CUDA collectors require fresh State")
	}
	for _, g := range s.runtime.graphs {
		if g.qkv != nil || g.ffn != nil {
			return fmt.Errorf("CUDA collectors require uncaptured State")
		}
	}
	s.runtime.collectors = p
	return nil
}
