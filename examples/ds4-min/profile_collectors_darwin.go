//go:build darwin && arm64

package main

import "fmt"

// Caller has waited for the layer's committed command buffer.
func (r *metalRuntime) collectMetalProfiles(layer, position int) error {
	if r.collectors == nil {
		return nil
	}
	var input, mid []float32
	if r.collectors.imatrix != nil {
		input = r.state.ffnNorm.HostSlice()
		mid = r.state.expertMid.HostSlice()
	}
	return r.collectors.observe(layer, position, 1, r.state.expertIDs.HostSlice(), r.state.expertW.HostSlice(), input, mid)
}

func (s *metalRequestState) setProfileCollectors(p *profileCollectors) error {
	if s == nil || s.closed || s.runtime == nil || s.runtime.pos != 0 {
		return fmt.Errorf("Metal collectors require fresh State")
	}
	s.runtime.collectors = p
	return nil
}
