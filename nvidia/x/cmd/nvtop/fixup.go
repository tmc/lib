package main

import "math"

// fixup derives device figures from the device's processes, as nvtop's
// gpuinfo_fix_dynamic_info_from_process_info does.
//
// The GPU utilization becomes the larger of the reported rate and the sum
// of the process rates, capped at 100. Encoder and decoder rates the device
// did not report become the capped sums of the process rates. When the
// device reports a total but no use, the use is the sum of the process
// memory and the free memory follows from it.
//
// Last, the effective load is the GPU utilization scaled by the ratio of
// the power drawn to the power limit, capped at 100.
func fixup(d *device) {
	for i := range d.procs {
		p := &d.procs[i]
		p.memPct = opt[uint32]{}
		if !d.memTotal.ok || !p.gpuMem.ok {
			continue
		}
		if p.gpuMem.v > d.memTotal.v {
			p.gpuMem = opt[uint64]{}
			continue
		}
		pct := math.Round(100 * float64(p.gpuMem.v) / float64(d.memTotal.v))
		p.memPct = some(uint32(min(pct, 100)))
	}

	reported := d.gpuUtil
	d.gpuUtil = opt[uint32]{}
	needEnc := !d.encUtil.ok
	needDec := !d.decUtil.ok
	needMem := !d.memUsed.ok && d.memTotal.ok
	for _, p := range d.procs {
		if p.gpuUtil.ok {
			d.gpuUtil = addCapped(d.gpuUtil, p.gpuUtil.v)
		}
		if needEnc && p.encUtil.ok {
			d.encUtil = addCapped(d.encUtil, p.encUtil.v)
		}
		if needDec && p.decUtil.ok {
			d.decUtil = addCapped(d.decUtil, p.decUtil.v)
		}
		if needMem && p.gpuMem.ok {
			d.memUsed = some(d.memUsed.v + p.gpuMem.v)
		}
	}
	if needMem && d.memUsed.ok && d.memUsed.v > d.memTotal.v {
		d.memUsed = opt[uint64]{}
	}
	if needMem && !d.memFree.ok && d.memUsed.ok {
		d.memFree = some(d.memTotal.v - d.memUsed.v)
	}
	if reported.ok && (!d.gpuUtil.ok || reported.v > d.gpuUtil.v) {
		d.gpuUtil = reported
	}

	// The effective load weighs the utilization by the power drawn.
	d.effLoad = opt[uint32]{}
	if d.gpuUtil.ok && d.power.ok && d.powerMax.ok && d.powerMax.v > 0 {
		f := float64(d.power.v) / float64(d.powerMax.v)
		d.effLoad = some(min(uint32(float64(d.gpuUtil.v)*f), 100))
	}
}

// addCapped returns sum+v, capped at 100.
func addCapped(sum opt[uint32], v uint32) opt[uint32] {
	return some(min(100, sum.v+v))
}
