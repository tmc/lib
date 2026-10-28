package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// overridePrefillTiles applies LLAMA_EGRAPH_PF_TILE, a comma-separated list
// of bm=, bn=, stages= and streamk= settings, to every prefill tile gemm. It is an
// experiment knob for sweeping tile shapes; unset, the plans are unchanged.
func overridePrefillTiles(pf *egraphPrefillPlans, prefillRows int) error {
	s := os.Getenv("LLAMA_EGRAPH_PF_TILE")
	if s == "" {
		return nil
	}
	var bm, bn, stages, streamK int
	for _, kv := range strings.Split(s, ",") {
		k, v, ok := strings.Cut(kv, "=")
		n, err := strconv.Atoi(v)
		if !ok || err != nil {
			return fmt.Errorf("LLAMA_EGRAPH_PF_TILE: bad setting %q", kv)
		}
		switch k {
		case "bm":
			bm = n
		case "bn":
			bn = n
		case "stages":
			stages = n
		case "streamk":
			streamK = n
		default:
			return fmt.Errorf("LLAMA_EGRAPH_PF_TILE: unknown key %q", k)
		}
	}
	if bm > 0 {
		pf.rows = (prefillRows + bm - 1) / bm * bm
	}
	for _, pl := range []*hopper.Plan{pf.attnIn, pf.post} {
		for _, k := range pl.Kernels {
			t := k.Tile
			if t == nil {
				continue
			}
			if bm > 0 {
				t.BM = bm
			}
			if bn > 0 {
				t.BN = bn
			}
			if stages > 0 {
				t.Stages = stages
			}
			if streamK > 0 {
				t.StreamK = streamK
			}
			if b := t.SharedBytes(); b > sm121SharedPerBlock {
				return fmt.Errorf("LLAMA_EGRAPH_PF_TILE %q: %s needs %d bytes of shared memory, over %d", s, t.Name, b, sm121SharedPerBlock)
			}
		}
	}
	return nil
}

// prefillWorkspaceFloats returns the FP32 words of the one Stream-K
// workspace the prefill tile gemms share, or 0 without Stream-K. The
// gemms run one after another on one stream and each leaves the arrival
// counters zero, so one buffer sized for the largest serves them all.
func prefillWorkspaceFloats(pf *egraphPrefillPlans) int {
	n := 0
	for _, pl := range []*hopper.Plan{pf.attnIn, pf.post} {
		for _, k := range pl.Kernels {
			if k.Tile != nil {
				n = max(n, k.Tile.WorkspaceBytes()/4)
			}
		}
	}
	return n
}
