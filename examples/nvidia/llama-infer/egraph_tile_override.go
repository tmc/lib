package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// overridePrefillTiles applies LLAMA_EGRAPH_PF_TILE, a comma-separated list
// of bm=, bn= and stages= settings, to every prefill tile gemm. It is an
// experiment knob for sweeping tile shapes; unset, the plans are unchanged.
func overridePrefillTiles(pf *egraphPrefillPlans, prefillRows int) error {
	s := os.Getenv("LLAMA_EGRAPH_PF_TILE")
	if s == "" {
		return nil
	}
	var bm, bn, stages int
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
			if b := t.SharedBytes(); b > sm121SharedPerBlock {
				return fmt.Errorf("LLAMA_EGRAPH_PF_TILE %q: %s needs %d bytes of shared memory, over %d", s, t.Name, b, sm121SharedPerBlock)
			}
		}
	}
	return nil
}
