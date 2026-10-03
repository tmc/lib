package main

import "fmt"

// These are the native agent's per-kind admission counters. Pointers keep an
// omitted field distinct from a counter that the producer reported as zero.
type captureKindLoss struct {
	Offered  *uint64 `json:"offered"`
	Retained *uint64 `json:"retained"`
	Dropped  *uint64 `json:"dropped"`
}

type captureKindLosses map[string]captureKindLoss

// activityLoss admits a denominator only when the producer's complete kind
// inventory agrees with the aggregate and independently decoded activity count.
// CUPTI's native loss occurs before admission and has no kind classification.
func (l captureLoss) activityLoss(activity uint64) (offered, dropped uint64, known bool, err error) {
	if l.byKind == nil {
		return 0, 0, false, nil
	}
	kinds := []string{"activity", "api", "external", "failure", "host", "module", "pcsample", "gemm"}
	if len(*l.byKind) != len(kinds) {
		return 0, 0, false, nil
	}
	var totals [3]uint64
	for _, name := range kinds {
		c, ok := (*l.byKind)[name]
		if !ok || c.Offered == nil || c.Retained == nil || c.Dropped == nil {
			return 0, 0, false, nil
		}
		if *c.Retained > *c.Offered || *c.Dropped != *c.Offered-*c.Retained {
			return 0, 0, false, fmt.Errorf("injected %s admission counters inconsistent", name)
		}
		for i, n := range []uint64{*c.Offered, *c.Retained, *c.Dropped} {
			if n > ^uint64(0)-totals[i] {
				return 0, 0, false, fmt.Errorf("injected per-kind counter overflow")
			}
			totals[i] += n
		}
	}
	if totals != [3]uint64{l.Offered, l.Retained, l.Queue} {
		return 0, 0, false, fmt.Errorf("injected per-kind and aggregate counters differ")
	}
	c := (*l.byKind)["activity"]
	if *c.Retained != activity {
		return 0, 0, false, fmt.Errorf("injected activity counter differs from decoded records")
	}
	if l.Native != 0 {
		return 0, 0, false, nil
	}
	return *c.Offered, *c.Dropped, true, nil
}
