package main

import "github.com/tmc/lib/nvidia/x/egraph/hopper"

// prefillNarrowRows is the most rows a [hopper.SkinnyMMA] takes, and so
// the largest chunk the narrow prefill plans run.
const prefillNarrowRows = 32

// planNarrow builds pf's narrow plans: a decoder layer over
// prefillNarrowRows rows whose projections are [hopper.SkinnyMMA]s, split
// along K to fill sms multiprocessors (0 means hopper's default).
func (pf *egraphPrefillPlans) planNarrow(cfg *ModelConfig, qkvBias bool, sms int) error {
	var err error
	pf.narrow = prefillNarrowRows
	if pf.narrowIn, pf.narrowPost, err = buildLayerPlans(cfg, qkvBias, true, "llama_pfn", pf.narrow, hopper.BF16); err != nil {
		return err
	}
	pf.narrowIn.UseSkinnyMMA(sms)
	pf.narrowPost.UseSkinnyMMA(sms)
	return nil
}
