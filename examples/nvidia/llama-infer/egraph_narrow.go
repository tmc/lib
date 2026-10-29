package main

import "github.com/tmc/lib/nvidia/x/egraph/hopper"

// prefillNarrowRows is the most rows a [hopper.SkinnyMMA] takes, and so
// the largest chunk the narrow prefill plans run.
const prefillNarrowRows = 32

// planNarrow builds pf's narrow plans: a decoder layer over
// prefillNarrowRows rows whose projections are [hopper.SkinnyMMA]s, split
// along K to fill sms multiprocessors (0 means hopper's default), and
// attention over groups of one row. pf.attn must be set.
//
// Each warp of an [hopper.MTPPagedAttention] CTA walks its query rows in
// turn, so a short chunk attends fastest one row per CTA: on GB10 a
// 21-row chunk's attention takes 7.7 us per layer that way, against 69
// us in groups of 15 rows.
func (pf *egraphPrefillPlans) planNarrow(cfg *ModelConfig, qkvBias bool, sms int) error {
	var err error
	pf.narrow = prefillNarrowRows
	if pf.narrowIn, pf.narrowPost, err = buildLayerPlans(cfg, qkvBias, true, "llama_pfn", pf.narrow, hopper.BF16, ""); err != nil {
		return err
	}
	pf.narrowIn.UseSkinnyMMA(sms)
	pf.narrowPost.UseSkinnyMMA(sms)
	pf.narrowQTok = 1
	pf.narrowAttn = pf.attn
	pf.narrowAttn.Name = "llama_pfn_attn"
	pf.narrowAttn.QTokensPerSeq = pf.narrowQTok
	return nil
}
