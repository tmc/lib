package dsv4tools

import "fmt"

// ExpertVariant identifies the routed-expert encoding of a complete Flash
// model. It describes model admission only; it does not select a backend
// kernel.
type ExpertVariant uint8

const (
	ExpertVariantIQ2Q2K ExpertVariant = iota + 1
	ExpertVariantIQ2Q4K
	ExpertVariantQ4K
	ExpertVariantMXFP4
)

// ClassifyExpertLayout validates the per-layer expert formats and classifies
// one of the Flash layouts documented by the upstream model recipes. The
// q2-q4 splice is the only mixed layout admitted: its final six layers are
// Q4_K and the first 37 retain IQ2_XXS/Q2_K.
//
// This is a host-side admission check. A successful classification does not
// imply that a CUDA or Metal backend has a kernel for the variant.
func ClassifyExpertLayout(records []ExpertLayoutRecord) (ExpertVariant, error) {
	if len(records) != 43 {
		return 0, fmt.Errorf("expert layout has %d layers, want 43", len(records))
	}
	allIQ2 := true
	allQ4 := true
	allMXFP4 := true
	spliced := true
	for i, record := range records {
		if record.Layer != i {
			return 0, fmt.Errorf("expert layer %d has ordinal %d", i, record.Layer)
		}
		if !sameDims(record.GateDims, 4096, 2048, 256) ||
			!sameDims(record.UpDims, 4096, 2048, 256) ||
			!sameDims(record.DownDims, 2048, 4096, 256) {
			return 0, fmt.Errorf("expert layer %d has invalid dimensions", i)
		}
		iq2 := record.GateType == "IQ2_XXS" && record.UpType == "IQ2_XXS" && record.DownType == "Q2_K"
		q4 := record.GateType == "Q4_K" && record.UpType == "Q4_K" && record.DownType == "Q4_K"
		mxfp4 := record.GateType == "MXFP4" && record.UpType == "MXFP4" && record.DownType == "MXFP4"
		allIQ2 = allIQ2 && iq2
		allQ4 = allQ4 && q4
		allMXFP4 = allMXFP4 && mxfp4
		wantSplice := i >= 37
		spliced = spliced && ((wantSplice && q4) || (!wantSplice && iq2))
		if !iq2 && !q4 && !mxfp4 {
			return 0, fmt.Errorf("expert layer %d has unsupported types %s/%s/%s", i, record.GateType, record.UpType, record.DownType)
		}
	}
	switch {
	case allIQ2:
		return ExpertVariantIQ2Q2K, nil
	case spliced:
		return ExpertVariantIQ2Q4K, nil
	case allQ4:
		return ExpertVariantQ4K, nil
	case allMXFP4:
		return ExpertVariantMXFP4, nil
	default:
		return 0, fmt.Errorf("expert layout is not a supported Flash variant")
	}
}
