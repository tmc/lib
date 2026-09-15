package cupticallback

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// Collection reads native storage whose shape comes from cupti_callbacks.h,
// so every admitted CUPTI version needs a measured witness for that header.
//
// Measured across the eight CUDA toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 24, 26, 28, 130001, 130101, 130202, 130301, and
// 130400: CUpti_CallbackData is 72 bytes, CUpti_ResourceData is 24 bytes,
// CUpti_ModuleResourceData is 24 bytes, CUpti_GraphData is 56 bytes,
// CUpti_SynchronizeData is 16 bytes, CUpti_NvtxData is 24 bytes, and
// CUpti_StreamAttrData is 24 bytes across all toolkits.
//
// peer-b's 2026-09-15 L4 report measured the same seven sizes against CUDA
// 12.9.41 headers, with CUPTI_API_VERSION and runtime version both 27. This
// admits that observed build; toolkit release names alone do not select a layout.
//
// 130200 is admitted on a different witness: it is [cupti.ApiVersion], the version
// the Go types in nvidia/cupti were generated from.
func qualifiedVersion(v uint32) bool {
	switch v {
	case 24, 26, 27, 28, 130001, 130101, 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// CheckVersion verifies that the loaded CUPTI library version matches an
// admitted layout witness.
func CheckVersion() error {
	v, err := cupti.GetVersion()
	if err != nil {
		return fmt.Errorf("cupticallback: get version: %w", err)
	}
	if !qualifiedVersion(v) {
		return fmt.Errorf("cupticallback: unqualified CUPTI version %d", v)
	}
	return nil
}
