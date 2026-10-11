package cuptisass

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiversion"
)

// Collection reads native storage whose shape comes from cupti_sass_metrics.h,
// so every qualified CUPTI version has a measured witness for that header.
//
// Measured across the eight CUDA toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 24, 26, 28, 130001, 130101, 130202, 130301, and
// 130400: cupti_sass_metrics.h is byte-identical in all eight, the same 19674
// bytes with md5 036af92478945463a6c50d934902fcd8. Identical bytes mean
// identical structures, which is a stronger witness than a field-by-field
// comparison. 130200 retains its historical generated-header witness.
//
// The witness is per version, not per family. Qualifying a new version means
// measuring its header. Unlisted runtimes are refused, except historical
// source admission of 130401 with layout 130400.
var qualified = []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400}

// Admission is exact. Runtime 130401 retains historical source admission
// with layout 130400; this is not fresh physical qualification.
func supportedVersion(v uint32) bool {
	_, ok := cuptiversion.AdmittedLayout(v, qualified...)
	return ok
}

// CheckVersion verifies that the loaded CUPTI library version matches an
// admitted layout witness.
func CheckVersion() error {
	v, err := cupti.GetVersion()
	if err != nil {
		return fmt.Errorf("cuptisass: get version: %w", err)
	}
	if !supportedVersion(v) {
		return fmt.Errorf("cuptisass: unsupported CUPTI version %d", v)
	}
	return nil
}
