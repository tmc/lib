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
// comparison. 130200 is admitted on a different witness: it is
// [cupti.ApiVersion], the version the Go types in nvidia/cupti were generated
// from.
//
// The witness is per version, not per family. Qualifying a new version means
// measuring its header; until then it runs on the layout below it.
var qualified = []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400}

// supportedVersion reports whether CUPTI version v has a layout: a
// qualified version, or a later release in the same major, which uses the
// newest qualified layout at or below it (see [cuptiversion.Layout]).
func supportedVersion(v uint32) bool {
	_, ok := cuptiversion.Layout(v, qualified...)
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
