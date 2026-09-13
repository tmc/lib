package cupticheckpoint

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// Collection reads native storage whose shape comes from cupti_checkpoint.h,
// so every admitted CUPTI version needs a measured witness for that header,
// not a range comparison.
//
// Measured across the eight CUDA toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 24, 26, 28, 130001, 130101, 130202, 130301, and
// 130400: CUpti_Checkpoint is identical across all eight toolkits, with
// struct size 48 bytes ending at pPriv.
//
// 130200 is admitted on a different witness: it is [cupti.ApiVersion], the version
// the Go types in nvidia/cupti were generated from.
func qualifiedVersion(v uint32) bool {
	switch v {
	case 24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400:
		return true
	}
	return false
}

// CheckVersion verifies that the loaded CUPTI library version matches an
// admitted layout witness.
func CheckVersion() error {
	v, err := cupti.GetVersion()
	if err != nil {
		return fmt.Errorf("cupticheckpoint: get version: %w", err)
	}
	if !qualifiedVersion(v) {
		return fmt.Errorf("cupticheckpoint: unqualified CUPTI version %d", v)
	}
	return nil
}
