package cuptipmsampling

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiversion"
)

// Collection reads native storage whose shape comes from cupti_pmsampling.h,
// so every qualified CUPTI version has a measured witness for that header.
//
// Measured across the eight CUDA toolkits installed on the GB10 host, whose
// CUPTI_API_VERSION values are 24, 26, 28, 130001, 130101, 130202, 130301, and
// 130400:
//   - CUDA 12.8 through 13.4 share identical struct definitions, where
//     CUpti_PmSampling_SetConfig_Params ends at hwBufferAppendMode (64 bytes).
//   - CUDA 12.6 (version 24) precedes hwBufferAppendMode, ending at
//     triggerMode (60 bytes).
//   - All other ten parameter structs are byte-identical across all versions.
//
// 130200 retains its historical generated-header witness.
var qualified = []uint32{24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400}

// Admission is exact. Runtime 130401 retains historical source admission
// with layout 130400; this is not fresh physical qualification.
func supportedVersion(v uint32) bool {
	_, ok := cuptiversion.AdmittedLayout(v, qualified...)
	return ok
}

// setConfigStructSize returns the version-specific prefix size for
// [cupti.PmSamplingSetConfigParams].
func setConfigStructSize(v uint32) uint {
	if l, _ := cuptiversion.AdmittedLayout(v, qualified...); l == 24 {
		// CUDA 12.6 layout ends at triggerMode
		var p cupti.PmSamplingSetConfigParams
		return uint(unsafe.Offsetof(p.TriggerMode) + unsafe.Sizeof(p.TriggerMode))
	}
	// CUDA 12.8+ layout ends at hwBufferAppendMode
	var p cupti.PmSamplingSetConfigParams
	return uint(unsafe.Offsetof(p.HwBufferAppendMode) + unsafe.Sizeof(p.HwBufferAppendMode))
}

// CheckVersion verifies that the loaded CUPTI library version matches an
// admitted layout witness.
func CheckVersion() (uint32, error) {
	v, err := cupti.GetVersion()
	if err != nil {
		return 0, fmt.Errorf("cuptipmsampling: get version: %w", err)
	}
	if !supportedVersion(v) {
		return 0, fmt.Errorf("cuptipmsampling: unsupported CUPTI version %d", v)
	}
	return v, nil
}
