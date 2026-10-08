package cudasanitizer

import "github.com/tmc/lib/nvidia/x/cudatrial"

// HardwareStatus indicates whether genuine physical GPU hardware is available.
type HardwareStatus = cudatrial.HardwareStatus

const (
	// StatusHardwareUnspecified is the zero value.
	StatusHardwareUnspecified = cudatrial.StatusHardwareUnspecified

	// StatusPendingHardware indicates that no physical GPU device is present,
	// qualified or locked for hardware coverage capture. Generating mock or
	// synthetic coverage bytes is strictly forbidden.
	StatusPendingHardware = cudatrial.StatusPendingHardware

	// StatusHardwareAvailable indicates that physical GPU hardware is present
	// and verified.
	StatusHardwareAvailable = cudatrial.StatusHardwareAvailable
)

// CheckHardwareStatus inspects the host environment for qualified physical GPU hardware.
// When no physical GPU is present, it explicitly reports StatusPendingHardware.
func CheckHardwareStatus() HardwareStatus {
	return cudatrial.CheckHardwareStatus()
}

// ValidateHardwareCoverage verifies that coverage bytes were produced by genuine
// physical hardware execution. If status is StatusPendingHardware or not StatusHardwareAvailable,
// it returns an error wrapping cudaerrors.ErrHardwarePending.
func ValidateHardwareCoverage(raw []byte, status HardwareStatus) error {
	return cudatrial.ValidateHardwareCoverage(raw, status)
}
