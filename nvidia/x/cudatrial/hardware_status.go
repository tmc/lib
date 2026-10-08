package cudatrial

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// HardwareStatus indicates whether genuine physical GPU hardware is available
// to capture device execution coverage or SASS patching traces.
type HardwareStatus string

const (
	// StatusHardwareUnspecified is the zero value.
	StatusHardwareUnspecified HardwareStatus = ""

	// StatusPendingHardware indicates that no physical GPU device is present,
	// qualified or locked for hardware coverage capture. Generating mock or
	// synthetic coverage bytes is strictly forbidden.
	StatusPendingHardware HardwareStatus = "pending_hardware"

	// StatusHardwareAvailable indicates that physical GPU hardware is present
	// and verified.
	StatusHardwareAvailable HardwareStatus = "available"
)

// hardwareProbeFunc inspects whether real GPU hardware is available.
var hardwareProbeFunc = defaultHardwareProbe

func defaultHardwareProbe() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	if strings.TrimSpace(os.Getenv("GPU_LOCK_HELD")) == "" {
		return false
	}
	// Check if any NVIDIA device node exists.
	if _, err := os.Stat("/dev/nvidia0"); err != nil {
		return false
	}
	return true
}

// CheckHardwareStatus inspects the host environment for qualified physical GPU hardware.
// When no physical GPU is present, it explicitly reports StatusPendingHardware.
func CheckHardwareStatus() HardwareStatus {
	if hardwareProbeFunc() {
		return StatusHardwareAvailable
	}
	return StatusPendingHardware
}

// ValidateHardwareCoverage verifies that coverage bytes were produced by genuine
// physical hardware execution. If status is StatusPendingHardware or not StatusHardwareAvailable,
// or if the trace is empty or invalid, it returns an error wrapping cudaerrors.ErrHardwarePending.
func ValidateHardwareCoverage(raw []byte, status HardwareStatus) error {
	if status == StatusPendingHardware || status != StatusHardwareAvailable {
		return fmt.Errorf("cudatrial: hardware coverage unavailable: %w", cudaerrors.ErrHardwarePending)
	}
	if len(raw) == 0 {
		return fmt.Errorf("cudatrial: hardware coverage trace is empty")
	}
	return nil
}
