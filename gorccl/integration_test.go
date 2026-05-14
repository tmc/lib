package gorccl_test

import (
	"os"
	"testing"
)

// RequireHardwareGate checks whether the hardware manual gate is enabled.
func RequireHardwareGate(t *testing.T) {
	t.Helper()
	if os.Getenv("GORCCL_TEST_HARDWARE") != "1" {
		t.Skip("set GORCCL_TEST_HARDWARE=1 to run hardware/provider integration tests")
	}
}

// RunHardwareGate is set by handwritten hardware integration tests.
var RunHardwareGate func(*testing.T)

func TestIntegrationHardwareGate(t *testing.T) {
	RequireHardwareGate(t)
	if RunHardwareGate == nil {
		t.Fatal("manual integration body is not registered")
	}
	RunHardwareGate(t)
}
