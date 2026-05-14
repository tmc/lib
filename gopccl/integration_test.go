package gopccl_test

import (
	"os"
	"testing"
)

// RequireHardwareGate checks whether the hardware manual gate is enabled.
func RequireHardwareGate(t *testing.T) {
	t.Helper()
	if os.Getenv("GOPCCL_TEST_HARDWARE") != "1" {
		t.Skip("set GOPCCL_TEST_HARDWARE=1 to run hardware/provider integration tests")
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

// RequireMultirankGate checks whether the multirank manual gate is enabled.
func RequireMultirankGate(t *testing.T) {
	t.Helper()
	if os.Getenv("GOPCCL_TEST_MULTIRANK") != "1" {
		t.Skip("set GOPCCL_TEST_MULTIRANK=1 with an external accept-pump harness to run multi-rank datapath acceptance tests")
	}
}

// RunMultirankGate is set by handwritten hardware integration tests.
var RunMultirankGate func(*testing.T)

func TestIntegrationMultirankGate(t *testing.T) {
	RequireMultirankGate(t)
	if RunMultirankGate == nil {
		t.Fatal("manual integration body is not registered")
	}
	RunMultirankGate(t)
}
