package kvm_test

import (
	"os"
	"testing"

	kvm "github.com/tmc/lib/kvm"
)

// RequireHardwareGate checks whether the kvm hardware manual gate is enabled.
func RequireHardwareGate(t *testing.T) {
	t.Helper()
	if os.Getenv("KVM_TEST_HARDWARE") != "1" {
		t.Skip("set KVM_TEST_HARDWARE=1 to open /dev/kvm and run hardware integration tests")
	}
}

// RunHardwareGate is set by handwritten hardware integration tests.
var RunHardwareGate func(*testing.T)

func runHardwareGate(t *testing.T) {
	dev, err := kvm.Open()
	if err != nil {
		t.Fatalf("open kvm device: %v", err)
	}
	defer dev.Close()
	version, err := dev.APIVersion()
	if err != nil {
		t.Fatalf("get kvm api version: %v", err)
	}
	if version != kvm.APIVersion {
		t.Fatalf("kvm api version = %d, want %d", version, kvm.APIVersion)
	}
}

func TestIntegrationHardwareGate(t *testing.T) {
	RequireHardwareGate(t)
	run := runHardwareGate
	if RunHardwareGate != nil {
		run = RunHardwareGate
	}
	if run == nil {
		t.Fatal("manual integration body is not registered")
	}
	run(t)
}
