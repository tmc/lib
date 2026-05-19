package virtiofs_test

import (
	"os"
	"testing"
)

func RequireVMGate(t *testing.T) {
	t.Helper()
	if os.Getenv("VIRTIOFS_TEST_VM") != "1" {
		t.Skip("set VIRTIOFS_TEST_VM=1 and provide a VM-backed RunVMGate hook")
	}
}

var RunVMGate func(*testing.T)

func TestIntegrationVMGate(t *testing.T) {
	RequireVMGate(t)
	if RunVMGate == nil {
		t.Fatal("manual integration body is not generated")
	}
	RunVMGate(t)
}
