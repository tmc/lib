//go:build ds4cpuactualrows

package dsv4cpu

import (
	"os"
	"testing"
)

func TestPinnedNativeQ80ActualRows(t *testing.T) {
	if os.Getenv("DS4_CPU_Q80_ROWS") == "" {
		t.Fatal("DS4_CPU_Q80_ROWS is required")
	}
	TestPinnedNativeQ80Projection(t)
}
