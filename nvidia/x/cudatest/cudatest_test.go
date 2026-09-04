package cudatest_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestRequireFilters(t *testing.T) {
	dev := cudatest.Require(t, cudatest.MinCompute(7, 0))
	if dev != nil {
		check, err := cudatest.BeginDeviceMemoryCheck(dev, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := check.Check(); err != nil {
				t.Error(err)
			}
		}()
		t.Logf("Acquired device: %d", dev.Ordinal())
	}
}
