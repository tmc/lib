package cudatest_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestRequireFilters(t *testing.T) {
	dev := cudatest.Require(t, cudatest.MinCompute(7, 0))
	if dev != nil {
		defer cudatest.VerifyNoLeaks(t)
		t.Logf("Acquired device: %d", dev.Ordinal())
	}
}
