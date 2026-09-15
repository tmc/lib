package cudamega_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestMegaKernelTemplate(t *testing.T) {
	tmpl := cudamega.MegaKernelTemplate
	if !strings.Contains(tmpl, "PersistentMegaKernel") {
		t.Error("expected PersistentMegaKernel in template")
	}
	if !strings.Contains(tmpl, "struct Task") {
		t.Error("expected struct Task in template")
	}
}

func TestDispatcher(t *testing.T) {
	dev := cudatest.Require(t)
	d, err := cudamega.NewDispatcher(dev, 128)
	if err != nil {
		t.Fatalf("NewDispatcher failed: %v", err)
	}
	defer d.Close()

	if d.Capacity() != 128 {
		t.Errorf("expected capacity 128, got %d", d.Capacity())
	}
}
