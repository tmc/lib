package gomlx

import (
	"github.com/gomlx/compute/support/backendtest"
	"os"
	"testing"
)

func TestNativeConformance(t *testing.T) {
	p := os.Getenv("TMC_COMPUTE_PLUGIN")
	if p == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(p)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	backendtest.RunAll(t, b, nil)
}
