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
	// Exercise static and implemented dynamic cases in one upstream run.
	// The production capability remains off until general replay is complete;
	// this test-only view also reaches dynamic cases that would otherwise skip.
	backendtest.RunAll(t, dynamicConformanceBackend{b}, nil)
}
