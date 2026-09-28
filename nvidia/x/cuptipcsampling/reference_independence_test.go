package cuptipcsampling_test

import (
	"fmt"
	"strings"
	"testing"
)

func checkPCReferenceDependencies(deps []string, hasGoBuildID bool) error {
	if hasGoBuildID {
		return fmt.Errorf("reference binary has a Go build ID")
	}
	for _, dep := range deps {
		lower := strings.ToLower(dep)
		if strings.Contains(lower, "lib/nvidia") || strings.Contains(lower, "github.com/tmc/lib/nvidia") {
			return fmt.Errorf("reference binary depends on lib/nvidia artifact %q", dep)
		}
	}
	return nil
}

func TestReferenceSampleIndependenceRejectsLibNvidia(t *testing.T) {
	if err := checkPCReferenceDependencies([]string{"libc.so.6"}, false); err != nil {
		t.Fatalf("independent C++ fixture rejected: %v", err)
	}
	if err := checkPCReferenceDependencies([]string{"/work/lib/nvidia/libcupti.so"}, false); err == nil {
		t.Fatal("lib/nvidia dependency accepted")
	}
	if err := checkPCReferenceDependencies(nil, true); err == nil {
		t.Fatal("Go build ID accepted")
	}
}
