package cuptirange_test

import (
	"fmt"
	"strings"
	"testing"
)

func checkReferenceDependencies(dependencies []string, hasGoBuildID bool) error {
	if hasGoBuildID {
		return fmt.Errorf("reference binary has a Go build ID")
	}
	for _, dep := range dependencies {
		lower := strings.ToLower(dep)
		if strings.Contains(lower, "lib/nvidia") || strings.Contains(lower, "github.com/tmc/lib/nvidia") {
			return fmt.Errorf("reference binary depends on lib/nvidia artifact %q", dep)
		}
	}
	return nil
}

func TestReferenceSampleIndependenceRejectsLibNvidia(t *testing.T) {
	if err := checkReferenceDependencies([]string{"libc.so.6"}, false); err != nil {
		t.Fatalf("independent C++ fixture rejected: %v", err)
	}
	if err := checkReferenceDependencies([]string{"/workspace/lib/nvidia/libcupti.so"}, false); err == nil {
		t.Fatal("lib/nvidia dependency was accepted")
	}
	if err := checkReferenceDependencies(nil, true); err == nil {
		t.Fatal("Go build ID was accepted")
	}
}
