//go:build poweroracle

package main

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestPowerExternalOracle(t *testing.T) {
	path := os.Getenv("DS4_POWER_ORACLE_SOURCE")
	if path == "" {
		t.Fatal("explicit poweroracle gate requires DS4_POWER_ORACLE_SOURCE")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7" {
		t.Fatalf("oracle source identity %s", got)
	}
	fixture, err := os.ReadFile("testdata/power-oracle-11e8cad.c")
	if err != nil {
		t.Fatal(err)
	}
	start := strings.Index(string(fixture), "static bool graph_power_throttle_enabled")
	if start < 0 || !strings.Contains(string(data), strings.TrimSpace(string(fixture[start:]))) {
		t.Fatal("extracted fixture does not match pinned oracle")
	}
}
