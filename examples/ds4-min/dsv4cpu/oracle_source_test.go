package dsv4cpu

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func pinnedCPUOracle(t *testing.T) []byte {
	t.Helper()
	path := os.Getenv("DS4_CPU_ORACLE_SOURCE")
	if path == "" {
		t.Skip("DS4_CPU_ORACLE_SOURCE not set")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7" {
		t.Fatal("oracle source SHA256 differs from pinned11e8cad")
	}
	blob, err := exec.Command("git", "-C", filepath.Dir(path), "rev-parse", "11e8cad4473716a41baf4d62e92ed9764a00fce6:ds4.c").Output()
	if err != nil || strings.TrimSpace(string(blob)) != "20b523b581ff6f67d80327e7680c71d8494c6408" {
		t.Fatalf("oracle Git blob: %v %s", err, blob)
	}
	return b
}
