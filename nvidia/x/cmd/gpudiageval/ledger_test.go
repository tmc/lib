package main

import (
	"bytes"
	"github.com/tmc/lib/nvidia/x/cudadiageval"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLedgerCommandPendingRoundTrip(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "ledger.json")
	cell := cudadiageval.QualificationCell{ID: "h100-native", CapabilityID: "nvml", Capability: "unknown", Gate: "native", Status: "pending", SourcePaths: []string{"nvidia/x/nvtelemetry/nvml_sampler.go"}, Consumers: []string{"nvidia/x/cmd/nvtelemetry/main.go"}, Owner: "native-owner", NextAction: "admit exact binary and target", Closure: cudadiageval.QualificationClosure{Candidate: strings.Repeat("1", 40), Tree: strings.Repeat("2", 40)}}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudadiageval.WriteQualificationLedger(f, cudadiageval.QualificationLedger{Version: 1, Cells: []cudadiageval.QualificationCell{cell}}); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if err := run([]string{"ledger", "-file", path, "-root", root}, &out, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"accepted":false`) || !strings.Contains(out.String(), `"status":"pending"`) {
		t.Fatalf("pending promoted: %s", &out)
	}
}
func TestLedgerCommandFlags(t *testing.T) {
	for _, args := range [][]string{{"ledger"}, {"ledger", "-file", "a"}, {"ledger", "-file", "a", "-root", "b", "extra"}} {
		if err := run(args, &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
