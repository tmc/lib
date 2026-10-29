package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/advisecorpus"
)

func TestSidecarCLI(t *testing.T) {
	root := t.TempDir()
	file := func(name string, v any) advisecorpus.Artifact {
		t.Helper()
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, name), b, 0600); err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(b)
		return advisecorpus.Artifact{Path: name, SHA256: hex.EncodeToString(h[:]), Bytes: int64(len(b))}
	}
	pair := map[string]advisecorpus.Artifact{"baseline": file("before.cu", "CPU_MOCK_BASELINE"), "candidate": file("after.cu", "CPU_MOCK_CANDIDATE")}
	record := map[string]any{"id": "CPU_MOCK_ONLY", "kind": "STATIC_COMPILE_CLOSURE", "source": pair, "mechanism": "producer issue hypothesis", "comparison": map[string]string{"boundary": "STATIC_ONLY"}, "observation": "STATIC_STOP_NO_NATIVE", "advise": map[string]string{"status": "NOT_EVALUATED"}}
	file("record.ndjson", record)
	index := 0
	link := advisecorpus.Link{Schema: "corpus-advise-link/v1", RecordID: "CPU_MOCK_ONLY", SourcePair: map[string]string{"baseline": pair["baseline"].SHA256, "candidate": pair["candidate"].SHA256}, Mechanism: json.RawMessage(`"producer issue hypothesis"`), Origin: "NONBLIND_CALIBRATION", Scope: "STATIC_ONLY", Outcome: "LEAD", InputArm: "baseline", InputSourceSHA256: pair["baseline"].SHA256, ItemIndex: &index}
	link.ToolSourceReceipt = file("tool-source.json", "CPU_MOCK_TOOL_SOURCE")
	link.ToolSourceSHA256 = link.ToolSourceReceipt.SHA256
	link.ToolBinary = file("tool.txt", "CPU_MOCK_NOT_EXECUTED")
	link.InputCapture = file("capture.json", "CPU_MOCK_NOT_NATIVE")
	link.AdviseReport = file("advice.json", map[string]any{"items": []any{map[string]string{"finding": "producer issue hypothesis", "evidence_level": "suspected"}}})
	link.DecisionReceipt = file("decision.json", map[string]any{"record_id": link.RecordID, "source_pair": link.SourcePair, "mechanism": "producer issue hypothesis", "advice_report_sha256": link.AdviseReport.SHA256, "outcome": "LEAD", "advice_sequence": 1, "decision_sequence": 2})
	for _, name := range []string{"linked", "unknown-id"} {
		t.Run(name, func(t *testing.T) {
			l := link
			if name == "unknown-id" {
				l.RecordID = "absent"
			}
			file("link.json", l)
			args, flags, stdout := os.Args, flag.CommandLine, os.Stdout
			defer func() { os.Args = args; flag.CommandLine = flags; os.Stdout = stdout }()
			flag.CommandLine = flag.NewFlagSet("advisecorpus", flag.ContinueOnError)
			os.Args = []string{"advisecorpus", "-root", root, "-link", filepath.Join(root, "link.json"), filepath.Join(root, "record.ndjson")}
			reader, writer, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			os.Stdout = writer
			err = run()
			writer.Close()
			raw, readErr := io.ReadAll(reader)
			reader.Close()
			if readErr != nil {
				t.Fatal(readErr)
			}
			if name == "unknown-id" {
				if err == nil || len(raw) != 0 {
					t.Fatalf("unknown id produced %s: %v", raw, err)
				}
				return
			}
			if err != nil || !strings.Contains(string(raw), "STATIC_LEAD_LOGGED_NOT_RUNTIME_WIN") {
				t.Fatalf("sidecar was not evaluated: %s %v", raw, err)
			}
		})
	}
}
