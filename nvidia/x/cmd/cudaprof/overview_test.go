package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestOverviewHTMLScopesAndEscaping(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Subtype: "kernel", Name: "<script>raw</script>", Start: 10, End: 20}}}
	report := overview.Analyze(records, "driver", nil)
	result := overviewResult{Capture: "<capture>", Completeness: "partial", Stationarity: "unassessed", Correctness: "unverified", Overview: report, Raw: overviewRaw(records, report)}
	var output bytes.Buffer
	if err := writeOverviewHTML(&output, result); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"&lt;capture&gt;", "&lt;script&gt;raw&lt;/script&gt;", "partial", "unassessed", "No intervals with this provenance.", "Summed kernel work: 10", "raw-"} {
		if !strings.Contains(output.String(), want) {
			t.Errorf("missing %q", want)
		}
	}
	if strings.Contains(output.String(), "<script>") {
		t.Fatal("unescaped raw metadata")
	}
}

func TestOverviewCLIObservedAndPartialReceipt(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	scratch := filepath.Join(home, "tmp")
	if err = os.MkdirAll(scratch, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(scratch, "overview-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	b := newJournalBuilder()
	text := `cudaprof:v1 {"phase":"decode","token":0,"layer":0,"operation":"gemv","materialization":true}`
	for _, m := range []struct {
		name  string
		flags uint32
		at    int
	}{{text, timeline.MarkerStart, 1}, {"", timeline.MarkerEnd, 4}, {`cudaprof:v1 {"phase":"decode","token":0,"event":"delivered"}`, timeline.MarkerInstantaneous, 21}} {
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": m.name, "marker_id": 1, "marker_flags": m.flags, "thread_id": 2, "process_id": 7, "start": m.at, "end": m.at})
	}
	b.append(wire.KindAPI, map[string]any{"kind": "api", "domain": "driver", "correlation": 1, "thread_id": 2, "process_id": 7, "start": 2, "end": 3})
	b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "kernel", "valid": true, "name": "gemv<float>", "correlation": 1, "start": 10, "end": 20})
	for _, completeness := range []string{"passed", "partial"} {
		t.Run(completeness, func(t *testing.T) {
			status := completeStatus()
			status["run_completeness"] = completeness
			path := filepath.Join(dir, completeness+".pprof")
			writeTestCaptureFile(t, path, testCapture(map[string][]byte{"capture.journal": b.finalize()}, status))
			var output bytes.Buffer
			htmlPath := filepath.Join(dir, completeness+".html")
			err := runOverview([]string{"-html", htmlPath, path}, &output)
			if (err != nil) != (completeness == "partial") {
				t.Fatalf("unexpected refusal: %v", err)
			}
			var result overviewResult
			if err := json.Unmarshal(output.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Completeness != completeness || len(result.SHA256) != 64 || result.Stationarity != "unassessed" {
				t.Fatalf("report=%s", output.String())
			}
			if completeness == "passed" && (len(result.Overview.Tokens) != 1 || result.Overview.Tokens[0].Delivery == nil) {
				t.Fatal("observed endpoint missing")
			}
			if completeness == "partial" && len(result.Overview.Tokens) != 0 {
				t.Fatal("reconstructed incomplete timeline")
			}
			html, err := os.ReadFile(htmlPath)
			if err != nil {
				t.Fatal(err)
			}
			if completeness == "passed" && !bytes.Contains(html, []byte("gemv&lt;float&gt;")) {
				t.Fatal("raw kernel name missing")
			}
		})
	}
}
func TestOverviewArguments(t *testing.T) {
	for _, args := range [][]string{{"-domain=x", "missing"}, {"-max-records=-1", "missing"}, {"-iteration-marker=[", "missing"}} {
		if err := runOverview(args, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestOverviewRawSequencePreserved(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "j", Sequence: 42, Subtype: "kernel", Start: 1, End: 2}}}
	report := overview.Analyze(records, "driver", nil)
	raw := overviewRaw(records, report)
	if len(raw) != 1 || raw[0].Sequence != 42 || raw[0].Reference.Index != 0 {
		t.Fatalf("source identity=%+v", raw)
	}
}
