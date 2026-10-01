package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func TestSubmissionReceiptsRetainUnmatchedEvidence(t *testing.T) {
	records := timeline.Records{Activities: []timeline.Activity{
		{Journal: "fixture", Sequence: 7, Subtype: "kernel", Name: "matched", Correlation: 1, Start: 100, End: 150},
		{Journal: "fixture", Sequence: 8, Subtype: "kernel", Name: "missing", Correlation: 2, Start: 200, End: 250},
	}, Calls: []timeline.Call{{Journal: "fixture", Sequence: 6, Domain: "driver", Correlation: 1, Start: 90, End: 95}}}
	var output bytes.Buffer
	identity := traceExportIdentity{sha256: strings.Repeat("a", 64), arm: "base"}
	if err := writeSubmissionTrace(&output, records, "driver", 80, 1<<48, 0, 0, identity); err != nil {
		t.Fatal(err)
	}
	if got := bytes.Count(output.Bytes(), []byte("cudaprof.correlation.v1")); got != len(records.Activities) {
		t.Fatalf("receipts=%d, want %d", got, len(records.Activities))
	}
	if got := bytes.Count(output.Bytes(), []byte("cudaprof.correlation_coverage.v1")); got != 1 {
		t.Fatalf("coverage receipts=%d", got)
	}
	for _, match := range timeline.Correlations(records, "driver") {
		data, err := json.Marshal(match)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(output.Bytes(), []byte(hex.EncodeToString(data))) {
			t.Fatalf("native evidence omitted: %+v", match)
		}
	}
}
