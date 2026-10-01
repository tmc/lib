package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
)

// Submission parity preserves native match results, including absence and
// ambiguity. SQL consumes supplied receipts; it never performs another join.
func TestTraceSubmissionSQLParity(t *testing.T) {
	processor := os.Getenv("CUDAPROF_TRACE_PROCESSOR")
	if processor == "" {
		t.Skip("set CUDAPROF_TRACE_PROCESSOR to local v58.2")
	}
	base := timeline.Records{
		Activities: []timeline.Activity{{Journal: "j", Sequence: 1, Subtype: "kernel", Correlation: 7, Start: 30, End: 40}},
		Calls:      []timeline.Call{{Journal: "j", Domain: "driver", Process: 1, Thread: 2, Correlation: 7, Start: 20, End: 25}},
		Ranges:     []timeline.Range{{Journal: "j", Process: 1, Thread: 2, Start: 10, End: 29, Name: `cudaprof:v1 {"phase":"decode","token":0,"operation":"gemv","dtype":"float32","cache_length":32}`}},
	}
	for _, tt := range []struct {
		name    string
		edit    func(*timeline.Records)
		invalid bool
	}{
		{"matched and zero unmatched", func(r *timeline.Records) {}, false},
		{"single capture without arm label", func(r *timeline.Records) {}, false},
		{"missing submission", func(r *timeline.Records) {
			r.Activities = append(r.Activities, timeline.Activity{Journal: "j", Sequence: 2, Subtype: "kernel", Correlation: 8, Start: 50, End: 60})
		}, false},
		{"ambiguous call", func(r *timeline.Records) { r.Calls = append(r.Calls, r.Calls[0]) }, false},
		{"unsupported graph provenance", func(r *timeline.Records) { r.Activities[0].GraphID = 1 }, false},
		{"duplicate source sequence", func(r *timeline.Records) { r.Activities = append(r.Activities, r.Activities[0]) }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			records := base
			records.Activities = append([]timeline.Activity(nil), base.Activities...)
			records.Calls = append([]timeline.Call(nil), base.Calls...)
			tt.edit(&records)
			session, err := traceanalysis.New(records, traceanalysis.Options{CollectorComplete: true, Domain: "driver"})
			if err != nil {
				t.Fatal(err)
			}
			var output bytes.Buffer
			arm := "base"
			if tt.name == "single capture without arm label" {
				arm = ""
			}
			if err := writeSubmissionTrace(&output, records, "driver", 10, 3<<48, 0, 0, traceExportIdentity{sha256: strings.Repeat("a", 64), arm: arm}); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"correlations", "unmatched"} {
				sql, err := traceanalysis.SQLDomain(name, "driver")
				if err != nil {
					t.Fatal(err)
				}
				receipt, err := queryTraceProcessor(output.Bytes(), processor, sql)
				if err != nil {
					t.Fatal(err)
				}
				if len(receipt.Rows) < 2 {
					t.Fatalf("missing coverage: %v", receipt.Rows)
				}
				if err := validateSubmissionQuery(receipt.Rows, "driver"); err != nil {
					t.Fatal(err)
				}
				columns := map[string]int{}
				for i, key := range receipt.Rows[0] {
					columns[key] = i
				}
				var observed []timeline.Correlation
				for _, row := range receipt.Rows[1:] {
					if tt.invalid {
						if row[columns["state"]] != "unavailable" {
							t.Fatalf("duplicate promoted: %v", row)
						}
						continue
					}
					if row[columns["row_kind"]] == "coverage" {
						if row[columns["state"]] != "available" {
							t.Fatalf("coverage: %v", row)
						}
						continue
					}
					data, err := hex.DecodeString(row[columns["native_match_hex"]])
					if err != nil {
						t.Fatal(err)
					}
					var match timeline.Correlation
					if err := json.Unmarshal(data, &match); err != nil {
						t.Fatal(err)
					}
					observed = append(observed, match)
				}
				if !tt.invalid {
					want, err := session.Query(name)
					if err != nil {
						t.Fatal(err)
					}
					expected := want.([]timeline.Correlation)
					if len(expected) != len(observed) || len(expected) > 0 && !reflect.DeepEqual(expected, observed) {
						t.Fatalf("%s native=%+v SQL=%+v", name, expected, observed)
					}
				}
			}
			// Domain selection cannot use retained driver receipts as runtime evidence.
			sql, err := traceanalysis.SQLDomain("correlations", "runtime")
			if err != nil {
				t.Fatal(err)
			}
			receipt, err := queryTraceProcessor(output.Bytes(), processor, sql)
			if err != nil {
				t.Fatal(err)
			}
			if len(receipt.Rows) != 1 {
				t.Fatalf("domain fallback: %v", receipt.Rows)
			}
		})
	}
}

func TestTraceSubmissionRejectsDisagreeingColumns(t *testing.T) {
	match := timeline.Correlation{Activity: 0, Call: 0, Range: -1, APIStatus: "matched", RangeStatus: "missing", MetadataStatus: "missing", Provenance: "submission"}
	data, err := json.Marshal(match)
	if err != nil {
		t.Fatal(err)
	}
	rows := [][]string{
		{"state", "row_kind", "capture_sha256", "arm", "journal", "raw_activity_sequence", "raw_activity_index", "domain", "api_status", "range_status", "metadata_status", "provenance", "native_match_hex", "reason"},
		{"available", "activity", strings.Repeat("a", 64), "base", "j", "1", "0", "driver", "matched", "missing", "missing", "submission", hex.EncodeToString(data), ""},
	}
	if err := validateSubmissionQuery(rows, "driver"); err != nil {
		t.Fatal(err)
	}
	rows[1][8] = "unsupported"
	if err := validateSubmissionQuery(rows, "driver"); err == nil {
		t.Fatal("accepted contradictory status column")
	}
}
