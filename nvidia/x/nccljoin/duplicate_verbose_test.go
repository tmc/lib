package nccljoin

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// verboseRecord reads the upstream verbose example after applying edit to its
// exact bytes, so Raw and the decoded facts always agree.
func verboseRecord(t *testing.T, old, new string) ncclinspector.Record {
	t.Helper()
	b, err := os.ReadFile("../ncclinspector/testdata/verbose.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if old != "" {
		edited := bytes.Replace(b, []byte(old), []byte(new), 1)
		if bytes.Equal(edited, b) {
			t.Fatalf("fixture lacks %q", old)
		}
		b = edited
	}
	records, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
	if err != nil || len(records) != 1 || records[0].Collective == nil {
		t.Fatalf("verbose fixture: %d records, %v", len(records), err)
	}
	return records[0]
}

func TestJoinIdenticalVerboseDuplicate(t *testing.T) {
	opts := Options{Scope: Scope{"capture", "job", "epoch"}}
	a, b := verboseRecord(t, "", ""), verboseRecord(t, "", "")
	if a.Collective.EventTraceSN == nil || a.Collective.EventTraceTS == nil || a.Collective.EventTraceTS == b.Collective.EventTraceTS {
		t.Fatal("fixture must decode independent event traces")
	}
	captures := []Capture{
		{ID: "a", Records: []ncclinspector.Record{a}},
		{ID: "b", Records: []ncclinspector.Record{b}},
	}
	got, err := Join(captures, opts)
	if err != nil {
		t.Fatalf("identical verbose duplicate refused: %v", err)
	}
	if len(got.Collectives) != 1 {
		t.Fatalf("collectives = %d, want 1", len(got.Collectives))
	}
	c := got.Collectives[0]
	if len(c.Observations) != 2 || c.Observations[0].Capture != "a" || c.Observations[1].Capture != "b" {
		t.Fatalf("duplicate observations not retained: %+v", c.Observations)
	}
	if len(c.Present) != 1 || c.Present[0] != 4 || len(c.Unobserved) != 7 {
		t.Fatalf("membership: present %v unobserved %v", c.Present, c.Unobserved)
	}
}

func TestJoinConflictingVerboseDuplicate(t *testing.T) {
	for _, tc := range []struct {
		name, old, new string
	}{
		{"coll_start_ts", `"coll_start_ts":1752867229235059`, `"coll_start_ts":1752867229235060`},
		{"kernel_stop_ts", `"kernel_stop_ts":1752867229275811`, `"kernel_stop_ts":1752867229275812`},
		{"kernel_stop_sn", `"kernel_stop_sn":48`, `"kernel_stop_sn":49`},
		{"coll_stop_sn", `"coll_stop_sn":2`, `"coll_stop_sn":3`},
		{"drop_event_trace_ts", `,"event_trace_ts":{"coll_start_ts":1752867229235059,"coll_stop_ts":1752867229235064,"kernel_events":[{"channel_id":0,"kernel_start_ts":1752867229235181,"kernel_stop_ts":1752867229275811,"kernel_record_ts":1752867229275811}]}`, ``},
		{"exec_time", `"coll_exec_time_us":41057`, `"coll_exec_time_us":41058`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			opts := Options{Scope: Scope{"capture", "job", "epoch"}}
			changed := verboseRecord(t, tc.old, tc.new)
			captures := []Capture{
				{ID: "a", Records: []ncclinspector.Record{verboseRecord(t, "", "")}},
				{ID: "b", Records: []ncclinspector.Record{changed}},
			}
			_, err := Join(captures, opts)
			if err == nil || !strings.Contains(err.Error(), "conflicting duplicate observation") {
				t.Fatalf("conflicting verbose duplicate: err = %v", err)
			}
		})
	}
}
