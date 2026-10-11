package nvtelemetry

import (
	"fmt"
	"strings"
	"testing"
)

func TestTrainingContext(t *testing.T) {
	raw := `{"version":1,"run_id":"r","updates":[{"attempt":"a","step":9007199254740993,"rank":0,"world_size":2,"outcome":"updated","quality":"accepted","quality_policy":"p","evidence":"log:1"},{"attempt":"a","step":9007199254740993,"rank":1,"world_size":2,"outcome":"updated","quality":"accepted","quality_policy":"p","evidence":"log:2"}]}`
	e, c, err := ReadTrainingContext(strings.NewReader(raw), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if string(e.Raw) != raw || len(e.RawSHA256) != 64 || c.Updates[0].Step != 9007199254740993 {
		t.Fatal("lost raw provenance or integer")
	}
	out, err := SummarizeTrainingContext(c, WorkloadLimits{})
	if err != nil || out.Accepted != 1 {
		t.Fatalf("%+v %v", out, err)
	}
	cases := []struct {
		name   string
		change func(*TrainingContext)
		want   TrainingContextSummary
	}{
		{"missing rank", func(c *TrainingContext) { c.Updates = c.Updates[:1] }, TrainingContextSummary{Incomplete: 1}},
		{"rejected quality", func(c *TrainingContext) { c.Updates[1].Quality = "rejected" }, TrainingContextSummary{Rejected: 1}},
		{"unknown quality", func(c *TrainingContext) { c.Updates[1].Quality = "unknown"; c.Updates[1].QualityPolicy = "" }, TrainingContextSummary{Unknown: 1}},
		{"overflow skipped", func(c *TrainingContext) { c.Updates[1].Outcome = "skipped" }, TrainingContextSummary{Skipped: 1}},
		{"failed", func(c *TrainingContext) { c.Updates[1].Outcome = "failed" }, TrainingContextSummary{Failed: 1}},
		{"resume separates", func(c *TrainingContext) { c.Updates[1].Attempt = "b" }, TrainingContextSummary{Incomplete: 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cc := c
			cc.Updates = append([]TrainingUpdate(nil), c.Updates...)
			tc.change(&cc)
			got, err := SummarizeTrainingContext(cc, WorkloadLimits{})
			if err != nil || got != tc.want {
				t.Fatalf("%+v %v", got, err)
			}
		})
	}
}

func TestTrainingContextRefusals(t *testing.T) {
	base := `{"version":1,"run_id":"r","updates":[{"attempt":"a","step":1,"rank":0,"world_size":1,"outcome":"updated","quality":"accepted","quality_policy":"p","evidence":"log"}]}`
	cases := []struct {
		name, raw string
		limits    WorkloadLimits
	}{
		{"duplicate key", strings.Replace(base, `"version":1`, `"version":1,"version":1`, 1), WorkloadLimits{}},
		{"future version", strings.Replace(base, `"version":1`, `"version":2`, 1), WorkloadLimits{}},
		{"unknown field", strings.Replace(base, `"version":1`, `"version":1,"optimizer_success":true`, 1), WorkloadLimits{}},
		{"trailing json", base + ` {}`, WorkloadLimits{}},
		{"run mismatch", strings.Replace(base, `"run_id":"r"`, `"run_id":"wrong"`, 1), WorkloadLimits{}},
		{"logging step outcome", strings.Replace(base, `"updated"`, `"logged"`, 1), WorkloadLimits{}},
		{"missing quality policy", strings.Replace(base, `"quality_policy":"p"`, `"quality_policy":""`, 1), WorkloadLimits{}},
		{"missing oracle", strings.Replace(base, `"evidence":"log"`, `"evidence":""`, 1), WorkloadLimits{}},
		{"invalid rank", strings.Replace(base, `"rank":0`, `"rank":1`, 1), WorkloadLimits{}},
		{"fractional step", strings.Replace(base, `"step":1`, `"step":1.1`, 1), WorkloadLimits{}},
		{"absent step", strings.Replace(base, `"step":1,`, "", 1), WorkloadLimits{}},
		{"null step", strings.Replace(base, `"step":1`, `"step":null`, 1), WorkloadLimits{}},
		{"absent rank", strings.Replace(base, `"rank":0,`, "", 1), WorkloadLimits{}},
		{"null rank", strings.Replace(base, `"rank":0`, `"rank":null`, 1), WorkloadLimits{}},
		{"negative step", strings.Replace(base, `"step":1`, `"step":-1`, 1), WorkloadLimits{}},
		{"byte limit", base, WorkloadLimits{MaxBytes: 10}},
		{"string limit", base, WorkloadLimits{MaxStringBytes: 2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := ReadTrainingContext(strings.NewReader(tc.raw), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, tc.limits)
			if err == nil {
				t.Fatal("accepted contradiction")
			}
		})
	}
	c := TrainingContext{Version: 1, RunID: "r", Updates: []TrainingUpdate{{Attempt: "a", Step: 1, WorldSize: 2, Outcome: "updated", Quality: "accepted", QualityPolicy: "p", Evidence: "x"}, {Attempt: "a", Step: 1, Rank: 1, WorldSize: 2, Outcome: "updated", Quality: "accepted", QualityPolicy: "q", Evidence: "y"}}}
	if _, err := SummarizeTrainingContext(c, WorkloadLimits{}); err == nil {
		t.Fatal("accepted policy change")
	}
	c.Updates[1] = c.Updates[0]
	if _, err := SummarizeTrainingContext(c, WorkloadLimits{}); err == nil {
		t.Fatal("accepted duplicate rank")
	}
}

func ExampleReadTrainingContext() {
	raw := `{"version":1,"run_id":"run","updates":[{"attempt":"first","step":0,"rank":0,"world_size":1,"outcome":"updated","quality":"unknown","evidence":"application-log"}]}`
	_, c, err := ReadTrainingContext(strings.NewReader(raw), WorkloadSource{Provider: "application", Version: "1", RunID: "run"}, WorkloadLimits{})
	if err != nil {
		panic(err)
	}
	out, _ := SummarizeTrainingContext(c, WorkloadLimits{})
	fmt.Println(out.Accepted, out.Unknown)
	// Output: 0 1
}

func ExampleSummarizeTrainingContext() {
	c := TrainingContext{Version: 1, RunID: "r", Updates: []TrainingUpdate{{Attempt: "resume", Step: 10, WorldSize: 2, Outcome: "updated", Quality: "unknown", Evidence: "rank0-log"}}}
	out, _ := SummarizeTrainingContext(c, WorkloadLimits{})
	fmt.Println(out.Incomplete)
	// Output: 1
}

func ExampleWriteTrainingContext() {
	c := TrainingContext{Version: 1, RunID: "r", Updates: []TrainingUpdate{}}
	var b strings.Builder
	err := WriteTrainingContext(&b, c, WorkloadLimits{})
	fmt.Println(b.String(), err)
	// Output: {"version":1,"run_id":"r","updates":[]} <nil>
}

type contextShortWriter struct{}

func (contextShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }
func TestTrainingContextWriter(t *testing.T) {
	c := TrainingContext{Version: 1, RunID: "r", Updates: []TrainingUpdate{}}
	if err := WriteTrainingContext(contextShortWriter{}, c, WorkloadLimits{}); err == nil {
		t.Fatal("accepted short write")
	}
	var b strings.Builder
	if err := WriteTrainingContext(&b, c, WorkloadLimits{MaxBytes: 10}); err == nil || b.Len() != 0 {
		t.Fatal("unbounded or partially emitted export")
	}
	if err := WriteTrainingContext(&b, c, WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	_, out, err := ReadTrainingContext(strings.NewReader(b.String()), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, WorkloadLimits{})
	if err != nil || out.RunID != c.RunID {
		t.Fatal(err)
	}
}

func TestServingContext(t *testing.T) {
	raw := `{"version":1,"run_id":"r","attempts":[{"request_id":"q","attempt":"1","worker":"w","worker_epoch":"e","role":"decode","outcome":"accepted","output_tokens":9007199254740993,"output_policy":"p","evidence":"log"}]}`
	e, c, err := ReadServingContext(strings.NewReader(raw), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, WorkloadLimits{})
	if err != nil || string(e.Raw) != raw || len(e.RawSHA256) != 64 {
		t.Fatal(err)
	}
	out, err := SummarizeServingContext(c, WorkloadLimits{})
	if err != nil || out.AcceptedOutputTokens != 9007199254740993 || out.KnownAcceptedTokenAttempts != 1 {
		t.Fatalf("%+v %v", out, err)
	}
	c.Attempts[0].OutputTokens = nil
	out, err = SummarizeServingContext(c, WorkloadLimits{})
	if err != nil || out.Accepted != 1 || out.KnownAcceptedTokenAttempts != 0 {
		t.Fatal("unknown token count lost")
	}
	for _, status := range []string{"failed", "cancelled", "partial", "unknown"} {
		c.Attempts[0].Outcome = status
		out, err = SummarizeServingContext(c, WorkloadLimits{})
		if err != nil || out.Accepted != 0 {
			t.Fatalf("%s: %+v %v", status, out, err)
		}
	}
	cases := []struct{ name, raw string }{
		{"duplicate", strings.Replace(raw, `"version":1`, `"version":1,"version":1`, 1)},
		{"unknown field", strings.Replace(raw, `"version":1`, `"version":1,"cache_blocks":20`, 1)},
		{"trailing", raw + ` {}`},
		{"run mismatch", strings.Replace(raw, `"run_id":"r"`, `"run_id":"s"`, 1)},
		{"missing epoch", strings.Replace(raw, `"worker_epoch":"e"`, `"worker_epoch":""`, 1)},
		{"missing policy", strings.Replace(raw, `"output_policy":"p"`, `"output_policy":""`, 1)},
		{"negative tokens", strings.Replace(raw, `9007199254740993`, `-1`, 1)},
		{"fractional tokens", strings.Replace(raw, `9007199254740993`, `1.1`, 1)},
		{"guessed role", strings.Replace(raw, `"decode"`, `"GPU"`, 1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := ReadServingContext(strings.NewReader(tc.raw), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, WorkloadLimits{})
			if err == nil {
				t.Fatal("accepted contradiction")
			}
		})
	}
	a := ServingAttempt{RequestID: "q", Attempt: "1", Worker: "w", WorkerEpoch: "e", Role: "decode", Outcome: "accepted", OutputPolicy: "p", Evidence: "log"}
	c = ServingContext{Version: 1, RunID: "r", Attempts: []ServingAttempt{a, a}}
	if _, err := SummarizeServingContext(c, WorkloadLimits{}); err == nil {
		t.Fatal("duplicate attempt accepted")
	}
	c.Attempts[1].Attempt = "2"
	n := ^uint64(0)
	c.Attempts[0].OutputTokens = &n
	c.Attempts[1].OutputTokens = &n
	if _, err := SummarizeServingContext(c, WorkloadLimits{}); err == nil {
		t.Fatal("token overflow accepted")
	}
	c.Attempts[0].OutputTokens = nil
	c.Attempts[1].OutputTokens = nil
	out, err = SummarizeServingContext(c, WorkloadLimits{})
	if err != nil || out.Accepted != 2 {
		t.Fatal("retry attempts merged")
	}
	if err := WriteServingContext(contextShortWriter{}, c, WorkloadLimits{}); err == nil {
		t.Fatal("short write accepted")
	}
}

func ExampleReadServingContext() {
	raw := `{"version":1,"run_id":"r","attempts":[]}`
	_, c, err := ReadServingContext(strings.NewReader(raw), WorkloadSource{Provider: "application", Version: "1", RunID: "r"}, WorkloadLimits{})
	fmt.Println(c.RunID, err)
	// Output: r <nil>
}

func ExampleWriteServingContext() {
	var b strings.Builder
	err := WriteServingContext(&b, ServingContext{Version: 1, RunID: "r", Attempts: []ServingAttempt{}}, WorkloadLimits{})
	fmt.Println(b.String(), err)
	// Output: {"version":1,"run_id":"r","attempts":[]} <nil>
}

func ExampleSummarizeServingContext() {
	c := ServingContext{Version: 1, RunID: "r", Attempts: []ServingAttempt{{RequestID: "q", Attempt: "1", Worker: "w", WorkerEpoch: "e", Role: "decode", Outcome: "partial", Evidence: "application-log"}}}
	out, _ := SummarizeServingContext(c, WorkloadLimits{})
	fmt.Println(out.Accepted, out.Partial)
	// Output: 0 1
}
