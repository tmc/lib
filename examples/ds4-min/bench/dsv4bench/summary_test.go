package dsv4bench

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestSummarizeMedianAndRange(t *testing.T) {
	rows := []Row{
		{PrefillTPS: 8, GenTPS: 3, GenFirstMS: 12, GenSteadyTPS: 4},
		{PrefillTPS: 10, GenTPS: 5, GenFirstMS: 10, GenSteadyTPS: 6},
		{PrefillTPS: 9, GenTPS: 4, GenFirstMS: 11, GenSteadyTPS: 5},
	}
	got, err := Summarize(rows)
	if err != nil {
		t.Fatal(err)
	}
	if got.Samples != 3 || got.PrefillTPS != (Range{Min: 8, Median: 9, Max: 10}) || got.GenTPS != (Range{Min: 3, Median: 4, Max: 5}) {
		t.Fatalf("summary = %#v", got)
	}
	if got.GenFirstMS != (Range{Min: 10, Median: 11, Max: 12}) || got.GenSteadyTPS != (Range{Min: 4, Median: 5, Max: 6}) {
		t.Fatalf("summary = %#v", got)
	}
	if rows[0].PrefillTPS != 8 {
		t.Fatal("Summarize mutated input")
	}
}

func TestReadSummaryRequiresStrictCSV(t *testing.T) {
	_, err := ReadSummary(strings.NewReader("prefill_tps\n8\n"))
	if err == nil {
		t.Fatal("accepted non-canonical CSV")
	}
}

func TestWriteSummaryJSON(t *testing.T) {
	var b bytes.Buffer
	if err := WriteSummaryJSON(&b, Summary{Samples: 1, PrefillTPS: Range{Min: 8, Median: 8, Max: 8}}); err != nil {
		t.Fatal(err)
	}
	if want := `{"samples":1,"prefill_tps":{"min":8,"median":8,"max":8},"gen_tps":{"min":0,"median":0,"max":0},"gen_first_ms":{"min":0,"median":0,"max":0},"gen_steady_tps":{"min":0,"median":0,"max":0}}` + "\n"; b.String() != want {
		t.Fatalf("JSON = %q, want %q", b.String(), want)
	}
}

func TestWriteSummaryJSONRejectsInvalidSummary(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Summary)
	}{
		{"zero samples", func(s *Summary) { s.Samples = 0 }},
		{"non-finite", func(s *Summary) { s.PrefillTPS.Median = math.NaN() }},
		{"negative", func(s *Summary) { s.GenTPS.Min = -1 }},
		{"unordered", func(s *Summary) { s.GenFirstMS = Range{Min: 2, Median: 1, Max: 3} }},
	}
	base := Summary{
		Samples:    1,
		PrefillTPS: Range{Min: 8, Median: 8, Max: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := base
			tt.mutate(&candidate)
			if err := WriteSummaryJSON(new(bytes.Buffer), candidate); err == nil {
				t.Fatal("invalid summary accepted")
			}
		})
	}
}

func TestSummarizeRejectsInvalidRows(t *testing.T) {
	_, err := Summarize([]Row{{PrefillTPS: math.NaN()}})
	if err == nil {
		t.Fatal("accepted NaN")
	}
	if _, err := Summarize(nil); err == nil {
		t.Fatal("accepted empty input")
	}
}
