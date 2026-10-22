package dsv4bench

import (
	"bytes"
	"encoding/csv"
	"math"
	"strings"
	"testing"
)

func TestWriteCSVHeaderAndRow(t *testing.T) {
	var b bytes.Buffer
	row := Row{CtxTokens: 2048, PrefillTokens: 2048, PrefillTPS: 825.76, GenTokens: 128, GenTPS: 18.05, GenFirstMS: 55.4, GenSteadyTokens: 127, GenSteadyTPS: 18.11, KVCacheBytes: 123, ContextTotalBytes: 456, ContextRawBytes: 78, ContextCompressedBytes: 90, ContextScratchBytes: 12, PeakRSSBytes: 345}
	if err := WriteCSV(&b, []Row{row}); err != nil {
		t.Fatal(err)
	}
	r := csv.NewReader(&b)
	records, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || len(records[0]) != 14 || records[0][0] != "ctx_tokens" || records[0][13] != "peak_rss_bytes" {
		t.Fatalf("header = %#v", records[0])
	}
	if records[1][0] != "2048" || records[1][2] != "825.76" || records[1][13] != "345" {
		t.Fatalf("row = %#v", records[1])
	}
}

func TestWriteCSVRejectsInvalidRows(t *testing.T) {
	for name, row := range map[string]Row{"negative count": {CtxTokens: -1}, "NaN": {PrefillTPS: math.NaN()}, "infinity": {GenTPS: math.Inf(1)}} {
		t.Run(name, func(t *testing.T) {
			var b bytes.Buffer
			if err := WriteCSV(&b, []Row{row}); err == nil {
				t.Fatal("invalid row accepted")
			}
			if strings.Contains(b.String(), "-1") && name == "negative count" {
				t.Fatal("invalid row was emitted")
			}
		})
	}
}

func TestWriteCSVEmpty(t *testing.T) {
	var b bytes.Buffer
	if err := WriteCSV(&b, nil); err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(b.String(), "\n"); lines != 1 {
		t.Fatalf("lines = %d, want 1", lines)
	}
}
