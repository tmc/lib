package dsv4tools

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRendererJSONLCorpusGate(t *testing.T) {
	dir := t.TempDir()
	oraclePath := filepath.Join(dir, "oracle.jsonl")
	candidatePath := filepath.Join(dir, "candidate.jsonl")
	records := rendererTestRecords()
	write := func(name string, in []RendererRecord) {
		t.Helper()
		f, err := os.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if err := WriteRendererJSONL(f, in); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	write(oraclePath, records)
	write(candidatePath, records)
	receipt, err := CompareRendererJSONL(oraclePath, candidatePath)
	if err != nil || receipt.Status != "PASS" || receipt.ComparedCases != 24 {
		t.Fatalf("gate = %#v, %v", receipt, err)
	}

	mutated := append([]RendererRecord(nil), records...)
	mutated[7].TokenIDs[0]++
	write(candidatePath, mutated)
	receipt, err = CompareRendererJSONL(oraclePath, candidatePath)
	if err != nil || receipt.Status != "FAIL" || receipt.Mismatches != 1 {
		t.Fatalf("token mutation = %#v, %v", receipt, err)
	}
}

func TestRendererJSONLCorpusRejectsMutations(t *testing.T) {
	dir := t.TempDir()
	oraclePath := filepath.Join(dir, "oracle.jsonl")
	candidatePath := filepath.Join(dir, "candidate.jsonl")
	records := rendererTestRecords()
	writeRecords := func(name string, records []RendererRecord) {
		t.Helper()
		var b bytes.Buffer
		if err := WriteRendererJSONL(&b, records); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(name, b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeRecords(oraclePath, records)
	writeRecords(candidatePath, records)
	for _, test := range []struct {
		name   string
		mutate func([]RendererRecord)
	}{
		{"input", func(r []RendererRecord) { r[3].RenderedInput += "!" }},
		{"duplicate ordinal", func(r []RendererRecord) { r[2].Ordinal = 1 }},
		{"duplicate case id", func(r []RendererRecord) { r[2].CaseID = r[1].CaseID }},
		{"schema", func(r []RendererRecord) { r[4].Schema = "other/v1" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			mutated := append([]RendererRecord(nil), records...)
			test.mutate(mutated)
			// WriteRendererJSONL repairs hashes and ordinals, so use JSONL text
			// directly for mutations that must be rejected on read.
			var lines []string
			for _, record := range mutated {
				b, err := json.Marshal(record)
				if err != nil {
					t.Fatal(err)
				}
				lines = append(lines, string(b))
			}
			if err := os.WriteFile(candidatePath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			receipt, err := CompareRendererJSONL(oraclePath, candidatePath)
			if err == nil || receipt.Status != "FAIL" {
				t.Fatalf("mutation = %#v, %v", receipt, err)
			}
		})
	}
}

func TestRendererJSONLCorpusAbsentIsUnmeasured(t *testing.T) {
	receipt, err := CompareRendererJSONL(filepath.Join(t.TempDir(), "missing"), filepath.Join(t.TempDir(), "missing"))
	if err != nil || receipt.Status != "UNMEASURED" {
		t.Fatalf("missing corpus = %#v, %v", receipt, err)
	}
}

func TestWriteRendererJSONLRequiresRegisteredSize(t *testing.T) {
	var b bytes.Buffer
	if err := WriteRendererJSONL(&b, rendererTestRecords()[:23]); err == nil {
		t.Fatal("accepted incomplete corpus")
	}
}

func rendererTestRecords() []RendererRecord {
	records := make([]RendererRecord, RendererCorpusCases)
	for i := range records {
		records[i] = RendererRecord{CaseID: "case-" + string(rune('a'+i)), RenderedInput: "rendered-" + string(rune('a'+i)), TokenIDs: []int{i, i + 1, -i - 1}}
	}
	return records
}
