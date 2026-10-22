package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4tools"
)

func TestRunPassAndFail(t *testing.T) {
	dir := t.TempDir()
	oracle, candidate := filepath.Join(dir, "oracle.jsonl"), filepath.Join(dir, "candidate.jsonl")
	records := testRecords()
	writeCorpus(t, oracle, records)
	writeCorpus(t, candidate, records)
	var out, errOut bytes.Buffer
	args := []string{"-oracle", oracle, "-candidate", candidate}
	if rc := run(args, &out, &errOut); rc != 0 {
		t.Fatalf("pass rc=%d stderr=%s", rc, errOut.String())
	}
	var receipt dsv4tools.RendererReceipt
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil || receipt.Status != "PASS" {
		t.Fatalf("receipt=%q parsed=%#v err=%v", out.String(), receipt, err)
	}
	records[0].TokenIDs[0]++
	writeCorpus(t, candidate, records)
	out.Reset()
	if rc := run(args, &out, &errOut); rc != 1 {
		t.Fatalf("fail rc=%d stderr=%s", rc, errOut.String())
	}
}

func TestRunMissingIsNotPass(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := run([]string{"-oracle", "missing", "-candidate", "missing"}, &out, &errOut); rc != 1 {
		t.Fatalf("missing rc=%d stderr=%s output=%s", rc, errOut.String(), out.String())
	}
	var receipt dsv4tools.RendererReceipt
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil || receipt.Status != "UNMEASURED" {
		t.Fatalf("missing receipt=%q parsed=%#v err=%v", out.String(), receipt, err)
	}
}

func TestRunUsageAndMalformed(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := run(nil, &out, &errOut); rc != 2 {
		t.Fatalf("usage rc=%d", rc)
	}
	dir := t.TempDir()
	oracle, candidate := filepath.Join(dir, "oracle"), filepath.Join(dir, "candidate")
	if err := os.WriteFile(oracle, []byte("not json\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(candidate, []byte("not json\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if rc := run([]string{"-oracle", oracle, "-candidate", candidate}, &out, &errOut); rc != 2 {
		t.Fatalf("malformed rc=%d", rc)
	}
}

func writeCorpus(t *testing.T, path string, records []dsv4tools.RendererRecord) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := dsv4tools.WriteRendererJSONL(f, records); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func testRecords() []dsv4tools.RendererRecord {
	records := make([]dsv4tools.RendererRecord, dsv4tools.RendererCorpusCases)
	for i := range records {
		records[i] = dsv4tools.RendererRecord{CaseID: string(rune('a' + i)), RenderedInput: "input", TokenIDs: []int{i, -i}}
	}
	return records
}
