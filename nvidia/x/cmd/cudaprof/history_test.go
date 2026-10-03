package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestHistoryCommandRegistration(t *testing.T) {
	for _, name := range []string{"history", "bisect"} {
		c, ok := lookupCommand(name)
		if !ok || c.usage == "" || c.run == nil {
			t.Fatalf("offline command %q not registered", name)
		}
		if err := c.run([]string{name}); err == nil || !strings.Contains(err.Error(), "usage: cudaprof "+name) {
			t.Fatalf("%s dispatch error = %v", name, err)
		}
	}
}

func TestHistoryOfflineHandlersBounded(t *testing.T) {
	dir := t.TempDir()
	s, err := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 2, MaxBytes: 4096, MaxRecordBytes: 2048})
	if err != nil {
		t.Fatal(err)
	}
	var snapshot bytes.Buffer
	s.WriteSnapshot(&snapshot)
	path := filepath.Join(dir, "snapshot.ndjson")
	if err := os.WriteFile(path, snapshot.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runHistoryCommand([]string{path}, &out); err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out.String()) != "[]" {
		t.Fatal(out.String())
	}
	request := cudaprofilehistory.BisectRequest{Candidates: []string{"first", "middle", "last"}, Predicate: "correctness", MaxCandidates: 3}
	b, _ := json.Marshal(request)
	path = filepath.Join(dir, "request.json")
	os.WriteFile(path, b, 0600)
	out.Reset()
	if err := runBisectCommand([]string{path}, &out); err != nil {
		t.Fatal(err)
	}
	var p cudaprofilehistory.BisectPlan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil || p.Next != "middle" || p.FirstBad != "" {
		t.Fatalf("offline plan invented execution: %+v %v", p, err)
	}
	out.Reset()
	if err := runBisectCommand([]string{"-max-bytes=8", path}, &out); err == nil || out.Len() != 0 {
		t.Fatal("oversized request published output")
	}
	out.Reset()
	if err := writeHistoryJSON(&out, request, 8); err == nil || out.Len() != 0 {
		t.Fatal("oversized report published output")
	}
	b = bytes.Replace(b, []byte(`"predicate"`), []byte(`"Predicate"`), 1)
	os.WriteFile(path, b, 0600)
	if err := runBisectCommand([]string{path}, &out); err == nil {
		t.Fatal("alias JSON accepted")
	}
}

func TestHistoryOfflineFailedObservationCompare(t *testing.T) {
	d := strings.Repeat("1", 64)
	compiler := cudacache.CompilerIdentity{NVRTCVersion: "13", DriverSONAME: "libcuda", TargetArch: "sm_90", OutputKind: nvrtcx.OutputPTX}
	workload := cudacache.WorkloadIdentity{ProgramID: "test", NumericalMode: "fp32"}
	receipt := []byte("compiler failure")
	hash := sha256.Sum256(receipt)
	row := cudaprofilehistory.Record{Version: 1, SourceDigest: d, InputDigest: d, PlanDigest: d, CompilerInputKey: "source", Compiler: compiler, CompilerKey: cudacache.CompilerCacheKey("source", compiler), Workload: workload, WorkloadKey: workload.Key(), NumericalMode: "fp32", Entry: "kernel", Target: "sm_90", Receipt: receipt, ReceiptDigest: hex.EncodeToString(hash[:]), Status: "compile_failed", Reason: "compiler failure", Correctness: cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}}
	data, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "record.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runHistoryCommand([]string{"compare", path, path}, &out); err != nil {
		t.Fatal(err)
	}
	var report cudaprofilehistory.Comparison
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Performance != cudaprofilehistory.Unknown || report.AfterCorrectness.Verdict != cudaprofilehistory.Unknown {
		t.Fatal("offline failure became qualified outcome")
	}
	store, err := cudaprofilehistory.NewStore(cudaprofilehistory.Limits{MaxRecords: 2, MaxBytes: 65536, MaxRecordBytes: 16384})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Append(row); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := store.WriteSnapshot(&out); err != nil {
		t.Fatal(err)
	}
	snapshot := filepath.Join(t.TempDir(), "snapshot")
	if err := os.WriteFile(snapshot, out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := runHistoryCommand([]string{snapshot}, &out); err != nil {
		t.Fatal(err)
	}
	var rows []cudaprofilehistory.Record
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 || rows[0].Status != "compile_failed" {
		t.Fatalf("failed observation omitted: %+v %v", rows, err)
	}
}
