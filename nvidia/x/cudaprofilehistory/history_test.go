package cudaprofilehistory

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func fixture(t testing.TB) Record {
	t.Helper()
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	compiler := cudacache.CompilerIdentity{NVRTCVersion: "13.0", DriverSONAME: "libcuda.so.1", TargetArch: "sm_90", Flags: []string{"--std=c++20"}, OutputKind: nvrtcx.OutputPTX}
	workload := cudacache.WorkloadIdentity{ProgramID: "affine", ShapeDims: []int{64}, NumericalMode: "fp32"}
	r := Record{Version: Version, SourceDigest: digest([]byte("source")), InputDigest: digest([]byte("input")), CompilerInputKey: "source-with-headers", Compiler: compiler, CompilerKey: cudacache.CompilerCacheKey("source-with-headers", compiler), Workload: workload, WorkloadKey: workload.Key(), NumericalMode: "fp32", Entry: "kernel", Target: "sm_90", PlanDigest: digest([]byte("plan")), Receipt: []byte("physical producer receipt fixture"), Status: "complete", Environment: Environment{Device: "gpu", Driver: "driver", CUDA: "13", Clock: "gpu", Domain: "same"}, Artifact: a}
	r.ReceiptDigest = digest(r.Receipt)
	r.Identity = cudaevidence.Identity{ArtifactDigest: a.Digest, Entry: r.Entry, Target: r.Target, CompilerKey: r.CompilerKey, WorkloadKey: r.WorkloadKey}
	r.Correctness = Correctness{Verdict: Pass, Method: "digest-exact", Oracle: "independent-affine", ExpectedDigest: digest([]byte("output")), ActualDigest: digest([]byte("output"))}
	r.Facts = []Fact{fact(t, r, "resource", "registers", 32, cudaevidence.Derived), fact(t, r, "codegen", "sass_count", 4, cudaevidence.Derived), fact(t, r, "performance", "gpu_busy", 100, cudaevidence.Measured)}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	return r
}
func fact(t testing.TB, r Record, axis, name string, n float64, class cudaevidence.Class) Fact {
	t.Helper()
	unit := "count"
	if axis == "performance" {
		unit = "ns"
	}
	v, err := cudaevidence.New(cudaevidence.Record{Class: class, Number: &n, Unit: unit, Basis: "explicit producer aggregate", Source: cudaevidence.Source{Identity: r.Identity, Producer: "fixture", Locator: name}, Loss: &cudaevidence.Loss{Total: 10}})
	if err != nil {
		t.Fatal(err)
	}
	return Fact{Axis: axis, Name: name, Value: v, Qualified: true, Samples: 10}
}
func limits() Limits { return Limits{MaxRecords: 100, MaxBytes: 1 << 20, MaxRecordBytes: 1 << 16} }
func policy() Policy {
	return Policy{Timing: "gpu_busy", RelativeThreshold: .1, NoiseNanos: 1, MinSamples: 3}
}

func TestHistoryRejectsIdentityCollision(t *testing.T) {
	s, _ := NewStore(limits())
	r := fixture(t)
	first, err := s.Append(r)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Append(r); err == nil {
		t.Fatal("duplicate overwrote row")
	}
	changed := fixture(t)
	changed.Compiler.NVRTCVersion = "14"
	if _, err := s.Append(changed); err == nil {
		t.Fatal("stale compiler key accepted")
	}
	changed.CompilerKey = cudacache.CompilerCacheKey(changed.CompilerInputKey, changed.Compiler)
	changed.Identity.CompilerKey = changed.CompilerKey
	changed.Facts = nil
	second, err := s.Append(changed)
	if err != nil || first == second {
		t.Fatalf("compiler-only change collapsed: %s %v", second, err)
	}
	r.Artifact.Raw[0] ^= 1
	r.Compiler.Flags[0] = "changed"
	r.Workload.ShapeDims[0] = 99
	r.Receipt[0] = 'X'
	rows, err := s.Records()
	if err != nil || len(rows) != 2 || rows[0].Artifact.Raw[0] != '.' || rows[0].Compiler.Flags[0] != "--std=c++20" || rows[0].Workload.ShapeDims[0] != 64 {
		t.Fatalf("store aliased caller: %+v %v", rows, err)
	}
	rows[0].Artifact.Raw[0] = 'X'
	again, err := s.Records()
	if err != nil || again[0].Artifact.Raw[0] != '.' {
		t.Fatal("snapshot mutated store")
	}
}

func TestHistoryPreservesFailedAndUnreadRows(t *testing.T) {
	r := fixture(t)
	r.Artifact = nil
	r.Identity = cudaevidence.Identity{}
	r.Status = "compile_failed"
	r.Reason = "compiler refused source"
	r.Correctness = Correctness{Verdict: Unknown}
	r.Facts = []Fact{{Axis: "resource", Name: "registers"}}
	s, _ := NewStore(limits())
	if _, err := s.Append(r); err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	if err := s.WriteSnapshot(&encoded); err != nil {
		t.Fatal(err)
	}
	restored, err := ReadSnapshot(&encoded, limits())
	if err != nil {
		t.Fatal(err)
	}
	rows, _ := restored.Records()
	if rows[0].Artifact != nil || rows[0].Facts[0].Value.Record().Number != nil || rows[0].Correctness.Verdict != Unknown {
		t.Fatal("failure became fabricated pass")
	}
	r.Correctness = fixture(t).Correctness
	if err := r.Validate(); err == nil {
		t.Fatal("missing image got correctness pass")
	}
}

func TestHistoryRejectsStaleEvidence(t *testing.T) {
	for _, kind := range []string{"bytes", "workload", "artifact missing", "artifact digest", "input", "fact", "unread fact", "receipt", "key", "empty provenance", "compile number"} {
		t.Run(kind, func(t *testing.T) {
			r := fixture(t)
			switch kind {
			case "bytes":
				r.Artifact.Raw[0] ^= 1
			case "workload":
				r.Workload.ProgramID = "different"
			case "artifact missing":
				r.Artifact = nil
			case "artifact digest":
				r.Artifact.Digest = ""
				r.Identity.ArtifactDigest = ""
				r.Facts = nil
			case "input":
				r.InputDigest = ""
			case "receipt":
				r.Receipt[0] ^= 1
			case "key":
				r.Key = strings.Repeat("0", 64)
			case "fact":
				v := r.Facts[0].Value.Record()
				v.Source.Identity.CompilerKey = strings.Repeat("0", 64)
				r.Facts[0].Value, _ = cudaevidence.New(v)
			case "unread fact":
				v := cudaevidence.Record{Reason: "unavailable", Source: cudaevidence.Source{Identity: r.Identity}}
				v.Source.Identity.CompilerKey = strings.Repeat("0", 64)
				r.Facts[0].Value, _ = cudaevidence.New(v)
			case "empty provenance":
				b, _ := json.Marshal(r)
				b = bytes.ReplaceAll(b, []byte(`"artifact_digest":"`+r.Artifact.Digest+`"`), []byte(`"artifact_digest":""`))
				if _, err := DecodeRecord(b, len(b)); err == nil {
					t.Fatal("empty non-unread provenance accepted")
				}
				return
			case "compile number":
				r.Artifact = nil
				r.Identity = cudaevidence.Identity{}
				r.Status = "compile_failed"
				r.Reason = "failed"
				r.Correctness = Correctness{Verdict: Unknown}
			}
			if err := r.Validate(); err == nil {
				t.Fatal("invalid record accepted")
			}
		})
	}
}

func TestHistoryStrictBoundedAtomicSnapshot(t *testing.T) {
	r := fixture(t)
	b, _ := json.Marshal(r)
	for _, bad := range [][]byte{bytes.Replace(b, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1), bytes.Replace(b, []byte(`"version":1`), []byte(`"Version":1`), 1), bytes.Replace(b, []byte(`"entry_point"`), []byte(`"Entry_point"`), 1), append(append([]byte(nil), b...), []byte(` {}`)...), append([]byte(`{"bogus":1,`), b[1:]...)} {
		if _, err := DecodeRecord(bad, len(bad)); err == nil {
			t.Fatal("hostile JSON accepted")
		}
	}
	if _, err := DecodeRecord(b, len(b)-1); err == nil {
		t.Fatal("original byte bound ignored")
	}
	s, _ := NewStore(limits())
	s.Append(r)
	var snapshot bytes.Buffer
	s.WriteSnapshot(&snapshot)
	invalid := append(append([]byte(nil), snapshot.Bytes()...), []byte("{}\n")...)
	if got, err := ReadSnapshot(bytes.NewReader(invalid), limits()); err == nil || got != nil {
		t.Fatal("partial import returned")
	}
	small := limits()
	small.MaxRecords = 1
	got, err := ReadSnapshot(bytes.NewReader(snapshot.Bytes()), small)
	if err != nil {
		t.Fatal(err)
	}
	r.InputDigest = digest([]byte("other input"))
	if _, err := got.Append(r); err == nil {
		t.Fatal("record capacity ignored")
	}
	if _, err := ReadSnapshot(bytes.NewReader(snapshot.Bytes()), Limits{MaxRecords: 10, MaxBytes: len(snapshot.Bytes()) - 1, MaxRecordBytes: len(snapshot.Bytes()) - 1}); err == nil {
		t.Fatal("snapshot byte cap ignored")
	}
	var zero Store
	if _, err := zero.Append(fixture(t)); err == nil {
		t.Fatal("zero store accepted")
	}
}

func TestCompareSeparatesCodegenResourceAndPerformance(t *testing.T) {
	a, b := fixture(t), fixture(t)
	b.Artifact.Raw = append(b.Artifact.Raw, []byte("// new code\n")...)
	b.Artifact.Digest = digest(b.Artifact.Raw)
	b.Identity.ArtifactDigest = b.Artifact.Digest
	b.Facts = []Fact{fact(t, b, "resource", "registers", 64, cudaevidence.Derived), fact(t, b, "performance", "gpu_busy", 140, cudaevidence.Measured)}
	report, err := Compare(a, b, policy())
	if err != nil || !report.CodeChanged || report.Performance != Fail || len(report.Resources) != 1 || len(report.Codegen) != 1 || report.Codegen[0].After != nil {
		t.Fatalf("axes not independent: %+v %v", report, err)
	}
	for _, kind := range []string{"unread", "inferred", "loss", "unknown loss", "unqualified", "undersampled", "correctness", "input", "environment", "zero", "noise"} {
		t.Run(kind, func(t *testing.T) {
			a, b := fixture(t), fixture(t)
			f := b.Facts[2]
			v := f.Value.Record()
			switch kind {
			case "unread":
				f.Value = cudaevidence.Value{}
			case "inferred":
				v.Class = cudaevidence.Inferred
				f.Value, _ = cudaevidence.New(v)
			case "loss":
				v.Loss.Dropped = 1
				f.Samples--
				f.Value, _ = cudaevidence.New(v)
			case "unknown loss":
				v.Loss = nil
				f.Value, _ = cudaevidence.New(v)
			case "unqualified":
				f.Qualified = false
			case "undersampled":
				f.Samples = 1
				v.Loss.Total = 1
				f.Value, _ = cudaevidence.New(v)
			case "correctness":
				b.Correctness = Correctness{Verdict: Unknown}
			case "input":
				b.InputDigest = digest([]byte("new input"))
			case "environment":
				b.Environment.Driver = "changed"
			case "zero":
				a.Facts[2] = fact(t, a, "performance", "gpu_busy", 0, cudaevidence.Measured)
			case "noise":
				f = fact(t, b, "performance", "gpu_busy", 110, cudaevidence.Measured)
			}
			b.Facts[2] = f
			got, err := Compare(a, b, policy())
			if err != nil || got.Performance != Unknown {
				t.Fatalf("unusable timing judged: %+v %v", got, err)
			}
		})
	}
}

func TestBisectPlannerIsDeterministic(t *testing.T) {
	r := BisectRequest{Candidates: []string{"a", "b", "c", "d", "e"}, Predicate: "correctness", MaxCandidates: 5}
	a, err := PlanBisect(r)
	b, _ := PlanBisect(r)
	if err != nil || a != b || a.Next != "c" {
		t.Fatalf("nondeterministic midpoint: %+v %v", a, err)
	}
}
func TestBisectDoesNotInventObservation(t *testing.T) {
	r := BisectRequest{Candidates: []string{"a", "b", "c"}, Predicate: "correctness", MaxCandidates: 3, AssumeMonotonic: true, Observations: []Observation{{Candidate: "c", Correctness: Fail, Performance: Unknown}}}
	p, err := PlanBisect(r)
	if err != nil || p.FirstBad != "" || p.Next == "c" {
		t.Fatalf("invented predecessor: %+v %v", p, err)
	}
	r.Observations = append(r.Observations, Observation{Candidate: "b", Correctness: Pass, Performance: Unknown})
	p, err = PlanBisect(r)
	if err != nil || p.FirstBad != "c" {
		t.Fatalf("observed endpoints ignored: %+v %v", p, err)
	}
	r.AssumeMonotonic = false
	p, _ = PlanBisect(r)
	if p.FirstBad != "" {
		t.Fatal("assumption invented")
	}
}
func TestBisectKeepsCorrectnessAndPerformanceVerdictsSeparate(t *testing.T) {
	d := digest([]byte("policy"))
	r := BisectRequest{Candidates: []string{"a", "b"}, Predicate: "performance", Baseline: "base", PolicyDigest: d, MaxCandidates: 2, AssumeMonotonic: true, Observations: []Observation{{Candidate: "a", Correctness: Pass, Performance: Pass, Baseline: "base", PolicyDigest: d}, {Candidate: "b", Correctness: Fail, Performance: Fail, Baseline: "base", PolicyDigest: d}}}
	p, err := PlanBisect(r)
	if err != nil || p.FirstBad != "" {
		t.Fatalf("correctness failure became performance regression: %+v %v", p, err)
	}
	r.Predicate = "correctness"
	p, err = PlanBisect(r)
	if err != nil || p.FirstBad != "b" {
		t.Fatal("correctness endpoints lost")
	}
	r.Observations = append(r.Observations, Observation{Candidate: "b", Correctness: Pass, Performance: Unknown})
	p, _ = PlanBisect(r)
	if !p.Conflict || p.FirstBad != "" {
		t.Fatal("conflict hidden")
	}
}

func TestHistoryConcurrentAppendRead(t *testing.T) {
	s, _ := NewStore(limits())
	base := fixture(t)
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			r := base
			r.InputDigest = digest([]byte(fmt.Sprint(i)))
			if _, err := s.Append(r); err != nil {
				t.Error(err)
			}
			if _, err := s.Records(); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	rows, _ := s.Records()
	if len(rows) != 20 {
		t.Fatal(len(rows))
	}
}

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }
func TestHistorySnapshotShortWrite(t *testing.T) {
	s, _ := NewStore(limits())
	if err := s.WriteSnapshot(shortWriter{}); err != io.ErrShortWrite {
		t.Fatal(err)
	}
}

type reentrantWriter struct {
	s *Store
	r Record
}

func (w reentrantWriter) Write(p []byte) (int, error) {
	if _, err := w.s.Append(w.r); err != nil {
		return 0, err
	}
	return len(p), nil
}
func TestHistorySnapshotRejectsDeletedReorderedRows(t *testing.T) {
	s, _ := NewStore(limits())
	r := fixture(t)
	s.Append(r)
	r.InputDigest = digest([]byte("second input"))
	s.Append(r)
	var b bytes.Buffer
	s.WriteSnapshot(&b)
	lines := bytes.Split(b.Bytes(), []byte{'\n'})
	for _, bad := range [][]byte{bytes.Join([][]byte{lines[0], lines[1], nil}, []byte{'\n'}), bytes.Join([][]byte{lines[0], lines[2], lines[1], nil}, []byte{'\n'})} {
		if got, err := ReadSnapshot(bytes.NewReader(bad), limits()); err == nil || got != nil {
			t.Fatal("snapshot deletion/reordering undetected")
		}
	}
	empty, _ := NewStore(limits())
	if err := empty.WriteSnapshot(reentrantWriter{empty, fixture(t)}); err != nil {
		t.Fatal(err)
	}
}

func TestHistorySnapshotHeaderBudgets(t *testing.T) {
	if _, err := NewStore(Limits{MaxRecords: 1, MaxBytes: 1, MaxRecordBytes: 1}); err == nil {
		t.Fatal("empty header exceeds bound")
	}
	tinyRows := Limits{MaxRecords: 1, MaxBytes: 4096, MaxRecordBytes: 1}
	s, err := NewStore(tinyRows)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := s.WriteSnapshot(&b); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadSnapshot(&b, tinyRows); err != nil {
		t.Fatal("row bound incorrectly applied to header")
	}
	r := fixture(t)
	r.Key, _ = ObservationKey(r)
	encoded, _ := json.Marshal(r)
	header, _ := json.Marshal(snapshotHeader{Version: Version, Count: 1, RowsDigest: digest(append(append([]byte(nil), encoded...), '\n'))})
	exact := Limits{MaxRecords: 1, MaxBytes: len(header) + 1 + len(encoded) + 1, MaxRecordBytes: len(encoded)}
	s, err = NewStore(exact)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Append(r); err != nil {
		t.Fatal(err)
	}
	b.Reset()
	if err := s.WriteSnapshot(&b); err != nil || b.Len() != exact.MaxBytes {
		t.Fatalf("exact budget mismatch bytes=%d %v", b.Len(), err)
	}
	exact.MaxBytes--
	s, _ = NewStore(exact)
	if _, err := s.Append(r); err == nil {
		t.Fatal("mandatory header bytes not reserved")
	}
}

func TestCompareRefusesChangedOracleAndSamplePopulation(t *testing.T) {
	for _, kind := range []string{"oracle", "expected output", "samples"} {
		t.Run(kind, func(t *testing.T) {
			a, b := fixture(t), fixture(t)
			switch kind {
			case "oracle":
				b.Correctness.Oracle = "other oracle"
			case "expected output":
				d := digest([]byte("other expected"))
				b.Correctness.ExpectedDigest = d
				b.Correctness.ActualDigest = d
			case "samples":
				b.Facts[2].Samples = 100
			}
			p, err := Compare(a, b, policy())
			if kind == "samples" {
				if err == nil {
					t.Fatal("fabricated sample population accepted")
				}
			} else if err != nil || p.Performance != Unknown {
				t.Fatalf("changed oracle qualified: %+v %v", p, err)
			}
		})
	}
}

func TestBisectRetriesUnknownAndKeepsNonmonotonicData(t *testing.T) {
	r := BisectRequest{Candidates: []string{"a", "b", "c"}, Predicate: "correctness", MaxCandidates: 3, Observations: []Observation{{Candidate: "b", Correctness: Unknown, Performance: Unknown}}}
	p, err := PlanBisect(r)
	if err != nil || p.Next != "b" {
		t.Fatalf("unknown candidate exhausted: %+v %v", p, err)
	}
	r.Observations = []Observation{{Candidate: "a", Correctness: Fail, Performance: Unknown}, {Candidate: "c", Correctness: Pass, Performance: Unknown}}
	p, err = PlanBisect(r)
	if err != nil || p.Conflict || p.Next != "b" {
		t.Fatalf("nonmonotonic data refused without assumption: %+v %v", p, err)
	}
	r.AssumeMonotonic = true
	p, _ = PlanBisect(r)
	if !p.Conflict {
		t.Fatal("monotonic contradiction hidden")
	}
}

func FuzzDecodeRecord(f *testing.F) {
	f.Add([]byte(`{"version":1}`))
	valid, _ := json.Marshal(fixture(f))
	f.Add(valid)
	f.Fuzz(func(t *testing.T, b []byte) {
		r, err := DecodeRecord(b, 65536)
		if err == nil {
			encoded, e := json.Marshal(r)
			if e != nil {
				t.Fatal(e)
			}
			again, e := DecodeRecord(encoded, 1<<20)
			if e != nil || !reflect.DeepEqual(r, again) {
				t.Fatal("noncanonical valid record")
			}
		}
	})
}

func TestHistoryArtifactFormatAdmission(t *testing.T) {
	for _, mode := range []string{"compiler", "artifact"} {
		t.Run(mode, func(t *testing.T) {
			r := fixture(t)
			if mode == "compiler" {
				r.Compiler.OutputKind = nvrtcx.OutputCUBIN
				r.CompilerKey = cudacache.CompilerCacheKey(r.CompilerInputKey, r.Compiler)
				r.Identity.CompilerKey = r.CompilerKey
				for i := range r.Facts {
					v := r.Facts[i].Value.Record()
					v.Source.Identity = r.Identity
					r.Facts[i].Value, _ = cudaevidence.New(v)
				}
			} else {
				r.Artifact.Kind = cudakernel.Kind(255)
			}
			if err := r.Validate(); err == nil {
				t.Fatal("inconsistent image format admitted")
			}
		})
	}
}
func TestCompareWhitespaceEnvironment(t *testing.T) {
	a, b := fixture(t), fixture(t)
	b.Receipt = []byte("different observation")
	b.ReceiptDigest = digest(b.Receipt)
	a.Environment = Environment{Device: " ", Driver: "\t", CUDA: "\n", Clock: " ", Domain: " "}
	b.Environment = a.Environment
	out, err := Compare(a, b, Policy{Timing: "gpu_busy", MinSamples: 3, RelativeThreshold: 0.1, NoiseNanos: 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Performance != Unknown {
		t.Fatalf("blank environment qualified: %+v", out)
	}
}
