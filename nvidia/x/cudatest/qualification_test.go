package cudatest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"strings"
	"testing"
	"testing/fstest"
)

// Synthetic records exercise structural admission only. None is physical
// qualification evidence; real rows require the actual executing runner files.
type qualificationFixture struct {
	matrix  QualificationMatrix
	files   fstest.MapFS
	records map[string]QualificationRun
}

func qualificationTestFile(files fstest.MapFS, path string, data []byte) QualificationFile {
	files[path] = &fstest.MapFile{Data: append([]byte(nil), data...)}
	return QualificationFile{Path: path, Digest: fmt.Sprintf("%x", sha256.Sum256(data))}
}
func qualificationTestJSON(t *testing.T, value any) []byte {
	t.Helper()
	b, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func qualificationTestLog(t *testing.T, pkg, test, outcome string) []byte {
	t.Helper()
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, event := range []map[string]any{{"Action": "start", "Package": pkg}, {"Action": "run", "Package": pkg, "Test": test}, {"Action": outcome, "Package": pkg, "Test": test}, {"Action": map[string]string{"pass": "pass", "skip": "pass", "fail": "fail"}[outcome], "Package": pkg}} {
		if err := enc.Encode(event); err != nil {
			t.Fatal(err)
		}
	}
	return b.Bytes()
}
func qualificationSyntheticFixture(t *testing.T) qualificationFixture {
	t.Helper()
	f := qualificationFixture{matrix: QualificationMatrix{Version: 1, SourceRevision: strings.Repeat("a", 40), Complete: true}, files: fstest.MapFS{}, records: map[string]QualificationRun{}}
	kinds := []string{"copy", "address-space", "dsmem", "cupti-range", "cupti-pc"}
	for _, kind := range kinds {
		workload := qualificationTestFile(f.files, kind+"/workload.json", []byte(`{"fixture":"synthetic-host-only"}`))
		inputs := qualificationTestFile(f.files, kind+"/inputs.bin", []byte("synthetic actual input bytes"))
		id := QualificationIdentity{SourceRevision: f.matrix.SourceRevision, WorkloadDigest: workload.Digest, InputDigest: inputs.Digest, Toolchain: "synthetic CUDA compiler", Driver: "synthetic driver", DeviceUUID: "synthetic-host-only", ComputeCapability: "12.1"}
		sources := []QualificationFile{qualificationTestFile(f.files, kind+"/source.go", []byte("synthetic source bytes"))}
		artifacts := []QualificationFile{qualificationTestFile(f.files, kind+"/kernel.ptx", []byte("synthetic artifact bytes"))}
		policy := qualificationTestFile(f.files, kind+"/policy.json", qualificationTestJSON(t, QualificationPolicy{Attempts: 1, RunTimeoutMillis: 1000, TotalTimeoutMillis: 2000, MinimumUserSamples: 1, SampleVariation: 0.25}))
		pkg := "github.com/tmc/lib/nvidia/x/cudatest"
		name := "TestSyntheticQualification"
		facts := QualificationFacts{PositiveRuns: 1, NegativeRuns: 1, Launches: 2, CompletedLaunches: 2, Synchronizations: 1, Expected: 1003, Checked: 1003, GuardsExpected: 2, GuardsChecked: 2, ScalarRuns: 1, VectorRuns: 1, CollapsedAliasRuns: 1, AliasRefusals: 1, ExplicitSharedCast: true, ABOutputsChecked: true, MatchedKernelInspection: true, RegistersA: 16, RegistersB: 18, RankGeometryMatched: true, PostReadOrdered: true, CooperativeOracleChecked: true, LocalWindowControlDetected: true, Offered: 150, User: 100, NonUser: 20, Dropped: 20, Unnamed: 10, ReferenceUser: 100, ExpectedActivity: 3, IsolationChecked: true, DropControlOffered: 1000, DropControlDropped: 900, WorkloadGeometryMatched: true, PrintedToleranceChecked: true}
		run := QualificationRun{Version: 1, Kind: kind, Test: name, Package: pkg, Identity: id, Workload: workload, Inputs: inputs, Sources: sources, Artifacts: artifacts, Policy: policy, Log: qualificationTestFile(f.files, kind+"/go-test.json", qualificationTestLog(t, pkg, name, "pass")), Command: []string{"/home/tmc/ml-explore/bin/gpu-lock", "env", "GOWORK=off", "go", "test", "-tags=native", "-count=1", "-run", "^" + name + "$", "-json", "-timeout=1s", "./x/cudatest"}, Result: "pass", Native: true, WallMillis: 20, Facts: facts}
		run.Facts.ConversionCountA = 1
		receipt := qualificationTestFile(f.files, kind+"/run.json", qualificationTestJSON(t, run))
		f.records[kind] = run
		f.matrix.Rows = append(f.matrix.Rows, QualificationRow{Kind: kind, Status: "qualified", Identity: id, Workload: workload, Inputs: inputs, Policy: policy, Sources: sources, Artifacts: artifacts, Runs: []QualificationFile{receipt}})
	}
	return f
}
func (f *qualificationFixture) changeRun(t *testing.T, index int, mutate func(*QualificationRun)) {
	t.Helper()
	row := &f.matrix.Rows[index]
	run := f.records[row.Kind]
	mutate(&run)
	f.records[row.Kind] = run
	row.Runs[0] = qualificationTestFile(f.files, row.Kind+"/run.json", qualificationTestJSON(t, run))
}

func TestNativeQualification_RejectsSkipAsPass(t *testing.T) {
	for _, mode := range []string{"outer-skip", "nested-skip", "non-native", "no-executed-controls", "no-source", "no-artifact", "no-receipt", "pending", "fake-command"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			switch mode {
			case "outer-skip":
				f.changeRun(t, 0, func(r *QualificationRun) {
					r.Log = qualificationTestFile(f.files, "copy/go-test.json", qualificationTestLog(t, r.Package, r.Test, "skip"))
				})
			case "nested-skip":
				f.changeRun(t, 0, func(r *QualificationRun) {
					raw := qualificationTestLog(t, r.Package, r.Test, "pass")
					event := qualificationTestJSON(t, map[string]any{"Action": "skip", "Package": r.Package, "Test": r.Test + "/required-control"})
					raw = append(append(event, '\n'), raw...)
					r.Log = qualificationTestFile(f.files, "copy/go-test.json", raw)
				})
			case "non-native":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Native = false })
			case "no-executed-controls":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Facts.PositiveRuns = 0 })
			case "no-source":
				f.matrix.Rows[0].Sources = nil
			case "no-artifact":
				f.matrix.Rows[0].Artifacts = nil
			case "no-receipt":
				f.matrix.Rows[0].Runs = nil
			case "pending":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Result = "unmeasured" })
			case "fake-command":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Command[3] = "echo" })
			}
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("fabricated qualification admitted")
			}
		})
	}
}

func TestNativeQualification_ArtifactAndSourceBinding(t *testing.T) {
	for _, mode := range []string{"source-tamper", "artifact-tamper", "receipt-tamper", "log-tamper", "policy-tamper", "revision", "run-binding", "observed-identity", "missing-driver", "missing-workload", "missing-hardware", "forged-lock", "wrong-selector", "wrong-package", "missing-timeout"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			switch mode {
			case "source-tamper":
				f.files["copy/source.go"].Data = []byte("changed source")
			case "artifact-tamper":
				f.files["copy/kernel.ptx"].Data = []byte("changed artifact")
			case "receipt-tamper":
				f.files["copy/run.json"].Data = []byte("changed run")
			case "log-tamper":
				f.files["copy/go-test.json"].Data = []byte("changed log")
			case "policy-tamper":
				f.files["copy/policy.json"].Data = []byte("changed policy")
			case "revision":
				f.matrix.SourceRevision = strings.Repeat("d", 40)
			case "run-binding":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Artifacts = nil })
			case "observed-identity":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Identity.InputDigest = strings.Repeat("d", 64) })
			case "missing-driver":
				f.matrix.Rows[0].Identity.Driver = " "
			case "missing-workload":
				f.matrix.Rows[0].Identity.WorkloadDigest = ""
			case "missing-hardware":
				f.matrix.Rows[0].Identity.DeviceUUID = ""
			case "forged-lock":
				f.changeRun(t, 0, func(r *QualificationRun) {
					r.Command = append(r.Command[:3], append([]string{"GPU_LOCK_HELD=1"}, r.Command[3:]...)...)
				})
			case "wrong-selector":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Command[8] = ".*" })
			case "wrong-package":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Command[len(r.Command)-1] = "./x/cuda" })
			case "missing-timeout":
				f.changeRun(t, 0, func(r *QualificationRun) { r.Command = append(r.Command[:10], r.Command[11:]...) })
			}
			if err := ValidateQualification(f.matrix, strings.Repeat("a", 40), f.files, f.files); err == nil {
				t.Fatal("stale/missing binding admitted")
			}
		})
	}
}

func TestNativeQualification_RejectsEmptyRequiredContent(t *testing.T) {
	for _, kind := range []string{"source", "artifact", "workload", "input"} {
		t.Run(kind, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			row := &f.matrix.Rows[0]
			switch kind {
			case "source":
				row.Sources[0] = qualificationTestFile(f.files, row.Sources[0].Path, nil)
			case "artifact":
				row.Artifacts[0] = qualificationTestFile(f.files, row.Artifacts[0].Path, nil)
			case "workload":
				row.Workload = qualificationTestFile(f.files, row.Workload.Path, nil)
				row.Identity.WorkloadDigest = row.Workload.Digest
			case "input":
				row.Inputs = qualificationTestFile(f.files, row.Inputs.Path, nil)
				row.Identity.InputDigest = row.Inputs.Digest
			}
			f.changeRun(t, 0, func(r *QualificationRun) {
				r.Sources, r.Artifacts = row.Sources, row.Artifacts
				r.Workload, r.Inputs, r.Identity = row.Workload, row.Inputs, row.Identity
			})
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil || !strings.Contains(err.Error(), "empty required") {
				t.Fatalf("empty %s admission error=%v", kind, err)
			}
		})
	}
}

func TestNativeQualification_UnmeasuredSuppliedPolicy(t *testing.T) {
	for _, mode := range []string{"valid", "absent", "stale", "invalid", "empty"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			row := &f.matrix.Rows[0]
			row.Status, row.Reason, row.Runs = "unmeasured", "no native attempts", nil
			f.matrix.Complete = false
			switch mode {
			case "absent":
				row.Policy = QualificationFile{}
			case "stale":
				f.files[row.Policy.Path].Data = []byte("changed")
			case "invalid":
				row.Policy = qualificationTestFile(f.files, row.Policy.Path, qualificationTestJSON(t, QualificationPolicy{}))
			case "empty":
				row.Policy = qualificationTestFile(f.files, row.Policy.Path, nil)
			}
			err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files)
			wantError := mode != "valid" && mode != "absent"
			if (err != nil) != wantError {
				t.Fatalf("policy %s error=%v wantError=%v", mode, err, wantError)
			}
		})
	}
}

func TestNativeQualification_DomainControlsAndStatus(t *testing.T) {
	mutations := []struct {
		name  string
		index int
		fn    func(*QualificationFacts)
	}{
		{"tail-prefix", 0, func(f *QualificationFacts) { f.Checked-- }},
		{"guard", 0, func(f *QualificationFacts) { f.GuardsChecked-- }},
		{"scalar", 0, func(f *QualificationFacts) { f.ScalarRuns = 0 }},
		{"vector", 0, func(f *QualificationFacts) { f.VectorRuns = 0 }},
		{"alias", 0, func(f *QualificationFacts) { f.AliasRefusals = 0 }},
		{"collapsed-alias", 0, func(f *QualificationFacts) { f.CollapsedAliasRuns = 0 }},
		{"cast", 1, func(f *QualificationFacts) { f.ExplicitSharedCast = false }},
		{"source-only-conversion", 1, func(f *QualificationFacts) { f.MatchedKernelInspection = false }},
		{"unchecked-ab", 1, func(f *QualificationFacts) { f.ABOutputsChecked = false }},
		{"vacuous-ab", 1, func(f *QualificationFacts) { f.ConversionCountA = 0 }},
		{"conversion-remains", 1, func(f *QualificationFacts) { f.ConversionCountB = 1 }},
		{"geometry", 2, func(f *QualificationFacts) { f.RankGeometryMatched = false }},
		{"read-order", 2, func(f *QualificationFacts) { f.PostReadOrdered = false }},
		{"local-window", 2, func(f *QualificationFacts) { f.LocalWindowControlDetected = false }},
		{"oracle", 2, func(f *QualificationFacts) { f.CooperativeOracleChecked = false }},
		{"launch-without-sync", 2, func(f *QualificationFacts) { f.CompletedLaunches-- }},
	}
	for _, tc := range mutations {
		t.Run(tc.name, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			f.changeRun(t, tc.index, func(r *QualificationRun) { tc.fn(&r.Facts) })
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("missing observed control admitted")
			}
		})
	}
	f := qualificationSyntheticFixture(t)
	if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"qualified-reason", "unknown-status", "missing-reason", "false-complete", "missing-row", "duplicate-row", "budget-overrun", "attempt-reuse"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			switch mode {
			case "qualified-reason":
				f.matrix.Rows[0].Reason = "not actually run"
			case "unknown-status":
				f.matrix.Rows[0].Status = "probably pass"
			case "missing-reason":
				f.matrix.Rows[0].Status = "unmeasured"
			case "false-complete":
				f.matrix.Complete = false
			case "missing-row":
				f.matrix.Rows = f.matrix.Rows[:4]
			case "duplicate-row":
				f.matrix.Rows[1] = f.matrix.Rows[0]
			case "budget-overrun":
				f.changeRun(t, 0, func(r *QualificationRun) { r.WallMillis = 1001 })
			case "attempt-reuse":
				f.matrix.Rows[0].Runs = append(f.matrix.Rows[0].Runs, f.matrix.Rows[0].Runs[0])
			}
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("contradiction admitted")
			}
		})
	}
}

func TestCUPTIQualification_IsolationAndDropDenominators(t *testing.T) {
	for _, index := range []int{3, 4} {
		for _, mode := range []string{"zero", "dead", "dropped-hidden", "unnamed-hidden", "overflow", "isolation", "drop-control", "geometry", "printed-tolerance", "reference-zero", "variation"} {
			t.Run(fmt.Sprint(index)+"/"+mode, func(t *testing.T) {
				f := qualificationSyntheticFixture(t)
				f.changeRun(t, index, func(r *QualificationRun) {
					v := &r.Facts
					switch mode {
					case "zero":
						v.Offered = 0
					case "dead":
						v.User = 0
					case "dropped-hidden":
						v.Dropped = 900
					case "unnamed-hidden":
						v.Unnamed = 0
					case "overflow":
						v.User = math.MaxUint64
					case "isolation":
						v.IsolationChecked = false
					case "drop-control":
						v.DropControlDropped = 0
					case "geometry":
						v.WorkloadGeometryMatched = false
					case "printed-tolerance":
						v.PrintedToleranceChecked = false
					case "reference-zero":
						v.ReferenceUser = 0
						if index == 3 {
							v.ExpectedActivity = 0
						}
					case "variation":
						v.User = 200
						v.Offered = 250
						if index == 3 {
							v.PrintedToleranceChecked = false
						}
					}
				})
				if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
					t.Fatal("dead/partial CUPTI collector admitted")
				}
			})
		}
	}
	if qualificationSampleVariation(math.MaxUint64, math.MaxUint64-1, 0) {
		t.Fatal("rounded uint64 sample delta disappeared")
	}
	if !qualificationSampleVariation(125, 100, 0.25) || qualificationSampleVariation(126, 100, 0.25) || !qualificationSampleVariation(75, 100, 0.25) || qualificationSampleVariation(74, 100, 0.25) {
		t.Fatal("variation boundary incorrect")
	}
}

func TestNativeQualification_PreservesPartialAndFailure(t *testing.T) {
	f := qualificationSyntheticFixture(t)
	f.matrix.Rows[0] = QualificationRow{Kind: "copy", Status: "unmeasured", Reason: "actual copy alias controls not executed"}
	f.matrix.Complete = false
	if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err != nil {
		t.Fatal(err)
	}
	for _, result := range []string{"failed", "invalid-config", "harness-error", "unsupported"} {
		t.Run(result, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			row := &f.matrix.Rows[0]
			row.Status = result
			row.Reason = "actual observed diagnostic"
			f.matrix.Complete = result == "unsupported"
			f.changeRun(t, 0, func(r *QualificationRun) {
				r.Result = map[string]string{"failed": "fail", "invalid-config": "invalid-config", "harness-error": "harness-error", "unsupported": "unsupported"}[result]
				r.Reason = "actual observed diagnostic"
				outcome := "fail"
				r.ExitCode = 1
				if result == "unsupported" {
					outcome = "skip"
					r.ExitCode = 0
				}
				r.Log = qualificationTestFile(f.files, "copy/go-test.json", qualificationTestLog(t, r.Package, r.Test, outcome))
			})
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err != nil {
				t.Fatal("honest nonqualified evidence dropped", err)
			}
		})
	}
}

func TestNativeQualification_StrictBoundedJSON(t *testing.T) {
	for _, s := range []string{`null`, `[]`, `{"version":1,"version":1}`, `{"Version":1}`, `{"verſion":1}`, `{"unknown":1}`, `{} {}`} {
		if _, err := ReadQualification(strings.NewReader(s)); err == nil {
			t.Fatalf("invalid schema accepted: %s", s)
		}
	}
	if _, err := ReadQualification(strings.NewReader(strings.Repeat(" ", qualificationJSONLimit+1))); err == nil {
		t.Fatal("overbyte matrix accepted")
	}
	if _, err := ReadQualification(nil); err == nil {
		t.Fatal("nil reader accepted")
	}
	f := qualificationSyntheticFixture(t)
	encoded := qualificationTestJSON(t, f.matrix)
	m, err := ReadQualification(bytes.NewReader(encoded))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateQualification(m, m.SourceRevision, f.files, f.files); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"../outside", "/absolute", "."} {
		f := qualificationSyntheticFixture(t)
		f.matrix.Rows[0].Sources[0].Path = path
		if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
			t.Fatal("invalid retained path accepted")
		}
	}
	var _ fs.FS = f.files
}

func TestNativeQualification_CurrentSourceNotArchivedCopy(t *testing.T) {
	f := qualificationSyntheticFixture(t)
	sources := fstest.MapFS{}
	for path, file := range f.files {
		sources[path] = &fstest.MapFile{Data: append([]byte(nil), file.Data...)}
	}
	sources["copy/source.go"].Data = []byte("actual executing checkout changed")
	if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, sources, f.files); err == nil {
		t.Fatal("archived receipt source copy replaced current source")
	}
}

func TestNativeQualification_InputWorkloadBytesAndUnknownIdentity(t *testing.T) {
	for _, mode := range []string{"inputs-tamper", "workload-tamper", "unbound-inputs", "unbound-workload", "unknown-driver", "unknown-toolchain", "unknown-hardware", "invalid-partial-digest"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			switch mode {
			case "inputs-tamper":
				f.files["copy/inputs.bin"].Data = []byte("different input")
			case "workload-tamper":
				f.files["copy/workload.json"].Data = []byte("different workload")
			case "unbound-inputs":
				f.matrix.Rows[0].Inputs = QualificationFile{}
			case "unbound-workload":
				f.matrix.Rows[0].Workload = QualificationFile{}
			case "unknown-driver":
				f.matrix.Rows[0].Identity.Driver = "unknown"
			case "unknown-toolchain":
				f.matrix.Rows[0].Identity.Toolchain = "unmeasured"
			case "unknown-hardware":
				f.matrix.Rows[0].Identity.DeviceUUID = "unread"
			case "invalid-partial-digest":
				f.matrix.Rows[0] = QualificationRow{Kind: "copy", Status: "unmeasured", Reason: "actual controls absent", Identity: QualificationIdentity{InputDigest: "unknown"}}
				f.matrix.Complete = false
			}
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("unbound or unknown actual identity admitted")
			}
		})
	}
}

func TestNativeQualification_RawOutcomeContradictionsAndFailedAttempts(t *testing.T) {
	for _, mode := range []string{"package-failed", "duplicate-package", "unexpected-test", "unexpected-package", "duplicate-action", "case-alias", "unterminated"} {
		t.Run(mode, func(t *testing.T) {
			f := qualificationSyntheticFixture(t)
			f.changeRun(t, 0, func(r *QualificationRun) {
				raw := qualificationTestLog(t, r.Package, r.Test, "pass")
				switch mode {
				case "package-failed":
					raw = bytes.Replace(raw, []byte(`{"Action":"pass","Package":"`+r.Package+`"}`), []byte(`{"Action":"fail","Package":"`+r.Package+`"}`), 1)
				case "duplicate-package":
					raw = append(raw, qualificationTestJSON(t, map[string]string{"Action": "pass", "Package": r.Package})...)
				case "unexpected-test":
					raw = append(raw, qualificationTestJSON(t, map[string]string{"Action": "skip", "Package": r.Package, "Test": "TestOther"})...)
				case "unexpected-package":
					raw = append(raw, qualificationTestJSON(t, map[string]string{"Action": "skip", "Package": "other/package", "Test": r.Test})...)
				case "duplicate-action":
					raw = append(raw, []byte(`{"Action":"skip","Action":"pass","Package":"`+r.Package+`"}`)...)
				case "case-alias":
					raw = append(raw, []byte(`{"action":"skip","Action":"pass","Package":"`+r.Package+`"}`)...)
				case "unterminated":
					raw = raw[:bytes.LastIndexByte(raw[:len(raw)-1], '\n')]
				}
				r.Log = qualificationTestFile(f.files, "copy/go-test.json", raw)
			})
			if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err == nil {
				t.Fatal("raw outcome contradiction admitted")
			}
		})
	}
	f := qualificationSyntheticFixture(t)
	f.matrix.Complete = false
	f.matrix.Rows[0].Status = "harness-error"
	f.matrix.Rows[0].Reason = "compiler failed before named test began"
	f.changeRun(t, 0, func(r *QualificationRun) {
		r.Result = "harness-error"
		r.Native = false
		r.ExitCode = 1
		r.Reason = "compiler failed before named test began"
		r.Log = qualificationTestFile(f.files, "copy/go-test.json", qualificationTestJSON(t, map[string]string{"Action": "fail", "Package": r.Package}))
	})
	if err := ValidateQualification(f.matrix, f.matrix.SourceRevision, f.files, f.files); err != nil {
		t.Fatal("honest pre-test harness failure lost", err)
	}
}

func FuzzReadQualification(f *testing.F) {
	f.Add(`{"version":1,"source_revision":"","complete":false,"rows":[]}`)
	f.Add(`null`)
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > qualificationJSONLimit+1 {
			return
		}
		m, err := ReadQualification(strings.NewReader(s))
		if err == nil {
			_ = ValidateQualification(m, strings.Repeat("a", 40), fstest.MapFS{}, fstest.MapFS{})
		}
	})
}
