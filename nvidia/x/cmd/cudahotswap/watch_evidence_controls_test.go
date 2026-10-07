package main

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// watchEvidenceCPUFixture is synthetic schema input, never a native result.
func watchEvidenceCPUFixture() watchEvidence {
	var e watchEvidence
	e.Version = 1
	e.WatchPath = "/synthetic/watch.cu"
	e.Process = nativereceipt.Process{PID: 42, Parent: 41, Group: 41, Start: 1, Executable: "/synthetic/prepared.test", ExecutableSHA256: watchDigest([]byte("synthetic executable"))}
	baselineCode := "extern \"C\" __global__ void replay(float *x, float *y, const float *w, unsigned n) { unsigned i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) y[i]=x[i]+w[i]; }"
	for i := range 4 {
		code := baselineCode
		if i > 0 {
			code = nativeWatchSource(i - 1)
		}
		row := watchCompilation{Source: cudahotswap.Source{Code: code, Filename: e.WatchPath, Entry: "replay"}, SourceSHA256: watchDigest([]byte(code)), PTXSHA256: watchDigest([]byte("synthetic PTX")), CUBINSHA256: watchDigest([]byte("synthetic CUBIN")), Key: code, PTXBytes: 13, CUBINBytes: 15}
		e.Compilations = append(e.Compilations, row)
	}
	e.Baseline = cudahotswap.State{Name: "kernel", Kind: "kernel", Generation: 1, Version: 1, ConfigRevision: 1, Label: "replay check", SourceDigest: e.Compilations[0].SourceSHA256, CompilerKey: e.Compilations[0].Key}
	for i, c := range []int{0, 1, 3} {
		e.Loads = append(e.Loads, watchLoad{Compilation: c, Function: uint64(100 + i), Context: 3})
	}
	selected := e.Baseline
	selected.ConfigRevision++
	selected.Label = "local watch"
	for i := range 3 {
		c := e.Compilations[i+1]
		state := selected
		state.Label = "watch: " + e.WatchPath
		if i == 2 {
			state.SourceDigest = c.SourceSHA256
			state.CompilerKey = c.Key
			state.Generation++
			state.Version++
			selected = state
		}
		report := cudahotswap.Report{Outcome: []cudahotswap.Outcome{cudahotswap.OutcomeWrongAnswer, cudahotswap.OutcomeABIMismatch, cudahotswap.OutcomeOK}[i], Generation: state.Generation, ConfigRevision: state.ConfigRevision, Compile: &cudahotswap.CompileReport{SourceSHA256: c.SourceSHA256, SourceBytes: len(c.Source.Code), PTXSHA256: c.PTXSHA256, PTXBytes: c.PTXBytes}}
		if i != 1 {
			want, got := watchExpectedOutputs(i)
			metric, _ := cudahotswap.MeasureReplay(got, want)
			noise, _ := cudahotswap.MeasureReplay(want, want)
			sign := float64(0.0625)
			replay := &cudahotswap.ReplayReport{Passed: i == 2, Native: true, Clock: "cuda_event_interval", Outputs: []cudahotswap.ReplayOutputReport{{Arg: 1, Passed: i == 2, Candidate: metric, Noise: noise, WriteOnly: true}}, CandidateWins: 5, SignP: &sign}
			for round := 1; round <= 5; round++ {
				arms := []string{"baseline", "candidate"}
				if round%2 == 0 {
					arms[0], arms[1] = arms[1], arms[0]
				}
				for _, arm := range arms {
					ns, digest := int64(20), watchOutputDigest(want)
					if arm == "candidate" {
						ns, digest = 10, watchOutputDigest(got)
					}
					replay.Trials = append(replay.Trials, cudahotswap.ReplayTrialReport{Arm: arm, Round: round, Complete: true, Digest: digest, DeviceNanos: &ns})
				}
			}
			report.Replay = replay
		} else {
			report.Compile.PTXSHA256 = ""
			report.Compile.PTXBytes = 0
		}
		event := watchEvent{Kernel: "kernel", Result: &cudahotswap.PushResult{State: state, Selected: selected, Published: i == 2, Report: report}}
		if i < 2 {
			event.Error = "synthetic refusal"
		}
		e.Events = append(e.Events, watchObservedEvent{Event: event, Selected: selected})
	}
	e.Cleanup.Attempted = true
	e.Cleanup.Complete = true
	e.Cleanup.ClosedService = true
	e.Cleanup.Attempts = []string{""}
	return e
}
func cloneWatchEvidence(t *testing.T, e watchEvidence) watchEvidence {
	t.Helper()
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var copy watchEvidence
	if err := json.Unmarshal(data, &copy); err != nil {
		t.Fatal(err)
	}
	return copy
}

func TestWatchNativeEvidenceCPU(t *testing.T) {
	baseline := watchEvidenceCPUFixture()
	if err := validateWatchEvidence(baseline); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		change func(*watchEvidence)
	}{
		{"missing_trials", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials = nil }},
		{"short_trials", func(e *watchEvidence) { r := e.Events[2].Event.Result.Report.Replay; r.Trials = r.Trials[:9] }},
		{"duplicate_round", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials[2].Round = 1 }},
		{"wrong_arm", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials[0].Arm = "candidate" }},
		{"trial_incomplete", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials[0].Complete = false }},
		{"trial_error", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials[0].Error = "failed" }},
		{"missing_device_time", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Trials[0].DeviceNanos = nil }},
		{"zero_device_time", func(e *watchEvidence) { *e.Events[2].Event.Result.Report.Replay.Trials[0].DeviceNanos = 0 }},
		{"trial_digest", func(e *watchEvidence) {
			e.Events[0].Event.Result.Report.Replay.Trials[1].Digest = watchDigest([]byte("wrong"))
		}},
		{"wrong_abs", func(e *watchEvidence) { *e.Events[0].Event.Result.Report.Replay.Outputs[0].Candidate.MaxAbs = 99 }},
		{"wrong_rms", func(e *watchEvidence) { *e.Events[0].Event.Result.Report.Replay.Outputs[0].Candidate.RMS = 99 }},
		{"wrong_ulp", func(e *watchEvidence) { *e.Events[0].Event.Result.Report.Replay.Outputs[0].Candidate.MaxULP = 1 }},
		{"noise", func(e *watchEvidence) { *e.Events[2].Event.Result.Report.Replay.Outputs[0].Noise.MaxAbs = 1 }},
		{"write_only", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Outputs[0].WriteOnly = false }},
		{"wins", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.CandidateWins-- }},
		{"baseline_wins", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.BaselineWins++ }},
		{"missing_sign", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.SignP = nil }},
		{"wrong_sign", func(e *watchEvidence) { *e.Events[2].Event.Result.Report.Replay.SignP = 1 }},
		{"missing_compile", func(e *watchEvidence) { e.Events[1].Event.Result.Report.Compile = nil }},
		{"abi_report_ptx", func(e *watchEvidence) {
			e.Events[1].Event.Result.Report.Compile.PTXSHA256 = e.Compilations[2].PTXSHA256
		}},
		{"abi_report_ptx_extent", func(e *watchEvidence) { e.Events[1].Event.Result.Report.Compile.PTXBytes = e.Compilations[2].PTXBytes }},
		{"abi_artifact_ptx", func(e *watchEvidence) { e.Compilations[2].PTXSHA256 = "" }},
		{"abi_artifact_ptx_extent", func(e *watchEvidence) { e.Compilations[2].PTXBytes = 0 }},
		{"abi_report_source", func(e *watchEvidence) {
			e.Events[1].Event.Result.Report.Compile.SourceSHA256 = watchDigest([]byte("other"))
		}},
		{"abi_report_source_extent", func(e *watchEvidence) { e.Events[1].Event.Result.Report.Compile.SourceBytes++ }},
		{"wrong_answer_report_ptx_missing", func(e *watchEvidence) { e.Events[0].Event.Result.Report.Compile.PTXSHA256 = "" }},
		{"correct_report_ptx_extent", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Compile.PTXBytes++ }},
		{"compile_source", func(e *watchEvidence) {
			e.Events[0].Event.Result.Report.Compile.SourceSHA256 = watchDigest([]byte("other"))
		}},
		{"compile_ptx", func(e *watchEvidence) {
			e.Events[2].Event.Result.Report.Compile.PTXSHA256 = watchDigest([]byte("other"))
		}},
		{"source_extent", func(e *watchEvidence) { e.Events[0].Event.Result.Report.Compile.SourceBytes++ }},
		{"report_generation", func(e *watchEvidence) { e.Events[0].Event.Result.Report.Generation++ }},
		{"report_revision", func(e *watchEvidence) { e.Events[0].Event.Result.Report.ConfigRevision++ }},
		{"wrong_state", func(e *watchEvidence) { e.Events[0].Event.Result.State.Version++ }},
		{"wrong_selected", func(e *watchEvidence) { e.Events[0].Event.Result.Selected.Version++ }},
		{"observed_status", func(e *watchEvidence) { e.Events[2].Selected.Generation++ }},
		{"repair_source", func(e *watchEvidence) { e.Events[2].Event.Result.State.SourceDigest = watchDigest([]byte("other")) }},
		{"wrong_publication", func(e *watchEvidence) { e.Events[0].Event.Result.Published = true }},
		{"missing_replay", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay = nil }},
		{"host_clock", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Native = false }},
		{"wrong_replay", func(e *watchEvidence) { e.Events[2].Event.Result.Report.Replay.Outputs[0].Candidate.Exact = false }},
		{"abi_replay", func(e *watchEvidence) {
			e.Events[1].Event.Result.Report.Replay = e.Events[2].Event.Result.Report.Replay
		}},
		{"compilation_body", func(e *watchEvidence) { e.Compilations[1].Source.Code = "other" }},
		{"compile_flags", func(e *watchEvidence) { e.Compilations[1].Source.Flags = []string{"--use_fast_math"} }},
		{"missing_image", func(e *watchEvidence) { e.Compilations[3].CUBINSHA256 = "" }},
		{"load_binding", func(e *watchEvidence) { e.Loads[2].Compilation = 2 }},
		{"missing_function", func(e *watchEvidence) { e.Loads[1].Function = 0 }},
		{"wrong_context", func(e *watchEvidence) { e.Loads[1].Context++ }},
		{"failed_load", func(e *watchEvidence) { e.Loads[1].LoadError = "partial" }},
		{"service_not_closed", func(e *watchEvidence) { e.Cleanup.ClosedService = false }},
		{"cleanup_attempt_error", func(e *watchEvidence) { e.Cleanup.Attempts[0] = "failure" }},
		{"missing_cleanup", func(e *watchEvidence) { e.Cleanup.Attempted = false }},
		{"failed_cleanup", func(e *watchEvidence) { e.Cleanup.Complete = false }},
		{"cleanup_error", func(e *watchEvidence) { e.Cleanup.Error = "failed" }},
		{"missing_pid", func(e *watchEvidence) { e.Process.PID = 0 }},
		{"qualification", func(e *watchEvidence) { e.Qualification = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := cloneWatchEvidence(t, baseline)
			tc.change(&e)
			if err := validateWatchEvidence(e); err == nil {
				t.Fatal("contradictory receipt accepted")
			}
		})
	}
}
func TestWatchSessionMissingLoaderFactory(t *testing.T) {
	service, close, err := newSessionWithLoader(context.Background(), config{}, nil)
	if err == nil || service != nil || close != nil {
		t.Fatal("missing factory not refused before device setup")
	}
}

func TestWatchSourceArithmeticOracle(t *testing.T) {
	want, got := watchExpectedOutputs(0)
	if len(want) != 32 || len(got) != 32 || want[0] != 4 || want[31] != 128 || got[0] != -2 || got[31] != -64 {
		t.Fatal("actual initialized addition/subtraction source oracle changed")
	}
	m, err := cudahotswap.MeasureReplay(got, want)
	if err != nil {
		t.Fatal(err)
	}
	// sum(i*i), i=1..32 is 11440; error is 6*i.
	if m.Exact || m.NonFinite != 0 || *m.MaxAbs != 192 || *m.RMS != math.Sqrt(12870) || *m.MaxULP != 0x85800000 {
		t.Fatalf("literal source arithmetic metrics differ: %+v", m)
	}
}
