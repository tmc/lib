package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func cliCapturePlan() cudaprof.CapturePlan {
	d := strings.Repeat("1", 64)
	return cudaprof.CapturePlan{Version: 1, Identity: cudaevidence.Identity{ArtifactDigest: d, Entry: "kernel", Target: "sm_90", CompilerKey: d, WorkloadKey: d}, Environment: cudaprof.CaptureEnvironment{Device: "device", Driver: "driver", CUDA: "cuda", CUPTI: "cupti", ClockMode: "gpu", CorrelationDomain: "domain"}, Budget: cudaprof.CaptureBudget{MaxRecords: 100, MaxBytes: 65536, MaxPasses: 8, MaxWallNanos: int64(8 * time.Second)}, Passes: []cudaprof.CapturePass{{ID: "activity", Replay: "baseline", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity}}}}
}

func TestCapturePlanDefaultRouteUnchanged(t *testing.T) {
	opts := captureOptions{Trace: true, PCSampling: true, Timeout: 13 * time.Second, command: []string{"target", "--arg"}}
	want := captureResult{Session: "original", CaptureSuccess: true, ActivityRecords: 9}
	sentinel := errors.New("original error")
	calls := 0
	got, err := runCaptureWithPlan(context.Background(), opts, func(ctx context.Context, o captureOptions) (captureResult, error) {
		calls++
		if !reflect.DeepEqual(o, opts) {
			t.Errorf("options changed: %+v", o)
		}
		return want, sentinel
	})
	if calls != 1 || !reflect.DeepEqual(got, want) || err != sentinel {
		t.Fatalf("calls=%d result=%+v err=%v", calls, got, err)
	}
}

func TestCapturePlanHelp(t *testing.T) {
	text := captureHelp()
	for _, want := range []string{"--capture-plan <path>", "--capture-plan-cooperative", "separate replay identities", "Missing or incomplete providers remain so"} {
		if !strings.Contains(text, want) {
			t.Errorf("capture help missing %q", want)
		}
	}
}

func TestCapturePlanCooperativeRemainingBudgets(t *testing.T) {
	plan := cliCapturePlan()
	plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "second", Replay: "replay", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity}})
	calls := 0
	_, err := executeCapturePlan(context.Background(), captureOptions{CapturePlanCooperative: true}, plan, func(ctx context.Context, o captureOptions) (captureResult, error) {
		wantRecords := plan.Budget.MaxRecords - uint64(calls)
		if got := lookupEnv(o.Env, "CUDAPROF_CAPTURE_MAX_RECORDS"); got != strconv.FormatUint(wantRecords, 10) {
			t.Fatalf("remaining records=%s, want %d", got, wantRecords)
		}
		if got := lookupEnv(o.Env, "CUDAPROF_CAPTURE_MAX_PASSES"); got != "7" {
			t.Fatalf("native reservation=%s, want 7", got)
		}
		deadline, err := strconv.ParseInt(lookupEnv(o.Env, "CUDAPROF_CAPTURE_DEADLINE_UNIX_NANOS"), 10, 64)
		if err != nil || deadline <= time.Now().UnixNano() {
			t.Fatalf("invalid target deadline=%d: %v", deadline, err)
		}
		maxBytes, err := strconv.Atoi(lookupEnv(o.Env, "CUDAPROF_CAPTURE_MAX_BYTES"))
		if err != nil || maxBytes != plan.Budget.MaxBytes {
			t.Fatalf("remaining bytes=%d, want %d: %v", maxBytes, plan.Budget.MaxBytes, err)
		}
		calls++
		return captureResult{Session: "observed", TargetPID: 3, ActivityRecords: 1}, nil
	})
	if err != nil || calls != 2 {
		t.Fatalf("calls=%d: %v", calls, err)
	}
}

func TestCapturePlanRejectsBeforeTarget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, []byte(`{"version":1,"unknown":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	calls := 0
	_, err := runCaptureWithPlan(context.Background(), captureOptions{CapturePlanFile: path}, func(context.Context, captureOptions) (captureResult, error) { calls++; return captureResult{}, nil })
	if err == nil || calls != 0 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
	opts, err := parseCaptureArgs([]string{"--capture-plan", path, "--no-inject", "--", "target"})
	if err != nil || opts.CapturePlanFile != path {
		t.Fatalf("opts=%+v err=%v", opts, err)
	}
}

func TestCapturePlanSerialReplayAndMissingProvider(t *testing.T) {
	plan := cliCapturePlan()
	plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "pc", Replay: "replay1", Providers: []cudaprof.CaptureProvider{cudaprof.CapturePC}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CapturePC: {Period: 9}}}, cudaprof.CapturePass{ID: "range", Replay: "replay2", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureRange}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureRange: {Metrics: []string{"metric"}, ConfigDigest: strings.Repeat("2", 64)}}})
	calls := 0
	raws := [][]byte{[]byte("first raw journal\x00"), []byte("second raw journal\xff")}
	got, err := executeCapturePlan(context.Background(), captureOptions{}, plan, func(ctx context.Context, o captureOptions) (captureResult, error) {
		if _, ok := ctx.Deadline(); !ok {
			t.Fatal("missing global deadline")
		}
		if calls == 0 && (!o.Trace || o.PCSampling) {
			t.Fatal("wrong activity options")
		}
		if calls == 1 && (o.Trace || !o.PCSampling || o.PCSamplingPeriod != 9) {
			t.Fatalf("wrong PC options: %+v", o)
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "capture.journal"), raws[calls], 0600); err != nil {
			t.Fatal(err)
		}
		calls++
		return captureResult{Session: []string{"first", "second"}[calls-1], ScratchDir: dir, TargetPID: 3, ActivityRecords: 2, DroppedRecords: 1, RecordLoss: &captureLoss{}, PCSampling: &capturePCSampling{Folded: 3, Total: 5, HardwareLost: 2}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || got.Session != "first" {
		t.Fatalf("calls=%d primary=%s", calls, got.Session)
	}
	m, err := cudaprof.ReadCaptureManifest(strings.NewReader(string(got.CapturePlanManifest)), plan.Budget.MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if m.Complete() || len(m.Results) != 3 {
		t.Fatalf("overclaimed manifest=%+v", m)
	}
	if m.Results[0].Replay != "baseline" || m.Results[1].Replay != "replay1" || m.Results[2].Replay != "replay2" {
		t.Fatal("lost replay identity")
	}
	if m.Results[2].Status != "unavailable" {
		t.Fatal("fabricated range")
	}
	if m.Results[0].Identity.ArtifactDigest != "" || m.Results[0].Environment.Device != "" {
		t.Fatal("copied expected identity into observed")
	}
	if m.Results[0].Counts.Dropped == nil || *m.Results[0].Counts.Dropped != 1 {
		t.Fatal("lost activity drop")
	}
	if m.Results[1].Counts.Retained == nil || *m.Results[1].Counts.Retained != 3 {
		t.Fatal("PC denominator used distinct PC count")
	}
	for i := 0; i < 2; i++ {
		var files []capturePlanRawFile
		if err := json.Unmarshal(m.Results[i].Raw, &files); err != nil {
			t.Fatal(err)
		}
		if len(files) != 1 || string(files[0].Data) != string(raws[i]) {
			t.Fatal("raw journal changed")
		}
	}
	compact := compactCaptureStatus(got)
	if len(compact.CapturePlanManifest) == 0 {
		t.Fatal("compact status dropped manifest")
	}
}

func TestCapturePlanDeadlineStopsLaterPasses(t *testing.T) {
	plan := cliCapturePlan()
	plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "later", Replay: "later", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity}})
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	got, err := executeCapturePlan(ctx, captureOptions{}, plan, func(context.Context, captureOptions) (captureResult, error) {
		calls++
		cancel()
		return captureResult{Session: "first"}, nil
	})
	if calls != 1 || err == nil || got.CapturePlanError == "" || got.CaptureSuccess {
		t.Fatalf("calls=%d result=%+v err=%v", calls, got, err)
	}
}

func TestCapturePlanBudgetDiagnostics(t *testing.T) {
	for _, kind := range []string{"records", "bytes"} {
		t.Run(kind, func(t *testing.T) {
			plan := cliCapturePlan()
			if kind == "records" {
				plan.Budget.MaxRecords = 1
			} else {
				plan.Budget.MaxBytes = 1
			}
			got, err := executeCapturePlan(context.Background(), captureOptions{}, plan, func(context.Context, captureOptions) (captureResult, error) {
				return captureResult{Session: "run", ActivityRecords: 2}, nil
			})
			if err == nil || got.CapturePlanError == "" || got.CaptureSuccess {
				t.Fatalf("result=%+v err=%v", got, err)
			}
		})
	}
}

func TestCapturePlanOuterOutputBoundBeforePublish(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "capture.pprof")
	budget := cliCapturePlan().Budget
	budget.MaxBytes = 1
	_, err := writeCapture(path, captureResult{Session: "bounded", planBudget: &budget}, captureArtifactBudget{FileBytes: 1, TotalBytes: 1})
	if err == nil {
		t.Fatal("published oversized outer capture")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("output exists: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(dir, ".cudaprof-status-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("temporary output retained: %v", matches)
	}
	// The legacy writer continues to admit mandatory metadata above its raw
	// attachment budget; the hard encoded limit is opt-in only.
	if _, err := writeCapture(path, captureResult{Session: "legacy"}, captureArtifactBudget{FileBytes: 1, TotalBytes: 1}); err != nil {
		t.Fatal(err)
	}
}

func TestCapturePlanExhaustionHasNoFabricatedLaterCounters(t *testing.T) {
	plan := cliCapturePlan()
	plan.Budget.MaxRecords = 1
	plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "later", Replay: "later", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity}})
	calls := 0
	got, err := executeCapturePlan(context.Background(), captureOptions{}, plan, func(context.Context, captureOptions) (captureResult, error) {
		calls++
		return captureResult{Session: "first", ActivityRecords: 1}, nil
	})
	if err == nil || calls != 1 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
	m, err := cudaprof.ReadCaptureManifest(strings.NewReader(string(got.CapturePlanManifest)), plan.Budget.MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if m.Results[1].Status != "failed" || m.Results[1].Counts.Retained != nil {
		t.Fatalf("unexecuted counters=%+v", m.Results[1])
	}
}

func TestCapturePlanOverWallPreservesAllProviderReasons(t *testing.T) {
	plan := cliCapturePlan()
	plan.Budget.MaxWallNanos = 1
	plan.Passes[0].Providers = []cudaprof.CaptureProvider{cudaprof.CapturePC}
	plan.Passes[0].Configs = map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CapturePC: {Period: 9}}
	plan.Passes = append(plan.Passes, cudaprof.CapturePass{ID: "range", Replay: "later", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureRange}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureRange: {Metrics: []string{"metric"}, ConfigDigest: strings.Repeat("2", 64)}}})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err := executeCapturePlan(ctx, captureOptions{}, plan, func(context.Context, captureOptions) (captureResult, error) {
		t.Fatal("expired context launched target")
		return captureResult{}, nil
	})
	if err == nil {
		t.Fatal("lost wall exhaustion")
	}
	m, err := cudaprof.ReadCaptureManifest(bytes.NewReader(got.CapturePlanManifest), plan.Budget.MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if m.ElapsedNanos <= plan.Budget.MaxWallNanos || m.Complete() {
		t.Fatalf("elapsed/completeness=%+v", m)
	}
	for _, r := range m.Results {
		if !strings.Contains(r.Reason, "budget") {
			t.Fatalf("lost wall reason %+v", r)
		}
	}
	if !strings.Contains(m.Results[1].Reason, "target-cooperative") {
		t.Fatal("lost unavailable cause")
	}
}

// This is a synthetic host publication, not native capture or loaded-image proof.
func cliPublicationFixture(t *testing.T) (cudaprof.CapturePlan, cudaprof.CapturePublication) {
	t.Helper()
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_70\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_70")
	if err != nil {
		t.Fatal(err)
	}
	workload := cudacache.WorkloadIdentity{ProgramID: "synthetic", ShapeDims: []int{64}, NumericalMode: "fp32"}
	id, err := cudaprof.BindEvidence(a, workload, strings.Repeat("2", 64))
	if err != nil {
		t.Fatal(err)
	}
	plan := cliCapturePlan()
	plan.Identity = id
	config := cudaprof.CaptureProviderConfig{Metrics: []string{"metric"}, ConfigDigest: strings.Repeat("3", 64)}
	plan.Passes = []cudaprof.CapturePass{{ID: "range", Replay: "replay", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureRange}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureRange: config}}}
	one, zero := uint64(1), uint64(0)
	result := cudaprof.CaptureProviderResult{Provider: cudaprof.CaptureRange, PassID: "range", Replay: "replay", Config: config, NativePasses: &cudaprof.CaptureNativePasses{Started: 1, Stopped: 1, AllSubmitted: true}, Scope: "target-context", Process: 3, Generation: 1, Context: "ctx", Identity: id, Environment: plan.Environment, Start: 1, Stop: 2, Status: "complete", EvidenceClass: "derived", Counts: cudaprof.CaptureCounts{Retained: &one, Offered: &one, Dropped: &zero, NonUser: &zero, Unnamed: &zero}, Raw: []byte{0, 255, 7}, Decoded: true, CleanupComplete: true}
	return plan, cudaprof.CapturePublication{Version: 1, Result: result, Artifact: a, Workload: workload, CompilerKey: id.CompilerKey}
}

func TestCapturePlanCooperativePublicationRemainsSourceBoundPartial(t *testing.T) {
	for _, kind := range []string{"valid", "corrupt artifact", "missing", "wrong process"} {
		t.Run(kind, func(t *testing.T) {
			plan, publication := cliPublicationFixture(t)
			var original bytes.Buffer
			if err := cudaprof.WriteCapturePublication(&original, plan, publication); err != nil {
				t.Fatal(err)
			}
			if kind == "corrupt artifact" {
				publication.Artifact.Raw[0] ^= 1
				data, err := json.Marshal(publication)
				if err != nil {
					t.Fatal(err)
				}
				original.Reset()
				original.Write(data)
			}
			if kind == "wrong process" {
				publication.Result.Process = 4
				data, err := json.Marshal(publication)
				if err != nil {
					t.Fatal(err)
				}
				original.Reset()
				original.Write(data)
			}
			calls := 0
			planPath := filepath.Join(t.TempDir(), "plan.json")
			got, err := executeCapturePlan(context.Background(), captureOptions{CapturePlanFile: planPath, CapturePlanCooperative: true}, plan, func(ctx context.Context, o captureOptions) (captureResult, error) {
				calls++
				if lookupEnv(o.Env, "CUDAPROF_CAPTURE_PLAN") != planPath || lookupEnv(o.Env, "CUDAPROF_CAPTURE_PASS") != "range" {
					t.Fatal("cooperative env lost")
				}
				dir := t.TempDir()
				if kind != "missing" {
					if err := os.WriteFile(filepath.Join(dir, "capture-provider-range.json"), original.Bytes(), 0600); err != nil {
						t.Fatal(err)
					}
				}
				return captureResult{Session: "cooperative", TargetPID: 3, ScratchDir: dir}, nil
			})
			if calls != 1 {
				t.Fatalf("calls=%d", calls)
			}
			if (kind == "corrupt artifact" || kind == "wrong process") && err == nil {
				t.Fatal("invalid publication accepted")
			}
			if kind == "valid" || kind == "missing" {
				if err != nil {
					t.Fatal(err)
				}
			}
			m, readErr := cudaprof.ReadCaptureManifest(bytes.NewReader(got.CapturePlanManifest), plan.Budget.MaxBytes)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if m.Complete() {
				t.Fatal("publication substituted for loaded-module proof")
			}
			if kind == "valid" {
				r := m.Results[0]
				if r.Status != "partial" || r.Identity != publication.Result.Identity || !bytes.Equal(cliPublicationBytes(t, r.Raw), original.Bytes()) {
					t.Fatal("publication provenance/raw changed")
				}
			}
			if kind == "missing" && m.Results[0].Status != "unavailable" {
				t.Fatal("missing publication fabricated")
			}
			if kind == "corrupt artifact" || kind == "wrong process" {
				if !bytes.Equal(cliPublicationBytes(t, m.Results[0].Raw), original.Bytes()) || m.Results[0].Status != "failed" {
					t.Fatal("failed publication raw diagnostics lost")
				}
			}
		})
	}
}

func TestCapturePlanPartialPublicationRetainsCode(t *testing.T) {
	plan, p := cliPublicationFixture(t)
	code, real := planCodeFixture(t)
	p.Artifact, p.Workload, p.CompilerKey = real.Artifact, real.Workload, real.CompilerKey
	p.Result.Identity, plan.Identity = real.Result.Identity, real.Result.Identity
	p.Result.Status, p.Result.Reason = "partial", "owner loss denominator unavailable"
	var publication bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&publication, plan, p); err != nil {
		t.Fatal(err)
	}
	packet := planCodePacket(t, code)
	dir := planCodeWrite(t, packet)
	path := filepath.Join(dir, "capture-provider-range.json")
	if err := os.WriteFile(path, publication.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	r, found, err := capturePlanPublication(captureResult{ScratchDir: dir, TargetPID: 3}, plan, plan.Passes[0], cudaprof.CaptureRange, plan.Budget.MaxBytes)
	if err != nil || !found || r.Status != "partial" {
		t.Fatalf("found=%v status=%s err=%v", found, r.Status, err)
	}
	var retained capturePlanCodePacket
	if err := json.Unmarshal(r.Raw, &retained); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(retained.Publication, publication.Bytes()) || !bytes.Equal(retained.CodeCapture, packet) {
		t.Fatal("partial publication lost exact code or owner bytes")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(dir, "capture-provider-range.pprof"), path); err != nil {
		t.Fatal(err)
	}
	if _, _, err := capturePlanPublication(captureResult{ScratchDir: dir}, plan, plan.Passes[0], cudaprof.CaptureRange, plan.Budget.MaxBytes); err == nil {
		t.Fatal("admitted publication symlink")
	}
}

func TestCapturePlanCooperativeFlagRequiresPlan(t *testing.T) {
	if _, err := parseCaptureArgs([]string{"--capture-plan-cooperative", "--no-inject", "--", "target"}); err == nil {
		t.Fatal("cooperation accepted without plan")
	}
	plan, _ := cliPublicationFixture(t)
	calls := 0
	got, err := executeCapturePlan(context.Background(), captureOptions{}, plan, func(context.Context, captureOptions) (captureResult, error) { calls++; return captureResult{}, nil })
	if err != nil || calls != 0 || got.CaptureSuccess {
		t.Fatalf("default unsupported route changed calls=%d err=%v", calls, err)
	}
}

func cliPublicationBytes(t *testing.T, data []byte) []byte {
	t.Helper()
	var packet capturePlanCodePacket
	if err := json.Unmarshal(data, &packet); err != nil {
		t.Fatal(err)
	}
	return packet.Publication
}

func TestCapturePlanInjectedPublicationReconciliation(t *testing.T) {
	for _, provider := range []cudaprof.CaptureProvider{cudaprof.CaptureActivity, cudaprof.CapturePC} {
		for _, kind := range []string{"unknown coverage", "retained", "context pointer", "interval", "generation", "missing loss", "period"} {
			t.Run(string(provider)+"/"+kind, func(t *testing.T) {
				one, zero := uint64(1), uint64(0)
				r := cudaprof.CaptureProviderResult{Provider: provider, Process: 3, Generation: 1, Context: "7", Start: 11, Stop: 12, Status: "complete", Config: cudaprof.CaptureProviderConfig{Period: 8}, Counts: cudaprof.CaptureCounts{Retained: &one, Offered: &one, Dropped: &zero, Unnamed: &zero, NonUser: &zero}}
				o := captureResult{TargetPID: 3, CollectorFinalized: true, RunCompleteness: runComplete, ActivityRecords: 1, RecordLoss: &captureLoss{Offered: 1, Retained: 1}, WorkerGenerations: []captureWorker{{Process: 3, Generation: 1, Finalized: true, Disposition: "finalized"}}, Activities: []captureActivity{{Process: 3, Context: 7, Start: 11, End: 12, Valid: true}}, PCSampling: &capturePCSampling{Period: 8, Folded: 1, Total: 1}}
				wantErr := false
				switch kind {
				case "retained":
					v := uint64(2)
					r.Counts.Retained = &v
					wantErr = true
				case "context pointer":
					r.Context = "0x7"
					wantErr = true
				case "interval":
					r.Stop = 13
					wantErr = true
				case "generation":
					r.Generation = 2
				case "missing loss":
					o.RecordLoss = nil
				case "period":
					r.Config.Period = 9
					wantErr = provider == cudaprof.CapturePC
				}
				err := reconcileCapturePlanJournal(o, &r)
				if (err != nil) != wantErr {
					t.Fatalf("error=%v wantError=%v result=%+v", err, wantErr, r)
				}
				if !wantErr && r.Status != "partial" {
					t.Fatalf("unknown native coverage accepted: %+v", r)
				}
			})
		}
	}
}

func TestCapturePlanCooperativeInjectedPublication(t *testing.T) {
	for _, provider := range []cudaprof.CaptureProvider{cudaprof.CaptureActivity, cudaprof.CapturePC} {
		t.Run(string(provider), func(t *testing.T) {
			plan, pub := cliPublicationFixture(t)
			config := cudaprof.CaptureProviderConfig{}
			if provider == cudaprof.CapturePC {
				config.Period = 8
			}
			plan.Passes = []cudaprof.CapturePass{{ID: "injected", Replay: "replay", Providers: []cudaprof.CaptureProvider{provider}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{provider: config}}}
			pub.Result.Provider = provider
			pub.Result.PassID = "injected"
			pub.Result.Config = config
			pub.Result.NativePasses = nil
			pub.Result.Context = "7"
			pub.Result.Scope = "process"
			if provider == cudaprof.CapturePC {
				pub.Result.Scope = "target-context-first"
			}
			var data bytes.Buffer
			if err := cudaprof.WriteCapturePublication(&data, plan, pub); err != nil {
				t.Fatal(err)
			}
			got, err := executeCapturePlan(context.Background(), captureOptions{CapturePlanCooperative: true}, plan, func(context.Context, captureOptions) (captureResult, error) {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "capture.journal"), []byte("native journal\x00"), 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, "capture-provider-"+string(provider)+".json"), data.Bytes(), 0600); err != nil {
					t.Fatal(err)
				}
				return captureResult{Session: "injected", TargetPID: 3, ScratchDir: dir, ActivityRecords: 1, PCSampling: &capturePCSampling{Period: 8, Folded: 1, Total: 1}, Activities: []captureActivity{{Process: 3, Context: 7, Valid: true, Start: 1, End: 2}}}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			m, err := cudaprof.ReadCaptureManifest(bytes.NewReader(got.CapturePlanManifest), plan.Budget.MaxBytes)
			if err != nil {
				t.Fatal(err)
			}
			r := m.Results[0]
			if r.Status != "partial" || r.Identity != pub.Result.Identity || !bytes.Equal(cliPublicationBytes(t, r.Raw), data.Bytes()) {
				t.Fatalf("publication not preserved: %+v", r)
			}
			var packet capturePlanCodePacket
			if err := json.Unmarshal(r.Raw, &packet); err != nil {
				t.Fatal(err)
			}
			var journals []capturePlanRawFile
			if err := json.Unmarshal(packet.Journals, &journals); err != nil {
				t.Fatal(err)
			}
			if len(journals) != 1 || !bytes.Equal(journals[0].Data, []byte("native journal\x00")) {
				t.Fatalf("native raw journal lost: %+v", journals)
			}
			if r.Counts.NonUser != nil {
				t.Fatal("publisher supplied unknown native non-user zero")
			}

		})
	}
}

func TestCapturePlanCooperativeRetainedCodeAdmission(t *testing.T) {
	c, pub := planCodeFixture(t)
	plan, base := cliPublicationFixture(t)
	plan.Identity = pub.Result.Identity
	pub.Result = base.Result
	pub.Result.Identity = plan.Identity
	var publication bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&publication, plan, pub); err != nil {
		t.Fatal(err)
	}
	code := planCodePacket(t, c)
	got, err := executeCapturePlan(context.Background(), captureOptions{CapturePlanCooperative: true}, plan, func(context.Context, captureOptions) (captureResult, error) {
		dir := t.TempDir()
		for name, data := range map[string][]byte{"capture-provider-range.json": publication.Bytes(), "capture-provider-range.pprof": code} {
			if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
		return captureResult{Session: "retained-code", TargetPID: 3, ScratchDir: dir}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	m, err := cudaprof.ReadCaptureManifest(bytes.NewReader(got.CapturePlanManifest), plan.Budget.MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Complete() || !got.CaptureSuccess {
		t.Fatalf("retained owner code rejected: %+v", m.Results)
	}
	var packet capturePlanCodePacket
	if err := json.Unmarshal(m.Results[0].Raw, &packet); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(packet.Publication, publication.Bytes()) || !bytes.Equal(packet.CodeCapture, code) {
		t.Fatal("byte-exact owner evidence changed")
	}
	// This is a host artifact-admission fixture, not proof of a physical run.
}
