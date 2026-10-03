package cudaprof_test

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func captureFixture() cudaprof.CaptureManifest {
	d := strings.Repeat("1", 64)
	id := cudaevidence.Identity{ArtifactDigest: d, Entry: "kernel", Target: "sm_90", CompilerKey: d, WorkloadKey: d}
	env := cudaprof.CaptureEnvironment{Device: "gpu", Driver: "driver", CUDA: "cuda", CUPTI: "cupti", ClockMode: "native", CorrelationDomain: "run1"}
	plan := cudaprof.CapturePlan{Version: 1, Identity: id, Environment: env, Budget: cudaprof.CaptureBudget{MaxRecords: 100, MaxBytes: 100000, MaxPasses: 4, MaxWallNanos: 100000}, Passes: []cudaprof.CapturePass{{ID: "p1", Replay: "r1", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureActivity, cudaprof.CapturePC}}, {ID: "p2", Replay: "r2", Providers: []cudaprof.CaptureProvider{cudaprof.CaptureRange}}}}
	plan.Passes[0].Configs = map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CapturePC: {Period: 5}}
	plan.Passes[1].Configs = map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{cudaprof.CaptureRange: {Metrics: []string{"cycles"}, ConfigDigest: d}}
	m := cudaprof.CaptureManifest{Version: 1, Plan: plan, ElapsedNanos: 10}
	for _, pass := range plan.Passes {
		for _, provider := range pass.Providers {
			scope := "target-context"
			if provider == cudaprof.CapturePC {
				scope = "target-context-first"
			}
			if provider == cudaprof.CaptureActivity {
				scope = "process"
			}
			var native *cudaprof.CaptureNativePasses
			if provider == cudaprof.CaptureRange {
				native = &cudaprof.CaptureNativePasses{Started: 1, Stopped: 1, AllSubmitted: true}
			}
			zero, one := uint64(0), uint64(1)
			n := float64(7)
			value, _ := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Measured, Number: &n, Basis: "retained native record", Source: cudaevidence.Source{Identity: id, Producer: "owner", Locator: "record/1"}})
			m.Results = append(m.Results, cudaprof.CaptureProviderResult{Provider: provider, PassID: pass.ID, Replay: pass.Replay, Scope: scope, Config: pass.Configs[provider], NativePasses: native, Process: 1, Generation: 1, Context: "ctx", Identity: id, Environment: env, Start: 1, Stop: 2, Status: "complete", EvidenceClass: "measured", Counts: cudaprof.CaptureCounts{Retained: &one, Offered: &one, Dropped: &zero, NonUser: &zero, Unnamed: &zero}, Raw: []byte{0, 255, '{', 'x', '}'}, Values: []cudaevidence.Value{value}, Decoded: true, CleanupComplete: true})
		}
	}
	return m
}
func TestCapturePlanProviderIsolation(t *testing.T) {
	m := captureFixture()
	if err := m.Plan.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, providers := range [][]cudaprof.CaptureProvider{{"unknown"}, {cudaprof.CaptureActivity, cudaprof.CaptureRange}, {cudaprof.CapturePM, cudaprof.CapturePC}, {cudaprof.CaptureSASS, cudaprof.CaptureActivity}, {cudaprof.CapturePC, cudaprof.CapturePC}} {
		p := captureFixture().Plan
		p.Passes[0].Providers = providers
		if p.Validate() == nil {
			t.Fatalf("invalid provider set accepted: %v", providers)
		}
	}
	p := m.Plan
	p.Passes[1].ID = p.Passes[0].ID
	if p.Validate() == nil {
		t.Fatal("duplicate pass accepted")
	}
}
func TestCaptureManifestBudget(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CaptureManifest)
	}{
		{"records", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxRecords = 2 }},
		{"bytes", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxBytes = 1 }},
		{"passes", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxPasses = 1 }},
		{"wall", func(m *cudaprof.CaptureManifest) { m.ElapsedNanos = 100001 }},
		{"zero records", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxRecords = 0 }},
		{"max records", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxRecords = math.MaxUint64 }},
		{"max bytes", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxBytes = math.MaxInt }},
		{"max wall", func(m *cudaprof.CaptureManifest) { m.Plan.Budget.MaxWallNanos = math.MaxInt64 }},
		{"counter overflow", func(m *cudaprof.CaptureManifest) { n := uint64(math.MaxUint64); m.Results[0].Counts.Offered = &n }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := captureFixture()
			tc.change(&m)
			var out bytes.Buffer
			if cudaprof.WriteCaptureManifest(&out, m) == nil {
				t.Fatal("budget accepted")
			}
			if out.Len() != 0 {
				t.Fatal("wrote bytes before rejection")
			}
		})
	}
	m := captureFixture()
	var out bytes.Buffer
	if err := cudaprof.WriteCaptureManifest(&out, m); err != nil {
		t.Fatal(err)
	}
	if _, err := cudaprof.ReadCaptureManifest(bytes.NewReader(out.Bytes()), out.Len()-1); err == nil {
		t.Fatal("read limit ignored")
	}
	if _, err := cudaprof.ReadCaptureManifest(bytes.NewReader(out.Bytes()), 0); err == nil {
		t.Fatal("zero read limit admitted")
	}
}
func TestCaptureManifestPreservesReplayIdentity(t *testing.T) {
	m := captureFixture()
	if !m.Complete() {
		t.Fatal("healthy independent captures not complete")
	}
	for _, change := range []func(*cudaprof.CaptureManifest){func(m *cudaprof.CaptureManifest) { m.Results[0].Replay = "stale" }, func(m *cudaprof.CaptureManifest) { m.Results = m.Results[:2] }, func(m *cudaprof.CaptureManifest) { m.Results[2] = m.Results[0] }, func(m *cudaprof.CaptureManifest) { m.Results[0].Identity.ArtifactDigest = strings.Repeat("2", 64) }, func(m *cudaprof.CaptureManifest) { m.Results[0].Environment.Driver = "other" }} {
		m := captureFixture()
		change(&m)
		if m.Validate() == nil || m.Complete() {
			t.Fatal("mismatched/truncated capture admitted")
		}
	}
}
func TestCaptureRefusesCrossPassJoinWithoutBasis(t *testing.T) {
	m := captureFixture()
	a, b := m.Results[0], m.Results[2]
	if cudaprof.CheckCaptureJoin(a, b, "") == nil {
		t.Fatal("cross-pass join admitted without basis")
	}
	if err := cudaprof.CheckCaptureJoin(a, b, "separate replay comparison, not causality"); err != nil {
		t.Fatal(err)
	}
	b = m.Results[1]
	if err := cudaprof.CheckCaptureJoin(a, b, "same process activity association"); err != nil {
		t.Fatal(err)
	}
	b.Replay = "other"
	if cudaprof.CheckCaptureJoin(a, b, "basis") == nil {
		t.Fatal("same-pass replay mismatch accepted")
	}
	b = m.Results[1]
	b.Environment.CorrelationDomain = "other"
	if cudaprof.CheckCaptureJoin(a, b, "basis") == nil {
		t.Fatal("different domain accepted")
	}
}
func TestCaptureReportsDroppedAndUnavailableProviders(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CaptureProviderResult)
		valid  bool
	}{
		{"dropped complete", func(r *cudaprof.CaptureProviderResult) { n := uint64(1); r.Counts.Dropped = &n }, false},
		{"unknown complete", func(r *cudaprof.CaptureProviderResult) { r.Counts.Dropped = nil }, false},
		{"forged zero retained", func(r *cudaprof.CaptureProviderResult) { n := uint64(0); r.Counts.Retained = &n }, false},
		{"unavailable", func(r *cudaprof.CaptureProviderResult) {
			r.Status = "unavailable"
			r.Reason = "owner not attached"
			r.Counts = cudaprof.CaptureCounts{}
			r.Raw = nil
			r.Decoded = false
		}, true},
		{"partial loss", func(r *cudaprof.CaptureProviderResult) {
			r.Status = "partial"
			r.Reason = "dropped record"
			n, two := uint64(1), uint64(2)
			r.Counts.Dropped = &n
			r.Counts.Offered = &two
		}, true},
		{"cleanup", func(r *cudaprof.CaptureProviderResult) { r.CleanupComplete = false }, false},
		{"decode", func(r *cudaprof.CaptureProviderResult) { r.Decoded = false }, false},
		{"boundaries", func(r *cudaprof.CaptureProviderResult) { r.Stop = r.Start }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := captureFixture()
			tc.change(&m.Results[0])
			err := m.Validate()
			if (err == nil) != tc.valid {
				t.Fatalf("valid=%t err=%v", tc.valid, err)
			}
			if m.Complete() {
				t.Fatal("loss/unavailable capture reported complete")
			}
		})
	}
}
func TestCapturePreservesRawProviderRecords(t *testing.T) {
	m := captureFixture()
	var out bytes.Buffer
	if err := cudaprof.WriteCaptureManifest(&out, m); err != nil {
		t.Fatal(err)
	}
	read, err := cudaprof.ReadCaptureManifest(bytes.NewReader(out.Bytes()), 100000)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m.Results {
		if !bytes.Equal(read.Results[i].Raw, m.Results[i].Raw) {
			t.Fatal("raw payload rewritten")
		}
	}
	read.Plan.Passes[0].Providers[0] = "other"
	read.Results[0].Raw[0] = 1
	*read.Results[0].Counts.Retained = 99
	if m.Plan.Passes[0].Providers[0] != cudaprof.CaptureActivity || m.Results[0].Raw[0] != 0 || *m.Results[0].Counts.Retained != 1 {
		t.Fatal("read aliases source")
	}
	planBytes, _ := json.Marshal(m.Plan)
	if _, err := cudaprof.ReadCapturePlan(bytes.NewReader(planBytes), 100000); err != nil {
		t.Fatal(err)
	}
}
func TestCaptureStrictJSON(t *testing.T) {
	m := captureFixture()
	raw, _ := json.Marshal(m)
	s := string(raw)
	for _, bad := range []string{
		strings.Replace(s, `"version":1`, `"version":1,"version":1`, 1),
		strings.Replace(s, `"version":1`, `"Version":1`, 1),
		strings.Replace(s, `"version":1`, `"unknown":1,"version":1`, 1),
		strings.Replace(s, `"retained":1`, `"retained":1,"Retained":0`, 1),
		strings.Replace(s, `"basis":`, `"Basis":`, 1),
		strings.Replace(s, `"number":7`, `"number":7,"number":8`, 1),
		s + ` {}`, "null",
	} {
		if _, err := cudaprof.ReadCaptureManifest(strings.NewReader(bad), 100000); err == nil {
			t.Fatalf("ambiguous JSON admitted: %s", bad)
		}
	}
}

func TestCaptureConfigNativePassAndScopeAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CaptureManifest)
	}{
		{"PC period", func(m *cudaprof.CaptureManifest) {
			m.Plan.Passes[0].Configs[cudaprof.CapturePC] = cudaprof.CaptureProviderConfig{Period: 32}
		}},
		{"missing range config", func(m *cudaprof.CaptureManifest) { delete(m.Plan.Passes[1].Configs, cudaprof.CaptureRange) }},
		{"duplicate metrics", func(m *cudaprof.CaptureManifest) {
			c := m.Plan.Passes[1].Configs[cudaprof.CaptureRange]
			c.Metrics = []string{"cycles", "cycles"}
			m.Plan.Passes[1].Configs[cudaprof.CaptureRange] = c
		}},
		{"actual config differs", func(m *cudaprof.CaptureManifest) { m.Results[2].Config.ConfigDigest = strings.Repeat("2", 64) }},
		{"missing native passes", func(m *cudaprof.CaptureManifest) { m.Results[2].NativePasses = nil }},
		{"native passes unbalanced", func(m *cudaprof.CaptureManifest) { m.Results[2].NativePasses.Stopped = 0 }},
		{"native not submitted", func(m *cudaprof.CaptureManifest) { m.Results[2].NativePasses.AllSubmitted = false }},
		{"native passes over budget", func(m *cudaprof.CaptureManifest) {
			m.Results[2].NativePasses.Started = 5
			m.Results[2].NativePasses.Stopped = 5
		}},
		{"PC overclaimed scope", func(m *cudaprof.CaptureManifest) { m.Results[1].Scope = "target-context" }},
		{"missing owner", func(m *cudaprof.CaptureManifest) { m.Results[1].Generation = 0 }},
		{"unexplained retention gap", func(m *cudaprof.CaptureManifest) { n := uint64(2); m.Results[0].Counts.Offered = &n }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := captureFixture()
			tc.change(&m)
			if m.Validate() == nil || m.Complete() {
				t.Fatal("invalid config/native pass/scope admitted")
			}
		})
	}
	m := captureFixture()
	a, b := m.Results[0], m.Results[1]
	b.Process++
	if cudaprof.CheckCaptureJoin(a, b, "matching run") == nil {
		t.Fatal("different process joined")
	}
	b = m.Results[1]
	b.Stop = b.Start
	if cudaprof.CheckCaptureJoin(a, b, "basis") == nil {
		t.Fatal("empty native interval joined")
	}
}
func TestCaptureUnknownObservedFailurePreserved(t *testing.T) {
	m := captureFixture()
	for i := range m.Results {
		r := &m.Results[i]
		r.Status = "unavailable"
		r.Reason = "wall budget exceeded before owner attached"
		r.Identity = cudaevidence.Identity{}
		r.Environment = cudaprof.CaptureEnvironment{}
		r.Counts = cudaprof.CaptureCounts{}
		r.Values = nil
		r.Raw = nil
		r.Decoded = false
		r.NativePasses = nil
		r.Config = cudaprof.CaptureProviderConfig{}
	}
	m.ElapsedNanos = m.Plan.Budget.MaxWallNanos + 1
	var out bytes.Buffer
	if err := cudaprof.WriteCaptureManifest(&out, m); err != nil {
		t.Fatal(err)
	}
	read, err := cudaprof.ReadCaptureManifest(bytes.NewReader(out.Bytes()), 100000)
	if err != nil {
		t.Fatal(err)
	}
	if read.Complete() || read.Results[0].Identity.ArtifactDigest != "" || read.Results[0].Environment.Device != "" || read.ElapsedNanos != m.ElapsedNanos {
		t.Fatal("unknown observation manufactured or elapsed clamped")
	}
	m.Results[0].Identity.ArtifactDigest = strings.Repeat("2", 64)
	if m.Validate() != nil {
		t.Fatal("truthful mismatch lost")
	}
	m.Results[0].Identity.ArtifactDigest = "malformed"
	if m.Validate() == nil {
		t.Fatal("malformed supplied digest accepted")
	}
}
func TestCapturePMAndSASSConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name     string
		provider cudaprof.CaptureProvider
		config   cudaprof.CaptureProviderConfig
		valid    bool
	}{
		{"PM valid", cudaprof.CapturePM, cudaprof.CaptureProviderConfig{Metrics: []string{"cycles"}, ConfigDigest: strings.Repeat("1", 64), SamplingInterval: 100}, true},
		{"PM interval absent", cudaprof.CapturePM, cudaprof.CaptureProviderConfig{Metrics: []string{"cycles"}, ConfigDigest: strings.Repeat("1", 64)}, false},
		{"PM trigger", cudaprof.CapturePM, cudaprof.CaptureProviderConfig{Metrics: []string{"cycles"}, ConfigDigest: strings.Repeat("1", 64), SamplingInterval: 100, TriggerMode: 99}, false},
		{"PM append", cudaprof.CapturePM, cudaprof.CaptureProviderConfig{Metrics: []string{"cycles"}, ConfigDigest: strings.Repeat("1", 64), SamplingInterval: 100, AppendMode: 99}, false},
		{"SASS GPU granularity", cudaprof.CaptureSASS, cudaprof.CaptureProviderConfig{SASSMetrics: []cudaprof.CaptureSASSMetric{{MetricID: 1, Granularity: 0}}}, true},
		{"SASS wrong granularity", cudaprof.CaptureSASS, cudaprof.CaptureProviderConfig{SASSMetrics: []cudaprof.CaptureSASSMetric{{MetricID: 1, Granularity: 3}}}, false},
		{"SASS fictitious image", cudaprof.CaptureSASS, cudaprof.CaptureProviderConfig{ConfigDigest: strings.Repeat("1", 64), SASSMetrics: []cudaprof.CaptureSASSMetric{{MetricID: 1, Granularity: 0}}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := captureFixture().Plan
			p.Passes = []cudaprof.CapturePass{{ID: "p", Replay: "r", Providers: []cudaprof.CaptureProvider{tc.provider}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{tc.provider: tc.config}}}
			if (p.Validate() == nil) != tc.valid {
				t.Fatal("policy validation wrong")
			}
		})
	}
}

func TestCaptureEncodingAndJoinBounds(t *testing.T) {
	m := captureFixture()
	var out bytes.Buffer
	if err := cudaprof.WriteCaptureManifest(&out, m); err != nil {
		t.Fatal(err)
	}
	m.Plan.Budget.MaxBytes = out.Len() + 100
	out.Reset()
	if err := cudaprof.WriteCaptureManifest(&out, m); err != nil {
		t.Fatal(err)
	}
	padded := append(out.Bytes(), bytes.Repeat([]byte(" "), 200)...)
	if _, err := cudaprof.ReadCaptureManifest(bytes.NewReader(padded), 100000); err == nil {
		t.Fatal("original whitespace bypassed plan byte limit")
	}
	m = captureFixture()
	m.Plan.Budget.MaxBytes = 1024
	m.Results[0].Raw = make([]byte, 2048)
	out.Reset()
	if cudaprof.WriteCaptureManifest(&out, m) == nil || out.Len() != 0 {
		t.Fatal("oversized caller raw admitted")
	}
	m = captureFixture()
	a, b := m.Results[1], m.Results[1]
	b.Start, b.Stop = 3, 4
	if cudaprof.CheckCaptureJoin(a, b, "") == nil {
		t.Fatal("disjoint native intervals joined")
	}
	if err := cudaprof.CheckCaptureJoin(a, b, "separate interval comparison"); err != nil {
		t.Fatal(err)
	}
	a = m.Results[0]
	a.Context = ""
	b = m.Results[1]
	if cudaprof.CheckCaptureJoin(a, b, "") == nil {
		t.Fatal("process-only activity attributed without basis")
	}
}

func TestCaptureRequiredNegativeControls(t *testing.T) {
	positive := captureFixture()
	if err := positive.Validate(); err != nil || !positive.Complete() {
		t.Fatalf("healthy fixture: complete=%v error=%v", positive.Complete(), err)
	}
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CaptureManifest)
	}{
		{"unknown provider", func(m *cudaprof.CaptureManifest) {
			m.Plan.Passes = m.Plan.Passes[:1]
			m.Plan.Passes[0].Providers = []cudaprof.CaptureProvider{"unknown"}
			m.Plan.Passes[0].Configs = map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{"unknown": {}}
			m.Results = m.Results[:1]
			m.Results[0].Provider = "unknown"
			m.Results[0].Scope = "target-context"
			m.Results[0].Config = cudaprof.CaptureProviderConfig{}
		}},
		{"internally consistent stale identity", func(m *cudaprof.CaptureManifest) {
			r := &m.Results[0]
			r.Identity.ArtifactDigest = strings.Repeat("2", 64)
			n := float64(7)
			v, err := cudaevidence.New(cudaevidence.Record{Class: cudaevidence.Measured, Number: &n, Basis: "retained native record", Source: cudaevidence.Source{Identity: r.Identity, Producer: "owner", Locator: "record/1"}})
			if err != nil {
				t.Fatal(err)
			}
			r.Values = []cudaevidence.Value{v}
		}},
		{"forged known zero", func(m *cudaprof.CaptureManifest) {
			zero := uint64(0)
			m.Results[0].Counts.Retained = &zero
			m.Results[0].Counts.Offered = &zero
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := captureFixture()
			tc.change(&m)
			if err := m.Validate(); err == nil {
				t.Fatal("invalid receipt accepted")
			}
			if m.Complete() {
				t.Fatal("invalid receipt marked complete")
			}
			var out bytes.Buffer
			if err := cudaprof.WriteCaptureManifest(&out, m); err == nil {
				t.Fatal("invalid receipt written")
			}
			if out.Len() != 0 {
				t.Fatal("bytes written before refusal")
			}
		})
	}
}

func TestCapturePassReservationBudget(t *testing.T) {
	m := captureFixture()
	p := m.Plan.Passes[0]
	p.ID, p.Replay = "p3", "r3"
	p.Providers = []cudaprof.CaptureProvider{cudaprof.CaptureActivity}
	p.Configs = nil
	m.Plan.Passes = append(m.Plan.Passes, p)
	r := m.Results[0]
	r.PassID, r.Replay = p.ID, p.Replay
	m.Results = append(m.Results, r)
	m.Results[2].NativePasses = &cudaprof.CaptureNativePasses{Started: 3, Stopped: 3, AllSubmitted: true}
	m.Plan.Budget.MaxPasses = 4
	if err := m.Validate(); err == nil {
		t.Fatal("three reservations plus two native extra passes accepted under budget four")
	}
	m.Plan.Budget.MaxPasses = 5
	if err := m.Validate(); err != nil || !m.Complete() {
		t.Fatalf("exact five-pass budget refused: %v", err)
	}
}
