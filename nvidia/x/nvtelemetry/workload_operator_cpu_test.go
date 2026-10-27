package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func cpuContextDigest(raw []byte) string {
	h := sha256.Sum256(raw)
	return hex.EncodeToString(h[:])
}

func cpuContextFixture() (nvtelemetry.OperatorCPUContext, nvtelemetry.WorkloadSource) {
	raw := []byte(`{"schemaVersion":1,"deviceProperties":[],"traceEvents":[]}`)
	code := []byte("constructed code fact, not a member relation")
	trace := nvtelemetry.OperatorReference{SHA256: cpuContextDigest(raw), Locator: "traceEvents"}
	c := nvtelemetry.OperatorCPUContext{Version: 1, RunID: "run", TraceSource: nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "run"}, Trace: trace,
		Artifacts: map[string][]byte{trace.SHA256: raw, cpuContextDigest(code): code}, CodeFacts: []nvtelemetry.OperatorReference{{SHA256: cpuContextDigest(code), Locator: "generated/cpp"}}}
	return c, nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-cpu-context-v1", RunID: "run"}
}

func TestOperatorCPUContextRoundTrip(t *testing.T) {
	c, source := cpuContextFixture()
	var wire bytes.Buffer
	if err := nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	export, got, err := nvtelemetry.ReadOperatorCPUContext(bytes.NewReader(wire.Bytes()), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if export.RawSHA256 != cpuContextDigest(wire.Bytes()) || !bytes.Equal(export.Raw, wire.Bytes()) || got.Trace != c.Trace || got.TraceSource.Version != "2.10.0" || got.CodeFacts[0] != c.CodeFacts[0] || !bytes.Equal(got.Artifacts[c.Trace.SHA256], c.Artifacts[c.Trace.SHA256]) {
		t.Fatal("CPU context lost raw bytes, identity or separate code facts")
	}
	c.Artifacts[c.Trace.SHA256][0] = 'x'
	if got.Artifacts[got.Trace.SHA256][0] != '{' {
		t.Fatal("decoded payload aliases caller metadata")
	}
}

func TestOperatorCPUContextRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*nvtelemetry.OperatorCPUContext)
	}{
		{"version", func(c *nvtelemetry.OperatorCPUContext) { c.Version = 2 }},
		{"run", func(c *nvtelemetry.OperatorCPUContext) { c.TraceSource.RunID = "other" }},
		{"old trace", func(c *nvtelemetry.OperatorCPUContext) { c.TraceSource.Version = "2.7.0" }},
		{"engine", func(c *nvtelemetry.OperatorCPUContext) { c.TraceSource.Engine = "invented" }},
		{"provider", func(c *nvtelemetry.OperatorCPUContext) { c.TraceSource.Provider = "application" }},
		{"raw hash", func(c *nvtelemetry.OperatorCPUContext) { c.Artifacts[c.Trace.SHA256][0] = 'x' }},
		{"missing trace", func(c *nvtelemetry.OperatorCPUContext) { delete(c.Artifacts, c.Trace.SHA256) }},
		{"missing code", func(c *nvtelemetry.OperatorCPUContext) { delete(c.Artifacts, c.CodeFacts[0].SHA256) }},
		{"unreferenced", func(c *nvtelemetry.OperatorCPUContext) {
			c.Artifacts[cpuContextDigest([]byte("extra"))] = []byte("extra")
		}},
		{"locator", func(c *nvtelemetry.OperatorCPUContext) { c.Trace.Locator = " " }},
		{"rank", func(c *nvtelemetry.OperatorCPUContext) { n := -1; c.TraceSource.Rank = &n }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, source := cpuContextFixture()
			tc.change(&c)
			var out bytes.Buffer
			if err := nvtelemetry.WriteOperatorCPUContext(&out, c, nvtelemetry.WorkloadLimits{}); err == nil || out.Len() != 0 {
				t.Fatal("invalid CPU context was written")
			}
			raw, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := nvtelemetry.ReadOperatorCPUContext(bytes.NewReader(raw), source, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("invalid CPU context was read")
			}
		})
	}
}

type cpuContextShortWriter struct{}

func (cpuContextShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestOperatorCPUContextLimitsAndProfiles(t *testing.T) {
	c, source := cpuContextFixture()
	for _, limits := range []nvtelemetry.WorkloadLimits{{MaxBytes: 100}, {MaxMetrics: 1}, {MaxStringBytes: 3}, {MaxBytes: -1}} {
		if err := nvtelemetry.WriteOperatorCPUContext(io.Discard, c, limits); err == nil {
			t.Fatal("CPU context limit ignored")
		}
	}
	if err := nvtelemetry.WriteOperatorCPUContext(cpuContextShortWriter{}, c, nvtelemetry.WorkloadLimits{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if err := nvtelemetry.WriteOperatorCPUContext(nil, c, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("nil writer accepted")
	}
	raw, _ := json.Marshal(c)
	for _, bad := range []string{strings.Replace(string(raw), `"version":1`, `"version":1,"version":1`, 1), strings.TrimSuffix(string(raw), "}") + `,"groups":[]}`, string(raw) + `{}`} {
		if _, _, err := nvtelemetry.ReadOperatorCPUContext(strings.NewReader(bad), source, nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("ambiguous or native metadata accepted as CPU context")
		}
	}
	if _, _, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(raw), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("native reader accepted CPU profile")
	}
	source.Version = "operator-context-v1"
	if _, _, err := nvtelemetry.ReadOperatorCPUContext(bytes.NewReader(raw), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("CPU reader accepted native profile")
	}
}

func TestOperatorCPUContextHashAllocationBound(t *testing.T) {
	// The offending input and ordinary contexts are allocated before warmup
	// and measurement. These are a few direct writer calls, not parser fuzzing.
	large := strings.Repeat("a", 1<<20)
	trace, _ := cpuContextFixture()
	trace.Trace.SHA256 = large
	trace.Artifacts = nil
	code, _ := cpuContextFixture()
	code.CodeFacts[0].SHA256 = large
	evidence, _ := cpuContextFixture()
	evidence.Evidence = []nvtelemetry.OperatorReference{{SHA256: large, Locator: "application"}}
	key, _ := cpuContextFixture()
	key.Artifacts[large] = nil
	for _, tc := range []struct {
		name string
		c    nvtelemetry.OperatorCPUContext
	}{
		{"trace without artifacts", trace},
		{"code fact", code},
		{"evidence", evidence},
		{"artifact key", key},
	} {
		t.Run(tc.name, func(t *testing.T) {
			limits := nvtelemetry.WorkloadLimits{MaxBytes: 16 << 10}
			if err := nvtelemetry.WriteOperatorCPUContext(io.Discard, tc.c, limits); err == nil {
				t.Fatal("oversized hash accepted during warmup")
			}
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			for range 3 {
				if err := nvtelemetry.WriteOperatorCPUContext(io.Discard, tc.c, limits); err == nil {
					t.Fatal("oversized hash accepted")
				}
			}
			runtime.ReadMemStats(&after)
			runtime.KeepAlive(tc.c)
			if n := after.TotalAlloc - before.TotalAlloc; n > 64<<10 {
				t.Fatalf("SHA256 validation allocated %d bytes for three refused writes, want at most 65536", n)
			}
		})
	}
}
