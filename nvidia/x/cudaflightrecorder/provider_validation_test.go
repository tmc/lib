package cudaflightrecorder_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"runtime/trace"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

func validProviderProfile(t *testing.T) []byte {
	t.Helper()
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "gpu", Unit: "nanoseconds"}}, Sample: []*profile.Sample{{Value: []int64{42}}}}
	var b bytes.Buffer
	if err := p.Write(&b); err != nil {
		t.Fatal(err)
	}
	if _, err := profile.ParseData(b.Bytes()); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func providerSnapshot(t *testing.T, name string, data []byte, limit int64) (*cudaflightrecorder.Manifest, *cudaflightrecorder.Bundle) {
	t.Helper()
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1, MaxProviderOutputBytes: limit})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	p := func(context.Context) ([]byte, error) { return data, nil }
	switch name {
	case "gpu.pprof":
		r.SetGPUPprofProvider(p)
	case "go.trace":
		r.SetGoTraceProvider(p)
	case "timeline.pftrace":
		r.SetPerfettoProvider(p)
	}
	var b bytes.Buffer
	m, err := r.Snapshot(context.Background(), &b, "provider control")
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := cudaflightrecorder.ReadBundle(&b)
	if err != nil {
		t.Fatal(err)
	}
	return m, bundle
}

func TestProviderProfileDecoder(t *testing.T) {
	good := validProviderProfile(t)
	corrupt := append([]byte(nil), good...)
	corrupt[len(corrupt)-1] ^= 1
	var invalid bytes.Buffer
	z := gzip.NewWriter(&invalid)
	_, _ = z.Write([]byte{0x0e, 0, 0, 0}) // Invalid protobuf wire type 6.
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		data  []byte
		valid bool
	}{
		{"genuine", good, true},
		{"gzip-header-only", []byte{0x1f, 0x8b, 0, 0}, false},
		{"corrupt-checksum", corrupt, false},
		{"truncated", good[:len(good)-1], false},
		{"invalid-protobuf", invalid.Bytes(), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, oracleErr := profile.ParseData(tc.data)
			if (oracleErr == nil) != tc.valid {
				t.Fatalf("independent decoder oracle: %v", oracleErr)
			}
			m, b := providerSnapshot(t, "gpu.pprof", tc.data, 4096)
			_, retained := b.Files["gpu.pprof"]
			if strings.HasPrefix(m.ProviderStatus["gpu.pprof"], "valid") != tc.valid || retained != tc.valid {
				t.Fatalf("retained=%t status=%q", retained, m.ProviderStatus["gpu.pprof"])
			}
		})
	}
}

func TestProviderDecompressionBudget(t *testing.T) {
	var b bytes.Buffer
	z := gzip.NewWriter(&b)
	_, _ = z.Write(bytes.Repeat([]byte{0}, 8192))
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	if b.Len() >= 1024 {
		t.Fatal("control not compressed below provider cap")
	}
	m, bundle := providerSnapshot(t, "gpu.pprof", b.Bytes(), 1024)
	if !strings.Contains(m.ProviderStatus["gpu.pprof"], "decompressed output exceeds") || bundle.Files["gpu.pprof"] != nil {
		t.Fatalf("budget not enforced before decode: %+v", m)
	}
	m, bundle = providerSnapshot(t, "gpu.pprof", bytes.Repeat([]byte{0}, 1025), 1024)
	if !strings.Contains(m.ProviderStatus["gpu.pprof"], "output 1025 bytes exceeds") || bundle.Files["gpu.pprof"] != nil {
		t.Fatalf("input budget not enforced before decode: %+v", m)
	}
}

func TestUndecodedFormatsRemainUnvalidated(t *testing.T) {
	var goTrace bytes.Buffer
	if err := trace.Start(&goTrace); err != nil {
		t.Fatal(err)
	}
	trace.Stop()
	var timeline bytes.Buffer
	if err := perfetto.NewWriter(&timeline).WriteTrackDescriptor(1, "control", 0); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"go.trace", goTrace.Bytes()},
		{"go.trace", goTrace.Bytes()[:len(goTrace.Bytes())/2]},
		{"go.trace", []byte("go 1.27 trace\x00not trace records")},
		{"timeline.pftrace", timeline.Bytes()},
		{"timeline.pftrace", []byte{0x0e, 0, 0, 0}},
		{"timeline.pftrace", []byte("not a protobuf trace")},
	} {
		m, b := providerSnapshot(t, tc.name, tc.data, 1<<20)
		if !strings.HasPrefix(m.ProviderStatus[tc.name], "unvalidated:") || !bytes.Equal(b.Files[tc.name], tc.data) {
			t.Fatalf("undecoded format claimed valid or raw data lost: %q", m.ProviderStatus[tc.name])
		}
		for _, stored := range []*cudaflightrecorder.Manifest{m, &b.Manifest} {
			if stored.ProviderStatus[tc.name] != m.ProviderStatus[tc.name] || !strings.Contains(strings.Join(stored.LossReasons, ";"), "unvalidated") || stored.GoSpan() != 0 || stored.OverlapSpan() != 0 || stored.OverlapReliable {
				t.Fatalf("bundle consumer lost unvalidated status or admitted timing: %+v", stored)
			}
		}
	}
}

func TestAttachmentCannotReplaceValidatedProvider(t *testing.T) {
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	r.SetGPUPprofProvider(func(context.Context) ([]byte, error) { return validProviderProfile(t), nil })
	r.AttachFile("gpu.pprof", []byte{0x1f, 0x8b, 0, 0})
	var b bytes.Buffer
	if _, err := r.Snapshot(context.Background(), &b, "replacement"); err == nil || b.Len() != 0 {
		t.Fatal("invalid attachment replaced validated provider or wrote partial output")
	}
}
