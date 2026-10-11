package cudadiag

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func ExampleCollectHostEvidence() {
	_, err := CollectHostEvidence(context.Background(), cudatracehost.Config{}, 0, 0, io.Discard, nil)
	fmt.Println(err != nil)
	// Output: true
}

func TestHostEvidenceAdmission(t *testing.T) {
	cfg := cudatracehost.Config{Symbols: []string{"cuLaunchKernel"}}
	if _, err := CollectHostEvidence(context.Background(), cfg, 1, 10, io.Discard, nil); err == nil {
		t.Fatal("unbounded capture accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := CollectHostEvidence(ctx, cfg, 1, 10, io.Discard, nil); err == nil {
		t.Fatal("cancellation accepted")
	}
	var raw bytes.Buffer
	w := hostRawWriter{writer: &raw, written: 64 << 20}
	if _, err := w.Write([]byte("x")); err == nil {
		t.Fatal("byte cap bypassed")
	}
	w = hostRawWriter{writer: shortHostWriter{}}
	if _, err := w.Write([]byte("x")); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write: %v", err)
	}
}

type shortHostWriter struct{}

func (shortHostWriter) Write(p []byte) (int, error) { return 0, nil }

func TestStructuredHostCall(t *testing.T) {
	manifest := cudatracehost.Manifest{Clock: "linux_monotonic", Epoch: "boot"}
	base := cudatracehost.CallInterval{Clock: manifest.Clock, Epoch: manifest.Epoch, EntryTimestamp: 1, ReturnTimestamp: 2, PairingStatus: "verified_thread_lifetime", ThreadGenerationKnown: true}
	for _, tc := range []struct {
		name                string
		edit                func(*cudatracehost.CallInterval)
		wantErr, wantReturn bool
	}{
		{"known zero birth", func(c *cudatracehost.CallInterval) {}, false, true},
		{"tracefs unknown birth", func(c *cudatracehost.CallInterval) {
			c.PairingStatus = "unverified_thread_lifetime"
			c.ThreadGenerationKnown = false
		}, false, false},
		{"unknown marked verified", func(c *cudatracehost.CallInterval) { c.ThreadGenerationKnown = false }, true, false},
		{"epoch change", func(c *cudatracehost.CallInterval) { c.Epoch = "other" }, true, false},
		{"process change", func(c *cudatracehost.CallInterval) { c.Process.Generation++ }, true, false},
		{"reversed", func(c *cudatracehost.CallInterval) { c.ReturnTimestamp = 0 }, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			call := base
			tc.edit(&call)
			got, err := observedHostCall("cuLaunchKernel", &call, manifest)
			if (err != nil) != tc.wantErr || got.HasReturn != tc.wantReturn {
				t.Fatalf("got %v, %v", got, err)
			}
		})
	}
}
