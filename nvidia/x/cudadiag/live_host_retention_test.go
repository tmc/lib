package cudadiag

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// hostEvidenceFake supplies controlled host facts only. It never attaches a
// probe, resolves a CUDA library, reads kernel counters or claims native truth.
type hostEvidenceFake struct {
	initial, final        cudatracehost.Manifest
	events                []cudatracehost.Result
	nextErrors            []error
	stats                 cudatracehost.StopStats
	stopError, closeError error
	calls                 []string
	stopped               bool
}

func (c *hostEvidenceFake) Next(context.Context) (cudatracehost.Result, error) {
	c.calls = append(c.calls, "next")
	if len(c.events) == 0 {
		return cudatracehost.Result{}, io.EOF
	}
	event := c.events[0]
	c.events = c.events[1:]
	var err error
	if len(c.nextErrors) > 0 {
		err = c.nextErrors[0]
		c.nextErrors = c.nextErrors[1:]
	}
	return event, err
}
func (c *hostEvidenceFake) Manifest() cudatracehost.Manifest {
	if c.stopped {
		c.calls = append(c.calls, "final-manifest")
		return c.final
	}
	c.calls = append(c.calls, "initial-manifest")
	return c.initial
}
func (c *hostEvidenceFake) StopAndStats() (cudatracehost.StopStats, error) {
	c.calls = append(c.calls, "stop")
	c.stopped = true
	return c.stats, c.stopError
}
func (c *hostEvidenceFake) Close() error { c.calls = append(c.calls, "close"); return c.closeError }

func hostRetentionFixture() *hostEvidenceFake {
	process := cudaprobe.Process{PID: 42, Generation: 7}
	initial := cudatracehost.Manifest{Process: process, Backend: "bpf", Clock: "linux_monotonic", Epoch: "controlled-host-epoch", ThreadScope: "attachment_snapshot", LossStatus: "unknown"}
	attempt := cudatracehost.ThreadRescan{Sequence: 1, Status: "generation_verified_attachment", Clock: "elapsed_since_discovery", BlindInterval: "between_scans_unknown", MissedCalls: "unknown", StartNanos: 10, EndNanos: 20, Threads: []cudatracehost.ThreadSnapshot{{TID: 43, Generation: 8}}, AttachmentAttempted: true}
	final := initial
	final.ThreadRescans = []cudatracehost.ThreadRescan{attempt}
	final.ThreadRescanInterval = 10000000
	final.MaxThreadRescans = 1
	return &hostEvidenceFake{initial: initial, final: final, events: []cudatracehost.Result{{ThreadRescan: &attempt}}, stats: cudatracehost.StopStats{Submitted: 3, Dropped: 1, Delivered: 2, Undelivered: 1, RingStatus: "measured"}}
}
func runHostRetention(c *hostEvidenceFake, w io.Writer, max int) (HostEvidence, error) {
	return collectHostEvidence(context.Background(), c, func() (cudaworkload.ProcessGenID, error) {
		return cudaworkload.ProcessGenID{PID: 42, Generation: 7}, nil
	}, 43, max, w, nil)
}
func hostRetentionWire(t *testing.T, out HostEvidence) map[string]json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	return wire
}
func TestHostEvidenceRetainsCoverageAndFinalCounters(t *testing.T) {
	for _, name := range []string{"rescan-only", "final-manifest", "final-stats"} {
		t.Run(name, func(t *testing.T) {
			c := hostRetentionFixture()
			var raw bytes.Buffer
			out, err := runHostRetention(c, &raw, 10)
			if !errors.Is(err, io.EOF) {
				t.Fatal(err)
			}
			switch name {
			case "rescan-only":
				var result cudatracehost.Result
				if err := json.Unmarshal(bytes.TrimSpace(raw.Bytes()), &result); err != nil {
					t.Fatal("rescan-only raw record lost", err)
				}
				if result.ThreadRescan == nil || !reflect.DeepEqual(*result.ThreadRescan, c.final.ThreadRescans[0]) {
					t.Fatal("rescan-only raw record changed", result)
				}
				if out.Records != 1 {
					t.Fatal("serialized coverage record not bounded by count", out.Records)
				}
				if fmt.Sprintf("%x", sha256.Sum256(raw.Bytes())) != out.RawSHA256 {
					t.Fatal("coverage raw hash mismatch")
				}
			case "final-manifest":
				if !reflect.DeepEqual(out.Manifest, c.final) {
					t.Fatal("final rescan manifest lost", out.Manifest.ThreadRescans)
				}
			case "final-stats":
				var stats cudatracehost.StopStats
				if err := json.Unmarshal(hostRetentionWire(t, out)["StopStats"], &stats); err != nil || !reflect.DeepEqual(stats, c.stats) {
					t.Fatal("final ring/output stats lost", stats, err)
				}
			}
		})
	}
}
func TestHostEvidenceFinalizationErrors(t *testing.T) {
	stopErr := errors.New("controlled stop failure")
	closeErr := errors.New("controlled close failure")
	for _, test := range []struct {
		name string
		edit func(*hostEvidenceFake)
		want error
	}{
		{"stop", func(c *hostEvidenceFake) { c.stopError = stopErr; c.stats.RingStatus = "unavailable" }, stopErr},
		{"close", func(c *hostEvidenceFake) { c.closeError = closeErr }, closeErr},
		{"stop-and-close", func(c *hostEvidenceFake) { c.stopError = stopErr; c.closeError = closeErr }, stopErr},
		{"partial-output-stats", func(c *hostEvidenceFake) { c.stopError = stopErr }, stopErr},
	} {
		t.Run(test.name, func(t *testing.T) {
			c := hostRetentionFixture()
			test.edit(c)
			out, err := runHostRetention(c, io.Discard, 10)
			if !errors.Is(err, io.EOF) || !errors.Is(err, test.want) {
				t.Fatal("finalization error identity lost", err)
			}
			if c.closeError != nil && !errors.Is(err, c.closeError) {
				t.Fatal("close identity lost", err)
			}
			wire := hostRetentionWire(t, out)
			if c.stopError != nil && !bytes.Contains(wire["StopError"], []byte("controlled stop failure")) {
				t.Fatal("stop failure not visible", wire)
			}
			if c.closeError != nil && !bytes.Contains(wire["CleanupError"], []byte("controlled close failure")) {
				t.Fatal("cleanup failure not visible", wire)
			}
			if !out.Receipt.Partial || !out.Receipt.LostRecords["call"].Unknown || !out.Receipt.LostRecords["sched"].Unknown {
				t.Fatal("controlled ring counts promoted lifecycle coverage", out.Receipt)
			}
			want := []string{"initial-manifest", "next", "next", "stop", "final-manifest", "close"}
			if !reflect.DeepEqual(c.calls, want) {
				t.Fatal("finalization order", c.calls, want)
			}
		})
	}
}
func TestHostEvidenceRetentionOnExitPaths(t *testing.T) {
	captureErr := errors.New("controlled rescan error")
	writeErr := errors.New("controlled sink error")
	identityErr := errors.New("controlled identity error")
	for _, name := range []string{"rescan-error", "record-limit", "writer-error", "identity-error", "final-identity-change"} {
		t.Run(name, func(t *testing.T) {
			c := hostRetentionFixture()
			var raw bytes.Buffer
			var sink io.Writer = &raw
			max := 10
			process := func() (cudaworkload.ProcessGenID, error) {
				return cudaworkload.ProcessGenID{PID: 42, Generation: 7}, nil
			}
			switch name {
			case "rescan-error":
				c.nextErrors = []error{captureErr}
			case "record-limit":
				max = 1
			case "writer-error":
				sink = hostFailWriter{writeErr}
			case "identity-error":
				process = func() (cudaworkload.ProcessGenID, error) { return cudaworkload.ProcessGenID{}, identityErr }
			case "final-identity-change":
				c.final.Process.Generation++
			}
			out, err := collectHostEvidence(context.Background(), c, process, 43, max, sink, nil)
			if err == nil {
				t.Fatal("failure reported success")
			}
			if name == "rescan-error" && !errors.Is(err, captureErr) {
				t.Fatal(err)
			}
			if name == "writer-error" && !errors.Is(err, writeErr) {
				t.Fatal(err)
			}
			if name == "identity-error" && !errors.Is(err, identityErr) {
				t.Fatal(err)
			}
			if name == "final-identity-change" && !strings.Contains(err.Error(), "final process generation mismatch") {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out.Manifest, c.final) {
				t.Fatal("exit path lost final manifest")
			}
			if !out.Receipt.Partial {
				t.Fatal("exit path claimed complete")
			}
			if len(c.calls) < 3 || !reflect.DeepEqual(c.calls[len(c.calls)-3:], []string{"stop", "final-manifest", "close"}) {
				t.Fatal("exit cleanup ordering", c.calls)
			}
		})
	}
}

type hostFailWriter struct{ err error }

func (w hostFailWriter) Write([]byte) (int, error) { return 0, w.err }
func TestHostEvidenceFinalMetadataRoundTrip(t *testing.T) {
	c := hostRetentionFixture()
	var sink bytes.Buffer
	out, _ := runHostRetention(c, &sink, 10)
	encoded, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	var replay HostEvidence
	if err := json.Unmarshal(encoded, &replay); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(replay.Manifest, c.final) || !replay.Receipt.Partial {
		t.Fatal("replay lost partial coverage")
	}
	var stats cudatracehost.StopStats
	if err := json.Unmarshal(hostRetentionWire(t, replay)["StopStats"], &stats); err != nil || !reflect.DeepEqual(stats, c.stats) {
		t.Fatal("replay lost stats", err)
	}
}
