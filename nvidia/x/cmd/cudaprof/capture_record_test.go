package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/journal"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestWriteCaptureStatusIsPortableNoCodeRecording(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudacapture.Read(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if c.Code.Source != "" || c.Code.PTX != "" || c.Code.CUBIN != "" || c.Code.SASS != "" {
		t.Errorf("code artifacts recorded: %+v", c.Code)
	}
	if c.CodeUnavailable["source"] == "" {
		t.Errorf("source unavailability not recorded: %+v", c.CodeUnavailable)
	}
	// On a host with a visible device the capture also carries device
	// provenance, which is not a code artifact. Assert the file set by name:
	// counting files made this test pass on a machine without CUDA and fail on
	// the GPU host it is meant to describe.
	if _, ok := c.Files["capture-status.json"]; !ok {
		t.Errorf("capture-status.json missing, files = %v", slices.Sorted(maps.Keys(c.Files)))
	}
	for name := range c.Files {
		if name != "capture-status.json" && name != "capture-provenance.json" {
			t.Errorf("unexpected file %q recorded with no code", name)
		}
	}
	if err := c.Profile.CheckValid(); err != nil {
		t.Fatal(err)
	}
}

// TestWriteCaptureStatusStampsBuild checks that capture-status.json names the
// cudaprof build that wrote it, with every field present even when the build
// recorded no VCS information.
func TestWriteCaptureStatusStampsBuild(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudacapture.Read(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Status struct {
			Cudaprof map[string]string `json:"cudaprof"`
		} `json:"status"`
	}
	if err := json.Unmarshal(c.Files["capture-status.json"], &file); err != nil {
		t.Fatal(err)
	}
	stamp := file.Status.Cudaprof
	for _, key := range []string{"module", "revision", "time", "modified", "go_version"} {
		if stamp[key] == "" {
			t.Errorf("capture-status.json cudaprof.%s is empty: %v", key, stamp)
		}
	}
	if stamp["go_version"] != runtime.Version() {
		t.Errorf("cudaprof.go_version = %q, want %q", stamp["go_version"], runtime.Version())
	}
}

func TestWriteCaptureStatusDoesNotReplaceExistingOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err == nil {
		t.Fatal("second write succeeded")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatal("existing capture was replaced")
	}
}

func TestWriteCaptureStatusPreservesActivityAttributionLabels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{
		Session: "session", TargetPID: 99, Outcome: captureOutcomeCollected,
		Activities: []captureActivity{{Subtype: "memcpy", Valid: true, Start: 10, End: 20, Bytes: 8, CopyKind: 2, Device: 1, Context: 3, Stream: 4, Correlation: 5, Queued: 6, Submitted: 7}},
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudacapture.Read(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Profile.Sample) != 1 {
		t.Fatalf("samples = %d", len(c.Profile.Sample))
	}
	labels := c.Profile.Sample[0].Label
	for key, want := range map[string]string{"device": "1", "context": "3", "stream": "4", "correlation": "5", "queued": "6", "submitted": "7", "copy_kind": "2", "pid": "99", "session": "session"} {
		if got := labels[key]; len(got) != 1 || got[0] != want {
			t.Errorf("label %s = %v, want %q", key, got, want)
		}
	}
}

func TestWriteCaptureStatusPreservesKernelGeometryLabels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{
		Session: "session-geom", TargetPID: 42, Outcome: captureOutcomeCollected,
		Activities: []captureActivity{
			{
				Subtype:             "kernel",
				Name:                "vectorAdd",
				Valid:               true,
				Start:               100,
				End:                 200,
				Device:              0,
				GridDim:             [3]uint32{128, 2, 1},
				BlockDim:            [3]uint32{64, 1, 1},
				RegistersPerThread:  32,
				DynamicSharedMemory: 2048,
			},
		},
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudacapture.Read(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Profile.Sample) != 1 {
		t.Fatalf("samples = %d", len(c.Profile.Sample))
	}
	labels := c.Profile.Sample[0].Label
	for key, want := range map[string]string{
		"grid":      "128,2,1",
		"block":     "64,1,1",
		"registers": "32",
		"smem":      "2048",
		"subtype":   "kernel",
		"name":      "vectorAdd",
	} {
		if got := labels[key]; len(got) != 1 || got[0] != want {
			t.Errorf("label %s = %v, want %q", key, got, want)
		}
	}
}

func TestObserveJournalPreservesKernelGeometry(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	journalPath := filepath.Join(dir, "capture.journal")
	records := []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{
			Kind:    wire.KindActivity,
			Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"kernelArray","valid":true,"start":100,"end":200,"grid":[256,1,1],"block":[32,1,1],"registers_per_thread":48,"dynamic_shared_memory":4096}`),
		},
		{
			Kind:    wire.KindActivity,
			Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"kernelCoords","valid":true,"start":300,"end":400,"grid_x":64,"grid_y":2,"grid_z":1,"block_x":128,"block_y":1,"block_z":1,"registers":24,"smem":1024}`),
		},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
	}
	appendRecords(t, journalPath, sid, 1234, 1, records)

	_, _, _, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("observeJournal failed: %v", err)
	}
	if len(activities) != 2 {
		t.Fatalf("got %d activities, want 2", len(activities))
	}

	k1 := activities[0]
	if k1.GridDim != [3]uint32{256, 1, 1} {
		t.Errorf("k1 GridDim = %v, want [256 1 1]", k1.GridDim)
	}
	if k1.BlockDim != [3]uint32{32, 1, 1} {
		t.Errorf("k1 BlockDim = %v, want [32 1 1]", k1.BlockDim)
	}
	if k1.RegistersPerThread != 48 {
		t.Errorf("k1 RegistersPerThread = %d, want 48", k1.RegistersPerThread)
	}
	if k1.DynamicSharedMemory != 4096 {
		t.Errorf("k1 DynamicSharedMemory = %d, want 4096", k1.DynamicSharedMemory)
	}

	k2 := activities[1]
	if k2.GridDim != [3]uint32{64, 2, 1} {
		t.Errorf("k2 GridDim = %v, want [64 2 1]", k2.GridDim)
	}
	if k2.BlockDim != [3]uint32{128, 1, 1} {
		t.Errorf("k2 BlockDim = %v, want [128 1 1]", k2.BlockDim)
	}
	if k2.RegistersPerThread != 24 {
		t.Errorf("k2 RegistersPerThread = %d, want 24", k2.RegistersPerThread)
	}
	if k2.DynamicSharedMemory != 1024 {
		t.Errorf("k2 DynamicSharedMemory = %d, want 1024", k2.DynamicSharedMemory)
	}
}

func TestWriteCaptureStatusRejectsUnrepresentableCopyBytes(t *testing.T) {
	const maxInt64 = uint64(^uint64(0) >> 1)
	for _, tc := range []struct {
		name  string
		bytes uint64
		want  bool
	}{
		{name: "max-int64", bytes: maxInt64, want: false},
		{name: "max-int64-plus-one", bytes: maxInt64 + 1, want: true},
		{name: "max-uint64", bytes: ^uint64(0), want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "capture.pprof")
			result := captureResult{Activities: []captureActivity{{Subtype: "memcpy", Valid: true, Start: 1, End: 2, Bytes: tc.bytes}}}
			err := writeCaptureStatus(path, result, captureArtifactBudget{})
			if (err != nil) != tc.want {
				t.Fatalf("writeCaptureStatus error = %v, want error %v", err, tc.want)
			}
			if tc.want {
				if _, statErr := os.Stat(path); !errors.Is(statErr, os.ErrNotExist) {
					t.Fatalf("output stat error = %v, want absent output", statErr)
				}
			}
		})
	}

	path := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(path, captureResult{}, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	err = writeCaptureStatus(path, captureResult{Activities: []captureActivity{{Subtype: "memcpy", Valid: true, Start: 1, End: 2, Bytes: maxInt64 + 1}}}, captureArtifactBudget{})
	if err == nil {
		t.Fatal("overflow write succeeded")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatal("overflow write replaced existing output")
	}
}

// TestRunCaptureCommandNamesTargetSignal checks the line a caller sees when
// the target dies by a signal: the signal's name, not its description.
func TestRunCaptureCommandNamesTargetSignal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	old := os.Args
	os.Args = []string{"cudaprof", "capture", "--no-inject", "-o", path, "--", "/bin/sh", "-c", "kill -SEGV $$"}
	t.Cleanup(func() { os.Args = old })
	err := runCaptureCommand(os.Args[2:])
	var exitErr *exitCodeError
	if !errors.As(err, &exitErr) || exitErr.Code != 128 || err.Error() != "target died by SIGSEGV" {
		t.Fatalf("error = %v, want exit 128 and \"target died by SIGSEGV\"", err)
	}
}

func TestRunCaptureCommandPreservesTargetFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	old := os.Args
	os.Args = []string{"cudaprof", "capture", "--no-inject", "-o", path, "--", "/bin/sh", "-c", "exit 7"}
	t.Cleanup(func() { os.Args = old })
	err := runCaptureCommand(os.Args[2:])
	var exitErr *exitCodeError
	if !errors.As(err, &exitErr) || exitErr.Code != 7 {
		t.Fatalf("error = %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
	// Confirm the status JSON is retained inside the recording.
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudacapture.Read(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Metadata) != 0 {
		t.Fatalf("status duplicated into Capture.Metadata: %d bytes", len(c.Metadata))
	}
	var status struct {
		Status captureResult `json:"status"`
	}
	if err := json.Unmarshal(c.Files["capture-status.json"], &status); err != nil {
		t.Fatal(err)
	}
	if status.Status.TargetExitCode != 7 {
		t.Fatalf("status = %+v", status.Status)
	}
}

func TestRunCaptureCommandRejectsUnverifiedSuccessfulTarget(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.pprof")
	err := runCaptureCommand([]string{"-o", path, "--", "true"})
	var exitErr *exitCodeError
	if !errors.As(err, &exitErr) || exitErr.Code != 2 {
		t.Fatalf("error = %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestObserveCaptureSessionRequiresVerifiedCompletion(t *testing.T) {
	dir := t.TempDir()
	base := captureResult{Outcome: captureOutcomeNoCUDA}
	got := observeCaptureSession(base, dir, "00112233445566778899aabbccddeeff", observeOptions{})
	if got.CaptureSuccess || got.TransportReady || got.BootstrapSeen {
		t.Fatalf("missing handshake = %+v", got)
	}
	h := captureHandshake{Protocol: 1, Session: "00112233445566778899aabbccddeeff", Process: 41, Generation: 2, TransportReady: true}
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	decoded, _ := hex.DecodeString(h.Session)
	copy(sid[:], decoded)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range []wire.Record{
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"state":"no_cuda_observed"}`)},
	} {
		if err := j.Append(record); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	base.TargetPID = 41
	got = observeCaptureSession(base, dir, h.Session, observeOptions{})
	if got.Outcome != captureOutcomeObserved || !got.TransportReady || !got.BootstrapSeen {
		t.Fatalf("verified no-cuda = %+v", got)
	}
	// Verified is not successful: the capture holds no records.
	if got.CaptureSuccess || got.RunCompleteness != runEmpty || !strings.Contains(got.CaptureError, "no activity records") {
		t.Fatalf("verified no-cuda reported success: %+v", got)
	}
}

func TestObserveCaptureSessionIsolatesWorkerGenerations(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	writeStream := func(t *testing.T, name string, process, generation uint64, activity string) {
		t.Helper()
		streamDir := filepath.Join(dir, name)
		if err := os.MkdirAll(streamDir, 0700); err != nil {
			t.Fatal(err)
		}
		h := captureHandshake{Protocol: 1, Session: session, Process: process, Generation: generation, TransportReady: true}
		b, _ := json.Marshal(h)
		if err := os.WriteFile(filepath.Join(streamDir, "handshake.json"), b, 0600); err != nil {
			t.Fatal(err)
		}
		j, err := journal.Create(filepath.Join(streamDir, "capture.journal"), sid, journal.Options{})
		if err != nil {
			t.Fatal(err)
		}
		records := []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		}
		if activity != "" {
			records = append(records, wire.Record{Kind: wire.KindActivity, Payload: []byte(activity)})
		}
		records = append(records, wire.Record{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected"}`)})
		for i := range records {
			records[i].Identity = wire.Identity{Session: sid, Process: process, Generation: generation}
			records[i].Sequence = uint64(i + 1)
			if err := j.Append(records[i]); err != nil {
				t.Fatal(err)
			}
		}
		if err := j.Close(); err != nil {
			t.Fatal(err)
		}
	}
	writeStream(t, "process-41-2", 41, 2, `{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)
	writeStream(t, "process-42-3", 42, 3, `{"kind":"activity","subtype":"kernel","valid":true,"start":30,"end":50}`)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if !got.CaptureSuccess || got.ActivityRecords != 2 || len(got.Activities) != 2 {
		t.Fatalf("target capture = %+v", got)
	}
	if got.Activities[1].Process != 42 {
		t.Fatalf("worker activity process = %+v", got.Activities)
	}
	if len(got.WorkerGenerations) != 1 || got.WorkerGenerations[0].Process != 42 || got.WorkerGenerations[0].Disposition != "finalized" {
		t.Fatalf("worker manifest = %+v", got.WorkerGenerations)
	}
}

// TestObserveCaptureSessionStatesTheProducersItDoesNotDescribe goes journal to
// published record for the multi-process case.
//
// The session merges every process's activity records into one set and
// publishes one capability record, taken from the target's journal. Whether
// that record describes the merged work depends on whether the other processes
// ran the same agent, and until producer_scope the published record said the
// same thing either way. The .so is resolved next to each executable, so this
// is not hypothetical: a capture that spawns a second binary can load a second
// agent.
//
// Both arms run here because either alone is satisfiable by a wrong
// implementation: an unconditional annotation passes the disagreement arm, and
// no annotation at all passes the agreement arm.
func TestObserveCaptureSessionStatesTheProducersItDoesNotDescribe(t *testing.T) {
	session := "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	// observe writes one journal per process, each naming its own agent, and
	// returns the capability record the session publishes.
	observe := func(t *testing.T, targetRevision, workerRevision string) []byte {
		t.Helper()
		dir := t.TempDir()
		writeStream := func(name string, process, generation uint64, revision string) {
			streamDir := filepath.Join(dir, name)
			if err := os.MkdirAll(streamDir, 0700); err != nil {
				t.Fatal(err)
			}
			h := captureHandshake{Protocol: 1, Session: session, Process: process, Generation: generation, TransportReady: true}
			b, _ := json.Marshal(h)
			if err := os.WriteFile(filepath.Join(streamDir, "handshake.json"), b, 0600); err != nil {
				t.Fatal(err)
			}
			j, err := journal.Create(filepath.Join(streamDir, "capture.journal"), sid, journal.Options{})
			if err != nil {
				t.Fatal(err)
			}
			capability := `{"kind":"capability","memsets":true}`
			if revision != "" {
				capability = fmt.Sprintf(`{"kind":"capability","agent_revision":%q,"memsets":true}`, revision)
			}
			records := []wire.Record{
				{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: wire.KindCapability, Payload: []byte(capability)},
				{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
				{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected"}`)},
			}
			for i := range records {
				records[i].Identity = wire.Identity{Session: sid, Process: process, Generation: generation}
				records[i].Sequence = uint64(i + 1)
				if err := j.Append(records[i]); err != nil {
					t.Fatal(err)
				}
			}
			if err := j.Close(); err != nil {
				t.Fatal(err)
			}
		}
		writeStream("process-41-2", 41, 2, targetRevision)
		writeStream("process-42-3", 42, 3, workerRevision)
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
		if !got.CaptureSuccess || len(got.Activities) != 2 {
			t.Fatalf("capture = %+v", got)
		}
		return got.Capability
	}

	same := observe(t, "c0ffee1", "c0ffee1")
	if bytes.Contains(same, []byte("producer_scope")) {
		t.Errorf("one agent across both processes, annotated anyway: %s\n\ta caveat on every capture is one a reader learns to skip", same)
	}
	if !bytes.Contains(same, []byte(`"agent_revision":"c0ffee1"`)) {
		t.Fatalf("published capability lost the agent revision: %s", same)
	}

	mixed := observe(t, "c0ffee1", "deadbee")
	var m map[string]json.RawMessage
	if err := json.Unmarshal(mixed, &m); err != nil {
		t.Fatalf("published capability does not parse: %v: %s", err, mixed)
	}
	var scope producerScope
	if rawScope, ok := m["producer_scope"]; !ok {
		t.Fatalf("two agents contributed records and the published capability says nothing: %s", mixed)
	} else if err := json.Unmarshal(rawScope, &scope); err != nil {
		t.Fatalf("producer_scope does not parse: %v", err)
	}
	if scope.Described != "c0ffee1" {
		t.Errorf("described = %q, want the target's agent %q", scope.Described, "c0ffee1")
	}
	want := []otherProducer{{Process: 42, Revision: "deadbee", Records: 1}}
	if !slices.Equal(scope.Others, want) {
		t.Errorf("others = %+v, want %+v", scope.Others, want)
	}
	if !strings.Contains(scope.Note, "deadbee") {
		t.Errorf("note does not name the agent it did not describe: %s", scope.Note)
	}
}

func TestObserveCaptureSessionRejectsHandshakeGenerationMismatch(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	h := captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: true}
	b, _ := json.Marshal(h)
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i, record := range []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"no_cuda_observed"}`)},
	} {
		record.Identity = wire.Identity{Session: sid, Process: 41, Generation: 3}
		record.Sequence = uint64(i + 1)
		if err := j.Append(record); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomeTransport {
		t.Fatalf("generation mismatch promoted capture: %+v", got)
	}
}

func TestObserveCaptureSessionAcceptsIncompletePreloadGeneration(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	h := captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: true}
	b, _ := json.Marshal(h)
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	records := []wire.Record{
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 1}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 1}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"state":"no_cuda_observed"}`)},
	}
	for i := range records {
		records[i].Sequence = uint64(i + 1)
		if err := j.Append(records[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomeObserved || len(got.WorkerGenerations) != 1 {
		t.Fatalf("preload generation = %+v", got)
	}
	if got.WorkerGenerations[0].Generation != 1 || got.WorkerGenerations[0].Disposition != "incomplete" {
		t.Fatalf("preload disposition = %+v", got.WorkerGenerations)
	}
}

func TestObserveCaptureSessionDoesNotPromoteInitializedNoActivity(t *testing.T) {
	dir := t.TempDir()
	h := captureHandshake{Protocol: 1, Session: "00112233445566778899aabbccddeeff", Process: 41, Generation: 2, TransportReady: true}
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	raw, _ := hex.DecodeString(h.Session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for seq, record := range []wire.Record{
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"state":"initialized_no_activity"}`)},
	} {
		record.Sequence = uint64(seq + 1)
		if err := j.Append(record); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, h.Session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != "initialized-no-activity" {
		t.Fatalf("initialized no activity = %+v", got)
	}
}

func TestObserveCaptureSessionRejectsIdentityMismatch(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), []byte(`{"protocol":1,"session":"bad","process":1,"generation":1,"transport_ready":true,"bootstrap_seen":true,"collector_finalized":true,"cuda_observed":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{})
	if got.Outcome != captureOutcomeTransport || got.CaptureSuccess {
		t.Fatalf("mismatch = %+v", got)
	}
}

func TestObserveCaptureSessionRequiresCompleteJournal(t *testing.T) {
	dir := t.TempDir()
	h := captureHandshake{Protocol: 1, Session: "00112233445566778899aabbccddeeff", Process: 41, Generation: 2}
	b, _ := json.Marshal(h)
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, h.Session, observeOptions{})
	if got.CaptureSuccess || got.Outcome != captureOutcomeTransport {
		t.Fatalf("missing journal = %+v", got)
	}

	var sid [16]byte
	raw, _ := hex.DecodeString(h.Session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := j.Append(wire.Record{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)}); err != nil {
		t.Fatal(err)
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(filepath.Join(dir, "capture.journal"), os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.Write([]byte{1, 2, 3})
	_ = f.Close()
	got = observeCaptureSession(captureResult{TargetPID: 41}, dir, h.Session, observeOptions{})
	if got.CaptureSuccess || got.Outcome != captureOutcomeTransport {
		t.Fatalf("truncated journal = %+v", got)
	}
}

func TestObserveCaptureSessionReportsUnsupportedCUPTI(t *testing.T) {
	dir := t.TempDir()
	h := captureHandshake{Protocol: 1, Session: "00112233445566778899aabbccddeeff", Process: 41, Generation: 2}
	b, _ := json.Marshal(h)
	_ = os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600)
	var sid [16]byte
	raw, _ := hex.DecodeString(h.Session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for _, record := range []wire.Record{
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindFailure, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"state":"cupti_unavailable"}`)},
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 4, Payload: []byte(`{"state":"initialized_no_activity"}`)},
	} {
		if err := j.Append(record); err != nil {
			t.Fatal(err)
		}
	}
	_ = j.Close()
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, h.Session, observeOptions{})
	if got.CaptureSuccess || got.Outcome != captureOutcomeUnsupported {
		t.Fatalf("unsupported CUPTI = %+v", got)
	}
}

func TestObserveCaptureSessionRejectsTerminalOrderAndUnknownSubtype(t *testing.T) {
	for _, tc := range []struct {
		name    string
		records []wire.Record
	}{
		{name: "activity after finalize", records: []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
			{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","start":1,"end":2}`)},
		}},
		{name: "unknown subtype", records: []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"bogus","start":1,"end":2}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected"}`)},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			session := "00112233445566778899aabbccddeeff"
			if err := os.WriteFile(filepath.Join(dir, "handshake.json"), []byte(`{"protocol":1,"session":"00112233445566778899aabbccddeeff","process":41,"generation":2}`), 0600); err != nil {
				t.Fatal(err)
			}
			var sid [16]byte
			raw, _ := hex.DecodeString(session)
			copy(sid[:], raw)
			j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
			if err != nil {
				t.Fatal(err)
			}
			for i, r := range tc.records {
				r.Identity = wire.Identity{Session: sid, Process: 41, Generation: 2}
				r.Sequence = uint64(i + 1)
				if err := j.Append(r); err != nil {
					t.Fatal(err)
				}
			}
			if err := j.Close(); err != nil {
				t.Fatal(err)
			}
			got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{})
			if got.Outcome != captureOutcomeTransport || got.CaptureSuccess {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestObserveCaptureSessionRejectsInvalidIntervals(t *testing.T) {
	for _, payload := range []string{
		`{"kind":"activity","subtype":"kernel","valid":false,"start":10,"end":20}`,
		`{"kind":"activity","subtype":"kernel","valid":true,"start":20,"end":10}`,
		`{"kind":"activity","subtype":"kernel","valid":true,"start":0,"end":20}`,
	} {
		t.Run(payload, func(t *testing.T) {
			dir := t.TempDir()
			session := "00112233445566778899aabbccddeeff"
			if err := os.WriteFile(filepath.Join(dir, "handshake.json"), []byte(`{"protocol":1,"session":"00112233445566778899aabbccddeeff","process":41,"generation":2}`), 0600); err != nil {
				t.Fatal(err)
			}
			var sid [16]byte
			raw, _ := hex.DecodeString(session)
			copy(sid[:], raw)
			j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
			if err != nil {
				t.Fatal(err)
			}
			for i, r := range []wire.Record{
				{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
				{Kind: wire.KindActivity, Payload: []byte(payload)},
				{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected"}`)},
			} {
				r.Identity = wire.Identity{Session: sid, Process: 41, Generation: 2}
				r.Sequence = uint64(i + 1)
				if err := j.Append(r); err != nil {
					t.Fatal(err)
				}
			}
			if err := j.Close(); err != nil {
				t.Fatal(err)
			}
			got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{})
			if got.CaptureSuccess || got.Outcome != captureOutcomeTransport {
				t.Fatalf("invalid interval accepted: %+v", got)
			}
		})
	}
}

func TestObserveCaptureSessionSurfacesCUPTIVersionMismatch(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	h := captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: true}
	b, _ := json.Marshal(h)
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	// Real agent sequence when CUPTI version mismatch occurs in init():
	// The agent records the failure immediately, never emits bootstrap/capability,
	// and emits an initialized_no_activity finalized record at process exit.
	records := []wire.Record{
		{Kind: wire.KindFailure, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"state":"cupti_unavailable","error":"CUPTI runtime/header version mismatch"}`)},
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"state":"initialized_no_activity","trigger":"exit"}`)},
	}
	for _, r := range records {
		if err := j.Append(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomeUnsupported {
		t.Fatalf("outcome = %q, want %q; got: %+v", got.Outcome, captureOutcomeUnsupported, got)
	}
	if !strings.Contains(got.CaptureError, "CUPTI runtime/header version mismatch") {
		t.Fatalf("captureError = %q, want containing version mismatch", got.CaptureError)
	}
	if strings.Contains(got.CaptureError, "invalid finalized ordering") {
		t.Fatalf("captureError falsely reported decode ordering error: %q", got.CaptureError)
	}
	if strings.Contains(got.CaptureError, "no identity-matching target handshake") {
		t.Fatalf("captureError falsely reported handshake missing: %q", got.CaptureError)
	}
}

func TestObserveCaptureSessionSurfacesInvalidActivityFailure(t *testing.T) {
	dir := t.TempDir()
	session := "00112233445566778899aabbccddeeff"
	h := captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: true}
	b, _ := json.Marshal(h)
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	j, err := journal.Create(filepath.Join(dir, "capture.journal"), sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	records := []wire.Record{
		{Kind: wire.KindBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k1"}`)}, // no valid field
		{Kind: wire.KindFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 4, Payload: []byte(`{"state":"collected"}`)},
	}
	for _, r := range records {
		if err := j.Append(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomeTransport {
		t.Fatalf("outcome = %q, want %q; got: %+v", got.Outcome, captureOutcomeTransport, got)
	}
	if !strings.Contains(got.CaptureError, "collected finalization without valid activity") {
		t.Fatalf("captureError = %q, want 'collected finalization without valid activity'", got.CaptureError)
	}
	if strings.Contains(got.CaptureError, "no identity-matching target handshake") {
		t.Fatalf("captureError falsely reported handshake missing: %q", got.CaptureError)
	}
}

func TestRunCaptureCommandOutputsCompactStatusWithoutRecordArrays(t *testing.T) {
	// Verify that compactCaptureStatus strips Activities and APIRecords arrays.
	res := captureResult{
		Protocol:       1,
		Session:        "test-session",
		CaptureSuccess: true,
		Activities: []captureActivity{
			{Name: "k1", Subtype: "kernel", Device: 0, Start: 100, End: 200},
			{Name: "k2", Subtype: "kernel", Device: 0, Start: 200, End: 400},
		},
		APIRecords: []captureAPI{
			{Domain: "runtime", CallbackID: 1},
		},
	}
	status := compactCaptureStatus(res)
	if status.ActivitySamples != 2 {
		t.Fatalf("ActivitySamples = %d, want 2", status.ActivitySamples)
	}
	if status.APIRecordCount != 1 {
		t.Fatalf("APIRecordCount = %d, want 1", status.APIRecordCount)
	}
	b, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	str := string(b)
	if strings.Contains(str, `"activities":`) {
		t.Errorf("compact status JSON unexpectedly contains activities array: %s", str)
	}
	if strings.Contains(str, `"api_records":`) {
		t.Errorf("compact status JSON unexpectedly contains api_records array: %s", str)
	}
	if !strings.Contains(str, `"activity_samples":2`) {
		t.Errorf("compact status JSON missing activity_samples count: %s", str)
	}
}

// TestObserveJournalKeepsNonTimelineSubtypes covers the two activity subtypes
// that are not device work: NVTX markers and capture-range edges. Both are
// written to the same record stream as kernels and memcpys, and observeJournal
// rejected the whole capture on the first one it saw.
//
// The two are not kept alike. A marker carries the name and the timestamps that
// nvtxRanges pairs into a range, so it must reach the activity slice or NVTX
// labelling has nothing to read. A capture-range edge is an edge, an api and an
// index, none of which ActivityRecord has a field for, so carrying one would
// keep a record that has lost everything that made it meaningful.
//
// The marker arm is not hypothetical. CUPTI's MARKER kind is enabled whenever
// the runtime supports it (cupti_agent.c, cuptiActivityEnable(MARKER)) rather
// than on request, and PyTorch, cuBLAS and NCCL all emit NVTX, so a capture of
// any of them carries marker records.
//
// The payloads below are the producer's, field for field: subtype is tested
// before validity, so a fixture that trimmed them would still exercise the
// rejection, but it would stop proving that the records a real run emits are
// the records this accepts.
func TestObserveJournalKeepsNonTimelineSubtypes(t *testing.T) {
	for _, tt := range []struct {
		name    string
		payload string
		carried bool // the record reaches the activity slice
	}{
		{"marker", `{"kind":"activity","subtype":"marker","name":"iteration","marker_domain":"torch","valid":true,"start":150,"end":150,"marker_id":3,"marker_flags":2,"object_kind":2,"process_id":1234,"thread_id":5678,"device":0,"context":0,"stream":0}`, true},
		{"capture_range", `{"kind":"activity","subtype":"capture_range","edge":"start","api":"cudaProfilerStart","index":0,"toggle_failures":0}`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			sessionHex := "0123456789abcdef0123456789abcdef"
			raw, err := hex.DecodeString(sessionHex)
			if err != nil {
				t.Fatal(err)
			}
			var sid [16]byte
			copy(sid[:], raw)

			journalPath := filepath.Join(dir, "capture.journal")
			appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
				{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
				{Kind: wire.KindActivity, Payload: []byte(tt.payload)},
				{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
				{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
			})

			_, _, _, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
			if err != nil {
				t.Fatalf("a %s record cost the whole capture: %v", tt.name, err)
			}
			want := 1
			if tt.carried {
				want = 2
			}
			if len(activities) != want {
				t.Fatalf("got %d activities, want %d for a %s record", len(activities), want, tt.name)
			}
			if !tt.carried {
				return
			}
			// Carried is not enough: the fields nvtxRanges pairs on must
			// survive the hop through the controller's payload struct, which
			// is where they were dropped.
			var marker captureActivity
			for _, a := range activities {
				if a.Subtype == "marker" {
					marker = a
				}
			}
			if marker.Name != "iteration" || marker.MarkerDomain != "torch" {
				t.Errorf("marker name/domain = %q/%q, want %q/%q", marker.Name, marker.MarkerDomain, "iteration", "torch")
			}
			if marker.MarkerID != 3 || marker.MarkerFlags != 2 {
				t.Errorf("marker id/flags = %d/%d, want 3/2", marker.MarkerID, marker.MarkerFlags)
			}
			if marker.ProcessID != 1234 || marker.ThreadID != 5678 {
				t.Errorf("marker pid/tid = %d/%d, want 1234/5678", marker.ProcessID, marker.ThreadID)
			}
		})
	}
}

// TestObserveJournalStillRejectsUnknownSubtypes pins the other half: admitting
// the two subtypes above must not turn the check into one that admits anything.
func TestObserveJournalStillRejectsUnknownSubtypes(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, _ := hex.DecodeString(sessionHex)
	var sid [16]byte
	copy(sid[:], raw)
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"teleport","valid":true,"start":100,"end":200}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})
	if _, _, _, _, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0); err == nil {
		t.Error("an unrecognized activity subtype was accepted")
	}
}

// TestObserveJournalCarriesMarkersToNVTXLabels walks the path production takes:
// journal bytes -> observeJournal -> BuildActivityProfile -> pprof labels.
//
// The NVTX labelling unit tests hand BuildActivityProfile marker records
// directly. That is the one thing the capture path does not do, so those tests
// passed while the label could never fire on a real capture: the controller
// dropped markers, the profile was built from what the controller returned, and
// the ranges are paired from that same slice. Nothing failed anywhere -- the
// label was simply always absent.
//
// The API records ride the same path: a kernel is placed by the host call
// that launched it, so the controller has to return them and the caller has
// to pass them on. The device times are chosen so that placing either kernel
// by its device start gets it wrong: "early" runs inside the range but was
// launched before it opened, "inside" was launched inside it but runs after
// it closed.
func TestObserveJournalCarriesMarkersToNVTXLabels(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	// markerFlags: 1<<1 opens a range, 1<<2 closes it.
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindAPI, Payload: []byte(`{"kind":"api","domain":"runtime","callback_id":211,"start":50,"end":60,"process_id":1234,"thread_id":7,"correlation":1}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"marker","name":"decode","marker_domain":"ds4","valid":true,"start":100,"end":100,"marker_id":1,"marker_flags":2,"process_id":1234,"thread_id":7}`)},
		{Kind: wire.KindAPI, Payload: []byte(`{"kind":"api","domain":"runtime","callback_id":211,"start":150,"end":160,"process_id":1234,"thread_id":7,"correlation":2}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"early","valid":true,"start":200,"end":300,"device":0,"correlation":1}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"marker","name":"decode","marker_domain":"ds4","valid":true,"start":400,"end":400,"marker_id":1,"marker_flags":4,"process_id":1234,"thread_id":7}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"inside","valid":true,"start":500,"end":600,"device":0,"correlation":2}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":4}`)},
	})

	_, _, _, activities, apis, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("observeJournal: %v", err)
	}
	p := cudacapture.BuildActivityProfile(sessionHex, 1234, activities, apis)

	got := map[string][]string{}
	for _, s := range p.Sample {
		if n := s.Label["name"]; len(n) == 1 {
			got[n[0]] = s.Label["nvtx"]
		}
	}
	if len(p.Sample) != 2 {
		t.Fatalf("got %d samples, want 2: only the two kernels are device work", len(p.Sample))
	}
	if want := []string{"ds4/decode"}; !reflect.DeepEqual(got["inside"], want) {
		t.Errorf("kernel launched inside the range has nvtx = %v, want %v", got["inside"], want)
	}
	if got["early"] != nil {
		t.Errorf("kernel launched before the range opened has nvtx = %v, want none", got["early"])
	}
}

// TestObserveJournalCarriesMemsets covers the consumer half of the MEMSET
// tier. The subtype check in observeJournal is an allowlist, so the producer
// side cannot ship without this one: a capture of any target that calls
// cudaMemsetAsync would be rejected outright rather than degraded.
//
// The payload is the producer's, field for field, including the fields a
// memset does not have. A memset carries no name and no launch geometry, and a
// fixture that omitted them would not show that the zero grid and block survive
// the hop without being mistaken for a kernel's.
func TestObserveJournalCarriesMemsets(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"memset","name":"","valid":true,"start":300,"end":420,"bytes":8192,"queued":0,"submitted":0,"copy_kind":0,"value":7,"runtime_correlation":0,"device":1,"context":1,"stream":7,"correlation":9,"grid":[0,0,0],"block":[0,0,0],"registers_per_thread":0,"dynamic_shared_memory":0,"graph_id":0,"graph_node_id":0}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("a memset record cost the whole capture: %v", err)
	}
	if len(activities) != 2 {
		t.Fatalf("got %d activities, want 2", len(activities))
	}
	var memset captureActivity
	var found bool
	for _, a := range activities {
		if a.Subtype == "memset" {
			memset, found = a, true
		}
	}
	if !found {
		t.Fatal("no memset in the activity slice")
	}
	if !memset.Valid {
		t.Error("memset valid = false")
	}
	// The span is what moves idle time into busy time downstream. Everything
	// else on this record is detail; this is the reason the tier exists.
	if memset.Start != 300 || memset.End != 420 {
		t.Errorf("memset span = %d..%d, want 300..420", memset.Start, memset.End)
	}
	if memset.Bytes != 8192 {
		t.Errorf("memset bytes = %d, want 8192", memset.Bytes)
	}
	if memset.Value != 7 {
		t.Errorf("memset value = %d, want 7: the fill byte was dropped between the producer and the record", memset.Value)
	}
	if memset.Device != 1 || memset.Stream != 7 || memset.Correlation != 9 {
		t.Errorf("memset device/stream/correlation = %d/%d/%d, want 1/7/9", memset.Device, memset.Stream, memset.Correlation)
	}
	if memset.GridDim != [3]uint32{0, 0, 0} || memset.BlockDim != [3]uint32{0, 0, 0} {
		t.Errorf("memset grid/block = %v/%v, want zero: a memset has no launch geometry", memset.GridDim, memset.BlockDim)
	}
	// A memset writes bytes; it does not copy them. Charging them to the copy
	// total would inflate a figure that names a transfer.
	if metrics.copyBytes != 0 {
		t.Errorf("copy bytes = %d, want 0: a memset is not a transfer", metrics.copyBytes)
	}
	// It is device work with a duration, so its time belongs in the duration
	// total beside the kernel's: 120 for the memset, 100 for the kernel.
	if metrics.duration != 220 {
		t.Errorf("duration = %d, want 220 (100 kernel + 120 memset)", metrics.duration)
	}
}

// externalJournal writes a journal holding one external-correlation record for
// externalID against correlation 9, a kernel on that correlation, and a second
// kernel on a correlation nothing pushed an ID for. It returns the activities
// observeJournal decoded from it.
func externalJournal(t *testing.T, sessionHex string, externals ...string) []captureActivity {
	t.Helper()
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)
	records := []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
	}
	for _, e := range externals {
		records = append(records, wire.Record{Kind: wire.KindExternal, Payload: []byte(e)})
	}
	records = append(records,
		wire.Record{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"attributed","valid":true,"start":100,"end":200,"device":0,"correlation":9}`)},
		wire.Record{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"unattributed","valid":true,"start":300,"end":400,"device":0,"correlation":11}`)},
		wire.Record{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
	)
	journalPath := filepath.Join(t.TempDir(), "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, records)
	_, _, _, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("observeJournal: %v", err)
	}
	return activities
}

// externalLabels reads the per-kernel external labels, and the profile
// comments, out of a written capture.
func externalLabels(t *testing.T, activities []captureActivity, sessionHex string) (map[string]map[string]string, []string) {
	t.Helper()
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Session: sessionHex, TargetPID: 1234, Activities: activities, Outcome: captureOutcomeCollected, CaptureSuccess: true}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatalf("writeCaptureStatus: %v", err)
	}
	c, _ := readWrittenCapture(t, out)
	if c.Profile == nil {
		t.Fatal("written capture holds no profile")
	}
	labels := map[string]map[string]string{}
	for _, s := range c.Profile.Sample {
		name := ""
		if n := s.Label["name"]; len(n) == 1 {
			name = n[0]
		}
		got := map[string]string{}
		for k, v := range s.Label {
			if strings.HasPrefix(k, "cupti.external.") || strings.HasPrefix(k, "cupti.collection.") {
				got[k] = strings.Join(v, ",")
			}
		}
		labels[name] = got
	}
	return labels, c.Profile.Comments
}

// TestCaptureProfileCarriesExternalCorrelation walks the whole path the
// framework-attribution feature needs: journal bytes -> observeJournal ->
// writeCaptureStatus -> the published pprof profile. The join it reaches
// (cuptiactivity.MatchExternal, through cudaprof.LabelExternal) was written and
// unit-tested with zero production callers, so every one of its own tests
// passed while no capture ever carried an external ID to it: the controller
// counted the records and dropped them.
//
// The second kernel is the control. Its correlation ID is one nothing pushed an
// external ID for, so a capture that starts claiming a match for it is claiming
// one the producer never recorded.
func TestCaptureProfileCarriesExternalCorrelation(t *testing.T) {
	const sessionHex = "0123456789abcdef0123456789abcdef"
	activities := externalJournal(t, sessionHex, `{"kind":"external_correlation","external_kind":3,"external_id":770,"correlation":9}`)
	labels, comments := externalLabels(t, activities, sessionHex)

	if got := labels["attributed"]["cupti.external.id"]; got != "770" {
		t.Errorf("kernel on the pushed correlation has cupti.external.id = %q, want %q", got, "770")
	}
	if got := labels["attributed"]["cupti.external.status"]; got != "matched" {
		t.Errorf("kernel on the pushed correlation has status %q, want matched", got)
	}
	if got := labels["attributed"]["cupti.external.kind"]; got != "3" {
		t.Errorf("labelled namespace = %q, want 3 (the kind the producer recorded)", got)
	}
	if got := labels["attributed"]["cupti.collection.session"]; got != sessionHex {
		t.Errorf("cupti.collection.session = %q, want the capture session", got)
	}
	if got := labels["attributed"]["cupti.collection.process"]; got != "1234" {
		t.Errorf("cupti.collection.process = %q, want the target PID", got)
	}

	if got, ok := labels["unattributed"]["cupti.external.id"]; ok {
		t.Errorf("kernel on an unpushed correlation claims cupti.external.id = %q", got)
	}
	if got := labels["unattributed"]["cupti.external.status"]; got != "not_observed" {
		t.Errorf("kernel on an unpushed correlation has status %q, want not_observed", got)
	}
	// Independence from the test below: a capture holding one namespace must
	// carry no omission note, or the note is boilerplate rather than a
	// warning and a reader stops reading it.
	for _, c := range comments {
		if strings.HasPrefix(c, "cupti: ") {
			t.Errorf("single-namespace capture carries the omission note %q", c)
		}
	}
}

// TestCaptureProfileWithoutExternalIDsClaimsNothing is the negative control for
// the test above: a capture whose target pushed no external IDs at all must
// publish no external labels, rather than a namespace it invented or a match it
// did not observe.
func TestCaptureProfileWithoutExternalIDsClaimsNothing(t *testing.T) {
	const sessionHex = "0123456789abcdef0123456789abcdef"
	labels, _ := externalLabels(t, externalJournal(t, sessionHex), sessionHex)
	if len(labels) != 2 {
		t.Fatalf("got %d kernels in the profile, want 2", len(labels))
	}
	for name, got := range labels {
		if len(got) != 0 {
			t.Errorf("kernel %q carries external labels %v from a capture with no external records", name, got)
		}
	}
}

// TestCaptureProfileStatesTheNamespaceItDidNotJoin covers the omission. A
// target that pushes into two external namespaces -- PyTorch pushes Custom0
// and Custom1 -- gets one of them joined, because the profile holds one
// cupti.external.kind per sample. The records in the other namespace are real
// collected data that this tool chose not to attribute, and a reader who sees
// only the coverage number reads that choice as work nothing was pushed for.
//
// The note is the difference between "nothing was pushed there" and "we did
// not join it".
func TestCaptureProfileStatesTheNamespaceItDidNotJoin(t *testing.T) {
	const sessionHex = "0123456789abcdef0123456789abcdef"
	activities := externalJournal(t, sessionHex,
		`{"kind":"external_correlation","external_kind":3,"external_id":770,"correlation":9}`,
		`{"kind":"external_correlation","external_kind":4,"external_id":880,"correlation":9}`,
		`{"kind":"external_correlation","external_kind":4,"external_id":881,"correlation":11}`)
	labels, comments := externalLabels(t, activities, sessionHex)

	if got := labels["attributed"]["cupti.external.id"]; got != "770" {
		t.Errorf("joined namespace = %q, want Custom0's id 770: the lowest-numbered namespace is the one joined", got)
	}
	want := "cupti: 2 external namespaces present (Custom0, Custom1); only Custom0 was joined, 2 record(s) in Custom1 left unattributed"
	found := false
	for _, c := range comments {
		if c == want {
			found = true
		}
	}
	if !found {
		t.Errorf("profile comments = %q, want one reading %q", comments, want)
	}
}

// TestCompactStatusKeepsExternalLinksOutOfTheDeviceList covers the aggregate
// an external record would otherwise pollute. The link carries no device -- it
// names a correlation id, not hardware -- so its zero Device field would
// report device 0 on a capture whose device work all ran elsewhere. This is
// the same rule the marker branch beside it applies, and it is only reachable
// now that these records travel in the activity slice.
func TestCompactStatusKeepsExternalLinksOutOfTheDeviceList(t *testing.T) {
	status := compactCaptureStatus(captureResult{Activities: []captureActivity{
		{Subtype: "kernel", Valid: true, Start: 100, End: 200, Device: 3, Correlation: 9},
		{Subtype: "external_correlation", Correlation: 9, ExternalKind: 3, ExternalID: 770},
	}})
	if len(status.Devices) != 1 || status.Devices[0] != 3 {
		t.Errorf("devices = %v, want [3]: an external correlation link names no device", status.Devices)
	}
	if status.ActivityKinds["external_correlation"] != 1 {
		t.Errorf("activity_kinds = %v, want the link still counted by subtype", status.ActivityKinds)
	}
}

// TestObserveJournalCarriesHostSamples covers the consumer half of the host
// run-state tier. observeJournal's kind switch ends in a default that rejects
// the whole capture, so the producer cannot ship without this: a capture of
// any target run with CUDAPROF_HOST_SAMPLE would fail outright rather than be
// read without the samples.
//
// A host sample is not device work and must not touch the activity totals.
// Counting it as one would inflate ActivityRecords with records that have no
// device, no stream and no duration, and the number named "activity" would
// stop meaning what every consumer of it assumes.
func TestObserveJournalCarriesHostSamples(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	// Two parts of one sample, exactly as the producer splits a thread set
	// that does not fit in one payload.
	const partOne = `{"kind":"host_sample","sample":0,"part":0,"wall":1000,"threads":[{"tid":10,"state":"R","utime_ticks":5,"stime_ticks":1}],"last":false,"in_part":1,"threads_total":2,"unreadable":0,"overflowed":0}`
	const partTwo = `{"kind":"host_sample","sample":0,"part":1,"wall":1000,"threads":[{"tid":11,"state":"D","utime_ticks":0,"stime_ticks":9}],"last":true,"in_part":1,"threads_total":2,"unreadable":0,"overflowed":0}`

	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindHost, Payload: []byte(partOne)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
		{Kind: wire.KindHost, Payload: []byte(partTwo)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("a host sample record cost the whole capture: %v", err)
	}
	if metrics.hostSamples != 2 {
		t.Errorf("host sample records = %d, want 2", metrics.hostSamples)
	}
	if len(activities) != 1 {
		t.Errorf("got %d activities, want 1: a host sample is not device work", len(activities))
	}
	if metrics.duration != 100 {
		t.Errorf("activity duration = %d, want 100 from the single kernel", metrics.duration)
	}
}

// TestObserveJournalRejectsEarlyHostSample. The sampler runs on its own timer,
// so its ordering against bootstrap and capability is a property the producer
// has to establish rather than one it gets for free. This is the check that
// makes a producer which got it wrong fail loudly instead of writing a journal
// whose first record is a sample.
func TestObserveJournalRejectsEarlyHostSample(t *testing.T) {
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	const sample = `{"kind":"host_sample","sample":0,"part":0,"wall":1000,"threads":[],"last":true,"in_part":0,"threads_total":0,"unreadable":0,"overflowed":0}`
	for _, tt := range []struct {
		name    string
		records []wire.Record
	}{
		{"before bootstrap", []wire.Record{
			{Kind: wire.KindHost, Payload: []byte(sample)},
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
		}},
		{"before capability", []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindHost, Payload: []byte(sample)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
		}},
		{"after finalized", []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
			{Kind: wire.KindHost, Payload: []byte(sample)},
		}},
		// A record carrying the host kind but another payload type is a
		// producer/consumer disagreement about what kind 8 means, which is
		// the failure a shared kind number makes possible.
		{"mistyped payload", []wire.Record{
			{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: wire.KindHost, Payload: []byte(`{"kind":"api","domain":"driver"}`)},
			{Kind: wire.KindFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			journalPath := filepath.Join(t.TempDir(), "capture.journal")
			appendRecords(t, journalPath, sid, 1234, 1, tt.records)
			if _, _, _, _, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0); err == nil {
				t.Fatal("observeJournal accepted an out-of-order host sample")
			}
		})
	}
}

// TestObserveJournalCarriesTheIdleExplainingTiers covers the consumer half of
// the OVERHEAD, GRAPH_TRACE and SYNCHRONIZATION tiers.
//
// The subtype check in observeJournal is an allowlist, which is the opposite
// failure mode to the denylist in gaps: the producer side cannot ship without
// this one, because a capture of any target that synchronizes a stream -- that
// is, every target -- would be rejected outright with "unknown activity
// subtype" rather than degraded. Producer and consumer have to land together.
//
// The second half of the test is the arithmetic. journalMetrics.duration is
// accumulated in an else arm that used to mean "not a memcpy", so every one of
// these three would have been charged to device duration: CUPTI's own cost,
// the host's waiting, and a replay envelope that already contains the kernels
// it spans.
func TestObserveJournalCarriesTheIdleExplainingTiers(t *testing.T) {
	dir := t.TempDir()
	sessionHex := "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200,"device":1,"context":4}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"overhead","name":"","valid":true,"start":300,"end":370,"overhead_kind":65536,"object_kind":2,"process_id":4242,"thread_id":99}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"graph_trace","name":"","valid":true,"start":400,"end":490,"device":1,"context":4,"stream":7,"correlation":11,"graph_id":5}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"sync","name":"","valid":true,"start":500,"end":560,"sync_type":3,"cuda_event_id":3,"context":4,"stream":7,"correlation":12}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":4}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("an overhead, graph trace or sync record cost the whole capture: %v", err)
	}
	if len(activities) != 4 {
		t.Fatalf("got %d activities, want 4", len(activities))
	}
	byKind := map[string]captureActivity{}
	for _, a := range activities {
		byKind[a.Subtype] = a
	}
	for _, want := range []string{"overhead", "graph_trace", "sync"} {
		if _, ok := byKind[want]; !ok {
			t.Fatalf("no %q record in the activity slice", want)
		}
	}

	o := byKind["overhead"]
	if o.Start != 300 || o.End != 370 {
		t.Errorf("overhead span = %d..%d, want 300..370", o.Start, o.End)
	}
	if o.OverheadKind != 65536 {
		t.Errorf("overhead kind = %d, want 65536: the span survived but what caused it did not", o.OverheadKind)
	}
	// ObjectKind is what makes the zero Device readable as "no device" rather
	// than as device 0. Dropping it leaves a record that looks device-scoped.
	if o.ObjectKind != 2 {
		t.Errorf("overhead object kind = %d, want 2 (thread)", o.ObjectKind)
	}
	if o.ProcessID != 4242 || o.ThreadID != 99 {
		t.Errorf("overhead process/thread = %d/%d, want 4242/99", o.ProcessID, o.ThreadID)
	}

	g := byKind["graph_trace"]
	if g.Start != 400 || g.End != 490 {
		t.Errorf("graph_trace span = %d..%d, want 400..490", g.Start, g.End)
	}
	if g.GraphID != 5 {
		t.Errorf("graph_trace graph id = %d, want 5: the envelope can no longer be joined to its nodes", g.GraphID)
	}
	if g.Device != 1 {
		t.Errorf("graph_trace device = %d, want 1", g.Device)
	}

	y := byKind["sync"]
	if y.Start != 500 || y.End != 560 {
		t.Errorf("sync span = %d..%d, want 500..560", y.Start, y.End)
	}
	if y.SyncType != 3 {
		t.Errorf("sync type = %d, want 3 (stream synchronize)", y.SyncType)
	}
	if y.CudaEventID != 3 {
		t.Errorf("sync cuda event id = %d, want 3", y.CudaEventID)
	}
	if y.Context != 4 {
		t.Errorf("sync context = %d, want 4: without it the record can never be placed on a device", y.Context)
	}
	if y.Device != 0 {
		t.Errorf("sync device = %d, want 0: the producer has no device to report for a sync and must not invent one", y.Device)
	}

	if metrics.copyBytes != 0 {
		t.Errorf("copy bytes = %d, want 0", metrics.copyBytes)
	}
	// 100 ns for the kernel and nothing else. 70+90+60 = 220 ns of the three
	// tiers is not device work: the overhead is the tool's, the sync is the
	// host waiting, and the graph envelope already contains the kernel.
	if metrics.duration != 100 {
		t.Errorf("duration = %d, want 100 (the kernel alone): %d ns of overhead, sync or replay-envelope time was charged to the device", metrics.duration, metrics.duration-100)
	}
}

// TestCaptureStatusDoesNotInventADeviceForDevicelessTiers covers the device
// roster in compactCaptureStatus, which is built by reading Device off every
// activity. An overhead is scoped to a host thread and a sync carries no
// device field at all, so on both the field is zero because nothing filled it.
// Reading it there would list device 0 as a device the capture saw work on --
// on every capture, including one that used no device 0.
func TestCaptureStatusDoesNotInventADeviceForDevicelessTiers(t *testing.T) {
	status := compactCaptureStatus(captureResult{Activities: []captureActivity{
		{Subtype: "kernel", Valid: true, Start: 100, End: 200, Device: 1},
		{Subtype: "overhead", Valid: true, Start: 300, End: 370, ObjectKind: 2},
		{Subtype: "sync", Valid: true, Start: 500, End: 560, Context: 4},
		{Subtype: "graph_trace", Valid: true, Start: 400, End: 490, Device: 1},
	}})
	if len(status.Devices) != 1 || status.Devices[0] != 1 {
		t.Errorf("devices = %v, want [1]: a record with no device to report was read as reporting device 0", status.Devices)
	}
	// They are still counted by kind: excluded from the device roster is not
	// excluded from the capture.
	for _, kind := range []string{"overhead", "sync", "graph_trace"} {
		if status.ActivityKinds[kind] != 1 {
			t.Errorf("activity kind %q counted %d time(s), want 1", kind, status.ActivityKinds[kind])
		}
	}
}

// TestActivityRecordsCountsActivityRecords pins what the published
// activity_records field means.
//
// The field is fed from wire.Stats.Records, and wire increments that for every
// record it decodes -- bootstrap, capability, API, host sample, marker,
// finalized. observeJournal overwrites it before returning with the count of
// activity records that carried device work. Nothing tested that overwrite, so
// deleting the line left the suite green while every published count silently
// became a total. That is what this test is for.
//
// The fixture is built so the two definitions cannot coincide: six records, of
// which two are device work. A journal holding only activities would pass
// under either definition and pin nothing.
//
// len(activities) is a third number again, and deliberately not equal to
// either. A marker becomes a timeline entry -- it is an NVTX range the target
// asked to see -- but it is not device work and is not counted. A reader who
// expects the published count to match the timeline length is reading the
// field as something it does not claim to be.
func TestActivityRecordsCountsActivityRecords(t *testing.T) {
	dir := t.TempDir()
	const sessionHex = "0123456789abcdef0123456789abcdef"
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)

	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, sid, 1234, 1, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"marker","name":"range","valid":true}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"memset","name":"","valid":true,"start":300,"end":420,"bytes":8192,"value":7,"device":1,"stream":7,"correlation":9,"grid":[0,0,0],"block":[0,0,0]}`)},
		{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
	})

	stats, state, _, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("observeJournal: %v", err)
	}
	if stats.Records != 2 {
		t.Errorf("activity_records = %d, want 2 (kernel + memset): a marker is an NVTX range on the host, not device work", stats.Records)
	}
	// The control. state.records is wire's own count of everything decoded,
	// which is what the published field would carry if the overwrite were
	// removed. If the two agreed, the assertion above would hold under either
	// definition and would be pinning nothing.
	if state.records != 6 {
		t.Fatalf("fixture decoded %d records, want 6: it no longer discriminates the two definitions", state.records)
	}
	if state.records == stats.Records {
		t.Fatalf("total records and activity records are both %d: the fixture cannot discriminate the two definitions", state.records)
	}
	// The marker is on the timeline and out of the count, so neither number
	// substitutes for the other.
	if len(activities) != 3 {
		t.Errorf("timeline holds %d entries, want 3: the marker belongs on it even though it is not counted as device work", len(activities))
	}
}

// These payloads were emitted by the native lifecycle collector on a real
// graph capture. The resource domain and duplicate replay correlation must
// remain journal metadata rather than poison API timing projection.
func TestObserveJournalNativeGraphLifecycle(t *testing.T) {
	payloads, err := os.ReadFile("testdata/graph-lifecycle-native.json")
	if err != nil {
		t.Fatal(err)
	}
	var lifecycle []json.RawMessage
	if err := json.Unmarshal(payloads, &lifecycle); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "capture.journal")
	var sid [16]byte
	sid[0] = 1
	records := []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability","graph_lifecycle":"collected","graph_lifecycle_coverage":"driver_partial"}`)},
	}
	for _, payload := range lifecycle {
		records = append(records, wire.Record{Kind: wire.KindAPI, Payload: payload})
	}
	records = append(records,
		wire.Record{Kind: wire.KindAPI, Payload: []byte(`{"kind":"api","domain":"driver","callback_id":514,"start":419067073160000,"end":419067073167000,"correlation":178}`)},
		wire.Record{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"graph_kernel","valid":true,"start":419067073170000,"end":419067073180000,"correlation":178}`)},
		wire.Record{Kind: wire.KindFinalized, Payload: []byte(`{"kind":"finalized","state":"collected","activity":1}`)},
	)
	appendRecords(t, path, sid, 1234, 1, records)
	_, state, _, activities, apis, _, err := observeJournal(path, hex.EncodeToString(sid[:]), 1234, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(apis) != 1 || apis[0].Domain != "driver" || apis[0].Correlation != 178 || len(activities) != 1 {
		t.Fatalf("projection = %v APIs, %v activities", apis, activities)
	}
	if !state.finalized || !bytes.Contains(state.capabilityData, []byte(`"graph_lifecycle":"collected"`)) || !bytes.Contains(state.capabilityData, []byte(`"graph_lifecycle_coverage":"driver_partial"`)) {
		t.Fatalf("capability/state = %+v", state)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, payload := range lifecycle {
		if !bytes.Contains(data, payload) {
			t.Fatal("native lifecycle payload lost from journal")
		}
	}
}

func TestObserveJournalGraphLifecycleRefusals(t *testing.T) {
	tests := []struct {
		name, payload                    string
		beforeCapability, afterFinalized bool
	}{
		{name: "missing event", payload: `{"kind":"graph_lifecycle","domain":"resource","start":10,"end":10}`},
		{name: "unknown domain", payload: `{"kind":"graph_lifecycle","domain":"unknown","lifecycle_event":"resource","start":10,"end":10}`},
		{name: "reversed timestamps", payload: `{"kind":"graph_lifecycle","domain":"driver","lifecycle_event":"replay","start":20,"end":10}`},
		{name: "before capability", payload: `{"kind":"graph_lifecycle","domain":"resource","lifecycle_event":"resource","start":10,"end":10}`, beforeCapability: true},
		{name: "after finalized", payload: `{"kind":"graph_lifecycle","domain":"resource","lifecycle_event":"resource","start":10,"end":10}`, afterFinalized: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "capture.journal")
			var sid [16]byte
			sid[0] = 1
			records := []wire.Record{{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)}}
			if !tt.beforeCapability {
				records = append(records, wire.Record{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)})
			}
			if tt.afterFinalized {
				records = append(records, wire.Record{Kind: wire.KindFinalized, Payload: []byte(`{"kind":"finalized","state":"no_cuda"}`)})
			}
			records = append(records, wire.Record{Kind: wire.KindAPI, Payload: []byte(tt.payload)})
			appendRecords(t, path, sid, 1234, 1, records)
			if _, _, _, _, _, _, err := observeJournal(path, hex.EncodeToString(sid[:]), 1234, 0); err == nil {
				t.Fatal("accepted invalid lifecycle envelope")
			}
		})
	}
}
