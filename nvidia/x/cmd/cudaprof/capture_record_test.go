package main

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"reflect"
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
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{
			Kind:    recordActivity,
			Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"kernelArray","valid":true,"start":100,"end":200,"grid":[256,1,1],"block":[32,1,1],"registers_per_thread":48,"dynamic_shared_memory":4096}`),
		},
		{
			Kind:    recordActivity,
			Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"kernelCoords","valid":true,"start":300,"end":400,"grid_x":64,"grid_y":2,"grid_z":1,"block_x":128,"block_y":1,"block_z":1,"registers":24,"smem":1024}`),
		},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
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
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"state":"no_cuda_observed"}`)},
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
	if !got.CaptureSuccess || got.Outcome != captureOutcomeObserved || !got.TransportReady || !got.BootstrapSeen {
		t.Fatalf("verified no-cuda = %+v", got)
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
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		}
		if activity != "" {
			records = append(records, wire.Record{Kind: recordActivity, Payload: []byte(activity)})
		}
		records = append(records, wire.Record{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)})
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
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"no_cuda_observed"}`)},
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
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 1}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 1}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"state":"no_cuda_observed"}`)},
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
	if !got.CaptureSuccess || got.Outcome != captureOutcomeObserved || len(got.WorkerGenerations) != 1 {
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
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Payload: []byte(`{"state":"initialized_no_activity"}`)},
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
	if err := j.Append(wire.Record{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)}); err != nil {
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
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFailure, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"state":"cupti_unavailable"}`)},
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 4, Payload: []byte(`{"state":"initialized_no_activity"}`)},
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
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","start":1,"end":2}`)},
		}},
		{name: "unknown subtype", records: []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"bogus","start":1,"end":2}`)},
			{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
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
				{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
				{Kind: recordActivity, Payload: []byte(payload)},
				{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
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
		{Kind: recordFailure, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"state":"cupti_unavailable","error":"CUPTI runtime/header version mismatch"}`)},
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"state":"initialized_no_activity","trigger":"exit"}`)},
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
		{Kind: recordBootstrap, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 1, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 2, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 3, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k1"}`)}, // no valid field
		{Kind: recordFinalized, Identity: wire.Identity{Session: sid, Process: 41, Generation: 2}, Sequence: 4, Payload: []byte(`{"state":"collected"}`)},
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
				{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
				{Kind: recordActivity, Payload: []byte(tt.payload)},
				{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
				{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":2}`)},
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
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"teleport","valid":true,"start":100,"end":200}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
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
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"marker","name":"decode","marker_domain":"ds4","valid":true,"start":100,"end":100,"marker_id":1,"marker_flags":2,"process_id":1234,"thread_id":7}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"inside","valid":true,"start":200,"end":300,"device":0}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"marker","name":"decode","marker_domain":"ds4","valid":true,"start":400,"end":400,"marker_id":1,"marker_flags":4,"process_id":1234,"thread_id":7}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"outside","valid":true,"start":500,"end":600,"device":0}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":4}`)},
	})

	_, _, _, activities, _, _, err := observeJournal(journalPath, sessionHex, 1234, 0)
	if err != nil {
		t.Fatalf("observeJournal: %v", err)
	}
	p := cudacapture.BuildActivityProfile(sessionHex, 1234, activities)

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
		t.Errorf("kernel inside the range has nvtx = %v, want %v", got["inside"], want)
	}
	if got["outside"] != nil {
		t.Errorf("kernel after the range closed has nvtx = %v, want none", got["outside"])
	}
}
