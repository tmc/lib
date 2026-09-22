package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestLaunchManifestDisabled(t *testing.T) {
	if got := newLaunchManifest(launchManifestOptions{}); got != nil {
		t.Fatal("disabled manifest is non-nil")
	}
}

func TestBuildLaunchManifestIsOptInCaptureMetadata(t *testing.T) {
	result := captureResult{
		APIRecords: []captureAPI{{Domain: "runtime", CallbackID: 7, Correlation: 11}},
		Activities: []captureActivity{{Subtype: "kernel", Name: "k", Correlation: 11, Stream: 3, GridDim: [3]uint32{4, 2, 1}, BlockDim: [3]uint32{32, 1, 1}, DynamicSharedMemory: 512}},
	}
	b := buildLaunchManifest(result)
	var got launchManifest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Completeness != "metadata-only" || len(got.Records) != 1 {
		t.Fatalf("manifest = %+v, want one metadata-only record", got)
	}
	if got.Records[0].Function != "k" || got.Records[0].ActivityCorrelation != 11 || got.Records[0].SharedMemory != 512 {
		t.Fatalf("record = %+v, want activity join", got.Records[0])
	}
	if got.Records[0].Parameters.Status != "UNREAD" || got.Records[0].Module != "UNREAD" {
		t.Fatalf("record claims unread data: %+v", got.Records[0])
	}
}

func TestLaunchManifestCaptureFlagAndStatusOutput(t *testing.T) {
	opts, err := parseCaptureArgs([]string{"--launch-manifest", "--no-inject", "--", "true"})
	if err != nil {
		t.Fatal(err)
	}
	if !opts.LaunchManifest {
		t.Fatal("--launch-manifest did not enable the option")
	}
	result := captureResult{LaunchManifest: buildLaunchManifest(captureResult{})}
	b, err := json.Marshal(compactCaptureStatus(result))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"launch_manifest"`) || !strings.Contains(string(b), `"metadata-only"`) {
		t.Fatalf("status omitted metadata-only launch manifest: %s", b)
	}
}

func TestLaunchManifestUnknownLayoutStaysUnread(t *testing.T) {
	m := newLaunchManifest(launchManifestOptions{Enabled: true, CUPTIVersion: 130400, MaxBytes: 16})
	m.addLaunchManifestRecord(launchManifestRecord{
		CallbackID: 7, ActivityCorrelation: 11, Correlation: 9,
		Domain: "runtime", Function: "cudaLaunchKernel", Stream: 3,
		Grid: [3]uint32{4, 2, 1}, Block: [3]uint32{32, 1, 1}, SharedMemory: 512,
	})
	b, err := m.marshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "parameters snapshot") || strings.Contains(string(b), "1111") {
		t.Fatalf("manifest claims an operand snapshot: %s", b)
	}
	var got launchManifest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Records[0].Parameters.Status != "UNREAD" {
		t.Fatalf("parameter status = %q, want UNREAD", got.Records[0].Parameters.Status)
	}
	if got.Records[0].Module != "UNREAD" {
		t.Fatalf("module = %q, want UNREAD", got.Records[0].Module)
	}
	if got.Completeness != "metadata-only" {
		t.Fatalf("completeness = %q, want metadata-only", got.Completeness)
	}
}

func TestLaunchManifestAAIsByteIdentical(t *testing.T) {
	makeManifest := func() []byte {
		m := newLaunchManifest(launchManifestOptions{Enabled: true, CUPTIVersion: 130400})
		m.addLaunchManifestRecord(launchManifestRecord{
			CallbackID: 3, ActivityCorrelation: 8, Correlation: 7,
			Domain: "driver", Function: "cuLaunchKernel", Symbol: "kernel",
			Stream: 4, Grid: [3]uint32{8, 1, 1}, Block: [3]uint32{32, 1, 1},
		})
		b, err := m.marshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	if got, want := string(makeManifest()), string(makeManifest()); got != want {
		t.Fatalf("A/A manifest differs:\n%s\n---\n%s", got, want)
	}
}

func TestLaunchManifestBoundedOmission(t *testing.T) {
	m := newLaunchManifest(launchManifestOptions{Enabled: true, MaxBytes: 4096, MaxRecords: 3})
	records := make([]launchManifestRecord, 8)
	for i := range records {
		records[i] = launchManifestRecord{
			CallbackID: uint32(i), Function: strings.Repeat("kernel", 20),
			Module: "UNREAD", Parameters: launchBytes{Status: "UNREAD"},
		}
	}
	b, err := marshalBoundedLaunchManifest(m, records)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) > 4096 {
		t.Fatalf("manifest size = %d, want <= 4096", len(b))
	}
	if m.OmittedRecords == 0 || m.Omission == "" {
		t.Fatalf("omission = %q records=%d, want explicit bounded omission", m.Omission, m.OmittedRecords)
	}
}

func TestQualifiedLaunchBytesRequiresMatchingVersionAndBound(t *testing.T) {
	tests := []struct {
		name, wantStatus     string
		version, wantVersion uint32
		decoded              []byte
		max                  int
	}{
		{name: "unknown version", wantStatus: "UNREAD", version: 0, wantVersion: 130400, decoded: []byte{1}, max: 4},
		{name: "version mismatch", wantStatus: "UNREAD", version: 130400, wantVersion: 130300, decoded: []byte{1}, max: 4},
		{name: "over bound", wantStatus: "UNREAD", version: 130400, wantVersion: 130400, decoded: []byte{1, 2}, max: 1},
		{name: "qualified", wantStatus: "DECODED", version: 130400, wantVersion: 130400, decoded: []byte{1, 2}, max: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := qualifiedLaunchBytes(tt.version, tt.wantVersion, tt.decoded, tt.max)
			if tt.name == "over bound" && err == nil {
				t.Fatal("over-bound bytes accepted")
			}
			if got.Status != tt.wantStatus {
				t.Fatalf("status = %q, want %q", got.Status, tt.wantStatus)
			}
		})
	}
}

// TestLaunchManifestByteLimitFitsWithOmission sweeps the byte budget across the
// whole range where records are dropped. Recording an omission adds fields that
// were not in the length the byte check measured, so a manifest that just fit
// could be pushed back over it and marshal as an error instead of a bounded
// result. The assertion is the property, not the absence of an error: whatever
// comes back must be within max_bytes and must say what it dropped.
func TestLaunchManifestByteLimitFitsWithOmission(t *testing.T) {
	records := make([]launchManifestRecord, 200)
	for i := range records {
		records[i] = launchManifestRecord{
			CallbackID:  uint32(i),
			Correlation: uint32(i),
			Domain:      "DRIVER",
			Function:    "someKernelNameOfModerateLength",
			Module:      "UNREAD",
			Parameters:  launchBytes{Status: "UNREAD"},
		}
	}
	// The floor is the envelope with the omission fields populated, not the
	// bare envelope: a manifest that drops records must have room to say so.
	floor := newLaunchManifest(launchManifestOptions{Enabled: true, MaxBytes: 1 << 20})
	floor.OmittedRecords = uint64(len(records))
	floor.Omission = "byte_limit"
	envelope, err := floor.marshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	for max := len(envelope); max < 4000; max++ {
		m := newLaunchManifest(launchManifestOptions{Enabled: true, MaxBytes: max})
		b, err := marshalBoundedLaunchManifest(m, records)
		if err != nil {
			t.Fatalf("max_bytes=%d: %v", max, err)
		}
		if len(b) > max {
			t.Fatalf("max_bytes=%d: manifest is %d bytes, over its own limit", max, len(b))
		}
		var got launchManifest
		if err := json.Unmarshal(b, &got); err != nil {
			t.Fatalf("max_bytes=%d: %v", max, err)
		}
		if len(got.Records) == len(records) {
			continue
		}
		if got.Omission == "" || got.OmittedRecords == 0 {
			t.Fatalf("max_bytes=%d: kept %d of %d records but omission=%q omitted=%d",
				max, len(got.Records), len(records), got.Omission, got.OmittedRecords)
		}
		if int(got.OmittedRecords) != len(records)-len(got.Records) {
			t.Fatalf("max_bytes=%d: omitted_records=%d, want %d",
				max, got.OmittedRecords, len(records)-len(got.Records))
		}
	}
}

// TestLaunchManifestErrorNamesItsCause checks that a failure reaches the status
// record as a reason. "marshal failed" told a reader nothing, and a consumer
// cannot otherwise tell a bounded omission from a defect.
func TestLaunchManifestErrorNamesItsCause(t *testing.T) {
	b := launchManifestError(errors.New("envelope exceeds max_bytes=8 with no records"))
	var got struct {
		Schema string `json:"schema"`
		Error  string `json:"error"`
	}
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "cudaprof-launch-manifest/v1" {
		t.Fatalf("schema = %q", got.Schema)
	}
	if got.Error == "marshal failed" || !strings.Contains(got.Error, "max_bytes=8") {
		t.Fatalf("error = %q, want the cause", got.Error)
	}
}
