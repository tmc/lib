package dsv4eval

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const seededModelHash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
const seededLogitA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
const seededLogitB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func seededTrace() SeededDecodeTrace {
	return SeededDecodeTrace{
		Metadata: SeededDecodeMetadata{
			Build:       "oracle-11e8cad",
			ModelSHA256: seededModelHash,
			Context:     4096, Temperature: 1, Seed: 12345, MTPDisabled: true,
			ArtifactOnly: true,
		},
		Steps: []SeededDecodeStep{
			{Step: 0, Token: 17, LogitsSHA256: seededLogitA},
			{Step: 1, Token: 19, LogitsSHA256: seededLogitA},
		},
	}
}

func TestCompareSeededDecodePassesIdenticalTrace(t *testing.T) {
	report, err := CompareSeededDecode(seededTrace(), seededTrace())
	if err != nil {
		t.Fatal(err)
	}
	if !report.Pass() || report.FirstLogitDifference != -1 || report.TokenMatches != 2 {
		t.Fatalf("report = %+v, want identical pass", report)
	}
}

func TestCompareSeededDecodeCControlIsRecordedAsZeroSpread(t *testing.T) {
	// A C-vs-C identical control is valid evidence that the retained artifacts
	// are stable, but it must not be used as a nonzero cross-implementation
	// tolerance. The report exposes that distinction through zero differing
	// logit digests and no first divergence.
	report, err := CompareSeededDecode(seededTrace(), seededTrace())
	if err != nil {
		t.Fatal(err)
	}
	if report.FirstLogitDifference != -1 || report.LogitDigestMatches != report.Steps {
		t.Fatalf("control report = %+v, want zero spread", report)
	}
}

func TestCompareSeededDecodeReportsFirstLogitDifference(t *testing.T) {
	left, right := seededTrace(), seededTrace()
	right.Steps[1].LogitsSHA256 = seededLogitB
	report, err := CompareSeededDecode(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Pass() || report.FirstLogitDifference != 1 || report.FirstTokenDifference != -1 {
		t.Fatalf("report = %+v, want logit divergence with token agreement", report)
	}
	right.Steps[1].Token = 23
	report, err = CompareSeededDecode(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if report.Pass() {
		t.Fatalf("report = %+v, token mismatch at first differing logits must fail", report)
	}
	right.Steps[1].Token = left.Steps[1].Token
	right.Steps = append(right.Steps, SeededDecodeStep{Step: 2, Token: 29, LogitsSHA256: seededLogitB})
	// A trace-length mismatch is rejected before a later attributable token
	// can be accepted; the comparator must never manufacture a missing step.
	if _, err := CompareSeededDecode(left, right); err == nil {
		t.Fatal("missing step was accepted")
	}
}

func TestCompareSeededDecodeRejectsTokenMismatchBeforeLogitDifference(t *testing.T) {
	left, right := seededTrace(), seededTrace()
	right.Steps[0].Token++
	if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), "before the first differing logits") {
		t.Fatalf("error = %v, want early token mismatch", err)
	}
}

func TestCompareSeededDecodeRejectsProvenanceAndShapeMutations(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*SeededDecodeTrace, *SeededDecodeTrace)
		want   string
	}{
		{"model", func(l, r *SeededDecodeTrace) { r.Metadata.ModelSHA256 = strings.Repeat("f", 64) }, "model hashes differ"},
		{"mtp", func(l, r *SeededDecodeTrace) { r.Metadata.MTPDisabled = false }, "MTP is enabled"},
		{"steps", func(l, r *SeededDecodeTrace) { r.Steps = r.Steps[:1] }, "step counts differ"},
		{"noncontiguous", func(l, r *SeededDecodeTrace) { r.Steps[1].Step = 4 }, "non-contiguous"},
		{"bad hash", func(l, r *SeededDecodeTrace) { r.Steps[0].LogitsSHA256 = "bad" }, "not 64 hexadecimal"},
		{"empty", func(l, r *SeededDecodeTrace) { r.Steps = nil }, "no steps"},
		{"missing build", func(l, r *SeededDecodeTrace) { r.Metadata.Build = "" }, "build is missing"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			left, right := seededTrace(), seededTrace()
			tt.mutate(&left, &right)
			if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestReadSeededDecodeTraceRoundTripsJSON(t *testing.T) {
	trace := seededTrace()
	data := `{"metadata":{"build":"oracle-11e8cad","model_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","context":4096,"temperature":1,"seed":12345,"mtp_disabled":true,"artifact_only":true},"steps":[{"step":0,"token":17,"logits_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"step":1,"token":19,"logits_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}`
	got, err := ReadSeededDecodeTrace(strings.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata != trace.Metadata || len(got.Steps) != len(trace.Steps) || got.Steps[1] != trace.Steps[1] {
		t.Fatalf("trace = %+v, want %+v", got, trace)
	}
}

func TestReadSeededDecodeTraceRejectsTrailingAndUnknownData(t *testing.T) {
	base := `{"metadata":{"build":"b","model_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","context":1,"temperature":1,"seed":1,"mtp_disabled":true,"artifact_only":true},"steps":[]}`
	for _, data := range []string{base + " {}", strings.Replace(base, `"steps":[]`, `"extra":1,"steps":[]`, 1)} {
		if _, err := ReadSeededDecodeTrace(strings.NewReader(data)); err == nil {
			t.Fatalf("accepted malformed trace %q", data)
		}
	}
}

func TestConvertSeededDecodeArtifactsHashesRawRows(t *testing.T) {
	dir := t.TempDir()
	logitsDir := filepath.Join(dir, "logits")
	if err := os.Mkdir(logitsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	for i, data := range [][]byte{{1, 2, 3, 4}, {5, 6, 7, 8}} {
		name := filepath.Join(logitsDir, fmt.Sprintf("step-%04d.f32", i))
		if err := os.WriteFile(name, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	logprobs := filepath.Join(dir, "logprobs.json")
	dump := map[string]any{"source": "ds4", "steps": []any{
		map[string]any{"step": 0, "selected": map[string]any{"id": 17}},
		map[string]any{"step": 1, "selected": map[string]any{"id": 19}},
	}}
	data, err := json.Marshal(dump)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(logprobs, data, 0o600); err != nil {
		t.Fatal(err)
	}
	trace, err := ConvertSeededDecodeArtifacts(SeededDecodeArtifactOptions{
		LogitsDir: logitsDir, LogprobsPath: logprobs, Build: "b",
		ModelSHA256: seededModelHash, Context: 4096, Temperature: 1,
		Seed: 12345, MTPDisabled: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(trace.Steps) != 2 || trace.Steps[0].Token != 17 || trace.Steps[1].Token != 19 {
		t.Fatalf("trace steps = %+v", trace.Steps)
	}
	if trace.Steps[0].LogitsSHA256 == trace.Steps[1].LogitsSHA256 {
		t.Fatal("distinct raw rows produced the same digest")
	}
}

func TestConvertSeededDecodeArtifactsRejectsMissingAndMalformedRows(t *testing.T) {
	dir := t.TempDir()
	logitsDir := filepath.Join(dir, "logits")
	if err := os.Mkdir(logitsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	logprobs := filepath.Join(dir, "logprobs.json")
	if err := os.WriteFile(logprobs, []byte(`{"steps":[{"step":0,"selected":{"id":17}}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	base := SeededDecodeArtifactOptions{
		LogitsDir: logitsDir, LogprobsPath: logprobs, Build: "b",
		ModelSHA256: seededModelHash, Context: 4096, Temperature: 1,
		Seed: 12345, MTPDisabled: true,
	}
	if _, err := ConvertSeededDecodeArtifacts(base); err == nil || !strings.Contains(err.Error(), "read logits step 0") {
		t.Fatalf("missing row error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(logitsDir, "step-0000.f32"), []byte{1, 2, 3}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertSeededDecodeArtifacts(base); err == nil || !strings.Contains(err.Error(), "invalid byte length") {
		t.Fatalf("malformed row error = %v", err)
	}
	if err := os.WriteFile(logprobs, []byte(`{"steps":[{"step":1,"selected":{"id":17}}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertSeededDecodeArtifacts(base); err == nil || !strings.Contains(err.Error(), "logprob step 0 is 1") {
		t.Fatalf("non-contiguous error = %v", err)
	}
	if err := os.WriteFile(filepath.Join(logitsDir, "step-0001.f32"), []byte{1, 2, 3, 4}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ConvertSeededDecodeArtifacts(base); err == nil || !strings.Contains(err.Error(), "unexpected logits row") {
		t.Fatalf("extra row error = %v", err)
	}
}

func TestConvertSeededDecodeArtifactsRejectsInvalidMetadata(t *testing.T) {
	dir := t.TempDir()
	logitsDir := filepath.Join(dir, "logits")
	if err := os.Mkdir(logitsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	logprobs := filepath.Join(dir, "logprobs.json")
	if err := os.WriteFile(logprobs, []byte(`{"steps":[{"step":0,"selected":{"id":17}}]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	base := SeededDecodeArtifactOptions{
		LogitsDir: logitsDir, LogprobsPath: logprobs, Build: "b",
		ModelSHA256: seededModelHash, Context: 4096, Temperature: 1,
		Seed: 12345, MTPDisabled: true,
	}
	for name, mutate := range map[string]func(*SeededDecodeArtifactOptions){
		"missing build":       func(o *SeededDecodeArtifactOptions) { o.Build = "" },
		"bad model hash":      func(o *SeededDecodeArtifactOptions) { o.ModelSHA256 = "bad" },
		"invalid context":     func(o *SeededDecodeArtifactOptions) { o.Context = 0 },
		"invalid temperature": func(o *SeededDecodeArtifactOptions) { o.Temperature = 0 },
		"missing seed":        func(o *SeededDecodeArtifactOptions) { o.Seed = 0 },
		"mtp enabled":         func(o *SeededDecodeArtifactOptions) { o.MTPDisabled = false },
	} {
		t.Run(name, func(t *testing.T) {
			opts := base
			mutate(&opts)
			if _, err := ConvertSeededDecodeArtifacts(opts); err == nil {
				t.Fatal("invalid metadata unexpectedly accepted")
			}
		})
	}
}

func TestCompareSeededDecodeChecksRawLogits(t *testing.T) {
	dir := t.TempDir()
	leftDir, rightDir := filepath.Join(dir, "left"), filepath.Join(dir, "right")
	if err := os.MkdirAll(leftDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(rightDir, 0o700); err != nil {
		t.Fatal(err)
	}
	writeRow := func(path string, values ...float32) {
		t.Helper()
		data := make([]byte, 4*len(values))
		for i, value := range values {
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeRow(filepath.Join(leftDir, "step-0000.f32"), 1, 2)
	writeRow(filepath.Join(rightDir, "step-0000.f32"), 1.00005, 2)
	writeRow(filepath.Join(leftDir, "step-0001.f32"), 3, 4)
	writeRow(filepath.Join(rightDir, "step-0001.f32"), 3, 4)
	left, right := seededTrace(), seededTrace()
	left.Metadata.ArtifactOnly, right.Metadata.ArtifactOnly = false, false
	left.Metadata.LogitsDir, right.Metadata.LogitsDir = leftDir, rightDir
	report, err := CompareSeededDecode(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !report.NumericChecked || !report.NumericPass || report.FirstNumericDifference != -1 {
		t.Fatalf("report = %+v, want passing raw comparison", report)
	}
	if report.MaxAbs == 0 || report.MaxRel == 0 {
		t.Fatalf("report = %+v, want numeric maxima", report)
	}
	writeRow(filepath.Join(rightDir, "step-0000.f32"), 1.01, 2)
	report, err = CompareSeededDecode(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Pass() || report.NumericPass || report.FirstNumericDifference != 0 {
		t.Fatalf("report = %+v, want diagnostic tolerance failure with attribution pass", report)
	}
}

func TestCompareSeededDecodeAttributesLaterTokenToNumericDifference(t *testing.T) {
	dir := t.TempDir()
	leftDir, rightDir := filepath.Join(dir, "left"), filepath.Join(dir, "right")
	if err := os.MkdirAll(leftDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(rightDir, 0o700); err != nil {
		t.Fatal(err)
	}
	writeRow := func(path string, value float32) {
		t.Helper()
		data := make([]byte, 4)
		binary.LittleEndian.PutUint32(data, math.Float32bits(value))
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeRow(filepath.Join(leftDir, "step-0000.f32"), 1)
	writeRow(filepath.Join(rightDir, "step-0000.f32"), 1.01)
	writeRow(filepath.Join(leftDir, "step-0001.f32"), 2)
	writeRow(filepath.Join(rightDir, "step-0001.f32"), 2)
	left, right := seededTrace(), seededTrace()
	left.Metadata.ArtifactOnly, right.Metadata.ArtifactOnly = false, false
	left.Metadata.LogitsDir, right.Metadata.LogitsDir = leftDir, rightDir
	right.Steps[0].LogitsSHA256 = seededLogitB
	right.Steps[1].Token++
	report, err := CompareSeededDecode(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Pass() || report.FirstLogitDifference != 0 || report.FirstTokenDifference != 1 || report.NumericPass {
		t.Fatalf("report = %+v, want later token attributed to raw-logit difference", report)
	}
}

func TestCompareSeededDecodeRawLogitMutationsFailClosed(t *testing.T) {
	left, right := seededTrace(), seededTrace()
	dir := t.TempDir()
	left.Metadata.ArtifactOnly, right.Metadata.ArtifactOnly = false, false
	left.Metadata.LogitsDir = filepath.Join(dir, "left")
	right.Metadata.LogitsDir = filepath.Join(dir, "right")
	if err := os.Mkdir(left.Metadata.LogitsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(right.Metadata.LogitsDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), "read left logits step 0") {
		t.Fatalf("missing raw row error = %v", err)
	}
	raw := func(path string, bits uint32) {
		t.Helper()
		data := make([]byte, 4)
		binary.LittleEndian.PutUint32(data, bits)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	raw(filepath.Join(left.Metadata.LogitsDir, "step-0000.f32"), math.Float32bits(1))
	raw(filepath.Join(right.Metadata.LogitsDir, "step-0000.f32"), math.Float32bits(float32(math.NaN())))
	if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), "non-finite logits") {
		t.Fatalf("non-finite raw row error = %v", err)
	}
	right.Metadata.LogitsDir = ""
	if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), "missing logits_dir and artifact_only") {
		t.Fatalf("one-sided raw row error = %v", err)
	}
}

func TestCompareSeededDecodeRequiresExplicitArtifactOnly(t *testing.T) {
	left, right := seededTrace(), seededTrace()
	left.Metadata.ArtifactOnly = false
	if _, err := CompareSeededDecode(left, right); err == nil || !strings.Contains(err.Error(), "missing logits_dir and artifact_only") {
		t.Fatalf("implicit digest-only trace error = %v", err)
	}
}
