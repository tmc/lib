package dsv4eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func writeTeacherState(t *testing.T, dir string, step int, hash string, logits []float32) {
	t.Helper()
	b, err := json.Marshal(TeacherState{AcceptedDecodeCount: step, ForcedTokensSHA256: testTeacherHash(hash), TokenIDsSHA256: testTeacherHash("tokens"), Logits: logits})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "state-"+strconv.Itoa(step)+".json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func testTeacherHash(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

func TestCompareTeacherStates(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	writeTeacherState(t, left, 0, "hash", []float32{2, 1})
	writeTeacherState(t, right, 0, "hash", []float32{2, 1.5})
	report, err := CompareTeacherStates(left, right)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Steps) != 1 || !report.Steps[0].Top1Agree || report.Steps[0].MaxAbs != 0.5 {
		t.Fatalf("report = %+v", report)
	}
}

func TestCompareTeacherStatesRejectsMutations(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	writeTeacherState(t, left, 0, "left", []float32{2, 1})
	writeTeacherState(t, right, 0, "right", []float32{2, 1})
	if _, err := CompareTeacherStates(left, right); err == nil {
		t.Fatal("mismatched forced-token hash accepted")
	}
	writeTeacherState(t, right, 0, "left", []float32{2, 1})
	writeTeacherState(t, right, 1, "left", []float32{2, 1})
	if _, err := CompareTeacherStates(left, right); err == nil {
		t.Fatal("mismatched step set accepted")
	}
}

func TestReadTeacherStatesRequiresForcedHash(t *testing.T) {
	dir := t.TempDir()
	b, err := json.Marshal(TeacherState{
		AcceptedDecodeCount: 0,
		TokenIDsSHA256:      testTeacherHash("tokens"),
		Logits:              []float32{1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "state-0.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadTeacherStates(dir); err == nil {
		t.Fatal("accepted token-id hash as forced-token provenance")
	}
}

func TestReadTeacherStatesRejectsMissingTokenHash(t *testing.T) {
	dir := t.TempDir()
	b, err := json.Marshal(TeacherState{AcceptedDecodeCount: 0, ForcedTokensSHA256: testTeacherHash("forced"), Logits: []float32{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "state-0.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadTeacherStates(dir); err == nil {
		t.Fatal("accepted missing token-id provenance")
	}
}

func TestReadTeacherStatesRejectsMalformedHashes(t *testing.T) {
	dir := t.TempDir()
	b, err := json.Marshal(TeacherState{
		AcceptedDecodeCount: 0,
		ForcedTokensSHA256:  "not-a-sha256",
		TokenIDsSHA256:      testTeacherHash("tokens"),
		Logits:              []float32{1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "state-0.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := ReadTeacherStates(dir); err == nil {
		t.Fatal("malformed forced-token hash unexpectedly passed")
	}
}

func TestReadTeacherStatesRejectsMissingStep(t *testing.T) {
	dir := t.TempDir()
	writeTeacherState(t, dir, 0, "hash", []float32{1, 2})
	writeTeacherState(t, dir, 2, "hash", []float32{1, 2})
	if _, _, err := ReadTeacherStates(dir); err == nil {
		t.Fatal("accepted sparse teacher steps")
	}
}

func TestCompareTeacherStatesRejectsTokenMutation(t *testing.T) {
	left, right := t.TempDir(), t.TempDir()
	writeTeacherState(t, left, 0, "forced", []float32{2, 1})
	b, err := json.Marshal(TeacherState{AcceptedDecodeCount: 0, ForcedTokensSHA256: testTeacherHash("forced"), TokenIDsSHA256: testTeacherHash("other"), Logits: []float32{2, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(right, "state-0.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := CompareTeacherStates(left, right); err == nil {
		t.Fatal("accepted mismatched token-id provenance")
	}
}
