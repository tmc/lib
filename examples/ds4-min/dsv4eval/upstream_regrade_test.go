package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestUpstreamReadsGeneratedTrace(t *testing.T) {
	source := os.Getenv("DS4_ORACLE_SOURCE")
	if source == "" {
		t.Skip("set DS4_ORACLE_SOURCE to the pinned oracle checkout")
	}
	for name, want := range SmokeSourceHashes() {
		data, err := os.ReadFile(filepath.Join(source, name))
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != want {
			t.Fatalf("unqualified oracle %s", name)
		}
	}
	dir := t.TempDir()
	core, err := os.ReadFile(filepath.Join(source, "ds4.c"))
	if err != nil || fmt.Sprintf("%x", sha256.Sum256(core)) != "62e458f2c416148c0882f7247c7b16f9820054b0c2ae41374a9209541fe7a4c7" {
		t.Fatal("unqualified oracle ds4.c")
	}
	// Extract complete source-owned leaf helpers, not replacement formatters.
	helpers := "#include <stdio.h>\n#include \"ds4.h\"\n"
	for _, signature := range []string{"const char *ds4_backend_name(", "int ds4_think_mode_level(", "const char *ds4_think_mode_name("} {
		start := strings.Index(string(core), signature)
		if start < 0 {
			t.Fatal("missing upstream helper", signature)
		}
		end := strings.Index(string(core)[start:], "\n}")
		if end < 0 {
			t.Fatal("unterminated upstream helper", signature)
		}
		helpers += string(core)[start:start+end+2] + "\n"
	}
	helperPath := filepath.Join(dir, "helpers.c")
	if err := os.WriteFile(helperPath, []byte(helpers), 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(dir, "regrade")
	args := []string{"-O2", "-ffunction-sections", "-fdata-sections", "-I", source, "testdata/extract_regrade.c", helperPath, filepath.Join(source, "ds4_eval_cases.c"), "-o", binary, "-lm", "-pthread"}
	if runtime.GOOS == "darwin" {
		args = append(args, "-Wl,-dead_strip")
	} else {
		args = append(args, "-Wl,--gc-sections", "-ldl")
	}
	if output, err := exec.Command("cc", args...).CombinedOutput(); err != nil {
		t.Fatalf("build actual upstream parser: %v\n%s", err, output)
	}
	actual, err := exec.Command(binary, "--write").Output()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ReadTrace(bytes.NewReader(actual))
	if err != nil || len(parsed.Cases) != 12 {
		t.Fatalf("actual upstream writer: %v, cases=%d", err, len(parsed.Cases))
	}
	choices := 0
	for _, c := range parsed.Cases {
		if len(c.Choices) != 0 {
			choices++
		}
	}
	if choices != 6 {
		t.Fatalf("choice cases=%d, want 6", choices)
	}
	for _, mutate := range []func(string) string{
		func(s string) string { return strings.Replace(s, "  B. ", "  A. ", 1) },
		func(s string) string { return strings.Replace(s, "choices:\n", "choices:\nchoices:\n", 1) },
		func(s string) string { return strings.Replace(s, "answer_kind: choice", "answer_kind: integer", 1) },
		func(s string) string { return strings.Replace(s, "  A. ", "MODEL_OUTPUT_BEGIN bytes=0\n", 1) },
	} {
		if _, err := ReadTrace(strings.NewReader(mutate(string(actual)))); err == nil {
			t.Fatal("accepted mutated actual C choices")
		}
	}
	var wire bytes.Buffer
	w := NewTraceWriter(&wire, TraceHeader{})
	if err := w.Case(TraceCase{Index: 1, Total: 1, Source: "GPQA Diamond", ID: "recNu3MXkvWUzHZr9", Fields: map[string]string{"status": "PASSED", "expected": "B", "picked": "B"}, Blocks: map[string]string{"MODEL_OUTPUT": strings.Repeat("reason", 22000) + "\nMODEL_OUTPUT_END\nAnswer: B\n"}}); err != nil {
		t.Fatal(err)
	}
	if err := w.Summary(map[string]string{"total": "1"}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "go.trace")
	if err := os.WriteFile(path, wire.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	output, err := exec.Command(binary, path).CombinedOutput()
	if err != nil || !bytes.Contains(output, []byte("passed=1 failed=0 changed=0 not_graded=0 unknown=0 parse_errors=0")) {
		t.Fatalf("actual C regrade: %v\n%s", err, output)
	}
	if _, err := Regrade(bytes.NewReader(wire.Bytes())); err != nil {
		t.Fatal(err)
	}
	// The actual upstream parser reads these metadata fields, not the header.
	bad := strings.Replace(wire.String(), "source: GPQA Diamond\n", "", 1)
	if err := os.WriteFile(path, []byte(bad), 0600); err != nil {
		t.Fatal(err)
	}
	output, err = exec.Command(binary, path).CombinedOutput()
	if err == nil || !bytes.Contains(output, []byte("parse_errors=1")) {
		t.Fatalf("missing identity mutation was not detected by C: %v\n%s", err, output)
	}
}
