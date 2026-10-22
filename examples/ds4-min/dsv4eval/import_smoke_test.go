package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func oracleSmokeFixture(t *testing.T) ([]byte, OracleSmokeProvenance) {
	t.Helper()
	p := OracleSmokeProvenance{Build: strings.Repeat("a", 40), BinarySHA256: fmt.Sprintf("%x", sha256.Sum256([]byte("binary"))), ModelSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte("model"))), RawModelPath: "oracle.gguf", Backend: "cuda", Context: 32768, Temperature: "0", Think: "HIGH"}
	var b bytes.Buffer
	w := NewTraceWriter(&b, TraceHeader{Fields: map[string]string{"suite": "hard-smoke", "model": p.RawModelPath, "backend": p.Backend, "ctx": "32768", "temperature": "0", "think_mode_requested": "high", "questions": "12", "soft_limit_reply_budget": "1024", "hard_limit_reply_budget": "512", "soft_limit_think_close_rank": "3"}})
	cs, _ := SelectN(Cases(), Filter{Suite: HardSmoke}, 0)
	for i, c := range cs {
		c.Answer = smokeAnswer(c)
		output := "Answer: " + c.Answer + "\n"
		picked, complete, pass := Grade(c, output)
		if !complete || !pass {
			t.Fatalf("fixture answer %s: %q", c.ID, output)
		}
		if err := w.Case(TraceCase{Index: i + 1, Total: 12, Source: c.Source, ID: c.ID, Fields: map[string]string{"status": "PASSED", "picked": picked, "expected": c.Answer, "think_close": "natural"}, Blocks: map[string]string{"MODEL_OUTPUT": output}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Summary(map[string]string{"passed": "12", "failed": "0", "incomplete": "0", "total": "12"}); err != nil {
		t.Fatal(err)
	}
	p.RawTraceSHA256 = fmt.Sprintf("%x", sha256.Sum256(b.Bytes()))
	return b.Bytes(), p
}

func TestImportOracleSmokePreservesCases(t *testing.T) {
	raw, p := oracleSmokeFixture(t)
	var out bytes.Buffer
	if err := importOracleSmoke(&out, raw, p); err != nil {
		t.Fatal(err)
	}
	before := raw[bytes.Index(raw, []byte("===== CASE ")):]
	after := out.Bytes()[bytes.Index(out.Bytes(), []byte("===== CASE ")):]
	if !bytes.Equal(before, after) {
		t.Fatal("case or summary bytes changed")
	}
	if _, err := CompareSmoke(bytes.NewReader(out.Bytes()), bytes.NewReader(out.Bytes())); err != nil {
		t.Fatal(err)
	}
	for _, mutation := range []string{"hash", "temperature", "context", "reserve", "missing summary", "attested header", "binary", "model"} {
		t.Run(mutation, func(t *testing.T) {
			data := append([]byte(nil), raw...)
			receipt := p
			switch mutation {
			case "hash":
				receipt.RawTraceSHA256 = strings.Repeat("0", 64)
			case "temperature":
				receipt.Temperature = "1"
			case "context":
				receipt.Context++
			case "reserve":
				data = bytes.Replace(data, []byte("soft_limit_reply_budget: 1024"), []byte("soft_limit_reply_budget: 512"), 1)
			case "missing summary":
				data = data[:bytes.Index(data, []byte("===== SUMMARY ====="))]
			case "attested header":
				data = bytes.Replace(data, []byte("# ds4-eval trace\n"), []byte("# ds4-eval trace\nbuild: injected\n"), 1)
			case "binary":
				receipt.BinarySHA256 = ""
			case "model":
				receipt.ModelSHA256 = ""
			}
			if mutation != "hash" {
				receipt.RawTraceSHA256 = fmt.Sprintf("%x", sha256.Sum256(data))
			}
			var b bytes.Buffer
			if err := importOracleSmoke(&b, data, receipt); err == nil {
				t.Fatal("mutation accepted")
			}
			if b.Len() != 0 {
				t.Fatal("failed admission wrote output")
			}
		})
	}
}

func TestImportOracleSmokeArtifactHashes(t *testing.T) {
	raw, p := oracleSmokeFixture(t)
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "trace"), filepath.Join(dir, "binary"), filepath.Join(dir, "model")}
	for i, b := range [][]byte{raw, []byte("binary"), []byte("model")} {
		if err := os.WriteFile(paths[i], b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	var out bytes.Buffer
	if err := ImportOracleSmokeFiles(&out, paths[0], paths[1], paths[2], p); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths[1], []byte("wrong binary"), 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := ImportOracleSmokeFiles(&out, paths[0], paths[1], paths[2], p); err == nil || out.Len() != 0 {
		t.Fatal("wrong binary admitted")
	}
}
