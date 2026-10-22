package dsv4eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"strconv"
)

// OracleSmokeProvenance is an independently recorded execution receipt.
// Paths passed to ImportOracleSmokeFiles must contain the actual retained
// oracle executable and model, not files named by an unverified trace.
type OracleSmokeProvenance struct {
	Build          string `json:"build"`
	BinarySHA256   string `json:"binary_sha256"`
	ModelSHA256    string `json:"model_sha256"`
	RawTraceSHA256 string `json:"raw_trace_sha256"`
	RawModelPath   string `json:"raw_model_path"`
	Backend        string `json:"backend"`
	Context        int    `json:"ctx"`
	Temperature    string `json:"temp"`
	Think          string `json:"think"`
}

// ImportOracleSmokeFiles verifies all three retained artifacts and attaches
// their provenance to a raw upstream trace. It does not modify case headers,
// fields, outputs, statuses or summary bytes. Build identifies the independently
// attested source revision; hashing a binary does not establish that revision.
func ImportOracleSmokeFiles(w io.Writer, rawPath, binaryPath, modelPath string, p OracleSmokeProvenance) error {
	for _, a := range []struct{ path, want string }{{rawPath, p.RawTraceSHA256}, {binaryPath, p.BinarySHA256}, {modelPath, p.ModelSHA256}} {
		if !validSHA(a.want) {
			return fmt.Errorf("import smoke: missing or malformed artifact hash")
		}
		f, err := os.Open(a.path)
		if err != nil {
			return fmt.Errorf("import smoke: open artifact: %w", err)
		}
		h := sha256.New()
		_, readErr := io.Copy(h, f)
		closeErr := f.Close()
		if readErr != nil {
			return fmt.Errorf("import smoke: hash artifact: %w", readErr)
		}
		if closeErr != nil {
			return closeErr
		}
		if hex.EncodeToString(h.Sum(nil)) != a.want {
			return fmt.Errorf("import smoke: artifact SHA-256 mismatch for %s", a.path)
		}
	}
	raw, err := os.ReadFile(rawPath)
	if err != nil {
		return err
	}
	return importOracleSmoke(w, raw, p)
}

func validSHA(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func importOracleSmoke(w io.Writer, raw []byte, p OracleSmokeProvenance) error {
	if !validSHA(p.BinarySHA256) || !validSHA(p.ModelSHA256) || !validSHA(p.RawTraceSHA256) || p.Build == "" || p.RawModelPath == "" || p.Backend == "" || p.Context <= 0 || p.Temperature != "0" || p.Think != "HIGH" {
		return fmt.Errorf("import smoke: incomplete execution provenance")
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != p.RawTraceSHA256 {
		return fmt.Errorf("import smoke: raw trace changed")
	}
	t, err := ReadTrace(bytes.NewReader(raw))
	if err != nil {
		return err
	}
	f := t.Header.Fields
	seen := make(map[string]bool)
	for _, c := range t.Cases {
		key := c.Source + "\x00" + c.ID
		if !hardSmokeCases[key] || seen[key] {
			return fmt.Errorf("import smoke: unexpected or duplicate hard-smoke case %s/%s", c.Source, c.ID)
		}
		seen[key] = true
	}
	for key, want := range map[string]string{"suite": "hard-smoke", "model": p.RawModelPath, "backend": p.Backend, "ctx": strconv.Itoa(p.Context), "temperature": p.Temperature, "think_mode_requested": "high", "questions": "12", "soft_limit_reply_budget": "1024", "hard_limit_reply_budget": "512", "soft_limit_think_close_rank": "3"} {
		if f[key] != want {
			return fmt.Errorf("import smoke: %s=%q, want %q", key, f[key], want)
		}
	}
	for _, key := range []string{"build", "model_sha256", "temp", "think", "oracle_binary_sha256", "raw_trace_sha256"} {
		if _, ok := f[key]; ok {
			return fmt.Errorf("import smoke: raw trace already contains attested field %s", key)
		}
	}
	first := bytes.Index(raw, []byte("===== CASE "))
	if first < 0 {
		return fmt.Errorf("import smoke: no cases")
	}
	f["build"], f["model_sha256"], f["temp"], f["think"] = p.Build, p.ModelSHA256, p.Temperature, p.Think
	f["oracle_binary_sha256"], f["raw_trace_sha256"] = p.BinarySHA256, p.RawTraceSHA256
	var out bytes.Buffer
	writer := NewTraceWriter(&out, TraceHeader{Fields: f})
	if writer.err != nil {
		return writer.err
	}
	out.Write(raw[first:])
	if _, err := readSmokeTrace(bytes.NewReader(out.Bytes()), "imported oracle"); err != nil {
		return fmt.Errorf("import smoke: %w", err)
	}
	n, err := w.Write(out.Bytes())
	if err == nil && n != out.Len() {
		return io.ErrShortWrite
	}
	return err
}
