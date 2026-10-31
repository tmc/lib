package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// OperatorCPUContext retains an imported CPU annotation trace and optional
// separately hashed code facts and evidence. References identify retained
// bytes, not native operations or contributing members. The zero value is
// invalid. Artifacts must contain exactly the referenced raw payloads.
type OperatorCPUContext struct {
	Version     int                 `json:"version"`
	RunID       string              `json:"run_id"`
	TraceSource WorkloadSource      `json:"trace_source"`
	Trace       OperatorReference   `json:"trace"`
	Artifacts   map[string][]byte   `json:"artifacts"`
	CodeFacts   []OperatorReference `json:"code_facts,omitempty"`
	Evidence    []OperatorReference `json:"evidence,omitempty"`
}

// ReadOperatorCPUContext reads application operator-cpu-context-v1 metadata.
// It checks bounded strict JSON and the closed raw-hash map, without opening
// reference locators or parsing another producer's trace. TraceSource must name
// PyTorch 2.10.0 in the same run; ReadPyTorchAnnotations owns trace validation.
func ReadOperatorCPUContext(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, OperatorCPUContext, error) {
	var c OperatorCPUContext
	if s.Provider != "application" || s.Version != "operator-cpu-context-v1" {
		return WorkloadExport{}, c, errors.New("operator CPU context requires application operator-cpu-context-v1 source")
	}
	e, l, err := readWorkload(r, s, l)
	if err != nil {
		return e, c, err
	}
	if err := strictOperatorJSON(e.Raw, &c); err != nil {
		return e, OperatorCPUContext{}, err
	}
	if err := validateOperatorCPUContext(c, s.RunID, l); err != nil {
		return e, OperatorCPUContext{}, err
	}
	return e, c, nil
}

// WriteOperatorCPUContext writes validated, bounded CPU-context metadata.
// Successful serialization does not authenticate its producer or establish a
// relationship between annotations and the separately referenced code facts.
func WriteOperatorCPUContext(w io.Writer, c OperatorCPUContext, l WorkloadLimits) error {
	l, err := workloadLimits(l)
	if err != nil {
		return err
	}
	if w == nil {
		return errors.New("missing operator CPU context writer")
	}
	if err := validateOperatorCPUContext(c, c.RunID, l); err != nil {
		return err
	}
	raw, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("operator CPU context byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		err = io.ErrShortWrite
	}
	return err
}

func validateOperatorCPUContext(c OperatorCPUContext, run string, l WorkloadLimits) error {
	if c.Version != 1 || strings.TrimSpace(c.RunID) == "" || c.RunID != run || c.TraceSource.Provider != "pytorch" || c.TraceSource.Version != "2.10.0" || c.TraceSource.Engine != "" || c.TraceSource.RunID != c.RunID {
		return errors.New("invalid operator CPU context version or run/source identity")
	}
	if c.TraceSource.Rank != nil && *c.TraceSource.Rank < 0 {
		return errors.New("negative operator CPU trace rank")
	}
	if len(c.Artifacts) > l.MaxMetrics || len(c.TraceSource.Units) > l.MaxMetrics || len(c.CodeFacts) > l.MaxMetrics-1 || len(c.Evidence) > l.MaxMetrics-1-len(c.CodeFacts) {
		return errors.New("operator CPU context collection limit exceeded")
	}
	// Bound caller-owned strings and base64 payloads before JSON allocation.
	// The fixed overhead covers field names, scalar values and punctuation;
	// per-reference/map overhead is charged separately below.
	remaining := l.MaxBytes
	charge := func(n int64) error {
		if n > remaining {
			return errors.New("operator CPU context byte limit exceeded")
		}
		remaining -= n
		return nil
	}
	stringBytes := func(s string) error {
		if len(s) > l.MaxStringBytes {
			return errors.New("operator CPU context string limit exceeded")
		}
		return charge(int64(len(s))*6 + 3)
	}
	if err := charge(512); err != nil {
		return err
	}
	for _, s := range []string{c.RunID, c.TraceSource.Provider, c.TraceSource.Version, c.TraceSource.Engine, c.TraceSource.RunID, c.TraceSource.Phase, c.TraceSource.ClockDomain, c.TraceSource.ClockEpochID} {
		if err := stringBytes(s); err != nil {
			return err
		}
	}
	for k, v := range c.TraceSource.Units {
		if err := stringBytes(k); err != nil {
			return err
		}
		if err := stringBytes(v); err != nil {
			return err
		}
	}
	referenced := make(map[string]bool)
	check := func(r OperatorReference) error {
		if len(r.SHA256) != 2*sha256.Size || !operatorHash(r.SHA256) || strings.TrimSpace(r.Locator) == "" {
			return errors.New("invalid operator CPU reference")
		}
		if _, ok := c.Artifacts[r.SHA256]; !ok {
			return errors.New("operator CPU reference has no retained bytes")
		}
		if err := charge(32); err != nil {
			return err
		}
		if err := stringBytes(r.SHA256); err != nil {
			return err
		}
		if err := stringBytes(r.Locator); err != nil {
			return err
		}
		referenced[r.SHA256] = true
		return nil
	}
	if err := check(c.Trace); err != nil {
		return err
	}
	for _, refs := range [][]OperatorReference{c.CodeFacts, c.Evidence} {
		for _, ref := range refs {
			if err := check(ref); err != nil {
				return err
			}
		}
	}
	for hash, raw := range c.Artifacts {
		if len(hash) != 2*sha256.Size || !referenced[hash] || !operatorHash(hash) || int64(len(raw)) > l.MaxBytes {
			return errors.New("unreferenced or oversized operator CPU artifact")
		}
		if err := stringBytes(hash); err != nil {
			return err
		}
		if err := charge((int64(len(raw))+2)/3*4 + 4); err != nil {
			return err
		}
		sum := sha256.Sum256(raw)
		if hex.EncodeToString(sum[:]) != hash {
			return errors.New("operator CPU raw artifact hash mismatch")
		}
	}
	// Reuse the existing source validation and copy rules. No producer payload
	// is parsed here; the empty reader is only a bounded source check.
	if _, _, err := readWorkload(bytes.NewReader(nil), c.TraceSource, l); err != nil {
		return fmt.Errorf("operator CPU trace source: %w", err)
	}
	return nil
}
