package cudasotaeval

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"reflect"
	"sort"
	"syscall"
)

const ledgerLimit = 4 << 20

// QualificationClosure pins the source, executable and observation environment.
// A receipt transfers only when every field matches; source identity alone does
// not transfer native evidence to another executable or target. Unknown fields
// may be empty in pending cells, but passed receipts require a complete closure.
type QualificationClosure struct {
	Candidate          string            `json:"candidate"`
	Tree               string            `json:"tree"`
	Base               string            `json:"base"`
	SignatureMapSHA256 string            `json:"signature_map_sha256"`
	BinarySHA256       string            `json:"binary_sha256"`
	DependenciesSHA256 string            `json:"dependencies_sha256"`
	InputsSHA256       string            `json:"inputs_sha256"`
	OutputsSHA256      string            `json:"outputs_sha256"`
	HostOS             string            `json:"host_os"`
	HostArch           string            `json:"host_arch"`
	Device             string            `json:"device"`
	Architecture       string            `json:"architecture"`
	ProducerVersion    string            `json:"producer_version"`
	Libraries          map[string]string `json:"libraries"`
	Toolchain          string            `json:"toolchain"`
	Topology           string            `json:"topology"`
	Workload           string            `json:"workload"`
	Deployment         string            `json:"deployment"`
	Observer           string            `json:"observer"`
}

// QualificationCell separates discovery from evidence for one declared target.
// Capability is supported, denied, conflicting, unsupported or unknown. Status
// is not-selected, unavailable, pending, failed or qualified. Gate is host, ABI,
// native, backend, operations, observer, diagnostic or comparison.
type QualificationCell struct {
	ID           string                 `json:"id"`
	CapabilityID string                 `json:"capability_id"`
	Capability   string                 `json:"capability"`
	Gate         string                 `json:"gate"`
	Status       string                 `json:"status"`
	SourcePaths  []string               `json:"source_paths"`
	Consumers    []string               `json:"consumers"`
	DependsOn    []string               `json:"depends_on,omitempty"`
	Closure      QualificationClosure   `json:"closure"`
	Attempts     []QualificationAttempt `json:"attempts,omitempty"`
	Owner        string                 `json:"owner"`
	NextAction   string                 `json:"next_action"`
}

// QualificationEvidence is the immutable receipt envelope stored under an
// artifact root. RawSHA256 authenticates a separately retained raw output.
// Verifier identifies the external verifier, not this consistency checker.
// A valid envelope does not establish that its author's measurements are true.
type QualificationEvidence struct {
	CellID    string               `json:"cell_id"`
	Gate      string               `json:"gate"`
	Status    string               `json:"status"`
	Closure   QualificationClosure `json:"closure"`
	RawPath   string               `json:"raw_path"`
	RawSHA256 string               `json:"raw_sha256"`
	Command   []string             `json:"command"`
	Verifier  string               `json:"verifier"`
}

// QualificationAttempt preserves a previous pass, failure or withheld attempt.
// SHA256 authenticates the envelope, which is checked against Evidence.
type QualificationAttempt struct {
	Path     string                `json:"path"`
	SHA256   string                `json:"sha256"`
	Evidence QualificationEvidence `json:"evidence"`
}

// QualificationLedger is a bounded, versioned inventory, not a test executor.
// Commands retained in receipts are never executed by this package.
type QualificationLedger struct {
	Version int                 `json:"version"`
	Cells   []QualificationCell `json:"cells"`
}

// QualificationResult reports evidence consistency separately from discovery.
// Accepted means the retained envelope and raw output match the selected cell;
// it does not certify physical measurements or authorize a profiling run.
type QualificationResult struct {
	CellID     string `json:"cell_id"`
	Capability string `json:"capability"`
	Status     string `json:"status"`
	Gate       string `json:"gate"`
	Accepted   bool   `json:"accepted"`
	Reason     string `json:"reason,omitempty"`
	Owner      string `json:"owner"`
	NextAction string `json:"next_action"`
}

// ReadQualificationLedger reads at most 4 MiB and rejects unknown fields,
// duplicate JSON keys, trailing values and invalid dependency graphs.
func ReadQualificationLedger(r io.Reader) (*QualificationLedger, error) {
	if r == nil {
		return nil, errors.New("read ledger: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, ledgerLimit+1))
	if err != nil {
		return nil, fmt.Errorf("read ledger: %w", err)
	}
	if len(b) > ledgerLimit {
		return nil, errors.New("read ledger: size limit exceeded")
	}
	var l QualificationLedger
	if err := ledgerDecode(b, &l); err != nil {
		return nil, fmt.Errorf("read ledger: %w", err)
	}
	if err := l.Validate(); err != nil {
		return nil, err
	}
	return &l, nil
}

// WriteQualificationLedger validates and serializes a ledger. A short write is
// an error. It does not verify receipt files; use VerifyQualificationLedger.
func WriteQualificationLedger(w io.Writer, l QualificationLedger) error {
	if w == nil {
		return errors.New("write ledger: nil writer")
	}
	if err := l.Validate(); err != nil {
		return err
	}
	b, err := json.Marshal(l)
	if err != nil {
		return err
	}
	if len(b)+1 > ledgerLimit {
		return errors.New("write ledger: size limit exceeded")
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	return err
}

// Validate checks structural consistency without accessing receipts or hardware.
func (l QualificationLedger) Validate() error {
	if l.Version != 1 || len(l.Cells) == 0 || len(l.Cells) > 4096 {
		return errors.New("ledger: invalid version or cell count")
	}
	cells := make(map[string]QualificationCell, len(l.Cells))
	attempts := 0
	for _, c := range l.Cells {
		if c.ID == "" || c.CapabilityID == "" || c.Owner == "" || c.NextAction == "" || len(c.SourcePaths) == 0 || len(c.Consumers) == 0 {
			return fmt.Errorf("ledger: incomplete cell %q", c.ID)
		}
		if _, ok := cells[c.ID]; ok {
			return fmt.Errorf("ledger: duplicate cell %q", c.ID)
		}
		if !ledgerOneOf(c.Capability, "supported", "denied", "conflicting", "unsupported", "unknown") || !ledgerOneOf(c.Status, "not-selected", "unavailable", "pending", "failed", "qualified") || !ledgerGate(c.Gate) {
			return fmt.Errorf("ledger: invalid state in %q", c.ID)
		}
		for _, p := range append(append([]string{}, c.SourcePaths...), c.Consumers...) {
			if !fs.ValidPath(p) {
				return fmt.Errorf("ledger: invalid source or consumer path in %q", c.ID)
			}
		}
		if err := c.Closure.validate(c.Status == "qualified"); err != nil {
			return fmt.Errorf("ledger: cell %q: %w", c.ID, err)
		}
		if c.Status == "qualified" && (c.Capability != "supported" || len(c.Attempts) == 0) {
			return fmt.Errorf("ledger: qualified cell %q lacks supported evidence", c.ID)
		}
		attempts += len(c.Attempts)
		if len(c.Attempts) > 64 || attempts > 4096 {
			return errors.New("ledger: attempt count limit exceeded")
		}
		paths := map[string]bool{}
		for _, a := range c.Attempts {
			e := a.Evidence
			if !fs.ValidPath(a.Path) || paths[a.Path] || !ledgerDigest(a.SHA256, 64) || !fs.ValidPath(e.RawPath) || !ledgerDigest(e.RawSHA256, 64) || e.Verifier == "" || len(e.Command) == 0 || len(e.Command) > 128 || e.CellID != c.ID || e.Gate != c.Gate || !ledgerOneOf(e.Status, "passed", "failed", "withheld") {
				return fmt.Errorf("ledger: invalid attempt in %q", c.ID)
			}
			if err := e.Closure.validate(e.Status == "passed"); err != nil {
				return fmt.Errorf("ledger: attempt in %q: %w", c.ID, err)
			}
			paths[a.Path] = true
		}
		cells[c.ID] = c
	}
	state := map[string]int{}
	var visit func(string) error
	visit = func(id string) error {
		c, ok := cells[id]
		if !ok {
			return fmt.Errorf("ledger: unknown dependency %q", id)
		}
		if state[id] == 1 {
			return fmt.Errorf("ledger: dependency cycle at %q", id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		seen := map[string]bool{}
		for _, dep := range c.DependsOn {
			if seen[dep] {
				return fmt.Errorf("ledger: duplicate dependency %q", dep)
			}
			seen[dep] = true
			if err := visit(dep); err != nil {
				return err
			}
		}
		state[id] = 2
		return nil
	}
	for _, c := range l.Cells {
		if err := visit(c.ID); err != nil {
			return err
		}
	}
	return nil
}

// VerifyQualificationLedger checks retained evidence under root using a
// traversal-resistant os.Root. Missing, corrupt or scope-incompatible evidence
// produces pending results, never a promoted pass. All attempts are checked;
// only the last attempt can qualify a cell. Dependencies must also be accepted.
// Each artifact is bounded to 4 MiB and the combined read budget is 64 MiB.
func VerifyQualificationLedger(ctx context.Context, root string, l QualificationLedger) ([]QualificationResult, error) {
	if ctx == nil {
		return nil, errors.New("verify ledger: nil context")
	}
	if err := l.Validate(); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r, err := os.OpenRoot(root)
	if err != nil {
		return nil, fmt.Errorf("open artifact root: %w", err)
	}
	defer r.Close()
	remaining := int64(64 << 20)
	budgetErr := errors.New("artifact aggregate read budget exceeded")
	read := func(path, digest string) ([]byte, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if remaining <= 1 {
			return nil, budgetErr
		}
		limit := min(int64(ledgerLimit), remaining-1)
		// Nonblocking open prevents a malicious FIFO substitution from hanging
		// before f.Stat can reject non-regular files.
		f, err := r.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		info, err := f.Stat()
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("artifact is not a regular file")
		}
		if info.Size() > limit {
			return nil, errors.New("artifact size limit exceeded")
		}
		b, err := io.ReadAll(io.LimitReader(f, limit+1))
		remaining -= int64(len(b))
		if err != nil {
			return nil, err
		}
		if int64(len(b)) > limit {
			return nil, errors.New("artifact size limit exceeded")
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != digest {
			return nil, errors.New("artifact digest mismatch")
		}
		return b, nil
	}
	out := make(map[string]QualificationResult, len(l.Cells))
	cells := map[string]QualificationCell{}
	for _, c := range l.Cells {
		cells[c.ID] = c
		v := QualificationResult{CellID: c.ID, Capability: c.Capability, Status: c.Status, Gate: c.Gate, Owner: c.Owner, NextAction: c.NextAction}
		valid := len(c.Attempts) > 0
		for _, a := range c.Attempts {
			b, err := read(a.Path, a.SHA256)
			if errors.Is(err, budgetErr) {
				return nil, err
			}
			var observed QualificationEvidence
			if err == nil {
				err = ledgerDecode(b, &observed)
			}
			if err == nil && !reflect.DeepEqual(observed, a.Evidence) {
				err = errors.New("receipt differs from declared evidence")
			}
			if err == nil {
				_, err = read(observed.RawPath, observed.RawSHA256)
				if errors.Is(err, budgetErr) {
					return nil, err
				}
			}
			if err != nil {
				valid = false
				v.Reason = "receipt verification failed"
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if c.Status == "qualified" {
			if valid {
				last := c.Attempts[len(c.Attempts)-1].Evidence
				if last.Status != "passed" || !reflect.DeepEqual(last.Closure, c.Closure) {
					valid = false
					v.Reason = "latest receipt does not qualify selected closure"
				}
			}
			if valid {
				v.Accepted = true
			} else {
				v.Status = "pending"
				if v.Reason == "" {
					v.Reason = "missing qualification evidence"
				}
			}
		}
		out[c.ID] = v
	}
	done := map[string]bool{}
	var resolve func(string)
	resolve = func(id string) {
		if done[id] {
			return
		}
		c := cells[id]
		v := out[id]
		for _, dep := range c.DependsOn {
			resolve(dep)
			if v.Accepted && !out[dep].Accepted {
				v.Accepted = false
				v.Status = "pending"
				v.Reason = "dependency evidence not accepted"
			}
		}
		out[id] = v
		done[id] = true
	}
	ids := make([]string, 0, len(cells))
	for id := range cells {
		resolve(id)
		ids = append(ids, id)
	}
	sort.Strings(ids)
	result := make([]QualificationResult, 0, len(ids))
	for _, id := range ids {
		result = append(result, out[id])
	}
	return result, nil
}

func (c QualificationClosure) validate(complete bool) error {
	if !ledgerDigest(c.Candidate, 40) || !ledgerDigest(c.Tree, 40) {
		return errors.New("missing candidate or tree identity")
	}
	for _, digest := range []string{c.BinarySHA256, c.DependenciesSHA256, c.InputsSHA256, c.OutputsSHA256, c.SignatureMapSHA256} {
		if digest != "" && !ledgerDigest(digest, 64) {
			return errors.New("invalid closure digest")
		}
		if complete && digest == "" {
			return errors.New("incomplete qualification closure")
		}
	}
	if c.Base != "" && !ledgerDigest(c.Base, 40) {
		return errors.New("invalid base identity")
	}
	if complete && !ledgerDigest(c.Base, 40) {
		return errors.New("missing base identity")
	}
	if complete && (c.HostOS == "" || c.HostArch == "" || c.Device == "" || c.Architecture == "" || c.ProducerVersion == "" || c.Libraries == nil || c.Toolchain == "" || c.Topology == "" || c.Workload == "" || c.Deployment == "" || c.Observer == "") {
		return errors.New("incomplete target closure")
	}

	for k, v := range c.Libraries {
		if k == "" || v == "" {
			return errors.New("incomplete library identity")
		}
	}
	return nil
}
func ledgerDigest(s string, n int) bool {
	if len(s) != n {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
func ledgerGate(s string) bool {
	return ledgerOneOf(s, "host", "ABI", "native", "backend", "operations", "observer", "diagnostic", "comparison")
}
func ledgerOneOf(s string, values ...string) bool {
	for _, v := range values {
		if s == v {
			return true
		}
	}
	return false
}

func ledgerDecode(b []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var value func() error
	value = func() error {
		t, err := d.Token()
		if err != nil {
			return err
		}
		if delim, ok := t.(json.Delim); ok {
			if delim == '{' {
				seen := map[string]bool{}
				for d.More() {
					k, err := d.Token()
					if err != nil {
						return err
					}
					s, ok := k.(string)
					if !ok || seen[s] {
						return errors.New("duplicate JSON field")
					}
					seen[s] = true
					if err := value(); err != nil {
						return err
					}
				}
			} else if delim == '[' {
				for d.More() {
					if err := value(); err != nil {
						return err
					}
				}
			} else {
				return errors.New("unexpected JSON delimiter")
			}
			_, err = d.Token()
			return err
		}
		return nil
	}
	if err := value(); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing JSON value")
	}
	d = json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	return d.Decode(v)
}
