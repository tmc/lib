package cudarepro

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"reflect"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// ReadFinding reads one serialized campaign SeedResult and checks its source,
// comparison identity and retained confirmation evidence. Input is limited to
// 1 MiB, 32 nesting levels and 65536 JSON tokens. Unknown or duplicate fields,
// trailing data and incomplete historical records are errors. Free-form input
// and geometry labels remain labels; this function does not turn them into a
// runnable Case or substitute historical confirmations for a fresh baseline.
func ReadFinding(reader io.Reader, source []byte) (cudaschedule.SeedResult, error) {
	var finding cudaschedule.SeedResult
	if reader == nil {
		return finding, fmt.Errorf("read finding: missing reader")
	}
	const limit = 1 << 20
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return finding, fmt.Errorf("read finding: %w", err)
	}
	if len(data) > limit {
		return finding, fmt.Errorf("read finding: record exceeds size limit")
	}
	if err := uniqueRecord(data); err != nil {
		return finding, fmt.Errorf("read finding: %w", err)
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err := d.Decode(&finding); err != nil {
		return finding, fmt.Errorf("read finding: %w", err)
	}
	if finding.Status != cudaschedule.SeedExecuted || finding.Run == nil || finding.Verdict != cudaschedule.VerdictMismatch {
		return finding, fmt.Errorf("read finding: missing executed mismatch record")
	}
	if finding.Report.Seed != finding.Seed {
		return finding, fmt.Errorf("read finding: seed and rewrite report disagree")
	}
	digest := sha256.Sum256(source)
	if len(source) == 0 || finding.PristineSourceDigest != hex.EncodeToString(digest[:]) {
		return finding, fmt.Errorf("read finding: pristine source digest mismatch")
	}
	identity, err := Identify(*finding.Run)
	if err != nil {
		return finding, fmt.Errorf("read finding: %w", err)
	}
	rewrite, err := cudaschedule.Replay(string(source), cudaschedule.Options{
		Seed: finding.Seed, Intensity: finding.Report.Intensity,
		MaxSleepNs: finding.MaxSleepNs, Focus: finding.Focus,
	}, finding.Report.Decisions)
	if err != nil {
		return finding, fmt.Errorf("read finding: replay recorded decisions: %w", err)
	}
	if !reflect.DeepEqual(rewrite.Report, finding.Report) {
		return finding, fmt.Errorf("read finding: rewrite report disagrees with recorded decisions")
	}
	loaded := sha256.Sum256([]byte(rewrite.PTX))
	if finding.Run.ExpectedPTXDigest != hex.EncodeToString(loaded[:]) {
		return finding, fmt.Errorf("read finding: loaded candidate differs from recorded source and decisions")
	}
	c := finding.Confirmation
	if c == nil || c.Attempts <= 0 || len(c.Runs) != c.Attempts || len(c.Outcomes) != c.Attempts {
		return finding, fmt.Errorf("read finding: missing confirmation execution records")
	}
	if !reflect.DeepEqual(c.Runs[0], *finding.Run) {
		return finding, fmt.Errorf("read finding: initial confirmation differs from seed run")
	}
	count := 0
	for n, run := range c.Runs {
		if run.ExpectedPTXDigest != finding.Run.ExpectedPTXDigest {
			return finding, fmt.Errorf("read finding: confirmation %d loaded another candidate", n+1)
		}
		v := cudaschedule.Classify(run)
		if v != c.Outcomes[n] {
			return finding, fmt.Errorf("read finding: confirmation %d verdict mismatch", n+1)
		}
		ok, err := identity.Matches(run)
		if err != nil {
			return finding, fmt.Errorf("read finding: confirmation %d: %w", n+1, err)
		}
		if ok {
			count++
		} else if v != cudaschedule.VerdictPass {
			return finding, fmt.Errorf("read finding: confirmation %d identity drift", n+1)
		}
	}
	if count != c.Findings || count == 0 {
		return finding, fmt.Errorf("read finding: inconsistent historical reproduction count")
	}
	if c.Confirmed && count != c.Attempts {
		return finding, fmt.Errorf("read finding: partial confirmations marked confirmed")
	}
	return finding, nil
}

func uniqueRecord(data []byte) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	remaining := 65536
	token := func() (json.Token, error) {
		if remaining == 0 {
			return nil, fmt.Errorf("too many json tokens")
		}
		remaining--
		return d.Token()
	}
	var value func(int) error
	value = func(depth int) error {
		if depth > 32 {
			return fmt.Errorf("json nesting exceeds limit")
		}
		t, err := token()
		if err != nil {
			return err
		}
		delim, ok := t.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			keys := make(map[string]bool)
			for d.More() {
				k, err := token()
				if err != nil {
					return err
				}
				key, ok := k.(string)
				if !ok {
					return fmt.Errorf("invalid object key")
				}
				for _, c := range key {
					if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_') {
						return fmt.Errorf("noncanonical field %q", key)
					}
				}
				if keys[key] {
					return fmt.Errorf("duplicate field %q", key)
				}
				keys[key] = true
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		case '[':
			for d.More() {
				if err := value(depth + 1); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("invalid json delimiter")
		}
		end, err := token()
		if err != nil {
			return err
		}
		if (delim == '{' && end != json.Delim('}')) || (delim == '[' && end != json.Delim(']')) {
			return fmt.Errorf("invalid closing delimiter")
		}
		return nil
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("trailing json data")
	}
	return nil
}
