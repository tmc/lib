package dcgmjson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/dcgm"
)

// UnsupportedError reports an unsupported producer version or document shape.
// Version is empty unless a version string was actually present in the input.
type UnsupportedError struct{ Version, Format string }

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("dcgmjson: unsupported %s profile, version %q", e.Format, e.Version)
}

// Limits bounds bytes and total categories/tests/results/warnings/info records.
// Zero fields use 4 MiB and 65536 records; maxima are 16 MiB and 262144.
type Limits struct{ Bytes, Records int }

// Report contains the actual producer version and independently available fields.
// CategoriesAvailable distinguishes omission from a reported empty list.
// Unknown retains root fields, including raw entity inventory and metadata.
type Report struct {
	Version             string
	CategoriesAvailable bool
	Categories          []Category
	Unknown             map[string]json.RawMessage
	DiagnosticUnknown   map[string]json.RawMessage
	Raw                 []byte
}

// Category groups named tests, preserving uninterpreted additional fields.
type Category struct {
	Name    string
	Tests   []Test
	Unknown map[string]json.RawMessage
}

// Test retains per-entity results and an optional aggregate summary. An omitted
// results field does not identify which GPUs did not run or why.
type Test struct {
	Name             string
	ResultsAvailable bool
	Results          []Result
	Summary          *Result
	Unknown          map[string]json.RawMessage
}

// Result preserves a literal status and optional entity identity. Summary rows
// normally omit entity identity. Warnings/info availability is independent of
// whether the corresponding list is empty. Status Fail may include DCGM WARN.
type Result struct {
	Status                           string
	Entity                           *Entity
	WarningsAvailable, InfoAvailable bool
	Warnings                         []Warning
	Info                             []string
	Unknown                          map[string]json.RawMessage
}

// Entity is an explicitly reported entity, not necessarily a GPU. The GPU
// group is identified by the producer's numeric entity_group_id value 1.
type Entity struct {
	Group       string
	GroupID, ID uint32
}

// Warning retains the producer's numeric diagnostic code, category and severity.
// Message does not supply a guessed code; unknown codes remain numeric values.
type Warning struct {
	Code               dcgm.ErrorCode
	Category, Severity uint32
	Message            string
	Unknown            map[string]json.RawMessage
}

// Read parses a complete bounded document without synthesizing missing fields.
// Only the source-pinned 4.6.1 single-iteration profile is supported. Multi-
// iteration, older gpu_id profiles and runtime-error-only documents are refused.
// The caller must cancel an io.Reader that can block.
func Read(r io.Reader, l Limits) (*Report, error) {
	if l.Bytes == 0 {
		l.Bytes = 4 << 20
	}
	if l.Records == 0 {
		l.Records = 65536
	}
	if r == nil || l.Bytes < 1 || l.Bytes > 16<<20 || l.Records < 1 || l.Records > 262144 {
		return nil, errors.New("dcgmjson: invalid reader or limits")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("dcgmjson read: %w", err)
	}
	if len(b) > l.Bytes {
		return nil, errors.New("dcgmjson: byte limit exceeded")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	if err := unique(d, 0); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("dcgmjson: trailing JSON")
	}
	root, err := object(b)
	if err != nil {
		return nil, err
	}
	out := &Report{Raw: bytes.Clone(b)}
	if metadata, ok := root["metadata"]; ok {
		m, e := object(metadata)
		if e != nil {
			return nil, e
		}
		if version, ok := m["version"]; ok {
			if e = json.Unmarshal(version, &out.Version); e != nil || out.Version == "" {
				return nil, errors.New("dcgmjson: invalid reported version")
			}
		}
	}
	unsupported := func() (*Report, error) {
		return nil, &UnsupportedError{Version: out.Version, Format: "single-iteration diagnostic"}
	}
	if out.Version != "4.6.1" {
		return unsupported()
	}
	body, ok := root["DCGM Diagnostic"]
	if !ok {
		return unsupported()
	}
	diag, err := object(body)
	if err != nil {
		return nil, err
	}
	if _, ok := diag["runtime_error"]; ok {
		return unsupported()
	}
	if _, ok := root["iterations"]; ok {
		return unsupported()
	}
	budget := l.Records
	take := func(n int) error {
		budget -= n
		if budget < 0 {
			return errors.New("dcgmjson: record limit exceeded")
		}
		return nil
	}
	categories, ok := diag["test_categories"]
	out.CategoriesAvailable = ok
	if ok {
		rows, e := array(categories)
		if e != nil {
			return nil, e
		}
		if e = take(len(rows)); e != nil {
			return nil, e
		}
		for _, row := range rows {
			m, e := object(row)
			if e != nil {
				return nil, e
			}
			var c Category
			if e = required(m, "category", &c.Name); e != nil || c.Name == "" {
				return nil, errors.New("dcgmjson: category name required")
			}
			tests, e := array(m["tests"])
			if e != nil {
				return nil, e
			}
			if e = take(len(tests)); e != nil {
				return nil, e
			}
			for _, test := range tests {
				tm, e := object(test)
				if e != nil {
					return nil, e
				}
				var t Test
				if e = required(tm, "name", &t.Name); e != nil || t.Name == "" {
					return nil, errors.New("dcgmjson: test name required")
				}
				if results, ok := tm["results"]; ok {
					t.ResultsAvailable = true
					rs, e := array(results)
					if e != nil {
						return nil, e
					}
					if e = take(len(rs)); e != nil {
						return nil, e
					}
					for _, rr := range rs {
						v, e := readResult(rr, false, out.Version, take)
						if e != nil {
							return nil, e
						}
						t.Results = append(t.Results, *v)
					}
				}
				if summary, ok := tm["test_summary"]; ok {
					if e = take(1); e != nil {
						return nil, e
					}
					t.Summary, e = readResult(summary, true, out.Version, take)
					if e != nil {
						return nil, e
					}
				}
				delete(tm, "results")
				delete(tm, "test_summary")
				t.Unknown = tm
				c.Tests = append(c.Tests, t)
			}
			delete(m, "tests")
			c.Unknown = m
			out.Categories = append(out.Categories, c)
		}
	}
	delete(diag, "test_categories")
	out.DiagnosticUnknown = diag
	delete(root, "DCGM Diagnostic")
	out.Unknown = root
	return out, nil
}

func readResult(raw []byte, summary bool, version string, take func(int) error) (*Result, error) {
	m, err := object(raw)
	if err != nil {
		return nil, err
	}
	v := &Result{}
	if err = required(m, "status", &v.Status); err != nil {
		return nil, err
	}
	switch v.Status {
	case "Pass", "Fail", "Skip":
	default:
		return nil, &UnsupportedError{Version: version, Format: "result status"}
	}
	_, a := m["entity_group"]
	_, b := m["entity_group_id"]
	_, c := m["entity_id"]
	if a || b || c {
		if !(a && b && c) {
			return nil, errors.New("dcgmjson: partial entity identity")
		}
		v.Entity = &Entity{}
		if err = required(m, "entity_group", &v.Entity.Group); err != nil || v.Entity.Group == "" {
			return nil, errors.New("dcgmjson: entity group required")
		}
		if err = required(m, "entity_group_id", &v.Entity.GroupID); err != nil {
			return nil, err
		}
		if err = required(m, "entity_id", &v.Entity.ID); err != nil {
			return nil, err
		}
	} else if !summary {
		return nil, errors.New("dcgmjson: result entity identity required")
	}
	if _, ok := m["gpu_id"]; ok {
		return nil, &UnsupportedError{Version: version, Format: "legacy gpu_id"}
	}
	if raw, ok := m["warnings"]; ok {
		v.WarningsAvailable = true
		rows, err := array(raw)
		if err != nil {
			return nil, err
		}
		if err = take(len(rows)); err != nil {
			return nil, err
		}
		for _, row := range rows {
			w, err := object(row)
			if err != nil {
				return nil, err
			}
			var warning Warning
			if err = required(w, "error_id", &warning.Code); err != nil {
				return nil, err
			}
			if err = required(w, "error_category", &warning.Category); err != nil {
				return nil, err
			}
			if err = required(w, "error_severity", &warning.Severity); err != nil {
				return nil, err
			}
			if err = required(w, "warning", &warning.Message); err != nil {
				return nil, err
			}
			warning.Unknown = w
			v.Warnings = append(v.Warnings, warning)
		}
	}
	if raw, ok := m["info"]; ok {
		v.InfoAvailable = true
		rows, err := array(raw)
		if err != nil {
			return nil, err
		}
		if err = take(len(rows)); err != nil {
			return nil, err
		}
		for _, row := range rows {
			var s string
			if err = json.Unmarshal(row, &s); err != nil || bytes.Equal(bytes.TrimSpace(row), []byte("null")) {
				return nil, errors.New("dcgmjson: info string required")
			}
			v.Info = append(v.Info, s)
		}
	}
	delete(m, "warnings")
	delete(m, "info")
	v.Unknown = m
	return v, nil
}

func object(b []byte) (map[string]json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return nil, errors.New("dcgmjson: object required")
	}
	return m, nil
}
func array(b []byte) ([]json.RawMessage, error) {
	var a []json.RawMessage
	if err := json.Unmarshal(b, &a); err != nil || a == nil {
		return nil, errors.New("dcgmjson: array required")
	}
	return a, nil
}
func required(m map[string]json.RawMessage, key string, value any) error {
	b, ok := m[key]
	if !ok || bytes.Equal(bytes.TrimSpace(b), []byte("null")) {
		return fmt.Errorf("dcgmjson: %s required", key)
	}
	if err := json.Unmarshal(b, value); err != nil {
		return fmt.Errorf("dcgmjson %s: %w", key, err)
	}
	delete(m, key)
	return nil
}

func unique(d *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("dcgmjson: nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			k, ok := t.(string)
			if !ok || seen[k] {
				return errors.New("dcgmjson: duplicate key")
			}
			seen[k] = true
			if err = unique(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err = unique(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("dcgmjson: invalid JSON delimiter")
	}
	_, err = d.Token()
	return err
}
