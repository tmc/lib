package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/gpucampaign"
)

const verifyCaptureUsage = "usage: cudaprof verify-capture [-json] <capture.json>..."

// CaptureQualification is the result of validating one capture receipt against
// the strict qualification gate.
type CaptureQualification struct {
	Path                   string   `json:"path,omitempty"`
	Status                 string   `json:"status"`
	Qualified              bool     `json:"qualified"`
	Reason                 string   `json:"reason,omitempty"`
	Reasons                []string `json:"reasons,omitempty"`
	CaptureSuccess         *bool    `json:"capture_success,omitempty"`
	RunCompleteness        string   `json:"run_completeness,omitempty"`
	TargetExitCode         *int     `json:"target_exit_code,omitempty"`
	CollectorFinalized     *bool    `json:"collector_finalized,omitempty"`
	UnqualifiedGenerations *uint64  `json:"unqualified_generations,omitempty"`
	ActivityRecords        *uint64  `json:"activity_records,omitempty"`
	DroppedRecords         *uint64  `json:"dropped_records,omitempty"`
	CaptureError           string   `json:"capture_error,omitempty"`
	TerminalError          string   `json:"terminal_error,omitempty"`
}

// VerifyCaptureReport is the JSON output of "cudaprof verify-capture -json".
type VerifyCaptureReport struct {
	Status                 string                 `json:"status"`
	Qualified              bool                   `json:"qualified"`
	Path                   string                 `json:"path,omitempty"`
	Reason                 string                 `json:"reason,omitempty"`
	Reasons                []string               `json:"reasons,omitempty"`
	CaptureSuccess         *bool                  `json:"capture_success,omitempty"`
	RunCompleteness        string                 `json:"run_completeness,omitempty"`
	TargetExitCode         *int                   `json:"target_exit_code,omitempty"`
	CollectorFinalized     *bool                  `json:"collector_finalized,omitempty"`
	UnqualifiedGenerations *uint64                `json:"unqualified_generations,omitempty"`
	ActivityRecords        *uint64                `json:"activity_records,omitempty"`
	DroppedRecords         *uint64                `json:"dropped_records,omitempty"`
	CaptureError           string                 `json:"capture_error,omitempty"`
	TerminalError          string                 `json:"terminal_error,omitempty"`
	Captures               []CaptureQualification `json:"captures"`
}

func runVerifyCaptureCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("verify-capture", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOutput := fs.Bool("json", false, "write a JSON qualification report")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			_, err = fmt.Fprintln(stdout, verifyCaptureUsage)
			return err
		}
		return fmt.Errorf("verify-capture: %w", err)
	}
	if fs.NArg() < 1 {
		return errors.New(verifyCaptureUsage)
	}

	results := make([]CaptureQualification, 0, fs.NArg())
	var failed []string
	for _, path := range fs.Args() {
		q := VerifyCaptureFile(path)
		results = append(results, q)
		if !q.Qualified {
			failed = append(failed, fmt.Sprintf("%s: %s", path, q.Reason))
		}
	}

	allQualified := len(failed) == 0
	if *jsonOutput {
		rep := VerifyCaptureReport{
			Status:    "PASS",
			Qualified: true,
			Captures:  results,
		}
		if !allQualified {
			rep.Status = "HOLD"
			rep.Qualified = false
			rep.Reasons = failed
			rep.Reason = strings.Join(failed, "; ")
		}
		if len(results) == 1 {
			c := results[0]
			rep.Path = c.Path
			rep.Reason = c.Reason
			rep.Reasons = c.Reasons
			rep.CaptureSuccess = c.CaptureSuccess
			rep.RunCompleteness = c.RunCompleteness
			rep.TargetExitCode = c.TargetExitCode
			rep.CollectorFinalized = c.CollectorFinalized
			rep.UnqualifiedGenerations = c.UnqualifiedGenerations
			rep.ActivityRecords = c.ActivityRecords
			rep.DroppedRecords = c.DroppedRecords
			rep.CaptureError = c.CaptureError
			rep.TerminalError = c.TerminalError
		}
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			return err
		}
	} else {
		for _, q := range results {
			if q.Qualified {
				var act uint64
				if q.ActivityRecords != nil {
					act = *q.ActivityRecords
				}
				if _, err := fmt.Fprintf(stdout, "%s: PASS (run_completeness=%s, activity_records=%d)\n", q.Path, q.RunCompleteness, act); err != nil {
					return err
				}
			} else {
				if _, err := fmt.Fprintf(stdout, "%s: HOLD (%s)\n", q.Path, q.Reason); err != nil {
					return err
				}
			}
		}
	}

	if !allQualified {
		return &exitCodeError{
			Code: 1,
			Err:  fmt.Errorf("verify-capture: %s", strings.Join(failed, "; ")),
		}
	}
	return nil
}

// VerifyCaptureFile reads path and validates it against the strict capture
// qualification gate.
func VerifyCaptureFile(path string) CaptureQualification {
	data, err := os.ReadFile(path)
	if err != nil {
		q := CaptureQualification{
			Path:      path,
			Status:    "HOLD",
			Qualified: false,
			Reason:    err.Error(),
			Reasons:   []string{err.Error()},
		}
		return q
	}
	q := VerifyCapture(data)
	q.Path = path
	return q
}

// VerifyCapture validates raw capture receipt JSON (or capture log containing a
// JSON receipt line) against the strict qualification gate:
//   - capture_success == true
//   - run_completeness == "complete"
//   - target_exit_code == 0
//   - collector_finalized == true
//   - unqualified_generations == 0
//   - activity_records > 0
//   - no terminal CUDA or CUPTI error in stdout/stderr/error/failure fields
func VerifyCapture(data []byte) CaptureQualification {
	q := CaptureQualification{Status: "HOLD"}
	addReason := func(msg string) {
		for _, r := range q.Reasons {
			if r == msg {
				return
			}
		}
		q.Reasons = append(q.Reasons, msg)
	}

	if term := gpucampaign.DetectTerminalCUDAError(data); term != "" {
		q.TerminalError = term
		addReason("terminal CUDA error: " + term)
	}

	records, parseErr := extractCaptureRecords(data)
	if parseErr != nil {
		addReason(parseErr.Error())
		q.Reason = strings.Join(q.Reasons, "; ")
		return q
	}
	if len(records) == 0 {
		addReason("missing capture completeness receipt")
		q.Reason = strings.Join(q.Reasons, "; ")
		return q
	}
	if len(records) > 1 {
		addReason(fmt.Sprintf("ambiguous capture completeness receipt: found %d records", len(records)))
		q.Reason = strings.Join(q.Reasons, "; ")
		return q
	}

	rec := records[0]

	// 1. capture_success == true
	if raw, ok := rec["capture_success"]; !ok {
		addReason("capture_success is missing")
	} else if b, ok := raw.(bool); !ok {
		addReason("capture_success is not a boolean")
	} else {
		q.CaptureSuccess = &b
		if !b {
			addReason("capture_success is false")
		}
	}

	// 2. run_completeness == "complete"
	if raw, ok := rec["run_completeness"]; !ok {
		addReason(`run_completeness is missing, want "complete"`)
	} else if s, ok := raw.(string); !ok {
		addReason(`run_completeness is not a string, want "complete"`)
	} else {
		q.RunCompleteness = s
		if s != runComplete {
			addReason(fmt.Sprintf("run_completeness is %q, want %q", s, runComplete))
		}
	}

	// 3. target_exit_code == 0
	if raw, ok := rec["target_exit_code"]; !ok {
		addReason("target_exit_code is missing")
	} else if code, ok := parseJSONInt(raw); !ok {
		addReason("target_exit_code is not an integer")
	} else {
		q.TargetExitCode = &code
		if code != 0 {
			addReason(fmt.Sprintf("target_exit_code is %d, want 0", code))
		}
	}

	// 4. collector_finalized == true
	if raw, ok := rec["collector_finalized"]; !ok {
		addReason("collector_finalized is missing")
	} else if b, ok := raw.(bool); !ok {
		addReason("collector_finalized is not a boolean")
	} else {
		q.CollectorFinalized = &b
		if !b {
			addReason("collector_finalized is false")
		}
	}

	// 5. unqualified_generations == 0
	if raw, ok := rec["unqualified_generations"]; !ok {
		zero := uint64(0)
		if wgRaw, hasWG := rec["worker_generations"]; hasWG {
			if b, err := json.Marshal(wgRaw); err == nil {
				var workers []captureWorker
				if json.Unmarshal(b, &workers) == nil {
					zero = unqualifiedGenerations(workers)
				}
			}
		}
		q.UnqualifiedGenerations = &zero
		if zero != 0 {
			addReason(fmt.Sprintf("unqualified_generations is %d, want 0", zero))
		}
	} else if u, ok := parseJSONUint64(raw); !ok {
		addReason("unqualified_generations is not a nonnegative integer")
	} else {
		q.UnqualifiedGenerations = &u
		if u != 0 {
			addReason(fmt.Sprintf("unqualified_generations is %d, want 0", u))
		}
	}

	// 6. activity_records > 0
	if raw, ok := rec["activity_records"]; !ok {
		addReason("activity_records is missing")
	} else if act, ok := parseJSONUint64(raw); !ok {
		addReason("activity_records is not a nonnegative integer")
	} else {
		q.ActivityRecords = &act
		if act == 0 {
			addReason("activity_records is 0, want > 0")
		}
	}

	// 7. dropped_records == 0 (when present)
	if raw, ok := rec["dropped_records"]; ok {
		if dropped, ok := parseJSONUint64(raw); !ok {
			addReason("dropped_records is not a nonnegative integer")
		} else {
			q.DroppedRecords = &dropped
			if dropped > 0 {
				addReason(fmt.Sprintf("dropped_records is %d, want 0", dropped))
			}
		}
	}

	// 8. capture_error check
	if raw, ok := rec["capture_error"]; ok {
		if s, ok := raw.(string); ok && s != "" {
			q.CaptureError = s
			addReason("capture_error: " + s)
		}
	}

	if len(q.Reasons) == 0 {
		q.Status = "PASS"
		q.Qualified = true
	} else {
		q.Status = "HOLD"
		q.Qualified = false
		q.Reason = strings.Join(q.Reasons, "; ")
	}
	return q
}

func extractCaptureRecords(data []byte) ([]map[string]any, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, errors.New("empty capture file")
	}
	if obj, err := decodeJSONObjectWithNumbers(trimmed); err == nil {
		if _, ok := obj["capture_success"]; ok {
			return []map[string]any{obj}, nil
		}
		if st, ok := obj["status"].(map[string]any); ok {
			if _, ok := st["capture_success"]; ok {
				return []map[string]any{st}, nil
			}
		}
		return nil, nil
	}

	var records []map[string]any
	for _, rawLine := range bytes.Split(data, []byte("\n")) {
		line := bytes.TrimSpace(rawLine)
		if len(line) == 0 || line[0] != '{' {
			continue
		}
		obj, err := decodeJSONObjectWithNumbers(line)
		if err != nil {
			continue
		}
		if _, ok := obj["capture_success"]; ok {
			records = append(records, obj)
			continue
		}
		if st, ok := obj["status"].(map[string]any); ok {
			if _, ok := st["capture_success"]; ok {
				records = append(records, st)
			}
		}
	}
	return records, nil
}

func decodeJSONObjectWithNumbers(b []byte) (map[string]any, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	var obj map[string]any
	if err := dec.Decode(&obj); err != nil {
		return nil, err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return nil, errors.New("trailing JSON tokens")
	}
	return obj, nil
}

func parseJSONInt(v any) (int, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	if err != nil {
		return 0, false
	}
	return int(i), true
}

func parseJSONUint64(v any) (uint64, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := n.Int64()
	if err != nil || i < 0 {
		return 0, false
	}
	return uint64(i), true
}
