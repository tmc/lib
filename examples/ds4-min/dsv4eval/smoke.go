package dsv4eval

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// SmokeReport is the host-side result of comparing two hard-smoke traces.
// It is a trace/provenance gate, not evidence that either implementation ran
// a model or matched its logits.
type SmokeReport struct {
	Cases                   int
	Ran, Skipped            int
	StatusMatches           int
	ThinkCloseMatches       int
	LeftPassed, RightPassed int
	LeftBuild, RightBuild   string
	ModelSHA256             string
	Context                 string
	Temperature             string
	Think                   string
}

// SmokeReceipt is the machine-readable G7 result. UNMEASURED is used when
// one of the named traces is absent; absence is not a successful smoke run.
type SmokeReceipt struct {
	Status            string `json:"status"`
	Reason            string `json:"reason"`
	Left              string `json:"left_trace"`
	Right             string `json:"right_trace"`
	LeftBuild         string `json:"left_build,omitempty"`
	RightBuild        string `json:"right_build,omitempty"`
	ModelSHA256       string `json:"model_sha256,omitempty"`
	Context           string `json:"ctx,omitempty"`
	Temperature       string `json:"temp,omitempty"`
	Think             string `json:"think,omitempty"`
	Cases             int    `json:"cases"`
	Ran               int    `json:"ran"`
	Skipped           int    `json:"skipped"`
	StatusMatches     int    `json:"status_matches"`
	ThinkCloseMatches int    `json:"think_close_matches"`
	LeftPassed        int    `json:"left_passed"`
	RightPassed       int    `json:"right_passed"`
}

// CompareSmokeFiles evaluates two retained traces and returns a receipt. A
// missing trace is an explicit UNMEASURED result; malformed or incompatible
// traces are FAIL results with a reason.
func CompareSmokeFiles(leftPath, rightPath string) SmokeReceipt {
	receipt := SmokeReceipt{Status: "UNMEASURED", Left: leftPath, Right: rightPath}
	left, err := os.Open(leftPath)
	if err != nil {
		receipt.Reason = fmt.Sprintf("left trace unavailable: %v", err)
		return receipt
	}
	right, err := os.Open(rightPath)
	if err != nil {
		_ = left.Close()
		receipt.Reason = fmt.Sprintf("right trace unavailable: %v", err)
		return receipt
	}
	report, compareErr := CompareSmoke(left, right)
	leftErr, rightErr := left.Close(), right.Close()
	if compareErr != nil {
		receipt.Status = "FAIL"
		receipt.Reason = compareErr.Error()
		return receipt
	}
	if leftErr != nil || rightErr != nil {
		receipt.Status = "FAIL"
		receipt.Reason = "closing trace failed"
		return receipt
	}
	receipt.Cases = report.Cases
	receipt.Ran = report.Ran
	receipt.Skipped = report.Skipped
	receipt.StatusMatches = report.StatusMatches
	receipt.ThinkCloseMatches = report.ThinkCloseMatches
	receipt.LeftPassed = report.LeftPassed
	receipt.RightPassed = report.RightPassed
	receipt.LeftBuild = report.LeftBuild
	receipt.RightBuild = report.RightBuild
	receipt.ModelSHA256 = report.ModelSHA256
	receipt.Context = report.Context
	receipt.Temperature = report.Temperature
	receipt.Think = report.Think
	if report.Pass() {
		receipt.Status = "PASS"
		receipt.Reason = "hard-smoke traces satisfy the registered shape and agreement rule"
	} else {
		receipt.Status = "FAIL"
		receipt.Reason = "hard-smoke traces do not satisfy the registered agreement rule"
	}
	return receipt
}

// WriteJSON writes a stable G7 receipt.
func (r SmokeReceipt) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(r)
}

// Pass reports whether both complete traces have the registered hard-smoke
// shape, at least eleven passing cases each, and satisfy the G7 agreement
// rule. Incomplete cases are counted as skipped and cannot pass.
func (r SmokeReport) Pass() bool {
	return r.Cases == 12 && r.Ran == 12 && r.Skipped == 0 &&
		r.StatusMatches >= 11 && r.ThinkCloseMatches == 12 &&
		r.LeftPassed >= 11 && r.RightPassed >= 11
}

// AgreementPass reports the Spec08 status-agreement rule with the registered
// additional complete-run requirement. Correct-answer counts are diagnostics;
// Pass separately retains the existing stronger quality threshold.
func (r SmokeReport) AgreementPass() bool {
	return r.Cases == 12 && r.Ran == 12 && r.Skipped == 0 &&
		r.StatusMatches >= 11 && r.ThinkCloseMatches == 12
}

// CompareSmoke compares two strict hard-smoke traces.  It requires each
// trace to carry suite, build, model hash, context, temperature and thinking
// metadata.  Missing metadata, incomplete cases and skipped cases are errors;
// they must never turn into an apparent pass.
func CompareSmoke(left, right io.Reader) (SmokeReport, error) {
	l, err := readSmokeTrace(left, "left")
	if err != nil {
		return SmokeReport{}, err
	}
	r, err := readSmokeTrace(right, "right")
	if err != nil {
		return SmokeReport{}, err
	}
	if l.Header.Fields["model_sha256"] != r.Header.Fields["model_sha256"] {
		return SmokeReport{}, fmt.Errorf("smoke traces: model hashes differ")
	}
	if l.Header.Fields["ctx"] != r.Header.Fields["ctx"] {
		return SmokeReport{}, fmt.Errorf("smoke traces: contexts differ")
	}
	if l.Header.Fields["temp"] != r.Header.Fields["temp"] {
		return SmokeReport{}, fmt.Errorf("smoke traces: temperatures differ")
	}
	if l.Header.Fields["think"] != r.Header.Fields["think"] {
		return SmokeReport{}, fmt.Errorf("smoke traces: thinking modes differ")
	}

	report := SmokeReport{
		Cases: len(l.Cases), StatusMatches: 0, ThinkCloseMatches: 0,
		LeftBuild: l.Header.Fields["build"], RightBuild: r.Header.Fields["build"],
		ModelSHA256: l.Header.Fields["model_sha256"],
		Context:     l.Header.Fields["ctx"], Temperature: l.Header.Fields["temp"],
		Think: l.Header.Fields["think"],
	}
	for i := range l.Cases {
		lc, rc := l.Cases[i], r.Cases[i]
		if lc.Source != rc.Source || lc.ID != rc.ID {
			return SmokeReport{}, fmt.Errorf("smoke traces: case %d identity differs", i+1)
		}
		if lc.Fields["status"] == rc.Fields["status"] {
			report.StatusMatches++
		}
		if lc.Fields["status"] == "INCOMPLETE" || rc.Fields["status"] == "INCOMPLETE" {
			report.Skipped++
		} else {
			report.Ran++
		}
		if lc.Fields["status"] == "PASSED" {
			report.LeftPassed++
		}
		if rc.Fields["status"] == "PASSED" {
			report.RightPassed++
		}
		if lc.Fields["think_close"] == rc.Fields["think_close"] {
			report.ThinkCloseMatches++
		}
	}
	return report, nil
}

func readSmokeTrace(r io.Reader, side string) (Trace, error) {
	t, err := ReadTrace(r)
	if err != nil {
		return Trace{}, fmt.Errorf("%s smoke trace: %w", side, err)
	}
	f := t.Header.Fields
	if f["suite"] != "hard-smoke" {
		return Trace{}, fmt.Errorf("%s smoke trace: suite %q, want hard-smoke", side, f["suite"])
	}
	for _, key := range []string{"build", "model_sha256", "ctx", "temp", "think"} {
		if strings.TrimSpace(f[key]) == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: missing %s provenance", side, key)
		}
	}
	if len(f["model_sha256"]) != 64 {
		return Trace{}, fmt.Errorf("%s smoke trace: model_sha256 is not 64 hex characters", side)
	}
	if _, err := hex.DecodeString(f["model_sha256"]); err != nil {
		return Trace{}, fmt.Errorf("%s smoke trace: invalid model_sha256: %w", side, err)
	}
	if ctx, err := strconv.Atoi(f["ctx"]); err != nil || ctx <= 0 {
		return Trace{}, fmt.Errorf("%s smoke trace: invalid ctx", side)
	}
	if f["temp"] != "0" && f["temp"] != "0.0" {
		return Trace{}, fmt.Errorf("%s smoke trace: temperature must be zero", side)
	}
	if len(t.Cases) != 12 {
		return Trace{}, fmt.Errorf("%s smoke trace: got %d cases, want 12", side, len(t.Cases))
	}
	counts := map[string]int{}
	seen := make(map[string]struct{}, len(t.Cases))
	for i, c := range t.Cases {
		if c.Index != i+1 || c.Total != 12 {
			return Trace{}, fmt.Errorf("%s smoke trace: invalid case numbering at %d", side, i+1)
		}
		identity := c.Source + "\x00" + c.ID
		if _, exists := seen[identity]; exists {
			return Trace{}, fmt.Errorf("%s smoke trace: duplicate case identity at %d", side, i+1)
		}
		seen[identity] = struct{}{}
		status := c.Fields["status"]
		if status != "PASSED" && status != "FAILED" && status != "INCOMPLETE" {
			return Trace{}, fmt.Errorf("%s smoke trace: case %d has nonterminal status %q", side, i+1, status)
		}
		if status == "INCOMPLETE" {
			counts[status]++
			continue
		}
		if c.Fields["think_close"] == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: case %d missing think_close", side, i+1)
		}
		if strings.TrimSpace(c.Fields["picked"]) == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: case %d missing picked answer", side, i+1)
		}
		if strings.TrimSpace(c.Fields["expected"]) == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: case %d missing expected answer", side, i+1)
		}
		output, ok := c.Blocks["MODEL_OUTPUT"]
		if !ok || strings.TrimSpace(output) == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: case %d missing model output", side, i+1)
		}
		counts[status]++
	}
	if err := validateSmokeAnswers(t, side); err != nil {
		return Trace{}, err
	}
	for _, key := range []string{"passed", "failed", "incomplete", "total"} {
		if strings.TrimSpace(t.Summary[key]) == "" {
			return Trace{}, fmt.Errorf("%s smoke trace: missing summary %s", side, key)
		}
	}
	for key, want := range map[string]int{
		"passed": counts["PASSED"], "failed": counts["FAILED"],
		"incomplete": counts["INCOMPLETE"], "total": len(t.Cases),
	} {
		got, err := strconv.Atoi(t.Summary[key])
		if err != nil || got != want {
			return Trace{}, fmt.Errorf("%s smoke trace: summary %s=%q, want %d", side, key, t.Summary[key], want)
		}
	}
	return t, nil
}

// validateSmokeAnswers checks the producer's status and picked fields against
// the same answer extractor used by the ordinary regrader.  Comparing status
// fields alone is not a gate: two traces can agree on a fabricated status.
func validateSmokeAnswers(t Trace, side string) error {
	known := make(map[string]Case, len(Cases()))
	for _, c := range Cases() {
		known[c.Source+"\x00"+c.ID] = c
	}
	for i, tc := range t.Cases {
		if tc.Fields["status"] == "INCOMPLETE" {
			continue
		}
		c, ok := known[tc.Source+"\x00"+tc.ID]
		if !ok {
			return fmt.Errorf("%s smoke trace: case %d is not in the case table", side, i+1)
		}
		c.Answer = tc.Fields["expected"]
		picked, complete, pass := Grade(c, tc.Blocks["MODEL_OUTPUT"])
		if !complete {
			return fmt.Errorf("%s smoke trace: case %d model output has no answer", side, i+1)
		}
		if picked != strings.TrimSpace(tc.Fields["picked"]) {
			return fmt.Errorf("%s smoke trace: case %d picked answer disagrees with model output", side, i+1)
		}
		statusPass := tc.Fields["status"] == "PASSED"
		if statusPass != pass {
			return fmt.Errorf("%s smoke trace: case %d status disagrees with model output", side, i+1)
		}
		if tc.Fields["status"] == "INCOMPLETE" {
			return fmt.Errorf("%s smoke trace: case %d is incomplete", side, i+1)
		}
	}
	return nil
}

// FormatSmoke returns the stable summary used by the command-line gate.
func FormatSmoke(r SmokeReport) string {
	verdict := "FAIL"
	if r.Pass() {
		verdict = "PASS"
	}
	return fmt.Sprintf("ds4-eval: smoke cases=%d ran=%d skipped=%d status_matches=%d/12 passed=%d/%d think_close_matches=%d/12 verdict=%s", r.Cases, r.Ran, r.Skipped, r.StatusMatches, r.LeftPassed, r.RightPassed, r.ThinkCloseMatches, verdict)
}
