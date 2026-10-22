package dsv4eval

import (
	"fmt"
	"io"
	"strings"
)

// RegradeReport describes a trace regrade. Cases is the number of case
// headers found; Parsed excludes cases that could not be matched or parsed.
type RegradeReport struct {
	Cases, Parsed        int
	Passed, Failed       int
	Changed, NotGraded   int
	Unknown, ParseErrors int
}

// Regrade reads a ds4-eval trace and grades the answer in each MODEL_OUTPUT
// block against the expected answer recorded for that case. It accepts both
// C and Go traces and does not require a model.
func Regrade(r io.Reader) (RegradeReport, error) {
	trace, err := ReadTrace(r)
	if err != nil {
		return RegradeReport{ParseErrors: 1}, err
	}
	known := make(map[string]Case)
	for _, c := range Cases() {
		known[c.Source+"\x00"+c.ID] = c
	}
	var report RegradeReport
	for _, current := range trace.Cases {
		report.Cases++
		c, ok := known[current.Source+"\x00"+current.ID]
		if !ok {
			report.Unknown++
			continue
		}
		status := current.Fields["status"]
		if status != "PASSED" && status != "FAILED" && status != "" {
			report.NotGraded++
			continue
		}
		expected := current.Fields["expected"]
		output, haveOutput := current.Blocks["MODEL_OUTPUT"]
		if expected == "" || !haveOutput {
			report.Unknown++
			continue
		}
		c.Answer = expected
		_, complete, pass := Grade(c, output)
		if !complete {
			report.Unknown++
		} else if pass {
			report.Passed++
		} else {
			report.Failed++
		}
		if (status == "PASSED") != pass {
			report.Changed++
		}
		report.Parsed++
	}
	if report.Cases == 0 {
		return report, fmt.Errorf("regrade trace: no cases")
	}
	return report, nil
}

// SuiteName converts the command-line spelling to a case-table filter.
func SuiteName(s string) (Suite, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "core":
		return Core, nil
	case "hard":
		return Hard, nil
	case "all", "":
		return Core | Hard, nil
	case "hard-smoke":
		return HardSmoke, nil
	default:
		return 0, fmt.Errorf("unknown suite %q", s)
	}
}

// FormatRegrade returns the stable upstream-style summary line.
func FormatRegrade(p RegradeReport) string {
	return fmt.Sprintf("ds4-eval: regraded %d cases: passed=%d failed=%d changed=%d not_graded=%d unknown=%d parse_errors=%d", p.Cases, p.Passed, p.Failed, p.Changed, p.NotGraded, p.Unknown, p.ParseErrors)
}
