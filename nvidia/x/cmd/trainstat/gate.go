package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"

	"golang.org/x/perf/benchproc"
)

// A rule is one line of a gate rules file: row unit op value.
type rule struct {
	line      int
	row, unit string
	op        string  // >=, <= or ==
	value     float64 // a bound on Δ, or for ==, on every value
	pct       bool    // value is a percentage: a bound on new/base-1
	text      string  // op and value as written
	bad       string  // why the rule is not understood, or ""
}

// A verdict is a rule's outcome.
type verdict struct {
	Verdict string   `json:"verdict"` // PASS, FAIL, INCONCLUSIVE or INSUFFICIENT
	Row     string   `json:"row"`
	Unit    string   `json:"unit"`
	Rule    string   `json:"rule"`
	Delta   *float64 `json:"delta,omitempty"` // Δ and its interval, when finite
	Lo      *float64 `json:"lo,omitempty"`
	Hi      *float64 `json:"hi,omitempty"`
	Detail  string   `json:"detail"`
}

func runGate(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("gate", flag.ContinueOnError)
	var o options
	o.register(fs)
	rulesFile := fs.String("rules", "", "read rules from `file`")
	lenient := fs.Bool("lenient", false, "pass inconclusive and insufficient rules")
	minN := fs.Int("min-n", 0, "fail rules with fewer than `n` samples per column")
	jsonOut := fs.Bool("json", false, "write verdicts as JSON")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: trainstat gate -rules file [flags] base new\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return usageError(err)
	}
	if *rulesFile == "" || fs.NArg() == 0 {
		fs.Usage()
		return usageError(errors.New("gate needs -rules and inputs"))
	}
	rules, err := readRules(*rulesFile)
	if err != nil {
		return usageError(err)
	}
	d, err := read(fs.Args())
	if err != nil {
		return err
	}
	// Each rule is judged on its own at level -alpha: its interval has
	// confidence 1-alpha, with no correction for the number of rules.
	o.confidence = 1 - o.alpha
	a, err := analyze(d, &o)
	if err != nil {
		return err
	}
	if len(a.cols) != 2 {
		return fmt.Errorf("gate needs two columns, have %d (columns: %s)", len(a.cols), a.colDesc)
	}
	var vs []verdict
	for _, r := range rules {
		vs = append(vs, a.gate(r, *minN)...)
	}

	var fail, inconclusive int
	for _, v := range vs {
		switch v.Verdict {
		case "FAIL":
			fail++
		case "INCONCLUSIVE", "INSUFFICIENT":
			inconclusive++
		}
	}
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "\t")
		enc.SetEscapeHTML(false)
		if err := enc.Encode(vs); err != nil {
			return err
		}
	} else {
		tw := tabwriter.NewWriter(stdout, 0, 8, 2, ' ', 0)
		fmt.Fprintf(tw, "%d rules, each at α=%g (%g%% CI), not corrected for multiple rules\n", len(rules), o.alpha, 100*o.confidence)
		for _, v := range vs {
			fmt.Fprintf(tw, "%s\t%s %s\t%s\t%s\n", v.Verdict, v.Row, v.Unit, v.Rule, v.Detail)
		}
		if fail+inconclusive == 0 {
			fmt.Fprintln(tw, "ok")
		} else {
			fmt.Fprintf(tw, "%d fail, %d inconclusive or insufficient\n", fail, inconclusive)
		}
		if err := tw.Flush(); err != nil {
			return err
		}
	}
	if fail > 0 || inconclusive > 0 && !*lenient {
		return &exitCodeError{Code: 1, Err: fmt.Errorf("gate: %d fail, %d inconclusive or insufficient", fail, inconclusive)}
	}
	return nil
}

// readRules reads a rules file: lines "row unit op value", where op is
// >=, <= or ==, value may end in % for >= and <=, and # starts a comment.
// A line that is not such a rule is kept, to be judged INCONCLUSIVE.
func readRules(name string) ([]rule, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rules []rule
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line, _, _ := strings.Cut(sc.Text(), "#")
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		rules = append(rules, parseRule(n, fields))
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(rules) == 0 {
		return nil, fmt.Errorf("%s: no rules", name)
	}
	return rules, nil
}

func parseRule(n int, fields []string) rule {
	r := rule{line: n, text: strings.Join(fields, " ")}
	if len(fields) != 4 {
		r.bad = "want row unit op value"
		return r
	}
	r.row, r.unit, r.op = fields[0], fields[1], fields[2]
	r.text = fields[2] + " " + fields[3]
	v, pct := strings.CutSuffix(fields[3], "%")
	x, err := strconv.ParseFloat(v, 64)
	switch {
	case r.op != ">=" && r.op != "<=" && r.op != "==":
		r.bad = "unknown operator " + r.op
	case err != nil:
		r.bad = "bad value " + fields[3]
	case pct && r.op == "==":
		r.bad = "== takes an absolute value"
	}
	r.value, r.pct = x, pct
	return r
}

// gate judges a rule in every table that has its row.
func (a *analysis) gate(r rule, minN int) []verdict {
	if r.bad != "" {
		return []verdict{{Verdict: "INCONCLUSIVE", Row: r.row, Unit: r.unit, Rule: r.text, Detail: fmt.Sprintf("line %d: %s", r.line, r.bad)}}
	}
	var vs []verdict
	for _, t := range a.tables {
		for _, b := range t.blocks {
			if b.unit != r.unit {
				continue
			}
			for _, row := range b.rows {
				if row.StringValues() == r.row {
					vs = append(vs, a.judge(r, b, row, minN))
				}
			}
		}
	}
	if len(vs) == 0 {
		vs = append(vs, verdict{Verdict: "FAIL", Row: r.row, Unit: r.unit, Rule: r.text, Detail: "no data"})
	}
	return vs
}

func (a *analysis) judge(r rule, b *block, row benchproc.Key, minN int) verdict {
	v := verdict{Row: r.row, Unit: r.unit, Rule: r.text}
	fail := func(detail string) verdict {
		v.Verdict, v.Detail = "FAIL", detail
		return v
	}
	var n [2]int
	for j, col := range a.cols {
		cell := b.cells[[2]benchproc.Key{row, col}]
		switch {
		case cell == nil:
			return fail("no data in " + col.StringValues())
		case cell.bad != "":
			return fail(cell.bad + " in " + col.StringValues())
		}
		n[j] = len(cell.xs)
	}
	if min(n[0], n[1]) < minN {
		return fail(fmt.Sprintf("n=%d+%d < -min-n %d", n[0], n[1], minN))
	}
	insufficient := func(detail string) verdict {
		v.Verdict, v.Detail = "INSUFFICIENT", detail
		return v
	}
	switch {
	case r.op == "==":
		// An exact check of every value needs no distribution.
	case a.d.isTime(r.unit) && min(n[0], n[1]) < 6:
		return insufficient(fmt.Sprintf("need >=6 samples, e.g. go test -bench -count 6 (n=%d+%d)", n[0], n[1]))
	case min(n[0], n[1]) < 2:
		return insufficient(fmt.Sprintf("need >=2 seeds per column for an interval (n=%d+%d)", n[0], n[1]))
	}

	if r.op == "==" {
		var bad, total int
		for _, col := range a.cols {
			for _, x := range b.cells[[2]benchproc.Key{row, col}].xs {
				total++
				if x != r.value {
					bad++
				}
			}
		}
		if bad > 0 {
			return fail(fmt.Sprintf("%d of %d values differ", bad, total))
		}
		v.Verdict, v.Detail = "PASS", fmt.Sprintf("%v in all %d values", r.value, total)
		return v
	}

	c, _ := a.compare(b, row, a.cols[1])
	scale, suffix := 1.0, ""
	if r.pct {
		base := a.summarize(b, b.cell(row, a.cols[0])).center
		scale, suffix = 100/math.Abs(base), "%"
	}
	delta, lo, hi := c.delta*scale, c.lo*scale, c.hi*scale
	detail := fmt.Sprintf("Δ %+.4g%s [%+.4g%s, %+.4g%s]  %s", delta, suffix, lo, suffix, hi, suffix, c.info())
	v.Delta, v.Lo, v.Hi = finite(delta), finite(lo), finite(hi)
	if c.note != "" {
		detail += " (" + c.note + ")"
	}
	v.Detail = detail
	pass, failed := lo >= r.value, hi < r.value
	if r.op == "<=" {
		pass, failed = hi <= r.value, lo > r.value
	}
	switch {
	case pass:
		v.Verdict = "PASS"
	case failed:
		v.Verdict = "FAIL"
	default:
		v.Verdict = "INCONCLUSIVE"
		v.Detail += "  CI straddles the bound"
	}
	return v
}

func finite(x float64) *float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return nil
	}
	return &x
}
