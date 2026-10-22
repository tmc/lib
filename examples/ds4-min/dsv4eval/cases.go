// Package dsv4eval contains model-free evaluation case selection and grading.
package dsv4eval

import (
	"bufio"
	_ "embed"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

//go:embed testdata/cases.tsv
var caseTable string

// AnswerKind identifies the normalizer used by Grade.
type AnswerKind uint8

const (
	Auto AnswerKind = iota
	Choice
	Integer
	Rational
	ExactText
	OrderedSequence
	LineSet
)

// Suite identifies the upstream evaluation suites.
type Suite uint8

const (
	Core Suite = 1 << iota
	Hard
	HardSmoke
)

// Case is one evaluation question. Cases loaded from Cases contain the
// public case index metadata; answer and choices are supplied by a licensed
// case source when grading is enabled.
type Case struct {
	Source, ID, Domain, Title, Question string
	Choices, Aliases                    []string
	Answer                              string
	Kind                                AnswerKind
	Suites                              Suite
	MaxTokens                           int
}

// Filter selects cases. Empty fields do not filter. ID is exact; source and
// domain are compared case-insensitively.
type Filter struct {
	Suite              Suite
	Source, Domain, ID string
}

// Cases returns the upstream 142-case index table.
func Cases() []Case {
	s := bufio.NewScanner(strings.NewReader(caseTable))
	var out []Case
	for s.Scan() {
		f := strings.Split(s.Text(), "\t")
		if len(f) != 6 {
			continue
		}
		n, _ := strconv.Atoi(f[0])
		kind := parseKind(f[4])
		suite := Core
		if n > 92 {
			suite = Hard
		}
		key := f[1] + "\x00" + f[3]
		if hardSmokeCases[key] {
			suite |= HardSmoke
		}
		out = append(out, Case{Source: f[1], Domain: f[2], ID: f[3], Kind: kind, Title: f[5], Suites: suite})
	}
	return out
}

// hardSmokeCases is the fixed upstream subset, copied by identity rather than
// by position so a case-table reorder cannot silently change the smoke gate.
var hardSmokeCases = map[string]bool{
	"MMLU-Pro\x002826":      true,
	"MMLU-Pro\x0010360":     true,
	"MMLU-Pro\x006034":      true,
	"MMLU-Pro\x0010567":     true,
	"MMLU-Pro\x006500":      true,
	"MMLU-Pro\x009487":      true,
	"OlympiadBench\x001606": true,
	"OlympiadBench\x001618": true,
	"OlympiadBench\x001716": true,
	"LiveBench\x00c833574b0ff6e8609bee19b36fabc48501cdcd9a64b99cd0422f54ada1acc53d": true,
	"NIST Juliet\x00juliet-cwe190-int-multiply-01":                                  true,
	"NIST Juliet\x00juliet-cwe121-array-index-01":                                   true,
}

func parseKind(s string) AnswerKind {
	switch s {
	case "choice":
		return Choice
	case "integer":
		return Integer
	case "rational", "reduced_rational":
		return Rational
	case "exact_text":
		return ExactText
	case "ordered_sequence":
		return OrderedSequence
	case "security_line_set":
		return LineSet
	}
	return Auto
}

// Select applies f and rejects a selection with no matches.
func Select(cs []Case, f Filter) ([]Case, error) {
	return SelectN(cs, f, 0)
}

// SelectN applies f and returns at most n cases. A zero n means all matching
// cases. A positive n is a required count: asking for more cases than match
// is an error rather than silently returning a partial selection.
func SelectN(cs []Case, f Filter, n int) ([]Case, error) {
	if n < 0 {
		return nil, fmt.Errorf("select cases: invalid question count %d", n)
	}
	var out []Case
	for _, c := range cs {
		if f.Suite != 0 && c.Suites&f.Suite == 0 {
			continue
		}
		if f.Source != "" && !strings.EqualFold(c.Source, f.Source) {
			continue
		}
		if f.Domain != "" && !strings.EqualFold(c.Domain, f.Domain) {
			continue
		}
		if f.ID != "" && c.ID != f.ID {
			continue
		}
		out = append(out, c)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("select cases: no matches")
	}
	if n > 0 {
		if n > len(out) {
			return nil, fmt.Errorf("select cases: requested %d questions, only %d match", n, len(out))
		}
		out = out[:n]
	}
	return out, nil
}

// Validate checks case-table shape and identity invariants.
func Validate(cs []Case) error {
	if len(cs) == 0 {
		return fmt.Errorf("validate cases: empty table")
	}
	seen := make(map[string]bool, len(cs))
	counts := map[Suite]int{}
	for i, c := range cs {
		if c.Source == "" || c.ID == "" || c.Domain == "" || c.Title == "" {
			return fmt.Errorf("validate cases: case %d has missing identity", i)
		}
		if seen[c.ID] {
			return fmt.Errorf("validate cases: duplicate id %q", c.ID)
		}
		seen[c.ID] = true
		if c.Kind == Auto {
			return fmt.Errorf("validate cases: case %q has unknown answer kind", c.ID)
		}
		for _, s := range []Suite{Core, Hard, HardSmoke} {
			if c.Suites&s != 0 {
				counts[s]++
			}
		}
	}
	if len(cs) != 142 || counts[Core] != 92 || counts[Hard] != 50 || counts[HardSmoke] != 12 {
		return fmt.Errorf("validate cases: got %d cases (%d core, %d hard, %d hard-smoke), want 142 (92 core, 50 hard, 12 hard-smoke)", len(cs), counts[Core], counts[Hard], counts[HardSmoke])
	}
	return nil
}

// Grade extracts the last non-empty Answer: line and compares it to c.
// complete is false when the output has no answer line.
func Grade(c Case, output string) (picked string, complete, pass bool) {
	// The upstream extractor grades only the visible suffix after the first
	// thinking close, which need not be followed by a newline.
	if i := strings.Index(output, "</think>"); i >= 0 {
		output = output[i+len("</think>"):]
	}
	lines := strings.Split(output, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "Answer:") {
			picked = strings.TrimSpace(strings.TrimPrefix(line, "Answer:"))
			break
		}
	}
	if picked == "" {
		return "", false, false
	}
	for _, a := range append([]string{c.Answer}, c.Aliases...) {
		if c.Source == "NIST Juliet" && c.Kind == LineSet && acceptedLines(a, picked) {
			return picked, true, true
		}
		if answerEqual(c.Kind, picked, a) {
			return picked, true, true
		}
	}
	return picked, true, false
}

// acceptedLines follows upstream compsec_answer_matches: a nonempty supplied
// set must be a subset of the audited acceptable line range.
func acceptedLines(expected, picked string) bool {
	parse := func(s string) ([256]bool, bool) {
		var set [256]bool
		any := false
		for i := 0; i < len(s); {
			if s[i] < '0' || s[i] > '9' {
				i++
				continue
			}
			j := i
			for i < len(s) && s[i] >= '0' && s[i] <= '9' {
				i++
			}
			a, err := strconv.Atoi(s[j:i])
			if err != nil {
				return set, false
			}
			b := a
			if i < len(s) && s[i] == '-' {
				i++
				for i < len(s) && strings.ContainsRune(" \t\r\n\v\f", rune(s[i])) {
					i++
				}
				sign := 1
				if i < len(s) && (s[i] == '+' || s[i] == '-') {
					if s[i] == '-' {
						sign = -1
					}
					i++
				}
				j = i
				for i < len(s) && s[i] >= '0' && s[i] <= '9' {
					i++
				}
				b = 0
				if i != j {
					b, err = strconv.Atoi(s[j:i])
					if err != nil {
						return set, false
					}
					b *= sign
				}
			}
			if a > b {
				a, b = b, a
			}
			if b > 255 {
				b = 255
			}
			if a < 0 {
				a = 0
			}
			for n := a; n <= b; n++ {
				set[n] = true
				any = true
			}
		}
		return set, any
	}
	e, ok := parse(expected)
	if !ok {
		return false
	}
	g, ok := parse(picked)
	if !ok {
		return false
	}
	for i, hit := range g {
		if hit && !e[i] {
			return false
		}
	}
	return true
}

// SelfTestExtractors checks the answer extractors with representative cases.
// It does not require a model or any case data beyond the local grader.
//
// The table intentionally includes both positive and negative examples. A
// grader change that accepts the wrong answer, uses the first Answer line, or
// treats incomplete output as a pass makes this check fail.
func SelfTestExtractors() error {
	tests := []struct {
		name         string
		c            Case
		output       string
		wantPicked   string
		wantComplete bool
		wantPass     bool
	}{
		{
			name:         "choice-last-answer",
			c:            Case{Kind: Choice, Answer: "B"},
			output:       "reason\nAnswer: A\nmore reasoning\nAnswer: B\n",
			wantPicked:   "B",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "choice-wrong",
			c:            Case{Kind: Choice, Answer: "B"},
			output:       "Answer: A\n",
			wantPicked:   "A",
			wantComplete: true,
		},
		{
			name:         "integer-normalization",
			c:            Case{Kind: Integer, Answer: "70"},
			output:       "Answer: 070\n",
			wantPicked:   "070",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "rational-whitespace",
			c:            Case{Kind: Rational, Answer: "3/4"},
			output:       "Answer: 3 / 4\n",
			wantPicked:   "3 / 4",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "exact-text",
			c:            Case{Kind: ExactText, Answer: "Ada Lovelace"},
			output:       "Answer: ada lovelace\n",
			wantPicked:   "ada lovelace",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "ordered-sequence",
			c:            Case{Kind: OrderedSequence, Answer: "red green blue"},
			output:       "Answer: RED   green blue\n",
			wantPicked:   "RED   green blue",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "line-set",
			c:            Case{Kind: LineSet, Answer: "alpha,beta"},
			output:       "Answer: beta, alpha\n",
			wantPicked:   "beta, alpha",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "alias",
			c:            Case{Kind: ExactText, Answer: "yes", Aliases: []string{"affirmative"}},
			output:       "Answer: affirmative\n",
			wantPicked:   "affirmative",
			wantComplete: true,
			wantPass:     true,
		},
		{
			name:         "incomplete",
			c:            Case{Kind: Choice, Answer: "A"},
			output:       "There is no final answer.\n",
			wantComplete: false,
		},
	}
	for _, test := range tests {
		picked, complete, pass := Grade(test.c, test.output)
		if picked != test.wantPicked || complete != test.wantComplete || pass != test.wantPass {
			return fmt.Errorf("extractor self-test %s: got picked=%q complete=%t pass=%t, want picked=%q complete=%t pass=%t", test.name, picked, complete, pass, test.wantPicked, test.wantComplete, test.wantPass)
		}
	}
	return nil
}

func answerEqual(k AnswerKind, a, b string) bool {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	switch k {
	case Choice:
		return strings.EqualFold(strings.TrimRight(a, "."), strings.TrimRight(b, "."))
	case Integer:
		return parseInt(a) == parseInt(b) && parseInt(a) != ""
	case Rational:
		return normalizeRational(a) == normalizeRational(b)
	case LineSet:
		return lineSet(a) == lineSet(b)
	case OrderedSequence:
		return strings.Join(strings.Fields(strings.ToLower(a)), " ") == strings.Join(strings.Fields(strings.ToLower(b)), " ")
	default:
		return strings.EqualFold(a, b)
	}
}
func parseInt(s string) string {
	if n, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64); err == nil {
		return strconv.FormatInt(n, 10)
	}
	return ""
}
func normalizeRational(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), "")
}

func lineSet(s string) string {
	var a []string
	for _, l := range strings.FieldsFunc(s, func(r rune) bool { return r == '\n' || r == ',' }) {
		if l = strings.TrimSpace(l); l != "" {
			a = append(a, strings.ToLower(l))
		}
	}
	sort.Strings(a)
	return strings.Join(a, "\n")
}

// WriteList writes the stable six-column case index used by G2.
func WriteList(w io.Writer, cs []Case) error {
	for i, c := range cs {
		if _, err := fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\t%s\n", i+1, c.Source, c.Domain, c.ID, kindName(c.Kind), c.Title); err != nil {
			return fmt.Errorf("write case list: %w", err)
		}
	}
	return nil
}
func kindName(k AnswerKind) string {
	switch k {
	case Choice:
		return "choice"
	case Integer:
		return "integer"
	case Rational:
		return "rational"
	case ExactText:
		return "exact_text"
	case OrderedSequence:
		return "ordered_sequence"
	case LineSet:
		return "security_line_set"
	}
	return "auto"
}
