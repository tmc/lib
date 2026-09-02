package treesitter

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unsafe"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// PredicateArg is one argument of a [Predicate]: either a capture, named
// without its leading @, or a string literal.
type PredicateArg struct {
	Capture string // capture name; "" when the argument is a literal
	Literal string // literal text; meaningful when Capture is ""
}

// IsCapture reports whether the argument names a capture.
func (a PredicateArg) IsCapture() bool { return a.Capture != "" }

// String returns the argument in query syntax.
func (a PredicateArg) String() string {
	if a.IsCapture() {
		return "@" + a.Capture
	}
	return strconv.Quote(a.Literal)
}

// Predicate is a query predicate such as (#eq? @a "b"). Operator is the
// name without its leading #, so it ends in ? for a predicate and ! for a
// directive: "eq?", "match?", "any-of?", "set!".
//
// [Query.Satisfies] evaluates the text predicates tree-sitter defines.
// Every other predicate, including #set! and #is?, is reported here for
// the caller to interpret and does not affect matching.
type Predicate struct {
	Operator string
	Args     []PredicateArg
}

// String returns the predicate in query syntax.
func (p Predicate) String() string {
	var b strings.Builder
	b.WriteString("(#")
	b.WriteString(p.Operator)
	for _, a := range p.Args {
		b.WriteByte(' ')
		b.WriteString(a.String())
	}
	b.WriteByte(')')
	return b.String()
}

// Property is a (#set! key value) directive, or an (#is? key value)
// assertion, attached either to a capture or to the pattern as a whole.
// Value is empty for a key with no value.
type Property struct {
	Key     string
	Value   string
	Capture string // "" when the property applies to the whole pattern
	Negate  bool   // true for #is-not?
}

// text predicate kinds.
const (
	predEqCapture = iota
	predEqString
	predMatchString
	predAnyString
)

// textPredicate is a compiled text predicate, evaluated per match.
type textPredicate struct {
	kind     int
	capture  int // capture id of the first argument
	capture2 int // capture id of the second argument, for predEqCapture
	value    string
	values   []string
	re       *regexp.Regexp
	positive bool
	matchAll bool
}

// parsePredicates decodes every pattern's predicate steps once, at compile
// time, so that a malformed predicate is a compile error rather than a
// surprise during matching.
func (q *Query) parsePredicates(source string) error {
	n := q.PatternCount()
	q.preds = make([][]Predicate, n)
	q.props = make([][]Property, n)
	q.text = make([][]textPredicate, n)
	for i := range n {
		preds, err := q.decodeSteps(i)
		if err != nil {
			return q.predicateError(source, i, err)
		}
		q.preds[i] = preds
		for _, p := range preds {
			switch p.Operator {
			case "set!", "is?", "is-not?":
				prop, err := parseProperty(p)
				if err != nil {
					return q.predicateError(source, i, err)
				}
				q.props[i] = append(q.props[i], prop)
				continue
			}
			tp, ok, err := q.compile(p)
			if err != nil {
				return q.predicateError(source, i, err)
			}
			if ok {
				q.text[i] = append(q.text[i], tp)
			}
		}
	}
	return nil
}

// predicateError reports err against the start of pattern i.
func (q *Query) predicateError(source string, pattern int, err error) error {
	e := newQueryError(QueryErrorStructure, source, q.StartByteForPattern(pattern))
	e.Message = err.Error()
	return e
}

// decodeSteps reads the predicate steps of pattern i and groups them into
// predicates, resolving capture and string ids to their names and values.
// A Done step ends each predicate.
func (q *Query) decodeSteps(i int) ([]Predicate, error) {
	var n uint32
	p := ts.QueryPredicatesForPattern(q.q, u32(i), &n)
	if p == 0 || n == 0 {
		return nil, nil
	}
	steps := unsafe.Slice((*ts.QueryPredicateStep)(cptr(p)), n)

	var out []Predicate
	var cur []PredicateArg
	for _, s := range steps {
		switch s.Type {
		case ts.QueryPredicateStepDone:
			if len(cur) == 0 {
				return nil, fmt.Errorf("empty predicate")
			}
			if cur[0].IsCapture() {
				return nil, fmt.Errorf("predicate name must be a literal, not @%s", cur[0].Capture)
			}
			out = append(out, Predicate{Operator: cur[0].Literal, Args: cur[1:]})
			cur = nil
		case ts.QueryPredicateStepCapture:
			cur = append(cur, PredicateArg{Capture: q.CaptureName(int(s.ValueId))})
		case ts.QueryPredicateStepString:
			cur = append(cur, PredicateArg{Literal: q.StringValue(int(s.ValueId))})
		}
	}
	if len(cur) != 0 {
		return nil, fmt.Errorf("unterminated predicate")
	}
	return out, nil
}

// compile turns a text predicate into its evaluable form. It reports ok
// false for an operator the package does not evaluate.
func (q *Query) compile(p Predicate) (tp textPredicate, ok bool, err error) {
	switch p.Operator {
	case "eq?", "not-eq?", "any-eq?", "any-not-eq?":
		if len(p.Args) != 2 {
			return tp, false, fmt.Errorf("#%s takes 2 arguments, got %d", p.Operator, len(p.Args))
		}
		if !p.Args[0].IsCapture() {
			return tp, false, fmt.Errorf("#%s: first argument must be a capture", p.Operator)
		}
		tp.positive = p.Operator == "eq?" || p.Operator == "any-eq?"
		tp.matchAll = !strings.HasPrefix(p.Operator, "any-")
		tp.capture, err = q.captureID(p.Args[0].Capture)
		if err != nil {
			return tp, false, err
		}
		if p.Args[1].IsCapture() {
			tp.kind = predEqCapture
			tp.capture2, err = q.captureID(p.Args[1].Capture)
			return tp, err == nil, err
		}
		tp.kind = predEqString
		tp.value = p.Args[1].Literal
		return tp, true, nil

	case "match?", "not-match?", "any-match?", "any-not-match?":
		if len(p.Args) != 2 {
			return tp, false, fmt.Errorf("#%s takes 2 arguments, got %d", p.Operator, len(p.Args))
		}
		if !p.Args[0].IsCapture() {
			return tp, false, fmt.Errorf("#%s: first argument must be a capture", p.Operator)
		}
		if p.Args[1].IsCapture() {
			return tp, false, fmt.Errorf("#%s: second argument must be a string, not @%s", p.Operator, p.Args[1].Capture)
		}
		tp.kind = predMatchString
		tp.positive = p.Operator == "match?" || p.Operator == "any-match?"
		tp.matchAll = !strings.HasPrefix(p.Operator, "any-")
		if tp.capture, err = q.captureID(p.Args[0].Capture); err != nil {
			return tp, false, err
		}
		tp.re, err = regexp.Compile(p.Args[1].Literal)
		if err != nil {
			return tp, false, fmt.Errorf("#%s: %w", p.Operator, err)
		}
		return tp, true, nil

	case "any-of?", "not-any-of?":
		if len(p.Args) < 1 {
			return tp, false, fmt.Errorf("#%s takes at least 1 argument", p.Operator)
		}
		if !p.Args[0].IsCapture() {
			return tp, false, fmt.Errorf("#%s: first argument must be a capture", p.Operator)
		}
		tp.kind = predAnyString
		tp.positive = p.Operator == "any-of?"
		if tp.capture, err = q.captureID(p.Args[0].Capture); err != nil {
			return tp, false, err
		}
		for _, a := range p.Args[1:] {
			if a.IsCapture() {
				return tp, false, fmt.Errorf("#%s: arguments after the first must be strings", p.Operator)
			}
			tp.values = append(tp.values, a.Literal)
		}
		return tp, true, nil
	}
	return tp, false, nil
}

// captureID returns the id of a capture name.
func (q *Query) captureID(name string) (int, error) {
	for i, s := range q.names {
		if s == name {
			return i, nil
		}
	}
	return 0, fmt.Errorf("unknown capture @%s", name)
}

// parseProperty decodes a #set!, #is?, or #is-not? predicate, whose
// arguments are an optional capture followed by a key and an optional
// value.
func parseProperty(p Predicate) (Property, error) {
	prop := Property{Negate: p.Operator == "is-not?"}
	args := p.Args
	if len(args) > 0 && args[0].IsCapture() {
		prop.Capture = args[0].Capture
		args = args[1:]
	}
	switch len(args) {
	case 2:
		prop.Value = args[1].Literal
		fallthrough
	case 1:
		prop.Key = args[0].Literal
	default:
		return prop, fmt.Errorf("#%s takes a key and an optional value", p.Operator)
	}
	for _, a := range args {
		if a.IsCapture() {
			return prop, fmt.Errorf("#%s: key and value must be strings", p.Operator)
		}
	}
	return prop, nil
}

// Predicates returns the predicates of pattern i, in source order,
// including those the package does not evaluate.
func (q *Query) Predicates(i int) []Predicate {
	if i < 0 || i >= len(q.preds) {
		return nil
	}
	return slices.Clone(q.preds[i])
}

// Properties returns the #set!, #is?, and #is-not? properties of pattern
// i. They carry information for the caller, such as a highlight name, and
// do not affect matching.
func (q *Query) Properties(i int) []Property {
	if i < 0 || i >= len(q.props) {
		return nil
	}
	return slices.Clone(q.props[i])
}

// Satisfies reports whether m meets the text predicates of its pattern:
// #eq?, #match?, and #any-of?, with their not-, any-, and any-not- forms.
// Other predicates are ignored.
//
// Evaluation reads the source text of captured nodes. When the tree has no
// source, as after [Parser.ParseInput], there is nothing to compare and
// Satisfies reports true.
func (q *Query) Satisfies(m QueryMatch) bool {
	if m.PatternIndex < 0 || m.PatternIndex >= len(q.text) || len(q.text[m.PatternIndex]) == 0 {
		return true
	}
	if len(m.Captures) > 0 && m.Captures[0].Node.t.Source() == nil {
		return true
	}
	for _, tp := range q.text[m.PatternIndex] {
		if !q.satisfiesOne(tp, m) {
			return false
		}
	}
	return true
}

func (q *Query) satisfiesOne(tp textPredicate, m QueryMatch) bool {
	switch tp.kind {
	case predEqCapture:
		n1 := m.nodes(tp.capture)
		n2 := m.nodes(tp.capture2)
		i := 0
		for ; i < len(n1) && i < len(n2); i++ {
			isMatch := string(n1[i].Text()) == string(n2[i].Text())
			if isMatch != tp.positive && tp.matchAll {
				return false
			}
			if isMatch == tp.positive && !tp.matchAll {
				return true
			}
		}
		// The predicate holds only if both captures ran out together.
		return len(n1) == len(n2)

	case predEqString:
		for _, c := range m.Captures {
			if c.Index != tp.capture {
				continue
			}
			isMatch := string(c.Node.Text()) == tp.value
			if isMatch != tp.positive && tp.matchAll {
				return false
			}
			if isMatch == tp.positive && !tp.matchAll {
				return true
			}
		}
		return true

	case predMatchString:
		for _, c := range m.Captures {
			if c.Index != tp.capture {
				continue
			}
			isMatch := tp.re.Match(c.Node.Text())
			if isMatch != tp.positive && tp.matchAll {
				return false
			}
			if isMatch == tp.positive && !tp.matchAll {
				return true
			}
		}
		return true

	case predAnyString:
		for _, c := range m.Captures {
			if c.Index != tp.capture {
				continue
			}
			if slices.Contains(tp.values, string(c.Node.Text())) != tp.positive {
				return false
			}
		}
		return true
	}
	return true
}

// nodes returns the nodes captured under the given capture id, in match
// order.
func (m QueryMatch) nodes(capture int) []Node {
	var out []Node
	for _, c := range m.Captures {
		if c.Index == capture {
			out = append(out, c.Node)
		}
	}
	return out
}
