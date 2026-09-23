// SPDX-License-Identifier: Apache-2.0

package cudagrammar

import (
	"fmt"
)

// frameKind identifies what a stack frame is in the middle of matching.
type frameKind int

const (
	fValue   frameKind = iota // a value of f.node has not started yet
	fObject                   // between the braces of an object
	fArray                    // between the brackets of an array
	fString                   // inside a string literal
	fEnum                     // inside a string literal, narrowing enum candidates
	fNumber                   // inside a number
	fLiteral                  // inside a fixed run of bytes: true, false, null, a key
)

// Sub-states of fObject.
const (
	objKey = iota
	objColon
	objValue
	objNext
)

// Sub-states of fArray.
const (
	arrFirst = iota // a value or the closing bracket
	arrValue        // a value, the array already has one
	arrNext         // a comma or the closing bracket
)

// Sub-states of fNumber.
const (
	numFirst = iota
	numSign
	numIntZero
	numInt
	numFracFirst
	numFrac
	numExpSign
	numExpFirst
	numExp
)

type frame struct {
	kind  frameKind
	node  *node
	state int

	lit string // fLiteral: the bytes still to match, from idx
	idx int

	prop int // fObject: index of the property being matched

	esc bool // fString: the previous byte was a backslash
	hex int  // fString: hex digits still expected after \u

	cands []int // fEnum: indices of enum values still viable
	n     int   // fEnum: bytes matched so far
}

// Matcher enforces a [Grammar] over a byte stream.
//
// It is a pushdown automaton: the stack holds the object and array nestings
// currently open, so nesting depth is bounded by the input rather than by the
// compiled grammar. A Matcher is not safe for concurrent use; use [Matcher.Clone]
// to explore alternatives, which is how [Matcher.TokenMask] works.
type Matcher struct {
	stack []frame
	pos   int
}

// Match starts a matcher at the beginning of a document.
func (g *Grammar) Match() *Matcher {
	return &Matcher{stack: []frame{{kind: fValue, node: g.root}}}
}

// Clone returns an independent copy of m.
func (m *Matcher) Clone() *Matcher {
	stack := make([]frame, len(m.stack))
	copy(stack, m.stack)
	for i := range stack {
		if stack[i].cands != nil {
			c := make([]int, len(stack[i].cands))
			copy(c, stack[i].cands)
			stack[i].cands = c
		}
	}
	return &Matcher{stack: stack, pos: m.pos}
}

// Done reports whether a complete document has been matched. Once done, a
// matcher accepts only trailing whitespace.
func (m *Matcher) Done() bool { return len(m.stack) == 0 }

// Pos returns the number of bytes accepted so far.
func (m *Matcher) Pos() int { return m.pos }

func (m *Matcher) errorf(c byte, want string) error {
	return fmt.Errorf("cudagrammar: unexpected %q at offset %d: expected %s", string(rune(c)), m.pos, want)
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func isHex(c byte) bool {
	return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

func (m *Matcher) push(f frame) { m.stack = append(m.stack, f) }

func (m *Matcher) pop() { m.stack = m.stack[:len(m.stack)-1] }

// AcceptByte advances the matcher over one byte. On error the matcher is left
// unchanged, and the error names the offset of the byte that could not be
// matched.
func (m *Matcher) AcceptByte(c byte) error {
	if err := m.step(c); err != nil {
		return err
	}
	m.pos++
	return nil
}

// Accept advances the matcher over a run of bytes, stopping at the first byte
// that does not match.
func (m *Matcher) Accept(b []byte) error {
	for _, c := range b {
		if err := m.AcceptByte(c); err != nil {
			return err
		}
	}
	return nil
}

// AcceptString is Accept for a string.
func (m *Matcher) AcceptString(s string) error {
	return m.Accept([]byte(s))
}

// step matches one byte without advancing the offset. It loops because a value
// can end on lookahead: a number is terminated by the byte after it, which the
// enclosing frame must then match itself.
func (m *Matcher) step(c byte) error {
	for {
		if len(m.stack) == 0 {
			if isSpace(c) {
				return nil
			}
			return m.errorf(c, "end of input")
		}
		f := &m.stack[len(m.stack)-1]
		redo, err := m.stepFrame(f, c)
		if err != nil {
			return err
		}
		if !redo {
			return nil
		}
	}
}

// stepFrame matches c against the top frame. It reports whether c still has to
// be matched by whatever frame is on top afterwards.
func (m *Matcher) stepFrame(f *frame, c byte) (redo bool, err error) {
	switch f.kind {
	case fValue:
		return m.stepValue(f, c)
	case fObject:
		return m.stepObject(f, c)
	case fArray:
		return m.stepArray(f, c)
	case fString:
		return false, m.stepString(f, c)
	case fEnum:
		return false, m.stepEnum(f, c)
	case fNumber:
		return m.stepNumber(f, c)
	case fLiteral:
		return false, m.stepLiteral(f, c)
	}
	return false, fmt.Errorf("cudagrammar: internal error: unknown frame %d", f.kind)
}

// complete pops a finished value frame and tells its parent a value landed.
func (m *Matcher) complete() {
	m.pop()
	if len(m.stack) == 0 {
		return
	}
	parent := &m.stack[len(m.stack)-1]
	switch parent.kind {
	case fObject:
		switch parent.state {
		case objKey:
			// The key literal finished; a colon comes next.
			parent.state = objColon
		case objValue:
			parent.prop++
			parent.state = objNext
		}
	case fArray:
		parent.state = arrNext
	}
}

func (m *Matcher) stepValue(f *frame, c byte) (bool, error) {
	if isSpace(c) {
		return false, nil
	}
	n := f.node
	switch n.kind {
	case kObject:
		if c != '{' {
			return false, m.errorf(c, "'{'")
		}
		*f = frame{kind: fObject, node: n, state: objKey}
		return false, nil
	case kArray:
		if c != '[' {
			return false, m.errorf(c, "'['")
		}
		*f = frame{kind: fArray, node: n, state: arrFirst}
		return false, nil
	case kString:
		if c != '"' {
			return false, m.errorf(c, "'\"'")
		}
		*f = frame{kind: fString}
		return false, nil
	case kEnum:
		if c != '"' {
			return false, m.errorf(c, "'\"'")
		}
		cands := make([]int, len(n.enum))
		for i := range cands {
			cands[i] = i
		}
		*f = frame{kind: fEnum, node: n, cands: cands}
		return false, nil
	case kNumber, kInteger:
		if c != '-' && !isDigit(c) {
			return false, m.errorf(c, "a number")
		}
		*f = frame{kind: fNumber, node: n, state: numFirst}
		return true, nil
	case kBool:
		switch c {
		case 't':
			*f = frame{kind: fLiteral, lit: "true"}
		case 'f':
			*f = frame{kind: fLiteral, lit: "false"}
		default:
			return false, m.errorf(c, "true or false")
		}
		return true, nil
	case kNull:
		if c != 'n' {
			return false, m.errorf(c, "null")
		}
		*f = frame{kind: fLiteral, lit: "null"}
		return true, nil
	}
	return false, fmt.Errorf("cudagrammar: internal error: unknown kind %v", n.kind)
}

func (m *Matcher) stepObject(f *frame, c byte) (bool, error) {
	if isSpace(c) {
		return false, nil
	}
	switch f.state {
	case objKey:
		// The key is a fixed run of bytes, matched by its own frame. The
		// object frame stays in objKey until it completes, which is what
		// tells complete a colon comes next.
		m.push(frame{kind: fLiteral, lit: `"` + f.node.props[f.prop].name + `"`})
		return true, nil
	case objColon:
		if c != ':' {
			return false, m.errorf(c, "':'")
		}
		f.state = objValue
		m.push(frame{kind: fValue, node: f.node.props[f.prop].node})
		return false, nil
	case objValue:
		m.push(frame{kind: fValue, node: f.node.props[f.prop].node})
		return true, nil
	case objNext:
		if f.prop < len(f.node.props) {
			if c != ',' {
				return false, m.errorf(c, fmt.Sprintf("',' before property %q", f.node.props[f.prop].name))
			}
			f.state = objKey
			return false, nil
		}
		if c != '}' {
			return false, m.errorf(c, "'}'")
		}
		m.complete()
		return false, nil
	}
	return false, fmt.Errorf("cudagrammar: internal error: object state %d", f.state)
}

func (m *Matcher) stepArray(f *frame, c byte) (bool, error) {
	if isSpace(c) {
		return false, nil
	}
	switch f.state {
	case arrFirst:
		if c == ']' {
			m.complete()
			return false, nil
		}
		f.state = arrValue
		m.push(frame{kind: fValue, node: f.node.items})
		return true, nil
	case arrValue:
		m.push(frame{kind: fValue, node: f.node.items})
		return true, nil
	case arrNext:
		switch c {
		case ',':
			f.state = arrValue
			return false, nil
		case ']':
			m.complete()
			return false, nil
		}
		return false, m.errorf(c, "',' or ']'")
	}
	return false, fmt.Errorf("cudagrammar: internal error: array state %d", f.state)
}

func (m *Matcher) stepString(f *frame, c byte) error {
	switch {
	case f.hex > 0:
		if !isHex(c) {
			return m.errorf(c, "a hexadecimal digit")
		}
		f.hex--
		return nil
	case f.esc:
		f.esc = false
		switch c {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			return nil
		case 'u':
			f.hex = 4
			return nil
		}
		return m.errorf(c, "an escape character")
	}
	switch {
	case c == '\\':
		f.esc = true
		return nil
	case c == '"':
		m.complete()
		return nil
	case c < 0x20:
		return m.errorf(c, "a string character")
	}
	return nil
}

func (m *Matcher) stepEnum(f *frame, c byte) error {
	if c == '"' {
		for _, i := range f.cands {
			if len(f.node.enum[i]) == f.n {
				m.complete()
				return nil
			}
		}
		return m.errorf(c, "more of an enumerated value")
	}
	keep := f.cands[:0]
	for _, i := range f.cands {
		v := f.node.enum[i]
		if f.n < len(v) && v[f.n] == c {
			keep = append(keep, i)
		}
	}
	if len(keep) == 0 {
		return m.errorf(c, "one of "+quoteList(f.node.enum))
	}
	f.cands = keep
	f.n++
	return nil
}

func (m *Matcher) stepLiteral(f *frame, c byte) error {
	if c != f.lit[f.idx] {
		return m.errorf(c, fmt.Sprintf("%q", f.lit))
	}
	f.idx++
	if f.idx == len(f.lit) {
		m.complete()
	}
	return nil
}

// numberDone reports whether a number can end in this state.
func numberDone(state int) bool {
	switch state {
	case numIntZero, numInt, numFrac, numExp:
		return true
	}
	return false
}

func (m *Matcher) stepNumber(f *frame, c byte) (bool, error) {
	integer := f.node.kind == kInteger
	switch f.state {
	case numFirst:
		if c == '-' {
			f.state = numSign
			return false, nil
		}
		fallthrough
	case numSign:
		switch {
		case c == '0':
			f.state = numIntZero
			return false, nil
		case isDigit(c):
			f.state = numInt
			return false, nil
		}
		return false, m.errorf(c, "a digit")
	case numIntZero, numInt:
		if f.state == numInt && isDigit(c) {
			return false, nil
		}
		switch c {
		case '.':
			if integer {
				return false, m.errorf(c, "an integer")
			}
			f.state = numFracFirst
			return false, nil
		case 'e', 'E':
			if integer {
				return false, m.errorf(c, "an integer")
			}
			f.state = numExpSign
			return false, nil
		}
	case numFracFirst:
		if !isDigit(c) {
			return false, m.errorf(c, "a digit")
		}
		f.state = numFrac
		return false, nil
	case numFrac:
		if isDigit(c) {
			return false, nil
		}
		if c == 'e' || c == 'E' {
			f.state = numExpSign
			return false, nil
		}
	case numExpSign:
		if c == '+' || c == '-' {
			f.state = numExpFirst
			return false, nil
		}
		if !isDigit(c) {
			return false, m.errorf(c, "a digit")
		}
		f.state = numExp
		return false, nil
	case numExpFirst:
		if !isDigit(c) {
			return false, m.errorf(c, "a digit")
		}
		f.state = numExp
		return false, nil
	case numExp:
		if isDigit(c) {
			return false, nil
		}
	}
	// The byte is not part of the number. It ends here if it may, and the
	// enclosing frame matches the byte instead.
	if !numberDone(f.state) {
		return false, m.errorf(c, "a digit")
	}
	m.complete()
	return true, nil
}

// Finish reports whether the document is complete, ending any number that runs
// to the end of the input.
//
// A number has no closing delimiter, so a matcher fed "42" is still inside the
// number and not Done. Finish settles that case; it is the check to make when
// no more bytes are coming.
func (m *Matcher) Finish() error {
	if len(m.stack) == 1 {
		f := &m.stack[0]
		if f.kind == fNumber && numberDone(f.state) {
			m.complete()
		}
	}
	if len(m.stack) != 0 {
		return fmt.Errorf("cudagrammar: incomplete document at offset %d", m.pos)
	}
	return nil
}

func quoteList(values []string) string {
	s := ""
	for i, v := range values {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%q", v)
	}
	return s
}
