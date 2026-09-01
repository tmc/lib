// SPDX-License-Identifier: Apache-2.0

package grammar

import (
	"strings"
)

// State represents the parser state in JSON grammar enforcement.
type State int

const (
	StateAny State = iota
	StateObjectStart
	StateKey
	StateColon
	StateValue
	StateCommaOrClose
	StateComplete
)

// JSONConstraint tracks syntax boundaries to enforce valid JSON generation.
type JSONConstraint struct {
	state       State
	stack       []rune // '{' or '['
	inString    bool
	isEscaping  bool
	outputAccum strings.Builder
}

// NewJSONConstraint creates a constraint enforcing a valid root JSON object.
func NewJSONConstraint() *JSONConstraint {
	return &JSONConstraint{
		state: StateObjectStart,
		stack: make([]rune, 0, 16),
	}
}

// IsValidToken evaluates if candidate token string is permissible in the current state.
func (c *JSONConstraint) IsValidToken(tok string) bool {
	if c.state == StateComplete {
		return strings.TrimSpace(tok) == ""
	}
	if c.state == StateObjectStart {
		trimmed := strings.TrimSpace(tok)
		return trimmed == "" || strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
	}
	return true
}

// Accept processes the selected token and transitions the internal syntax state.
func (c *JSONConstraint) Accept(tok string) {
	c.outputAccum.WriteString(tok)

	for _, r := range tok {
		if c.inString {
			if c.isEscaping {
				c.isEscaping = false
				continue
			}
			if r == '\\' {
				c.isEscaping = true
				continue
			}
			if r == '"' {
				c.inString = false
				if len(c.stack) > 0 && c.stack[len(c.stack)-1] == '{' && c.state == StateKey {
					c.state = StateColon
				} else {
					c.state = StateCommaOrClose
				}
			}
			continue
		}

		switch r {
		case '"':
			c.inString = true
			if c.state == StateObjectStart || c.state == StateKey {
				c.state = StateKey
			} else {
				c.state = StateValue
			}
		case '{', '[':
			c.stack = append(c.stack, r)
			c.state = StateKey
		case ':', '=':
			c.state = StateValue
		case ',', ';':
			if len(c.stack) > 0 && c.stack[len(c.stack)-1] == '{' {
				c.state = StateKey
			} else {
				c.state = StateValue
			}
		case '}', ']':
			if len(c.stack) > 0 {
				c.stack = c.stack[:len(c.stack)-1]
			}
			if len(c.stack) == 0 {
				c.state = StateComplete
			} else {
				c.state = StateCommaOrClose
			}
		}
	}
}

// State returns the current JSON parser state.
func (c *JSONConstraint) State() State {
	return c.state
}

// Result returns the accumulated JSON string output.
func (c *JSONConstraint) Result() string {
	return c.outputAccum.String()
}
