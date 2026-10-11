package egraphcore

import (
	"fmt"
	"strings"
	"unicode"
)

// ParsePattern parses one S-expression structural pattern. Commas are
// separators and variable names begin with '?'. The parser uses an explicit
// stack so deeply nested patterns do not consume the Go call stack.
func ParsePattern(source string) (*Pattern, error) {
	tokens := patternTokens(source)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty pattern")
	}

	var stack []*Pattern
	for i := 0; i < len(tokens); {
		token := tokens[i]
		i++
		switch token {
		case "(":
			if i == len(tokens) || tokens[i] == "(" || tokens[i] == ")" {
				return nil, fmt.Errorf("missing operator after '('")
			}
			op := tokens[i]
			i++
			stack = append(stack, &Pattern{Op: op})
		case ")":
			if len(stack) == 0 {
				return nil, fmt.Errorf("unexpected ')' in pattern")
			}
			last := len(stack) - 1
			completed := stack[last]
			stack = stack[:last]
			if len(stack) != 0 {
				parent := stack[len(stack)-1]
				parent.Children = append(parent.Children, completed)
				continue
			}
			if i != len(tokens) {
				return nil, fmt.Errorf("unparsed trailing tokens in pattern: %v", tokens[i:])
			}
			return completed, nil
		default:
			var atom *Pattern
			if strings.HasPrefix(token, "?") {
				atom = &Pattern{Op: token, Variable: true}
			} else {
				atom = &Pattern{Op: token}
			}
			if len(stack) == 0 {
				if i != len(tokens) {
					return nil, fmt.Errorf("unparsed trailing tokens in pattern: %v", tokens[i:])
				}
				return atom, nil
			}
			parent := stack[len(stack)-1]
			parent.Children = append(parent.Children, atom)
		}
	}
	if len(stack) != 0 {
		return nil, fmt.Errorf("unclosed parenthesis in pattern")
	}
	return nil, fmt.Errorf("unexpected end of tokens")
}

func patternTokens(source string) []string {
	var tokens []string
	var current strings.Builder
	flush := func() {
		if current.Len() != 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}
	for _, r := range source {
		switch {
		case r == '(' || r == ')' || r == ',':
			flush()
			if r != ',' {
				tokens = append(tokens, string(r))
			}
		case unicode.IsSpace(r):
			flush()
		default:
			current.WriteRune(r)
		}
	}
	flush()
	return tokens
}
