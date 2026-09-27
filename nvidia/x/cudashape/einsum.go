package cudashape

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	// ErrRepeatedRank reports a dimension rank repeated within one tensor.
	ErrRepeatedRank = errors.New("cudashape: repeated einsum rank")
	// ErrMultiLetterPrefix reports a token whose prefix has more than one letter.
	ErrMultiLetterPrefix = errors.New("cudashape: einsum rank prefix must be one letter")
	// ErrUnbalancedGroup reports unmatched parentheses in a dimension string.
	ErrUnbalancedGroup = errors.New("cudashape: unbalanced einsum rank group")
	// ErrInvalidEquation reports a malformed einsum equation.
	ErrInvalidEquation = errors.New("cudashape: invalid einsum equation")
)

// DimToken is one normalized einsum rank, such as "A", "B12", or "(P+R)".
type DimToken string

// Validate reports whether t has the grammar accepted by [ParseDimTokens].
// A rank prefix is one letter; parenthesized groups must be balanced.
func (t DimToken) Validate() error {
	s := string(t)
	if len(s) == 0 {
		return fmt.Errorf("%w: empty token", ErrMultiLetterPrefix)
	}
	if strings.HasPrefix(s, "(") {
		depth := 0
		for _, r := range s {
			switch r {
			case '(':
				depth++
			case ')':
				depth--
				if depth < 0 {
					return fmt.Errorf("%w: %q", ErrUnbalancedGroup, s)
				}
			}
		}
		if depth != 0 || !strings.HasSuffix(s, ")") {
			return fmt.Errorf("%w: %q", ErrUnbalancedGroup, s)
		}
		depth = 0
		for i, r := range s {
			if r == '(' {
				depth++
			} else if r == ')' {
				depth--
				if depth == 0 && i != len(s)-1 {
					return fmt.Errorf("%w: %q", ErrUnbalancedGroup, s)
				}
			}
		}
		return nil
	}
	runes := []rune(s)
	if len(runes) == 0 || !unicode.IsLetter(runes[0]) {
		return fmt.Errorf("%w: %q", ErrMultiLetterPrefix, s)
	}
	i := 1
	for i < len(runes) && unicode.IsDigit(runes[i]) {
		i++
	}
	if i != len(runes) {
		return fmt.Errorf("%w: %q", ErrMultiLetterPrefix, s)
	}
	return nil
}

// ParseDimTokens parses a SOLAR-style einsum dimension string. Each rank is
// one letter with an optional integer suffix, or a balanced parenthesized
// group. Tokens are uppercased. Non-letter characters outside groups are
// skipped. When validate is true, repeated normalized tokens are rejected.
func ParseDimTokens(dims string, validate bool) ([]DimToken, error) {
	var tokens []DimToken
	seen := make(map[DimToken]struct{})
	for i := 0; i < len(dims); {
		r, size := runeAt(dims, i)
		if r == '(' {
			start, depth := i, 1
			i += size
			for i < len(dims) && depth != 0 {
				r, size = runeAt(dims, i)
				switch r {
				case '(':
					depth++
				case ')':
					depth--
				}
				i += size
			}
			if depth != 0 {
				return nil, fmt.Errorf("%w: %q", ErrUnbalancedGroup, dims[start:])
			}
			token := DimToken(strings.ToUpper(dims[start:i]))
			if validate {
				if _, ok := seen[token]; ok {
					return nil, fmt.Errorf("%w: %s", ErrRepeatedRank, token)
				}
				seen[token] = struct{}{}
			}
			tokens = append(tokens, token)
			continue
		}
		if r == ')' {
			return nil, fmt.Errorf("%w: unexpected close parenthesis", ErrUnbalancedGroup)
		}
		if !unicode.IsLetter(r) {
			i += size
			continue
		}
		start := i
		i += size
		for i < len(dims) {
			r, size = runeAt(dims, i)
			if !unicode.IsDigit(r) {
				break
			}
			i += size
		}
		token := DimToken(strings.ToUpper(dims[start:i]))
		if validate {
			if _, ok := seen[token]; ok {
				return nil, fmt.Errorf("%w: %s", ErrRepeatedRank, token)
			}
			seen[token] = struct{}{}
		}
		tokens = append(tokens, token)
	}
	return tokens, nil
}

// ParseEinsumEquation splits an equation into per-operand and output ranks.
// Repetition is checked independently within each tensor; a shared rank
// across operands is the normal signal for contraction.
func ParseEinsumEquation(eq string) (inputs [][]DimToken, output []DimToken, err error) {
	if strings.Count(eq, "->") != 1 {
		return nil, nil, fmt.Errorf("%w: expected one ->", ErrInvalidEquation)
	}
	parts := strings.SplitN(eq, "->", 2)
	lhs, rhs := strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1])
	if lhs == "" {
		return nil, nil, fmt.Errorf("%w: input ranks are required", ErrInvalidEquation)
	}
	for _, operand := range strings.Split(lhs, ",") {
		operand = strings.TrimSpace(operand)
		if operand == "" {
			return nil, nil, fmt.Errorf("%w: empty input operand", ErrInvalidEquation)
		}
		tokens, parseErr := ParseDimTokens(operand, true)
		if parseErr != nil {
			return nil, nil, fmt.Errorf("%w: input %q: %w", ErrInvalidEquation, operand, parseErr)
		}
		inputs = append(inputs, tokens)
	}
	output, err = ParseDimTokens(rhs, true)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: output %q: %w", ErrInvalidEquation, rhs, err)
	}
	return inputs, output, nil
}

func runeAt(s string, i int) (rune, int) {
	r := rune(s[i])
	if r >= 0x80 {
		r, size := utf8.DecodeRuneInString(s[i:])
		return r, size
	}
	return r, 1
}
