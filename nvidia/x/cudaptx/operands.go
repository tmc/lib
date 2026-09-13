package cudaptx

import "strings"

// SplitOperands splits an instruction's operand text at top-level commas.
// Commas inside bracketed, braced, or parenthesized operands are retained.
func SplitOperands(s string) []string {
	var ops []string
	var cur strings.Builder
	depthSquare, depthCurly, depthParen := 0, 0, 0
	for _, r := range s {
		switch r {
		case '[':
			depthSquare++
			cur.WriteRune(r)
		case ']':
			if depthSquare > 0 {
				depthSquare--
			}
			cur.WriteRune(r)
		case '{':
			depthCurly++
			cur.WriteRune(r)
		case '}':
			if depthCurly > 0 {
				depthCurly--
			}
			cur.WriteRune(r)
		case '(':
			depthParen++
			cur.WriteRune(r)
		case ')':
			if depthParen > 0 {
				depthParen--
			}
			cur.WriteRune(r)
		case ',':
			if depthSquare == 0 && depthCurly == 0 && depthParen == 0 {
				ops = append(ops, strings.TrimSpace(cur.String()))
				cur.Reset()
				continue
			}
			cur.WriteRune(r)
		default:
			cur.WriteRune(r)
		}
	}
	if trimmed := strings.TrimSpace(cur.String()); trimmed != "" {
		ops = append(ops, trimmed)
	}
	return ops
}
