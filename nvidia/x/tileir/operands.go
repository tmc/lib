package tileir

import "strings"

// SplitOperands splits an operation's operand string at top-level commas.
// Commas nested inside angle brackets <...>, curly braces {...}, parentheses (...),
// square brackets [...], or quoted strings "..." are preserved.
func SplitOperands(s string) []string {
	var ops []string
	var cur strings.Builder
	angle, curly, paren, square := 0, 0, 0, 0
	inQuote := false
	var prev rune

	for _, r := range s {
		if inQuote {
			cur.WriteRune(r)
			if r == '"' && prev != '\\' {
				inQuote = false
			}
			prev = r
			continue
		}

		switch r {
		case '"':
			inQuote = true
			cur.WriteRune(r)
		case '<':
			angle++
			cur.WriteRune(r)
		case '>':
			if prev == '-' {
				cur.WriteRune(r)
				prev = r
				continue
			}
			if angle > 0 {
				angle--
			}
			cur.WriteRune(r)
		case '{':
			curly++
			cur.WriteRune(r)
		case '}':
			if curly > 0 {
				curly--
			}
			cur.WriteRune(r)
		case '(':
			paren++
			cur.WriteRune(r)
		case ')':
			if paren > 0 {
				paren--
			}
			cur.WriteRune(r)
		case '[':
			square++
			cur.WriteRune(r)
		case ']':
			if square > 0 {
				square--
			}
			cur.WriteRune(r)
		case ',':
			if angle == 0 && curly == 0 && paren == 0 && square == 0 {
				if trimmed := strings.TrimSpace(cur.String()); trimmed != "" {
					ops = append(ops, trimmed)
				}
				cur.Reset()
				prev = r
				continue
			}
			cur.WriteRune(r)
		default:
			cur.WriteRune(r)
		}
		prev = r
	}

	if trimmed := strings.TrimSpace(cur.String()); trimmed != "" {
		ops = append(ops, trimmed)
	}
	return ops
}
