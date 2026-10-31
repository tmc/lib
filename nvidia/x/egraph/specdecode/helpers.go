package specdecode

import (
	"strconv"
	"strings"
)

// cIdent reports whether s is a C identifier that cannot collide with the
// prelude's eg_ names.
func cIdent(s string) bool {
	if s == "" || len(s) > 64 || strings.HasPrefix(s, "eg_") || strings.HasPrefix(s, "__") {
		return false
	}
	for i, r := range s {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

func paramNames(params []string) []string {
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	return names
}

func floatLiteral(v float32) string {
	s := strconv.FormatFloat(float64(v), 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s + "f"
}
