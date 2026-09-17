package cudalowering

import (
	"fmt"
	"strings"
)

// DefinitionLine returns the 1-based line of source that defines the kernel
// carried in a CUBIN as the symbol sym.
//
// The CUBIN spells a mangled symbol and the source spells the name alone, so
// the symbol is first reduced to its qualified name; an extern "C" name passes
// through, and every instantiation of a template reduces to the one definition
// they were compiled from.
//
// Reporting a line requires evidence that the name is defined there. A comment
// mentioning the kernel, a forward declaration and a launch site all name it
// without being its source location, and attributing a kernel to one of those
// misplaces every line the capture reports for it. Two definitions of the same
// name are ambiguous and are refused rather than resolved to the first.
func DefinitionLine(source, sym string) (int, error) {
	name := MatchKey(sym)
	if i := strings.LastIndex(name, "::"); i >= 0 {
		name = name[i+len("::"):]
	}
	if name == "" {
		return 0, fmt.Errorf("unreadable kernel symbol %q", sym)
	}
	found := 0
	for i, line := range strings.Split(source, "\n") {
		// Drop a trailing line comment first: a declaration commented as
		// "void compute(int *a);   // forward declaration" no longer ends in
		// ";" and would otherwise read as a definition.
		if j := strings.Index(line, "//"); j >= 0 {
			line = line[:j]
		}
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "", strings.HasPrefix(trimmed, "/*"), strings.HasPrefix(trimmed, "*"):
			continue
		case strings.HasSuffix(trimmed, ";"):
			// A declaration or a call, not a definition. A one-line
			// definition ends in "}".
			continue
		case !namesIdentifier(trimmed, name):
			continue
		}
		if found != 0 {
			return 0, fmt.Errorf("ambiguous CUDA definition of %s: lines %d and %d", name, found, i+1)
		}
		found = i + 1
	}
	if found == 0 {
		return 0, fmt.Errorf("missing CUDA definition of %s", name)
	}
	return found, nil
}

// namesIdentifier reports whether line applies the whole identifier name to an
// argument list. A substring match would accept my_compute for compute.
func namesIdentifier(line, name string) bool {
	for i := 0; ; {
		j := strings.Index(line[i:], name)
		if j < 0 {
			return false
		}
		j += i
		i = j + len(name)
		if j > 0 && isIdentByte(line[j-1]) {
			continue
		}
		rest := strings.TrimLeft(line[i:], " \t")
		if strings.HasPrefix(rest, "(") {
			return true
		}
	}
}

func isIdentByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}
