package hopper

import (
	"fmt"
	"strings"
)

// pdlEntry is the statement pair AddPDLEntry puts at the top of a kernel.
const pdlEntry = "\n    eg_pdl_wait(0u);\n    eg_pdl_trigger(0u);"

// AddPDLEntry returns src with every __global__ kernel definition opening
// with eg_pdl_wait and then eg_pdl_trigger, and the names of those kernels.
// src must already contain [PDLHelpers].
//
// A kernel rewritten this way may be the consumer of a programmatic
// dependency (a graph edge of type programmatic, or a stream launch with
// programmatic stream serialization): it waits for the full completion and
// memory flush of its predecessor grid before its first instruction that
// could read or write memory, so it is correct whether or not it overlaps the
// predecessor. Its trigger lets its own successor launch once all of its CTAs
// have passed the wait. Since every kernel waits on its immediate
// predecessor, a chain of such kernels orders each kernel after all earlier
// ones. A successor that does not wait (a kernel not passed through
// AddPDLEntry) must still depend on it through an ordinary edge.
//
// What PDL saves is the launch latency between grids: the successor's CTAs
// are scheduled while the predecessor drains. A kernel with independent
// prologue work could overlap more by waiting later; AddPDLEntry does not
// look for such work.
//
// The helpers emit griddepcontrol only for __CUDA_ARCH__ >= 900, so the
// rewritten source still compiles for older targets, where the host must not
// ask for programmatic launches.
func AddPDLEntry(src string) (string, []string, error) {
	if !strings.Contains(src, "EG_PDL_DEVICE_HELPERS") {
		return "", nil, fmt.Errorf("pdl entry: source lacks the PDL helpers")
	}
	var b strings.Builder
	var names []string
	rest := src
	for {
		i := globalIndex(rest)
		if i < 0 {
			b.WriteString(rest)
			break
		}
		name, open, err := kernelHeader(rest[i:])
		if err != nil {
			return "", nil, fmt.Errorf("pdl entry: %w", err)
		}
		if open < 0 { // a declaration
			b.WriteString(rest[:i+len("__global__")])
			rest = rest[i+len("__global__"):]
			continue
		}
		end := i + open + 1
		b.WriteString(rest[:end])
		b.WriteString(pdlEntry)
		names = append(names, name)
		rest = rest[end:]
	}
	if len(names) == 0 {
		return "", nil, fmt.Errorf("pdl entry: no kernel definitions")
	}
	return b.String(), names, nil
}

// globalIndex returns the index in s of the first __global__ that begins a
// line, after an optional extern "C", or -1.
func globalIndex(s string) int {
	off := 0
	for {
		j := strings.Index(s[off:], "__global__")
		if j < 0 {
			return -1
		}
		j += off
		k := strings.LastIndexByte(s[:j], '\n') + 1
		lead := strings.TrimSpace(s[k:j])
		if lead == "" || lead == `extern "C"` {
			return j
		}
		off = j + len("__global__")
	}
}

// kernelHeader parses the kernel header at the start of s, which begins with
// __global__. It returns the kernel's name and the index of its body's
// opening brace, or -1 if the header ends in a semicolon.
func kernelHeader(s string) (name string, open int, err error) {
	i := len("__global__")
	ident := func() string {
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
			i++
		}
		j := i
		for i < len(s) && (s[i] == '_' || 'a' <= s[i] && s[i] <= 'z' || 'A' <= s[i] && s[i] <= 'Z' || '0' <= s[i] && s[i] <= '9') {
			i++
		}
		return s[j:i]
	}
	parens := func() bool {
		for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
			i++
		}
		if i >= len(s) || s[i] != '(' {
			return false
		}
		depth := 0
		for ; i < len(s); i++ {
			switch s[i] {
			case '(':
				depth++
			case ')':
				depth--
				if depth == 0 {
					i++
					return true
				}
			}
		}
		return false
	}
	for name == "" {
		switch id := ident(); {
		case id == "":
			return "", 0, fmt.Errorf("kernel header %.40q: no name", s)
		case id == "void" || id == "static" || id == "inline":
		case strings.HasPrefix(id, "__") && strings.HasSuffix(id, "__"):
			// An attribute: __launch_bounds__(...), __cluster_dims__(...),
			// __maxnreg__(...).
			if !parens() {
				return "", 0, fmt.Errorf("kernel header %.40q: bad attribute %s", s, id)
			}
		default:
			name = id
		}
	}
	if !parens() {
		return "", 0, fmt.Errorf("kernel %s: no parameter list", name)
	}
	for i < len(s) && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
		i++
	}
	switch {
	case i < len(s) && s[i] == '{':
		return name, i, nil
	case i < len(s) && s[i] == ';':
		return name, -1, nil
	}
	return "", 0, fmt.Errorf("kernel %s: no body", name)
}
