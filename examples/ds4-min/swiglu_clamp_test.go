package main

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func swigluClampArray(t uint32, values ...any) *gguf.File {
	return &gguf.File{Metadata: map[string]gguf.Value{
		swigluClampKey: {TypeID: t, Value: values},
	}}
}

func swigluClampFloats(n int, v float32) []any {
	out := make([]any, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// TestValidateSwiGLUClamp is the positive control for the load-time gate. The
// checkpoint ds4-min runs agrees with swigluClampExp on every layer, so the
// gate never fires in production; these cases are the only evidence that it
// can fire at all.
func TestValidateSwiGLUClamp(t *testing.T) {
	const layers = 4
	disagree := swigluClampFloats(layers, swigluClampExp)
	disagree[2] = float32(0)
	tests := []struct {
		name string
		file *gguf.File
		want string // substring of the expected error; empty means accept
	}{
		{"exact", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers, swigluClampExp)...), ""},
		{"float64", swigluClampArray(gguf.TypeArray, float64(10), float64(10), float64(10), float64(10)), ""},
		{"longer than the layer count", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers+8, swigluClampExp)...), ""},
		{"within tolerance", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers, swigluClampExp+5e-6)...), ""},
		{"missing", &gguf.File{Metadata: map[string]gguf.Value{}}, "missing or not an array"},
		{"not an array", &gguf.File{Metadata: map[string]gguf.Value{
			swigluClampKey: {TypeID: gguf.TypeFloat32, Value: float32(10)},
		}}, "missing or not an array"},
		{"too short", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers-1, swigluClampExp)...), "has 3 entries, want at least 4"},
		{"wrong element type", swigluClampArray(gguf.TypeArray, uint32(10), uint32(10), uint32(10), uint32(10)), "layer 0 is uint32"},
		{"clamping disabled", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers, 0)...), "layer 0 is 0, want 10"},
		{"one layer disagrees", swigluClampArray(gguf.TypeArray, disagree...), "layer 2 is 0, want 10"},
		{"outside tolerance", swigluClampArray(gguf.TypeArray, swigluClampFloats(layers, swigluClampExp+1e-3)...), "layer 0 is 10.001"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateSwiGLUClamp(tt.file, layers)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateSwiGLUClamp = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateSwiGLUClamp = nil, want an error naming %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validateSwiGLUClamp = %v, want an error naming %q", err, tt.want)
			}
		})
	}
}

// swigluClampCall matches a call that could be clamping one half of the SwiGLU
// activation, in CUDA C or in a Metal shader embedded in a Go string.
var swigluClampCall = regexp.MustCompile(`\b(?:fminf|fmaxf|min|max|clamp)\s*\(`)

// swigluClampOperand matches the gate or up operand by name.
var swigluClampOperand = regexp.MustCompile(`\b(?:gate|up)\b`)

// swigluClampNumber matches a decimal literal, with the optional float suffix
// the kernel sources spell.
var swigluClampNumber = regexp.MustCompile(`-?[0-9]+\.[0-9]+f?`)

// TestSwiGLUClampLiteralsMatchConstant asserts that every clamp bound written
// as a literal in a kernel source agrees with swigluClampExp.
//
// The kernels cannot reference a Go constant, so the value is spelled out in
// each of them. That is the actual parity gap: validateSwiGLUClamp checks the
// model file against swigluClampExp, and this test checks the kernels against
// the same constant, so the two ends cannot drift apart silently.
//
// Like logit_extent_test.go, this reads the directory rather than the build's
// file set. Half the clamp literals live in metal_ops_darwin.go and half in
// .cu files compiled only on linux; a scan whose denominator is one GOOS sees
// one half and reports the package clean. os.ReadDir has no build tags.
func TestSwiGLUClampLiteralsMatchConstant(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	scanned, found := 0, 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			continue
		}
		switch {
		case strings.HasSuffix(name, ".go"), strings.HasSuffix(name, ".cu"), strings.HasSuffix(name, ".cuh"):
		default:
			continue
		}
		b, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		scanned++
		if name == "swiglu_clamp_test.go" {
			continue // holds the patterns themselves
		}
		for i, line := range strings.Split(string(b), "\n") {
			if !swigluClampCall.MatchString(line) || !swigluClampOperand.MatchString(line) {
				continue
			}
			for _, m := range swigluClampNumber.FindAllStringIndex(line, -1) {
				if !swigluClampArgument(line, m[0], m[1]) {
					continue // part of a larger expression, not a clamp bound
				}
				found++
				text := strings.TrimSuffix(line[m[0]:m[1]], "f")
				v, err := strconv.ParseFloat(text, 64)
				if err != nil {
					t.Fatalf("%s:%d: parse %q: %v", name, i+1, text, err)
				}
				// The sign is half the bound. An earlier version of this
				// scanner stripped a leading minus before comparing, so
				// fmaxf(up[i], -10.0f) and fmaxf(up[i], 10.0f) were the same
				// literal to it -- the second collapses the interval to a
				// point and the scanner counted it as agreeing. Which bound a
				// literal is depends on the call it is an argument to, not on
				// the line it sits on: a single line spells both.
				want, kind := swigluClampBound(line, m[0])
				if want != 0 && v != want {
					t.Errorf("%s:%d: %s bound is %v, want %v (swigluClampExp is %v):\n\t%s",
						name, i+1, kind, v, want, float64(swigluClampExp), strings.TrimSpace(line))
					continue
				}
				if want == 0 && v != swigluClampExp && v != -swigluClampExp {
					t.Errorf("%s:%d clamps at %v, but swigluClampExp is %v:\n\t%s",
						name, i+1, v, float64(swigluClampExp), strings.TrimSpace(line))
				}
			}
		}
	}
	// A scan that matched nothing because it read nothing, or because the
	// literals moved to a spelling these patterns no longer recognize, would
	// otherwise pass in silence. Both denominators are stated.
	if scanned < 200 {
		t.Fatalf("scanned %d source files, expected the whole package; the scan's denominator is wrong", scanned)
	}
	if found < 18 {
		t.Fatalf("found %d clamp literals in %d files, expected at least 18; the patterns no longer match the kernels", found, scanned)
	}
	t.Logf("scanned %d source files, checked %d clamp literals against swigluClampExp=%v", scanned, found, float64(swigluClampExp))
}

// swigluClampArgument reports whether line[start:end] stands alone as a call
// argument, rather than being one term of a larger expression such as the
// 1.0f in 1.0f+exp(-x).
// swigluClampBound reports the bound a literal at start is, judged by the call
// it is an argument to. A maximum takes the lower bound and a minimum the
// upper, so the two differ by sign and not only by magnitude. kind names the
// call for the failure message; a zero want means the enclosing call was not
// one whose direction is known, and the caller falls back to magnitude.
func swigluClampBound(line string, start int) (want float64, kind string) {
	switch swigluClampEnclosingCall(line, start) {
	case "fmaxf", "max":
		return -float64(swigluClampExp), "lower"
	case "fminf", "min":
		return float64(swigluClampExp), "upper"
	}
	return 0, ""
}

// swigluClampEnclosingCall names the function whose argument list contains the
// byte at start, by walking left to the unmatched open paren and reading the
// identifier in front of it.
func swigluClampEnclosingCall(line string, start int) string {
	depth := 0
	for i := start - 1; i >= 0; i-- {
		switch line[i] {
		case ')':
			depth++
		case '(':
			if depth == 0 {
				end := i
				for end > 0 && (line[end-1] == ' ' || line[end-1] == '\t') {
					end--
				}
				begin := end
				for begin > 0 && (isSwiGLUIdentByte(line[begin-1])) {
					begin--
				}
				return line[begin:end]
			}
			depth--
		}
	}
	return ""
}

func isSwiGLUIdentByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

func swigluClampArgument(line string, start, end int) bool {
	before := byte('?')
	for i := start - 1; i >= 0; i-- {
		if line[i] != ' ' && line[i] != '\t' {
			before = line[i]
			break
		}
	}
	after := byte('?')
	for i := end; i < len(line); i++ {
		if line[i] != ' ' && line[i] != '\t' {
			after = line[i]
			break
		}
	}
	return (before == '(' || before == ',') && (after == ')' || after == ',')
}
