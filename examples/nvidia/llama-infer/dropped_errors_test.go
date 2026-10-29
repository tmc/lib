package main

import (
	"bytes"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrace"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

var (
	errFakeClose  = errors.New("fake close failed")
	errFakeWrite  = errors.New("fake trace write failed")
	errFakeDecode = errors.New("fake decode failed")
	errFakeArch   = errors.New("fake compute capability failed")
)

// closeFailing is a checkpoint whose Close closes the real one and then
// returns err.
type closeFailing struct {
	tensorFile
	err    error
	closes int
}

func (c *closeFailing) Close() error {
	c.closes++
	return errors.Join(c.tensorFile.Close(), c.err)
}

// TestLoadModelCloseError checks that loadModel closes the checkpoint once
// and returns an error from that close, joined with any load error and
// without weights.
func TestLoadModelCloseError(t *testing.T) {
	dropNorm := func(m map[string]safetensors.Tensor) { delete(m, "model.norm.weight") }
	for _, tt := range []struct {
		name     string
		edit     func(map[string]safetensors.Tensor)
		closeErr error
		want     []error  // errors the result must wrap
		wantText []string // text the error must contain
	}{
		{"positive control: close succeeds", nil, nil, nil, nil},
		{"close fails", nil, errFakeClose, []error{errFakeClose}, []string{"close safetensors"}},
		{"load and close fail", dropNorm, errFakeClose, []error{errFakeClose}, []string{"missing tensor model.norm.weight", "close safetensors"}},
		{"positive control: load fails", dropNorm, nil, nil, []string{"missing tensor model.norm.weight"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := tinyCheckpoint(t, tt.edit)
			var c *closeFailing
			open := func(dir string) (tensorFile, error) {
				f, err := openSharded(dir)
				if err != nil {
					return nil, err
				}
				c = &closeFailing{tensorFile: f, err: tt.closeErr}
				return c, nil
			}
			w, cfg, err := loadModel(dir, open)
			if c == nil {
				t.Fatal("checkpoint not opened")
			}
			if c.closes != 1 {
				t.Errorf("checkpoint closed %d times, want 1", c.closes)
			}
			for _, e := range tt.want {
				if !errors.Is(err, e) {
					t.Errorf("error %v does not wrap %v", err, e)
				}
			}
			for _, s := range tt.wantText {
				if err == nil || !strings.Contains(err.Error(), s) {
					t.Errorf("error %v does not contain %q", err, s)
				}
			}
			if failed := len(tt.wantText) > 0; failed != (err != nil) {
				t.Errorf("err = %v, want error %v", err, failed)
			}
			if err != nil && (w != nil || cfg != nil) {
				t.Errorf("weights or config returned with error %v", err)
			}
			if err == nil && (w == nil || cfg == nil) {
				t.Error("no weights or config returned without an error")
			}
		})
	}
}

// failWriter fails every write.
type failWriter struct{}

func (failWriter) Write([]byte) (int, error) { return 0, errFakeWrite }

// decodeFailing is a tokenizer whose Decode fails.
type decodeFailing struct{ Tokenizer }

func (decodeFailing) Decode([]int32) (string, error) { return "", errFakeDecode }

// TestEncodePromptTraceError checks that encodePrompt returns an error
// when the span around the encoding cannot be written, instead of
// returning no tokens and no error.
func TestEncodePromptTraceError(t *testing.T) {
	tok := NewByteTokenizer()
	want, err := tok.Encode("abc")
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	for _, tt := range []struct {
		name    string
		tracer  *cudatrace.Tracer
		wantErr bool
	}{
		{"positive control: no tracer", nil, false},
		{"positive control: tracer", cudatrace.New(&buf), false},
		{"trace write fails", cudatrace.New(failWriter{}), true},
	} {
		got, err := encodePrompt(tt.tracer, tok, "abc")
		if tt.wantErr {
			if !errors.Is(err, errFakeWrite) || got != nil {
				t.Errorf("%s: got %v, %v; want no tokens and an error wrapping %v", tt.name, got, err, errFakeWrite)
			}
			continue
		}
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("%s: got %v, %v; want %v", tt.name, got, err, want)
		}
	}
	if buf.Len() == 0 {
		t.Error("positive control: tracer wrote nothing")
	}
}

// TestRunCPUInferenceErrors checks that runCPUInference returns an error
// from decoding a generated token and from writing its trace span.
func TestRunCPUInferenceErrors(t *testing.T) {
	// runCPUInference streams to standard output; keep it out of the
	// test output. No test in the package runs in parallel.
	devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	stdout := os.Stdout
	os.Stdout = devNull
	t.Cleanup(func() {
		os.Stdout = stdout
		if err := devNull.Close(); err != nil {
			t.Error(err)
		}
	})

	cfg := DefaultModelConfig()
	w := GenerateDeterministicWeights(&cfg)
	var buf bytes.Buffer
	for _, tt := range []struct {
		name   string
		tok    Tokenizer
		tracer *cudatrace.Tracer
		want   error
	}{
		{"positive control", NewByteTokenizer(), cudatrace.New(&buf), nil},
		{"decode fails", decodeFailing{NewByteTokenizer()}, nil, errFakeDecode},
		{"trace write fails", NewByteTokenizer(), cudatrace.New(failWriter{}), errFakeWrite},
	} {
		err := runCPUInference(w, &cfg, tt.tok, []int32{1, 2}, tt.tracer, nil)
		if tt.want == nil && err != nil || !errors.Is(err, tt.want) {
			t.Errorf("%s: err = %v, want %v", tt.name, err, tt.want)
		}
	}
	if buf.Len() == 0 {
		t.Error("positive control: tracer wrote nothing")
	}
}

// TestArchOr checks that archOr keeps the default compute capability when
// the device cannot report one.
func TestArchOr(t *testing.T) {
	for _, tt := range []struct {
		name         string
		cc           func() (int, int, error)
		major, minor int
		wantErr      bool
	}{
		{"reported", func() (int, int, error) { return 9, 0, nil }, 9, 0, false},
		{"failed", func() (int, int, error) { return 0, 0, errFakeArch }, 8, 0, true},
		{"failed with a partial answer", func() (int, int, error) { return 7, 5, errFakeArch }, 8, 0, true},
	} {
		major, minor, err := archOr(tt.cc, 8, 0)
		if major != tt.major || minor != tt.minor || (err != nil) != tt.wantErr {
			t.Errorf("%s: archOr = %d, %d, %v; want %d, %d, error %v", tt.name, major, minor, err, tt.major, tt.minor, tt.wantErr)
		}
	}
}

// blankAllowed counts, by file and statement text, the non-test statements
// that discard the last result of a call on purpose.
var blankAllowed = map[string]int{
	// filepath.Glob fails only on a malformed pattern. A home directory
	// containing glob syntax then finds no Hugging Face snapshot, and
	// findDefaultModelPath goes on to its fixed candidates.
	"main.go: matches, _ := filepath.Glob(pattern)": 1,
}

// blankResults parses src and returns the number of assignments and
// declarations it holds whose value is a single call, and the statements
// among them that assign the call's last result, the error by convention,
// to the blank identifier. The check is syntactic: it cannot tell a call
// that returns an error from one that does not, nor see an expression
// statement that drops an error.
func blankResults(name string, src []byte) (n int, bad []discard, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return 0, nil, err
	}
	check := func(node ast.Node, lhs []ast.Expr, rhs []ast.Expr) {
		if len(rhs) != 1 {
			return
		}
		if _, ok := ast.Unparen(rhs[0]).(*ast.CallExpr); !ok {
			return
		}
		n++
		if id, ok := lhs[len(lhs)-1].(*ast.Ident); ok && id.Name == "_" {
			text := string(src[fset.Position(node.Pos()).Offset:fset.Position(node.End()).Offset])
			bad = append(bad, discard{key: name + ": " + text, pos: fset.Position(node.Pos()).String()})
		}
	}
	ast.Inspect(f, func(node ast.Node) bool {
		switch s := node.(type) {
		case *ast.AssignStmt:
			check(s, s.Lhs, s.Rhs)
		case *ast.ValueSpec:
			lhs := make([]ast.Expr, len(s.Names))
			for i, id := range s.Names {
				lhs[i] = id
			}
			check(s, lhs, s.Values)
		}
		return true
	})
	return n, bad, nil
}

// TestNoBlankErrors checks every non-test .go file in the package, whatever
// its build constraints, for a call whose last result is discarded.
func TestNoBlankErrors(t *testing.T) {
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var discards []discard
	perFile := make(map[string]int)
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		n, bad, err := blankResults(p, src)
		if err != nil {
			t.Fatal(err)
		}
		perFile[p] = n
		discards = append(discards, bad...)
	}
	// The files that held the dropped errors must have been read and
	// must hold call assignments, or a clean result means nothing was
	// looked at.
	for _, name := range []string{"main.go", "model.go"} {
		if perFile[name] == 0 {
			t.Errorf("%s: no call assignments found", name)
		}
	}
	bad, stale := unallowed(discards, blankAllowed)
	for _, b := range bad {
		t.Errorf("call result discarded: %s", b)
	}
	for _, s := range stale {
		t.Errorf("stale allowance: %s", s)
	}
}

// TestBlankResultsControls checks blankResults on small sources: forms
// that keep the last result, and must-fail forms that each discard it.
func TestBlankResultsControls(t *testing.T) {
	for _, tt := range []struct {
		name string
		body string
		n    int
		bad  bool
	}{
		{"positive control: checked", `
		name, err := dev.Name()
		if err != nil {
			return err
		}`, 1, false},
		{"positive control: first result blank", `
		_, err := f()`, 1, false},
		{"positive control: map index", `
		v, _ := m[k]`, 0, false},
		{"positive control: type assertion", `
		v, _ := x.(T)`, 0, false},
		{"positive control: blank non-call", `
		_ = x`, 0, false},
		{"name discarded", `
		name, _ := dev.Name()`, 1, true},
		{"compute capability discarded", `
		major, minor, _ := dev.ComputeCapability()`, 1, true},
		{"reassigned with discard", `
		archMajor, archMinor, _ = dev.ComputeCapability()`, 1, true},
		{"span discarded", `
		_ = tracer.Span(nil, "x", "y", func() {})`, 1, true},
		{"discarded in a literal", `
		show := func(tok int32) {
			chunk, _ := tokenizer.Decode([]int32{tok})
			fmt.Print(chunk)
		}`, 1, true},
		{"var declaration", `
		var chunk, _ = tokenizer.Decode(toks)`, 1, true},
		{"parenthesized call", `
		_ = (f())`, 1, true},
	} {
		src := "package p\nfunc f() error {\n" + tt.body + "\nreturn nil\n}\n"
		n, bad, err := blankResults("control.go", []byte(src))
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if n != tt.n {
			t.Errorf("%s: %d call assignments found, want %d", tt.name, n, tt.n)
		}
		if got := len(bad) > 0; got != tt.bad {
			t.Errorf("%s: flagged %v, want %v (%v)", tt.name, got, tt.bad, bad)
		}
	}
}
