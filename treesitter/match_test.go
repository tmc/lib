package treesitter

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
)

// matchSrc declares two functions and calls one of them twice.
const matchSrc = `int add(int a, int b) { return a + b; }
int main() { return add(1, add(2, 3)); }
`

func TestMatches(t *testing.T) {
	lang := testGrammar(t)
	tree, err := Parse(lang, []byte(matchSrc))
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if got, want := string(tree.Source()), matchSrc; got != want {
		t.Errorf("Source() = %q, want %q", got, want)
	}

	q, err := NewQuery(lang, `(call_expression function: (identifier) @callee)`)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()

	var got []string
	for m := range q.Matches(tree.RootNode()) {
		if m.Pattern != 0 {
			t.Errorf("pattern = %d, want 0", m.Pattern)
		}
		got = append(got, string(m.Text("callee")))
		if n := m.Node("no_such_capture"); !n.IsNull() {
			t.Errorf("Node(missing) = %v, want null", n)
		}
		if n := m.Nodes("callee"); len(n) != 1 {
			t.Errorf("Nodes(callee) = %d nodes, want 1", len(n))
		}
	}
	if want := "add,add"; strings.Join(got, ",") != want {
		t.Errorf("callees = %q, want %q", strings.Join(got, ","), want)
	}

	// Breaking out of the loop must release the cursor, which the race
	// detector and a second run confirm.
	for range q.Matches(tree.RootNode()) {
		break
	}
	if n := count(q.Matches(tree.RootNode())); n != 2 {
		t.Errorf("second pass = %d matches, want 2", n)
	}

	// Captures reports the same nodes in document order.
	got = got[:0]
	for _, c := range q.Captures(tree.RootNode()) {
		got = append(got, c.Name+"="+string(c.Node.Text()))
	}
	if want := "callee=add,callee=add"; strings.Join(got, ",") != want {
		t.Errorf("captures = %q, want %q", strings.Join(got, ","), want)
	}
}

func TestPredicates(t *testing.T) {
	lang := testGrammar(t)
	tree, err := Parse(lang, []byte(matchSrc))
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	root := tree.RootNode()

	tests := []struct {
		name    string
		pattern string
		want    string
	}{
		{"none", `(call_expression function: (identifier) @c)`, "add,add"},
		{"eq", `((call_expression function: (identifier) @c) (#eq? @c "add"))`, "add,add"},
		{"eq-miss", `((call_expression function: (identifier) @c) (#eq? @c "sub"))`, ""},
		{"not-eq", `((call_expression function: (identifier) @c) (#not-eq? @c "add"))`, ""},
		{"match", `((call_expression function: (identifier) @c) (#match? @c "^a.d$"))`, "add,add"},
		{"not-match", `((call_expression function: (identifier) @c) (#not-match? @c "^a"))`, ""},
		{"any-of", `((call_expression function: (identifier) @c) (#any-of? @c "sub" "add"))`, "add,add"},
		{"not-any-of", `((call_expression function: (identifier) @c) (#not-any-of? @c "add"))`, ""},
		{"unknown", `((call_expression function: (identifier) @c) (#lua-match? @c "%a+"))`, "add,add"},
		{"set", `((call_expression function: (identifier) @c) (#set! kind "call"))`, "add,add"},
		{
			"eq-capture",
			`((function_definition
			    type: (primitive_type) @t
			    declarator: (function_declarator declarator: (identifier) @n))
			  (#eq? @t "int"))`,
			"int,int",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := NewQuery(lang, tt.pattern)
			if err != nil {
				t.Fatal(err)
			}
			defer q.Close()
			var got []string
			for m := range q.Matches(root) {
				got = append(got, string(m.Captures[0].Node.Text()))
			}
			if s := strings.Join(got, ","); s != tt.want {
				t.Errorf("captures = %q, want %q", s, tt.want)
			}
		})
	}
}

func TestPredicateReporting(t *testing.T) {
	lang := testGrammar(t)
	q, err := NewQuery(lang, `((identifier) @a
	  (#eq? @a "x")
	  (#set! @a kind "name")
	  (#is-not? local)
	  (#lua-match? @a "%a"))`)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()

	preds := q.Predicates(0)
	if len(preds) != 4 {
		t.Fatalf("Predicates = %v, want 4", preds)
	}
	if got, want := preds[0].String(), `(#eq? @a "x")`; got != want {
		t.Errorf("Predicates[0] = %s, want %s", got, want)
	}
	if got, want := preds[3].Operator, "lua-match?"; got != want {
		t.Errorf("Predicates[3].Operator = %q, want %q", got, want)
	}

	props := q.Properties(0)
	want := []Property{
		{Key: "kind", Value: "name", Capture: "a"},
		{Key: "local", Negate: true},
	}
	if len(props) != len(want) {
		t.Fatalf("Properties = %+v, want %+v", props, want)
	}
	for i := range want {
		if props[i] != want[i] {
			t.Errorf("Properties[%d] = %+v, want %+v", i, props[i], want[i])
		}
	}
}

func TestPredicateErrors(t *testing.T) {
	lang := testGrammar(t)
	tests := []struct {
		name    string
		pattern string
		want    string
	}{
		{"arity", `((identifier) @a (#eq? @a))`, "takes 2 arguments"},
		{"literal-first", `((identifier) @a (#eq? "a" "b"))`, "must be a capture"},
		{"capture-regexp", `((identifier) @a (#match? @a @a))`, "must be a string"},
		{"bad-regexp", `((identifier) @a (#match? @a "a("))`, "error parsing regexp"},
		{"any-of-capture", `((identifier) @a (#any-of? @a @a))`, "must be strings"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q, err := NewQuery(lang, tt.pattern)
			if err == nil {
				q.Close()
				t.Fatal("NewQuery succeeded, want error")
			}
			var qe *QueryError
			if !errors.As(err, &qe) {
				t.Fatalf("error is %T, want *QueryError", err)
			}
			if qe.Kind != QueryErrorStructure {
				t.Errorf("kind = %v, want structure", qe.Kind)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

func TestTreeErrors(t *testing.T) {
	lang := testGrammar(t)
	clean, err := Parse(lang, []byte("int x;\n"))
	if err != nil {
		t.Fatal(err)
	}
	defer clean.Close()
	if clean.HasError() {
		t.Error("clean tree reports an error")
	}
	if n := count(clean.Errors()); n != 0 {
		t.Errorf("clean tree has %d errors, want 0", n)
	}

	broken, err := Parse(lang, []byte("int x = ;\nvoid f( {}\n"))
	if err != nil {
		t.Fatal(err)
	}
	defer broken.Close()
	if !broken.HasError() {
		t.Fatal("broken tree reports no error")
	}
	n := 0
	for e := range broken.Errors() {
		if !e.IsError() && !e.IsMissing() {
			t.Errorf("Errors() yielded %v, which is neither an error nor missing", e)
		}
		n++
	}
	if n == 0 {
		t.Error("Errors() yielded nothing for a broken tree")
	}
}

func TestParseFile(t *testing.T) {
	testGrammar(t)
	_, name := testGrammarSpec(t)
	if name != "c" {
		t.Skipf("test grammar is %s, not c", name)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "hello.c")
	if err := os.WriteFile(path, []byte(matchSrc), 0o666); err != nil {
		t.Fatal(err)
	}
	t.Setenv(GrammarPathEnv, filepath.Dir(mustGrammarPath(t)))

	tree, err := ParseFile(path)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if got := tree.RootNode().Type(); got != "translation_unit" {
		t.Errorf("root = %q, want translation_unit", got)
	}
	if string(tree.Source()) != matchSrc {
		t.Error("tree source does not match the file")
	}

	if _, err := ParseFile(filepath.Join(dir, "x.unknown-extension")); err == nil {
		t.Error("ParseFile accepted an unknown extension")
	}
}

// mustGrammarPath links the test grammar into a temporary directory under
// the name FindGrammar looks for, and returns the link.
func mustGrammarPath(t *testing.T) string {
	t.Helper()
	src, name := testGrammarSpec(t)
	dir := t.TempDir()
	dst := filepath.Join(dir, name+filepath.Ext(src))
	if err := os.Symlink(src, dst); err != nil {
		t.Fatal(err)
	}
	return dst
}

func TestParseReaderAt(t *testing.T) {
	p, _ := testParser(t)
	r := strings.NewReader(matchSrc)
	tree, err := p.ParseReaderAt(r, r.Size(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if tree.HasError() {
		t.Errorf("parse error: %s", tree.RootNode())
	}
	if got, want := tree.RootNode().EndByte(), len(matchSrc); got != want {
		t.Errorf("end byte = %d, want %d", got, want)
	}
	// A reader-parsed tree keeps no source.
	if tree.Source() != nil {
		t.Error("Source() is not nil")
	}
	if tree.RootNode().Text() != nil {
		t.Error("Text() is not nil")
	}

	// A short size parses only that prefix.
	short, err := p.ParseReaderAt(strings.NewReader(matchSrc), 20, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer short.Close()
	if got := short.RootNode().EndByte(); got > 20 {
		t.Errorf("end byte = %d, want at most 20", got)
	}

	if _, err := p.ParseReaderAt(bytes.NewReader(nil), -1, nil); err == nil {
		t.Error("ParseReaderAt accepted a negative size")
	}
}

func TestNewQueryFS(t *testing.T) {
	lang := testGrammar(t)
	fsys := fstest.MapFS{
		"queries/calls.scm": {Data: []byte(`(call_expression function: (identifier) @callee)`)},
		"queries/bad.scm":   {Data: []byte(`(no_such_node) @x`)},
	}
	q, err := NewQueryFS(lang, fsys, "queries/calls.scm")
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if got := q.PatternCount(); got != 1 {
		t.Errorf("pattern count = %d, want 1", got)
	}
	if _, err := NewQueryFS(lang, fsys, "queries/missing.scm"); err == nil {
		t.Error("NewQueryFS accepted a missing file")
	}
	if _, err := NewQueryFS(lang, fsys, "queries/bad.scm"); err == nil {
		t.Error("NewQueryFS accepted an invalid query")
	} else if !strings.Contains(err.Error(), "bad.scm") {
		t.Errorf("error %v does not name the file", err)
	}
}

func TestMatchesContext(t *testing.T) {
	lang := testGrammar(t)
	tree, err := Parse(lang, []byte(matchSrc))
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	q, err := NewQuery(lang, `(call_expression function: (identifier) @callee)`)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()

	n := 0
	for m, err := range q.MatchesContext(context.Background(), tree.RootNode()) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if string(m.Text("callee")) != "add" {
			t.Errorf("callee = %q, want add", m.Text("callee"))
		}
		n++
	}
	if n != 2 {
		t.Errorf("got %d matches, want 2", n)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	sawErr := false
	for _, err := range q.MatchesContext(ctx, tree.RootNode()) {
		if err != nil {
			sawErr = true
		}
	}
	if !sawErr {
		t.Error("cancelled context produced no error")
	}
}

func TestSlogLogger(t *testing.T) {
	p, _ := testParser(t)
	var buf bytes.Buffer
	h := slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	p.SetLogger(SlogLogger(slog.New(h)))
	if _, err := p.Parse([]byte("int x;"), nil); err != nil {
		t.Fatal(err)
	}
	p.SetLogger(nil)
	if !strings.Contains(buf.String(), "kind=") {
		t.Errorf("log output %q lacks a kind attribute", buf.String())
	}
	if SlogLogger(nil) != nil {
		t.Error("SlogLogger(nil) is not nil")
	}
}
