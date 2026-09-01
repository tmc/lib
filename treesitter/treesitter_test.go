package treesitter

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testGrammar locates a C grammar for the live tests. TREE_SITTER_TEST_GRAMMAR
// names one explicitly as "path:name"; otherwise the parsers Neovim installs
// through Homebrew are tried. Tests skip when neither the library nor a
// grammar is available.
func testGrammar(t *testing.T) *Language {
	t.Helper()
	if err := Load(""); err != nil {
		t.Skipf("libtree-sitter unavailable: %v", err)
	}
	path, name := "", "c"
	if spec := os.Getenv("TREE_SITTER_TEST_GRAMMAR"); spec != "" {
		p, n, ok := strings.Cut(spec, ":")
		if !ok {
			t.Fatalf("TREE_SITTER_TEST_GRAMMAR=%q: want path:name", spec)
		}
		path, name = p, n
	} else {
		prefixes := []string{os.Getenv("HOMEBREW_PREFIX"), "/opt/homebrew", "/usr/local", "/usr"}
		for _, prefix := range prefixes {
			if prefix == "" {
				continue
			}
			candidate := filepath.Join(prefix, "lib", "nvim", "parser", "c.so")
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
				break
			}
		}
	}
	if path == "" {
		t.Skip("no C grammar found; set TREE_SITTER_TEST_GRAMMAR=path:name")
	}
	lang, err := LoadLanguage(path, name)
	if err != nil {
		t.Fatalf("LoadLanguage(%s, %s): %v", path, name, err)
	}
	return lang
}

func testParser(t *testing.T) (*Parser, *Language) {
	t.Helper()
	lang := testGrammar(t)
	p, err := NewParser()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(p.Close)
	if err := p.SetLanguage(lang); err != nil {
		t.Fatal(err)
	}
	return p, lang
}

const helloC = "int main() {\n  return 0;\n}\n"

func TestParse(t *testing.T) {
	p, _ := testParser(t)
	src := []byte(helloC)
	tree, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()

	root := tree.RootNode()
	if root.IsNull() {
		t.Fatal("root is null")
	}
	if got := root.Type(); got != "translation_unit" {
		t.Errorf("root type = %q, want translation_unit", got)
	}
	if root.HasError() {
		t.Errorf("unexpected error in %s", root)
	}
	if got, want := root.EndByte(), uint32(len(src)); got != want {
		t.Errorf("root end byte = %d, want %d", got, want)
	}
	if got, want := root.EndPoint(), (Point{Row: 3, Column: 0}); got != want {
		t.Errorf("root end point = %v, want %v", got, want)
	}
	if got := root.ChildCount(); got != 1 {
		t.Fatalf("root child count = %d, want 1", got)
	}
	fn := root.Child(0)
	if got := fn.Type(); got != "function_definition" {
		t.Errorf("child type = %q, want function_definition", got)
	}
	if !fn.Parent().Equal(root) {
		t.Error("child's parent is not root")
	}
	decl := fn.ChildByFieldName("declarator")
	if decl.IsNull() || decl.Type() != "function_declarator" {
		t.Fatalf("declarator = %v", decl)
	}
	name := decl.ChildByFieldName("declarator")
	if got := string(name.Text(src)); got != "main" {
		t.Errorf("function name = %q, want main", got)
	}
	if !strings.Contains(root.String(), "(function_definition") {
		t.Errorf("S-expression %q lacks function_definition", root.String())
	}
	if got := len(root.Children()); got != 1 {
		t.Errorf("Children() len = %d, want 1", got)
	}
	if got := len(fn.NamedChildren()); got != 3 {
		t.Errorf("NamedChildren() of function = %d, want 3 (type, declarator, body)", got)
	}
	if (Node{}).IsNull() != true {
		t.Error("zero Node is not null")
	}
	if got := root.Parent(); !got.IsNull() {
		t.Errorf("root parent = %v, want null", got)
	}
}

func TestParseErrors(t *testing.T) {
	p, _ := testParser(t)
	tree, err := p.Parse([]byte("int main( {"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if !tree.RootNode().HasError() {
		t.Errorf("expected syntax error in %s", tree.RootNode())
	}
}

func TestNoLanguage(t *testing.T) {
	testGrammar(t)
	p, err := NewParser()
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if _, err := p.Parse([]byte("x"), nil); !errors.Is(err, ErrNoLanguage) {
		t.Errorf("Parse without language: err = %v, want ErrNoLanguage", err)
	}
}

func TestIncrementalParse(t *testing.T) {
	p, _ := testParser(t)
	src := []byte(helloC)
	old, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer old.Close()

	// Turn "return 0;" into "return 0 + 1;", which changes the structure
	// of the tree and not just a token, so a range is reported as changed.
	edited := []byte(strings.Replace(helloC, "return 0;", "return 0 + 1;", 1))
	edit := InputEdit{
		StartByte:   22,
		OldEndByte:  23,
		NewEndByte:  27,
		StartPoint:  Point{1, 9},
		OldEndPoint: Point{1, 10},
		NewEndPoint: Point{1, 14},
	}
	old.Edit(edit)
	tree, err := p.Parse(edited, old)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if tree.RootNode().HasError() {
		t.Fatalf("edited parse has errors: %s", tree.RootNode())
	}
	ret := tree.RootNode().Child(0).ChildByFieldName("body").NamedChild(0)
	if got := ret.NamedChild(0).Type(); got != "binary_expression" {
		t.Errorf("return value type after edit = %q, want binary_expression", got)
	}
	changed := old.ChangedRanges(tree)
	if len(changed) == 0 {
		t.Fatal("no changed ranges")
	}
	if changed[0].StartByte > 22 || changed[0].EndByte < 27 {
		t.Errorf("changed range %+v does not cover the edit", changed[0])
	}
	if got := tree.IncludedRanges(); len(got) != 1 {
		t.Errorf("included ranges = %v, want one range", got)
	}
}

func TestIncludedRanges(t *testing.T) {
	p, _ := testParser(t)
	if got := p.IncludedRanges(); len(got) != 1 {
		t.Fatalf("default included ranges = %v", got)
	}
	src := []byte("<% int x; %> text <% int y; %>")
	ranges := []Range{
		{StartByte: 2, EndByte: 10, StartPoint: Point{0, 2}, EndPoint: Point{0, 10}},
		{StartByte: 20, EndByte: 28, StartPoint: Point{0, 20}, EndPoint: Point{0, 28}},
	}
	if err := p.SetIncludedRanges(ranges); err != nil {
		t.Fatal(err)
	}
	if got := p.IncludedRanges(); len(got) != 2 || got[1].StartByte != 20 {
		t.Errorf("included ranges = %+v", got)
	}
	tree, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	root := tree.RootNode()
	if root.HasError() || root.NamedChildCount() != 2 {
		t.Errorf("ranged parse: %s", root)
	}
	var ire *IncludedRangesError
	err = p.SetIncludedRanges([]Range{{StartByte: 5, EndByte: 1}})
	if !errors.As(err, &ire) || ire.Index != 0 {
		t.Errorf("inverted range: err = %v, want IncludedRangesError{Index: 0}", err)
	}
	err = p.SetIncludedRanges([]Range{{StartByte: 0, EndByte: 10}, {StartByte: 5, EndByte: 20}})
	if !errors.As(err, &ire) || ire.Index != 1 {
		t.Errorf("overlapping range: err = %v, want IncludedRangesError{Index: 1}", err)
	}
	if err := p.SetIncludedRanges(nil); err != nil {
		t.Fatal(err)
	}
}

func TestTreeCursor(t *testing.T) {
	p, lang := testParser(t)
	tree, err := p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	c := tree.RootNode().Walk()
	defer c.Close()
	if !c.GotoFirstChild() || c.CurrentNode().Type() != "function_definition" {
		t.Fatalf("first child: %v", c.CurrentNode())
	}
	if got := c.CurrentDepth(); got != 1 {
		t.Errorf("depth = %d, want 1", got)
	}
	var fields []string
	if c.GotoFirstChild() {
		for {
			if f := c.CurrentFieldName(); f != "" {
				fields = append(fields, f)
				if id := c.CurrentFieldID(); lang.FieldName(id) != f {
					t.Errorf("field id %d names %q, want %q", id, lang.FieldName(id), f)
				}
			}
			if !c.GotoNextSibling() {
				break
			}
		}
	}
	if got := strings.Join(fields, ","); got != "type,declarator,body" {
		t.Errorf("fields = %q, want type,declarator,body", got)
	}
	if !c.GotoParent() || !c.GotoParent() || c.GotoParent() {
		t.Error("GotoParent did not stop at the root")
	}
	if _, ok := c.GotoFirstChildForByte(0); !ok {
		t.Error("GotoFirstChildForByte(0) found nothing")
	}
	cp := c.Copy()
	defer cp.Close()
	if !cp.CurrentNode().Equal(c.CurrentNode()) {
		t.Error("copy is at a different node")
	}
	c.Reset(tree.RootNode())
	if c.CurrentDepth() != 0 {
		t.Error("Reset did not return to depth 0")
	}
}

func TestQuery(t *testing.T) {
	p, lang := testParser(t)
	src := []byte("int add(int a, int b) { return a + b; }\nint main() { return add(1, 2); }\n")
	tree, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()

	q, err := NewQuery(lang, `(function_definition declarator: (function_declarator declarator: (identifier) @name)) @func
(call_expression function: (identifier) @callee (#eq? @callee "add"))`)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	if got := q.PatternCount(); got != 2 {
		t.Errorf("pattern count = %d, want 2", got)
	}
	if got := q.CaptureCount(); got != 3 {
		t.Errorf("capture count = %d, want 3", got)
	}
	if got := q.CaptureName(0); got != "name" {
		t.Errorf("capture 0 = %q, want name", got)
	}
	steps := q.PredicatesForPattern(1)
	if len(steps) != 4 || steps[0].Type != PredicateStepString || q.StringValue(steps[0].ValueID) != "eq?" || steps[3].Type != PredicateStepDone {
		t.Errorf("predicates = %+v", steps)
	}
	if !q.IsPatternRooted(0) {
		t.Error("pattern 0 not rooted")
	}

	c, err := NewQueryCursor()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.Exec(q, tree.RootNode())
	var names []string
	for {
		m, ok := c.NextMatch()
		if !ok {
			break
		}
		for _, cap := range m.Captures {
			if q.CaptureName(cap.Index) == "name" || q.CaptureName(cap.Index) == "callee" {
				names = append(names, string(cap.Node.Text(src)))
			}
		}
	}
	if got := strings.Join(names, ","); got != "add,main,add" {
		t.Errorf("captures = %q, want add,main,add", got)
	}

	c.Exec(q, tree.RootNode())
	if !c.SetByteRange(40, uint32(len(src))) {
		t.Fatal("SetByteRange rejected")
	}
	var count int
	for {
		_, _, ok := c.NextCapture()
		if !ok {
			break
		}
		count++
	}
	if count != 3 {
		t.Errorf("captures in second function = %d, want 3 (func, name, callee)", count)
	}

	_, err = NewQuery(lang, "(identifier) @a\n  (nonexistent_node) @x")
	var qe *QueryError
	if !errors.As(err, &qe) || qe.Kind != QueryErrorNodeType || qe.Offset != 19 || qe.Row != 1 || qe.Column != 3 {
		t.Errorf("bad query error = %+v, want node type error at byte 19, 1:3", err)
	}
}

func TestQueryExecContext(t *testing.T) {
	p, lang := testParser(t)
	var b strings.Builder
	for i := 0; i < 2000; i++ {
		fmt.Fprintf(&b, "int f%d() { return %d; }\n", i, i)
	}
	tree, err := p.Parse([]byte(b.String()), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	q, err := NewQuery(lang, "(identifier) @id")
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()
	c, err := NewQueryCursor()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	c.ExecContext(context.Background(), q, tree.RootNode())
	all := 0
	for {
		if _, ok := c.NextMatch(); !ok {
			break
		}
		all++
	}
	if all != 2000 {
		t.Fatalf("matches with live context = %d, want 2000", all)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.ExecContext(ctx, q, tree.RootNode())
	cancelled := 0
	for {
		if _, ok := c.NextMatch(); !ok {
			break
		}
		cancelled++
	}
	if cancelled >= all {
		t.Errorf("matches with cancelled context = %d, want fewer than %d", cancelled, all)
	}

	// Exec after a cancelled ExecContext runs to completion again.
	c.Exec(q, tree.RootNode())
	again := 0
	for {
		if _, ok := c.NextMatch(); !ok {
			break
		}
		again++
	}
	if again != all {
		t.Errorf("matches after Exec = %d, want %d", again, all)
	}
}

func TestPrintDotGraphs(t *testing.T) {
	p, _ := testParser(t)
	f, err := os.CreateTemp(t.TempDir(), "dot")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := p.PrintDotGraphs(f); err != nil {
		t.Fatal(err)
	}
	tree, err := p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	tree.Close()
	p.StopPrintingDotGraphs()
	p.StopPrintingDotGraphs() // idempotent
	out, err := os.ReadFile(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out), "digraph") {
		t.Errorf("dot output (%d bytes) has no digraph", len(out))
	}
}

func TestParseInput(t *testing.T) {
	p, _ := testParser(t)
	src := []byte(helloC)
	var calls int
	read := func(byteIndex uint32, pos Point) []byte {
		calls++
		if int(byteIndex) >= len(src) {
			return nil
		}
		end := min(int(byteIndex)+4, len(src))
		return src[byteIndex:end]
	}
	tree, err := p.ParseInput(read, UTF8, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if calls < 2 {
		t.Errorf("read called %d times; chunks were not exercised", calls)
	}
	want, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer want.Close()
	if got, w := tree.RootNode().String(), want.RootNode().String(); got != w {
		t.Errorf("chunked parse:\n%s\nwant:\n%s", got, w)
	}
}

func TestParseContext(t *testing.T) {
	p, _ := testParser(t)
	tree, err := p.ParseContext(context.Background(), []byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	tree.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.ParseContext(ctx, []byte(helloC), nil); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled parse: err = %v, want context.Canceled", err)
	}
	p.Reset()
}

func TestLogger(t *testing.T) {
	p, _ := testParser(t)
	var msgs []string
	p.SetLogger(func(kind LogType, msg string) {
		msgs = append(msgs, kind.String()+": "+msg)
	})
	tree, err := p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	tree.Close()
	if len(msgs) == 0 {
		t.Fatal("logger received nothing")
	}
	if !strings.HasPrefix(msgs[0], "parse: ") && !strings.HasPrefix(msgs[0], "lex: ") {
		t.Errorf("first message = %q", msgs[0])
	}
	p.SetLogger(nil)
	n := len(msgs)
	tree, err = p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	tree.Close()
	if len(msgs) != n {
		t.Error("logger still receiving after removal")
	}
}

func TestLanguage(t *testing.T) {
	lang := testGrammar(t)
	if v := lang.ABIVersion(); v < MinCompatibleLanguageVersion || v > LanguageVersion {
		t.Errorf("ABI version %d out of range", v)
	}
	if name := lang.Name(); name != "" && name != "c" {
		t.Errorf("name = %q", name)
	}
	if lang.SymbolCount() == 0 || lang.StateCount() == 0 || lang.FieldCount() == 0 {
		t.Error("empty language counts")
	}
	sym := lang.SymbolForName("identifier", true)
	if sym == 0 {
		t.Fatal("no identifier symbol")
	}
	if got := lang.SymbolName(sym); got != "identifier" {
		t.Errorf("symbol name = %q", got)
	}
	if got := lang.SymbolType(sym); got != SymbolRegular {
		t.Errorf("identifier symbol type = %v", got)
	}
	if lang.SymbolForName("no_such_node", true) != 0 {
		t.Error("unknown symbol resolved")
	}
	id := lang.FieldID("declarator")
	if id == 0 || lang.FieldName(id) != "declarator" {
		t.Errorf("field declarator: id %d name %q", id, lang.FieldName(id))
	}
	if sup := lang.Supertypes(); len(sup) > 0 {
		if sub := lang.Subtypes(sup[0]); len(sub) == 0 {
			t.Errorf("supertype %s has no subtypes", lang.SymbolName(sup[0]))
		}
	}
	if NewLanguage(0) != nil {
		t.Error("NewLanguage(0) != nil")
	}
	if _, err := LoadLanguage("/nonexistent/grammar.so", "c"); err == nil {
		t.Error("LoadLanguage of a missing file succeeded")
	}
}

func TestLookahead(t *testing.T) {
	p, lang := testParser(t)
	tree, err := p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	// The state after the return type of a function accepts a declarator.
	typ := tree.RootNode().Child(0).ChildByFieldName("type")
	it, err := NewLookaheadIterator(lang, typ.NextParseState())
	if err != nil {
		t.Fatal(err)
	}
	defer it.Close()
	var names []string
	for it.Next() {
		names = append(names, it.CurrentSymbolName())
		if lang.SymbolName(it.CurrentSymbol()) != it.CurrentSymbolName() {
			t.Errorf("symbol %d name mismatch", it.CurrentSymbol())
		}
	}
	if len(names) == 0 {
		t.Fatal("no lookahead symbols")
	}
	found := false
	for _, n := range names {
		if n == "identifier" {
			found = true
		}
	}
	if !found {
		t.Errorf("identifier not among valid symbols %v", names)
	}
	if it.Language().Pointer() != lang.Pointer() {
		t.Error("iterator language differs")
	}
	if !it.ResetState(typ.NextParseState()) {
		t.Error("ResetState failed")
	}
}

func TestTreeCopy(t *testing.T) {
	p, _ := testParser(t)
	tree, err := p.Parse([]byte(helloC), nil)
	if err != nil {
		t.Fatal(err)
	}
	cp := tree.Copy()
	tree.Close()
	defer cp.Close()
	if cp.RootNode().Type() != "translation_unit" {
		t.Error("copy unusable after original closed")
	}
	if cp.Language() == nil {
		t.Error("copy has no language")
	}
}

func TestTypesWithoutLibrary(t *testing.T) {
	if got := (Point{Row: 3, Column: 7}).String(); got != "3:7" {
		t.Errorf("Point.String() = %q", got)
	}
	if got := QuantifierOneOrMore.String(); got != "+" {
		t.Errorf("Quantifier.String() = %q", got)
	}
	if got := (&QueryError{Kind: QueryErrorField, Offset: 9, Row: 1, Column: 2}).Error(); !strings.Contains(got, "field error at 1:2 (byte 9)") {
		t.Errorf("QueryError = %q", got)
	}
	if got := invalidRange([]Range{{StartByte: 0, EndByte: 4}, {StartByte: 4, EndByte: 4}, {StartByte: 3, EndByte: 9}}); got != 2 {
		t.Errorf("invalidRange = %d, want 2", got)
	}
	if got := (&IncludedRangesError{Index: 1}).Error(); !strings.Contains(got, "range 1") {
		t.Errorf("IncludedRangesError = %q", got)
	}
	if got := SymbolSupertype.String(); got != "supertype" {
		t.Errorf("SymbolType.String() = %q", got)
	}
}
