package treesitter

import (
	"bytes"
	"context"
	"testing"
)

// The parse benchmarks run over the same source through each input path,
// so their difference is the cost of the path rather than the grammar.
// Parse hands tree-sitter one slice; ParseContext adds a progress
// callback; ParseReaderAt adds a read callback for every window.

func BenchmarkParse(b *testing.B) {
	p, _ := testParser(b)
	src := []byte(longSource())
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for range b.N {
		tree, err := p.Parse(src, nil)
		if err != nil {
			b.Fatal(err)
		}
		tree.Close()
	}
}

func BenchmarkParseContext(b *testing.B) {
	p, _ := testParser(b)
	src := []byte(longSource())
	ctx := context.Background()
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for range b.N {
		tree, err := p.ParseContext(ctx, src, nil)
		if err != nil {
			b.Fatal(err)
		}
		tree.Close()
	}
}

func BenchmarkParseReaderAt(b *testing.B) {
	p, _ := testParser(b)
	src := []byte(longSource())
	r := bytes.NewReader(src)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for range b.N {
		tree, err := p.ParseReaderAt(r, int64(len(src)), nil)
		if err != nil {
			b.Fatal(err)
		}
		tree.Close()
	}
}

func BenchmarkMatches(b *testing.B) {
	lang := testGrammar(b)
	tree, err := Parse(lang, []byte(longSource()))
	if err != nil {
		b.Fatal(err)
	}
	defer tree.Close()
	q, err := NewQuery(lang, `(parameter_declaration declarator: (identifier) @name)`)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	root := tree.RootNode()
	b.ResetTimer()
	for range b.N {
		for range q.Matches(root) {
		}
	}
}
