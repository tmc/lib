package treesitter_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/treesitter"
)

// Example parses a C source file and prints the name of each function it
// defines. It needs libtree-sitter and a C grammar, so it has no fixed output.
func Example() {
	lang, err := treesitter.LoadLanguage("/opt/homebrew/lib/nvim/parser/c.so", "c")
	if err != nil {
		log.Fatal(err)
	}
	p, err := treesitter.NewParser()
	if err != nil {
		log.Fatal(err)
	}
	defer p.Close()
	if err := p.SetLanguage(lang); err != nil {
		log.Fatal(err)
	}

	src := []byte("int add(int a, int b) { return a + b; }\nint main() { return add(1, 2); }\n")
	tree, err := p.Parse(src, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()

	q, err := treesitter.NewQuery(lang, `(function_definition
	  declarator: (function_declarator declarator: (identifier) @name))`)
	if err != nil {
		log.Fatal(err)
	}
	defer q.Close()
	c, err := treesitter.NewQueryCursor()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()
	c.Exec(q, tree.RootNode())
	for {
		m, ok := c.NextMatch()
		if !ok {
			break
		}
		for _, cap := range m.Captures {
			fmt.Println(string(cap.Node.Text(src)))
		}
	}
}

// ExampleTree_Edit re-parses a document incrementally after a change.
func ExampleTree_Edit() {
	lang, err := treesitter.LoadLanguage("/opt/homebrew/lib/nvim/parser/c.so", "c")
	if err != nil {
		log.Fatal(err)
	}
	p, err := treesitter.NewParser()
	if err != nil {
		log.Fatal(err)
	}
	defer p.Close()
	if err := p.SetLanguage(lang); err != nil {
		log.Fatal(err)
	}

	old, err := p.Parse([]byte("int x;"), nil)
	if err != nil {
		log.Fatal(err)
	}
	defer old.Close()

	// Replace "x" with "count".
	old.Edit(treesitter.InputEdit{
		StartByte: 4, OldEndByte: 5, NewEndByte: 9,
		StartPoint:  treesitter.Point{Row: 0, Column: 4},
		OldEndPoint: treesitter.Point{Row: 0, Column: 5},
		NewEndPoint: treesitter.Point{Row: 0, Column: 9},
	})
	tree, err := p.Parse([]byte("int count;"), old)
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()
	for _, r := range old.ChangedRanges(tree) {
		fmt.Printf("changed bytes %d-%d\n", r.StartByte, r.EndByte)
	}
}
