package treesitter_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/treesitter"
)

// Example parses a C source file and prints the name of each function it
// defines. It needs libtree-sitter and a C grammar, so it has no fixed output.
func Example() {
	lang, err := treesitter.FindLanguage("c")
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
	for _, cap := range c.Captures() {
		fmt.Println(string(cap.Node.Text(src)))
	}
}

// ExampleFindLanguage parses a Swift file that uses MLX and lists the
// types it declares, the functions and methods with their parameter types,
// and every call into the MLX namespace. The Swift grammar is found through
// the usual grammar directories or TREE_SITTER_GRAMMAR_PATH.
func ExampleFindLanguage() {
	src := []byte(`import MLX
import MLXNN
import MLXRandom

class MLP: Module {
    @ModuleInfo var up: Linear
    @ModuleInfo var down: Linear

    init(dimensions: Int, hidden: Int) {
        up = Linear(dimensions, hidden)
        down = Linear(hidden, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(gelu(up(x)))
    }
}

func loss(model: MLP, x: MLXArray, y: MLXArray) -> MLXArray {
    MLX.mean(MLX.square(model(x) - y))
}

let model = MLP(dimensions: 8, hidden: 32)
let x = MLXRandom.normal([16, 8])
let grads = MLX.grad(loss)(model, x, x)
eval(grads)
`)

	lang, err := treesitter.FindLanguage("swift")
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
	tree, err := p.Parse(src, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()
	if tree.RootNode().HasError() {
		log.Fatal("parse error")
	}

	// Declarations: a class or struct, and each function with its
	// parameters' types.
	q, err := treesitter.NewQuery(lang, `
(class_declaration name: (type_identifier) @class)
(function_declaration name: (simple_identifier) @func)
(function_declaration
  (parameter (simple_identifier) @param . (user_type) @param.type))
(navigation_expression
  target: (simple_identifier) @module
  suffix: (navigation_suffix (simple_identifier) @member)
  (#eq? @module "MLX"))`)
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
	for m := range c.Matches() {
		for _, cap := range m.Captures {
			name := q.CaptureName(cap.Index)
			if name == "module" {
				continue // matched by the #eq? guard below
			}
			// Predicates are not evaluated by tree-sitter itself: apply
			// the #eq? on the module capture by hand.
			if name == "member" && !matchesModule(m, q, src, "MLX") {
				continue
			}
			fmt.Printf("%s %s at %v\n", name, cap.Node.Text(src), cap.Node.StartPoint())
		}
	}

	// Walk the tree directly for the imports.
	for n := range tree.RootNode().Children() {
		if n.Type() == "import_declaration" {
			fmt.Println("import", string(n.NamedChild(0).Text(src)))
		}
	}
}

// matchesModule reports whether the match's @module capture spells module.
func matchesModule(m treesitter.QueryMatch, q *treesitter.Query, src []byte, module string) bool {
	for _, cap := range m.Captures {
		if q.CaptureName(cap.Index) == "module" {
			return string(cap.Node.Text(src)) == module
		}
	}
	return false
}

// ExampleTree_Edit re-parses a document incrementally after a change.
func ExampleTree_Edit() {
	lang, err := treesitter.FindLanguage("c")
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
