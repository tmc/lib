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
	src := []byte("int add(int a, int b) { return a + b; }\nint main() { return add(1, 2); }\n")
	tree, err := treesitter.Parse(lang, src)
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
	for m := range q.Matches(tree.RootNode()) {
		fmt.Printf("%s\n", m.Text("name"))
	}
}

// ExampleParseFile parses a file chosen by its name and reports any
// syntax errors it contains.
func ExampleParseFile() {
	tree, err := treesitter.ParseFile("main.go")
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()
	for n := range tree.Errors() {
		fmt.Printf("%v: %s\n", n.StartPoint(), n.Type())
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
	tree, err := treesitter.Parse(lang, src)
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()
	if tree.HasError() {
		log.Fatal("parse error")
	}

	// Declarations: a class or struct, each function with its parameters'
	// types, and calls into the MLX namespace. The #eq? predicate on the
	// last pattern is applied by Matches.
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

	for m := range q.Matches(tree.RootNode()) {
		for _, c := range m.Captures {
			if c.Name == "module" {
				continue // reported through @member
			}
			fmt.Printf("%s %s at %v\n", c.Name, c.Node.Text(), c.Node.StartPoint())
		}
	}

	// Walk the tree directly for the imports.
	for n := range tree.RootNode().Children() {
		if n.Type() == "import_declaration" {
			fmt.Printf("import %s\n", n.NamedChild(0).Text())
		}
	}
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
