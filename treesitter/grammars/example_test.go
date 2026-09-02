package grammars_test

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/lib/treesitter"
	"github.com/tmc/lib/treesitter/grammars"
)

// Example installs a grammar and then parses with it. Installing needs the
// network and a C compiler, so this has no fixed output.
func Example() {
	if _, err := grammars.Install(context.Background(), "go"); err != nil {
		log.Fatal(err)
	}
	tree, err := treesitter.ParseFile("main.go")
	if err != nil {
		log.Fatal(err)
	}
	defer tree.Close()
	fmt.Println(tree.RootNode().Type())
}

// ExampleAll lists the grammars that can be installed.
func ExampleAll() {
	for _, g := range grammars.All() {
		fmt.Println(g)
	}
	// Output:
	// bash v0.25.1
	// c v0.24.2
	// cpp v0.23.4
	// css v0.25.0
	// go v0.25.0
	// html v0.23.2
	// java v0.23.5
	// javascript v0.25.0
	// json v0.24.8
	// python v0.25.0
	// ruby v0.23.1
	// rust v0.24.2
	// tsx v0.23.2
	// typescript v0.23.2
}

// ExampleGrammar_Build installs a grammar that is not in the manifest.
// Pinning the archive's SHA-256 is what makes the build repeatable; take
// it from the release you intend to use.
func ExampleGrammar_Build() {
	zig := grammars.Grammar{
		Name:    "zig",
		Version: "v1.1.2",
		URL:     "https://github.com/tree-sitter-grammars/tree-sitter-zig/archive/refs/tags/v1.1.2.tar.gz",
		SHA256:  "612d67059faa90ec7691e5d786d70d8f7c2c8b15b83de901b9b801122ad4cf25",
	}
	dir, err := grammars.Dir()
	if err != nil {
		log.Fatal(err)
	}
	path, err := zig.Build(context.Background(), dir)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}
