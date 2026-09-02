// Package grammars builds tree-sitter grammars from pinned source
// archives and installs them where [treesitter.FindLanguage] looks.
//
// A grammar is a shared library, and the upstream projects publish C
// source rather than binaries, so installing one downloads an archive,
// checks it against a recorded SHA-256, compiles it, and writes the
// result to the tree-sitter cache directory:
//
//	if _, err := grammars.Install(ctx, "go"); err != nil {
//		log.Fatal(err)
//	}
//	tree, err := treesitter.ParseFile("main.go")
//
// Installing needs network access and a C compiler, named by CC or found
// as cc, and it verifies the result by loading it, so libtree-sitter must
// be available too. Parsing needs none of that: this package is separate
// from treesitter precisely so that a program which only parses does not
// link an HTTP client, and so that no parse can reach the network.
//
// [All] lists what is pinned. A grammar that is not listed can still be
// installed by filling in a [Grammar] and calling [Grammar.Build].
package grammars
