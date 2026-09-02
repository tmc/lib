// Package treesitter provides Go bindings for the tree-sitter parsing library.
//
// The package loads libtree-sitter at run time with purego, so it needs no
// cgo. Grammars are shared libraries too, each exporting a function named
// tree_sitter_<name>. [FindLanguage] locates one by name in the directories
// [GrammarDirs] lists and [LoadLanguage] opens one by path; [DetectLanguage]
// guesses the name from a file name.
//
// Typical use:
//
//	lang, err := treesitter.FindLanguage("c")
//	if err != nil {
//		log.Fatal(err)
//	}
//	p, err := treesitter.NewParser()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer p.Close()
//	if err := p.SetLanguage(lang); err != nil {
//		log.Fatal(err)
//	}
//	tree, err := p.Parse(src, nil)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer tree.Close()
//	for n := range tree.RootNode().Descendants() {
//		if n.IsError() {
//			fmt.Println("syntax error at", n.StartPoint())
//		}
//	}
//
// Children, descendants, query matches and captures, and lookahead symbols
// are iterators for use with range.
//
// Parsers, trees, queries, and cursors own C memory and must be released
// with Close. Their zero values are not usable; obtain them from the New
// functions and from Parse. Close is idempotent. A [Node] is a value that
// refers into its tree and is valid only while that tree is open; its zero
// value is the null node, which IsNull reports and on which every method
// returns its zero result.
//
// The library is found through TREE_SITTER_LIB_PATH, then the platform's
// default search path, then common Homebrew locations. [Load] directs the
// search explicitly.
//
// The package builds on the platforms purego supports for calling C with
// struct arguments: Linux and macOS on amd64 and arm64. Other systems have
// no library search names and Load fails on them. See manual-gates.md for
// the configurations the live tests have been run on.
package treesitter
