// Package treesitter provides Go bindings for the tree-sitter parsing library.
//
// It does the same job as the established cgo bindings, with two
// differences: it loads libtree-sitter and its grammars at run time with
// purego, so building it needs no C toolchain, and its API is shaped for Go
// rather than translated from C.
//
// Grammars are shared libraries, each exporting a function named
// tree_sitter_<name>. [FindLanguage] locates one by name in the directories
// [GrammarDirs] lists and [LoadLanguage] opens one by path; [DetectLanguage]
// guesses the name from a file name, and [ParseFile] puts the two together.
// Package [github.com/tmc/lib/treesitter/grammars] builds a grammar from a
// pinned source archive when none is installed.
//
// Typical use:
//
//	tree, err := treesitter.ParseFile("hello.c")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer tree.Close()
//	for n := range tree.Errors() {
//		fmt.Println("syntax error at", n.StartPoint())
//	}
//
// A tree remembers the source it was parsed from, so [Node.Text] takes no
// argument. Children, descendants, errors, query matches and captures, and
// lookahead symbols are iterators for use with range. Counts and byte
// offsets are int.
//
// Queries return named captures and have their text predicates applied:
//
//	q, err := treesitter.NewQuery(lang, `
//	    (call_expression function: (identifier) @callee
//	      (#match? @callee "^str"))`)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer q.Close()
//	for m := range q.Matches(tree.RootNode()) {
//		fmt.Printf("%s at %v\n", m.Text("callee"), m.Node("callee").StartPoint())
//	}
//
// [Query.Matches] evaluates #eq?, #match?, and #any-of? with their not-,
// any-, and any-not- forms, and rejects a malformed one when the query is
// compiled. Predicates it does not evaluate, such as #set! and #is?, are
// reported by [Query.Predicates] and [Query.Properties] for the caller to
// interpret. [QueryCursor] runs a query without any of that.
//
// Parsers, trees, queries, and cursors own C memory and must be released
// with Close. Their zero values are not usable; obtain them from the New
// functions and from Parse. Close is idempotent. A [Node] is a value that
// refers into its tree and is valid only while that tree is open; its zero
// value is the null node, which IsNull reports and on which every method
// returns its zero result.
//
// tree-sitter reads its input by byte offset and revisits earlier offsets,
// so streaming input is [io.ReaderAt], not [io.Reader]; see
// [Parser.ParseReaderAt].
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
