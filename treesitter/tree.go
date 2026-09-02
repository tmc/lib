package treesitter

import (
	"iter"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// Tree is a syntax tree. Trees own C memory and must be closed.
//
// A tree keeps a reference to the source text it was parsed from, so
// [Node.Text] needs no argument. The source must not be modified while
// the tree is open. A tree parsed by [Parser.ParseInput] has no source.
type Tree struct {
	t   ts.Tree
	src []byte
}

// Close frees the tree. Nodes obtained from it become invalid.
func (t *Tree) Close() {
	if t.t == 0 {
		return
	}
	ts.TreeDelete(t.t)
	t.t = 0
	t.src = nil
}

// Copy returns a shallow copy that can be closed and edited independently,
// for use from another goroutine. The copy shares the original's source.
func (t *Tree) Copy() *Tree {
	return &Tree{t: ts.Tree(ts.TreeCopy(t.t)), src: t.src}
}

// Source returns the text the tree was parsed from, or nil if it is not
// known. The result aliases the caller's slice and must not be modified.
func (t *Tree) Source() []byte {
	if t == nil {
		return nil
	}
	return t.src
}

// RootNode returns the tree's root.
func (t *Tree) RootNode() Node {
	return Node{t: t, n: ts.TreeRootNode(t.t)}
}

// RootNodeWithOffset returns the root with its position shifted by the
// given byte offset and point, for a tree that describes part of a larger
// document.
func (t *Tree) RootNodeWithOffset(offsetBytes int, offsetExtent Point) Node {
	return Node{t: t, n: ts.TreeRootNodeWithOffset(t.t, u32(offsetBytes), toPoint(offsetExtent))}
}

// Language returns the language the tree was parsed with.
func (t *Tree) Language() *Language {
	return NewLanguage(ts.TreeLanguage(t.t))
}

// IncludedRanges returns the ranges of the document the tree covers.
func (t *Tree) IncludedRanges() []Range {
	var n uint32
	p := ts.TreeIncludedRanges(t.t, &n)
	rs := rangeSlice(p, n)
	if p != 0 {
		cFree(p)
	}
	return rs
}

// Edit records a change to the source text so that the tree's node
// positions stay correct and a following incremental parse can reuse
// unchanged nodes. It does not update [Tree.Source]; pass the new text to
// the next parse.
func (t *Tree) Edit(edit InputEdit) {
	e := toInputEdit(edit)
	ts.TreeEdit(t.t, &e)
}

// ChangedRanges returns the ranges whose syntax differs between t and
// newTree, where newTree came from an incremental parse using t.
func (t *Tree) ChangedRanges(newTree *Tree) []Range {
	var n uint32
	p := ts.TreeGetChangedRanges(t.t, newTree.t, &n)
	rs := rangeSlice(p, n)
	if p != 0 {
		cFree(p)
	}
	return rs
}

// HasError reports whether the parse found any syntax error. tree-sitter
// always produces a tree, so this is how a failed parse is detected.
func (t *Tree) HasError() bool {
	return t.RootNode().HasError()
}

// Errors returns an iterator over the tree's error and missing nodes, in
// document order. It descends only into subtrees that contain an error,
// so a tree that parsed cleanly costs one check.
func (t *Tree) Errors() iter.Seq[Node] {
	return func(yield func(Node) bool) {
		if root := t.RootNode(); root.HasError() {
			yieldErrors(root, yield)
		}
	}
}

// yieldErrors reports n and its descendants that are errors, stopping
// early if yield returns false, which it reports.
func yieldErrors(n Node, yield func(Node) bool) bool {
	if n.IsError() || n.IsMissing() {
		return yield(n)
	}
	for c := range n.Children() {
		if !c.HasError() && !c.IsMissing() {
			continue
		}
		if !yieldErrors(c, yield) {
			return false
		}
	}
	return true
}

// PrintDotGraph writes a Graphviz graph of the tree to fd.
func (t *Tree) PrintDotGraph(fd int) {
	ts.TreePrintDotGraph(t.t, int32(fd))
}
