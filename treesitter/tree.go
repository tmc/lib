package treesitter

import "github.com/tmc/lib/treesitter/internal/ts"

// Tree is a syntax tree. Trees own C memory and must be closed. A tree
// does not hold the source text it was parsed from.
type Tree struct {
	t ts.Tree
}

// Close frees the tree. Nodes obtained from it become invalid.
func (t *Tree) Close() {
	if t.t == 0 {
		return
	}
	ts.TreeDelete(t.t)
	t.t = 0
}

// Copy returns a shallow copy that can be closed and edited independently,
// for use from another goroutine.
func (t *Tree) Copy() *Tree {
	return &Tree{t: ts.Tree(ts.TreeCopy(t.t))}
}

// RootNode returns the tree's root.
func (t *Tree) RootNode() Node {
	return Node{n: ts.TreeRootNode(t.t)}
}

// RootNodeWithOffset returns the root with its position shifted by the
// given byte offset and point, for a tree that describes part of a larger
// document.
func (t *Tree) RootNodeWithOffset(offsetBytes uint32, offsetExtent Point) Node {
	return Node{n: ts.TreeRootNodeWithOffset(t.t, offsetBytes, toPoint(offsetExtent))}
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
// unchanged nodes.
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

// PrintDotGraph writes a Graphviz graph of the tree to fd.
func (t *Tree) PrintDotGraph(fd int) {
	ts.TreePrintDotGraph(t.t, int32(fd))
}
