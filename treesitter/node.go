package treesitter

import (
	"iter"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// Node is a node in a [Tree]. It is a small value that refers into the
// tree and is valid only while the tree is open. The zero Node is null;
// every method accepts a null node and returns its zero result, so a
// missing child or field can be used without checking first.
type Node struct {
	n ts.Node
}

// IsNull reports whether the node is null, which is how absent parents,
// children, and siblings are reported.
func (n Node) IsNull() bool {
	return n.n.Id == 0
}

// Type returns the node's type name.
func (n Node) Type() string {
	if n.IsNull() {
		return ""
	}
	return ts.NodeType(n.n)
}

// Symbol returns the node's type as a symbol.
func (n Node) Symbol() Symbol {
	if n.IsNull() {
		return 0
	}
	return Symbol(ts.NodeSymbol(n.n))
}

// GrammarType returns the type name as written in the grammar, before any
// alias is applied.
func (n Node) GrammarType() string {
	if n.IsNull() {
		return ""
	}
	return ts.NodeGrammarType(n.n)
}

// GrammarSymbol returns the symbol as written in the grammar, before any
// alias is applied.
func (n Node) GrammarSymbol() Symbol {
	if n.IsNull() {
		return 0
	}
	return Symbol(ts.NodeGrammarSymbol(n.n))
}

// Language returns the language of the node's tree.
func (n Node) Language() *Language {
	if n.IsNull() {
		return nil
	}
	return NewLanguage(ts.NodeLanguage(n.n))
}

// StartByte returns the byte offset where the node begins.
func (n Node) StartByte() uint32 {
	return ts.NodeStartByte(n.n)
}

// EndByte returns the byte offset just past the node's end.
func (n Node) EndByte() uint32 {
	if n.IsNull() {
		return 0
	}
	return ts.NodeEndByte(n.n)
}

// StartPoint returns the position where the node begins.
func (n Node) StartPoint() Point {
	return fromPoint(ts.NodeStartPoint(n.n))
}

// EndPoint returns the position just past the node's end.
func (n Node) EndPoint() Point {
	if n.IsNull() {
		return Point{}
	}
	return fromPoint(ts.NodeEndPoint(n.n))
}

// Range returns the node's extent in bytes and points.
func (n Node) Range() Range {
	return Range{
		StartPoint: n.StartPoint(),
		EndPoint:   n.EndPoint(),
		StartByte:  n.StartByte(),
		EndByte:    n.EndByte(),
	}
}

// Text returns the node's source text, given the document it was parsed
// from. It returns nil for a null node or one that lies outside src.
func (n Node) Text(src []byte) []byte {
	if n.IsNull() {
		return nil
	}
	start, end := int(n.StartByte()), int(n.EndByte())
	if start > end || end > len(src) {
		return nil
	}
	return src[start:end]
}

// String returns the node's subtree as an S-expression.
func (n Node) String() string {
	if n.IsNull() {
		return ""
	}
	p := ts.NodeString(n.n)
	if p == 0 {
		return ""
	}
	s := goString(p)
	cFree(p)
	return s
}

// IsNamed reports whether the node is a named node rather than an
// anonymous one such as punctuation.
func (n Node) IsNamed() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeIsNamed(n.n)
}

// IsMissing reports whether the parser inserted the node to recover from
// an error.
func (n Node) IsMissing() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeIsMissing(n.n)
}

// IsExtra reports whether the node is an extra, such as a comment, that
// the grammar allows anywhere.
func (n Node) IsExtra() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeIsExtra(n.n)
}

// HasChanges reports whether the node was edited since the tree was
// parsed.
func (n Node) HasChanges() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeHasChanges(n.n)
}

// HasError reports whether the node or any descendant is a syntax error.
func (n Node) HasError() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeHasError(n.n)
}

// IsError reports whether the node itself is a syntax error.
func (n Node) IsError() bool {
	if n.IsNull() {
		return false
	}
	return ts.NodeIsError(n.n)
}

// ParseState returns the parse state the node was created in.
func (n Node) ParseState() StateID {
	if n.IsNull() {
		return 0
	}
	return StateID(ts.NodeParseState(n.n))
}

// NextParseState returns the parse state after the node.
func (n Node) NextParseState() StateID {
	if n.IsNull() {
		return 0
	}
	return StateID(ts.NodeNextParseState(n.n))
}

// Parent returns the node's parent, or a null node at the root.
func (n Node) Parent() Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeParent(n.n)}
}

// ChildWithDescendant returns the child that contains descendant.
func (n Node) ChildWithDescendant(descendant Node) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeChildWithDescendant(n.n, descendant.n)}
}

// Child returns the i'th child, or a null node when out of range.
func (n Node) Child(i uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeChild(n.n, i)}
}

// ChildCount returns the number of children, named and anonymous.
func (n Node) ChildCount() uint32 {
	if n.IsNull() {
		return 0
	}
	return ts.NodeChildCount(n.n)
}

// NamedChild returns the i'th named child.
func (n Node) NamedChild(i uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeNamedChild(n.n, i)}
}

// NamedChildCount returns the number of named children.
func (n Node) NamedChildCount() uint32 {
	if n.IsNull() {
		return 0
	}
	return ts.NodeNamedChildCount(n.n)
}

// Children returns an iterator over all children in order.
func (n Node) Children() iter.Seq[Node] {
	return func(yield func(Node) bool) {
		if n.ChildCount() == 0 {
			return
		}
		c := n.Walk()
		defer c.Close()
		for ok := c.GotoFirstChild(); ok; ok = c.GotoNextSibling() {
			if !yield(c.CurrentNode()) {
				return
			}
		}
	}
}

// NamedChildren returns an iterator over the named children in order.
func (n Node) NamedChildren() iter.Seq[Node] {
	return func(yield func(Node) bool) {
		for c := range n.Children() {
			if c.IsNamed() && !yield(c) {
				return
			}
		}
	}
}

// Descendants returns an iterator over every node below n in document
// (pre-order) order. n itself is not included.
func (n Node) Descendants() iter.Seq[Node] {
	return func(yield func(Node) bool) {
		if n.ChildCount() == 0 {
			return
		}
		c := n.Walk()
		defer c.Close()
		if !c.GotoFirstChild() {
			return
		}
		for {
			if !yield(c.CurrentNode()) {
				return
			}
			if c.GotoFirstChild() {
				continue
			}
			for !c.GotoNextSibling() {
				if !c.GotoParent() || c.CurrentDepth() == 0 {
					return
				}
			}
		}
	}
}

// FieldNameForChild returns the field name of the i'th child, or "" if
// the child has none.
func (n Node) FieldNameForChild(i uint32) string {
	if n.IsNull() {
		return ""
	}
	return ts.NodeFieldNameForChild(n.n, i)
}

// FieldNameForNamedChild returns the field name of the i'th named child,
// or "" if it has none.
func (n Node) FieldNameForNamedChild(i uint32) string {
	if n.IsNull() {
		return ""
	}
	return ts.NodeFieldNameForNamedChild(n.n, i)
}

// ChildByFieldName returns the child with the given field name, or a null
// node.
func (n Node) ChildByFieldName(name string) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeChildByFieldName(n.n, stringPtr(name), uint32(len(name)))}
}

// ChildByFieldID returns the child with the given field id, or a null
// node.
func (n Node) ChildByFieldID(id FieldID) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeChildByFieldId(n.n, ts.FieldId(id))}
}

// NextSibling returns the following sibling, or a null node.
func (n Node) NextSibling() Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeNextSibling(n.n)}
}

// PrevSibling returns the preceding sibling, or a null node.
func (n Node) PrevSibling() Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodePrevSibling(n.n)}
}

// NextNamedSibling returns the following named sibling, or a null node.
func (n Node) NextNamedSibling() Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeNextNamedSibling(n.n)}
}

// PrevNamedSibling returns the preceding named sibling, or a null node.
func (n Node) PrevNamedSibling() Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodePrevNamedSibling(n.n)}
}

// FirstChildForByte returns the first child that extends past the byte
// offset.
func (n Node) FirstChildForByte(b uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeFirstChildForByte(n.n, b)}
}

// FirstNamedChildForByte returns the first named child that extends past
// the byte offset.
func (n Node) FirstNamedChildForByte(b uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeFirstNamedChildForByte(n.n, b)}
}

// DescendantCount returns the number of descendants, including the node
// itself.
func (n Node) DescendantCount() uint32 {
	if n.IsNull() {
		return 0
	}
	return ts.NodeDescendantCount(n.n)
}

// DescendantForByteRange returns the smallest descendant spanning the
// byte range.
func (n Node) DescendantForByteRange(start, end uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeDescendantForByteRange(n.n, start, end)}
}

// DescendantForPointRange returns the smallest descendant spanning the
// point range.
func (n Node) DescendantForPointRange(start, end Point) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeDescendantForPointRange(n.n, toPoint(start), toPoint(end))}
}

// NamedDescendantForByteRange returns the smallest named descendant
// spanning the byte range.
func (n Node) NamedDescendantForByteRange(start, end uint32) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeNamedDescendantForByteRange(n.n, start, end)}
}

// NamedDescendantForPointRange returns the smallest named descendant
// spanning the point range.
func (n Node) NamedDescendantForPointRange(start, end Point) Node {
	if n.IsNull() {
		return Node{}
	}
	return Node{n: ts.NodeNamedDescendantForPointRange(n.n, toPoint(start), toPoint(end))}
}

// Edit adjusts the node's position for an edit already applied to its
// tree with [Tree.Edit]. It is needed only for nodes retained across the
// edit; nodes fetched afterwards are already correct.
func (n *Node) Edit(edit InputEdit) {
	e := toInputEdit(edit)
	ts.NodeEdit(&n.n, &e)
}

// Equal reports whether two nodes are the same node of the same tree.
func (n Node) Equal(other Node) bool {
	return ts.NodeEq(n.n, other.n)
}

// Walk returns a cursor positioned at the node.
func (n Node) Walk() *TreeCursor {
	return NewTreeCursor(n)
}
