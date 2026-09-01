package treesitter

import (
	"github.com/tmc/lib/treesitter/internal/ts"
)

// Node is a node in a [Tree]. It is a small value that refers into the
// tree and is valid only while the tree is open. The zero Node is null.
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
	return ts.NodeType(n.n)
}

// Symbol returns the node's type as a symbol.
func (n Node) Symbol() Symbol {
	return Symbol(ts.NodeSymbol(n.n))
}

// GrammarType returns the type name as written in the grammar, before any
// alias is applied.
func (n Node) GrammarType() string {
	return ts.NodeGrammarType(n.n)
}

// GrammarSymbol returns the symbol as written in the grammar, before any
// alias is applied.
func (n Node) GrammarSymbol() Symbol {
	return Symbol(ts.NodeGrammarSymbol(n.n))
}

// Language returns the language of the node's tree.
func (n Node) Language() *Language {
	return NewLanguage(ts.NodeLanguage(n.n))
}

// StartByte returns the byte offset where the node begins.
func (n Node) StartByte() uint32 {
	return ts.NodeStartByte(n.n)
}

// EndByte returns the byte offset just past the node's end.
func (n Node) EndByte() uint32 {
	return ts.NodeEndByte(n.n)
}

// StartPoint returns the position where the node begins.
func (n Node) StartPoint() Point {
	return fromPoint(ts.NodeStartPoint(n.n))
}

// EndPoint returns the position just past the node's end.
func (n Node) EndPoint() Point {
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
// from. It returns nil when the node lies outside src.
func (n Node) Text(src []byte) []byte {
	start, end := int(n.StartByte()), int(n.EndByte())
	if start > end || end > len(src) {
		return nil
	}
	return src[start:end]
}

// String returns the node's subtree as an S-expression.
func (n Node) String() string {
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
func (n Node) IsNamed() bool { return ts.NodeIsNamed(n.n) }

// IsMissing reports whether the parser inserted the node to recover from
// an error.
func (n Node) IsMissing() bool { return ts.NodeIsMissing(n.n) }

// IsExtra reports whether the node is an extra, such as a comment, that
// the grammar allows anywhere.
func (n Node) IsExtra() bool { return ts.NodeIsExtra(n.n) }

// HasChanges reports whether the node was edited since the tree was
// parsed.
func (n Node) HasChanges() bool { return ts.NodeHasChanges(n.n) }

// HasError reports whether the node or any descendant is a syntax error.
func (n Node) HasError() bool { return ts.NodeHasError(n.n) }

// IsError reports whether the node itself is a syntax error.
func (n Node) IsError() bool { return ts.NodeIsError(n.n) }

// ParseState returns the parse state the node was created in.
func (n Node) ParseState() StateID { return StateID(ts.NodeParseState(n.n)) }

// NextParseState returns the parse state after the node.
func (n Node) NextParseState() StateID { return StateID(ts.NodeNextParseState(n.n)) }

// Parent returns the node's parent, or a null node at the root.
func (n Node) Parent() Node { return Node{n: ts.NodeParent(n.n)} }

// ChildWithDescendant returns the child that contains descendant.
func (n Node) ChildWithDescendant(descendant Node) Node {
	return Node{n: ts.NodeChildWithDescendant(n.n, descendant.n)}
}

// Child returns the i'th child, or a null node when out of range.
func (n Node) Child(i uint32) Node { return Node{n: ts.NodeChild(n.n, i)} }

// ChildCount returns the number of children, named and anonymous.
func (n Node) ChildCount() uint32 { return ts.NodeChildCount(n.n) }

// NamedChild returns the i'th named child.
func (n Node) NamedChild(i uint32) Node { return Node{n: ts.NodeNamedChild(n.n, i)} }

// NamedChildCount returns the number of named children.
func (n Node) NamedChildCount() uint32 { return ts.NodeNamedChildCount(n.n) }

// Children returns all children.
func (n Node) Children() []Node {
	count := n.ChildCount()
	if count == 0 {
		return nil
	}
	c := n.Walk()
	defer c.Close()
	out := make([]Node, 0, count)
	if c.GotoFirstChild() {
		out = append(out, c.CurrentNode())
		for c.GotoNextSibling() {
			out = append(out, c.CurrentNode())
		}
	}
	return out
}

// NamedChildren returns the named children.
func (n Node) NamedChildren() []Node {
	var out []Node
	for _, c := range n.Children() {
		if c.IsNamed() {
			out = append(out, c)
		}
	}
	return out
}

// FieldNameForChild returns the field name of the i'th child, or "" if
// the child has none.
func (n Node) FieldNameForChild(i uint32) string {
	return ts.NodeFieldNameForChild(n.n, i)
}

// FieldNameForNamedChild returns the field name of the i'th named child,
// or "" if it has none.
func (n Node) FieldNameForNamedChild(i uint32) string {
	return ts.NodeFieldNameForNamedChild(n.n, i)
}

// ChildByFieldName returns the child with the given field name, or a null
// node.
func (n Node) ChildByFieldName(name string) Node {
	return Node{n: ts.NodeChildByFieldName(n.n, stringPtr(name), uint32(len(name)))}
}

// ChildByFieldID returns the child with the given field id, or a null
// node.
func (n Node) ChildByFieldID(id FieldID) Node {
	return Node{n: ts.NodeChildByFieldId(n.n, ts.FieldId(id))}
}

// NextSibling returns the following sibling, or a null node.
func (n Node) NextSibling() Node { return Node{n: ts.NodeNextSibling(n.n)} }

// PrevSibling returns the preceding sibling, or a null node.
func (n Node) PrevSibling() Node { return Node{n: ts.NodePrevSibling(n.n)} }

// NextNamedSibling returns the following named sibling, or a null node.
func (n Node) NextNamedSibling() Node { return Node{n: ts.NodeNextNamedSibling(n.n)} }

// PrevNamedSibling returns the preceding named sibling, or a null node.
func (n Node) PrevNamedSibling() Node { return Node{n: ts.NodePrevNamedSibling(n.n)} }

// FirstChildForByte returns the first child that extends past the byte
// offset.
func (n Node) FirstChildForByte(b uint32) Node {
	return Node{n: ts.NodeFirstChildForByte(n.n, b)}
}

// FirstNamedChildForByte returns the first named child that extends past
// the byte offset.
func (n Node) FirstNamedChildForByte(b uint32) Node {
	return Node{n: ts.NodeFirstNamedChildForByte(n.n, b)}
}

// DescendantCount returns the number of descendants, including the node
// itself.
func (n Node) DescendantCount() uint32 { return ts.NodeDescendantCount(n.n) }

// DescendantForByteRange returns the smallest descendant spanning the
// byte range.
func (n Node) DescendantForByteRange(start, end uint32) Node {
	return Node{n: ts.NodeDescendantForByteRange(n.n, start, end)}
}

// DescendantForPointRange returns the smallest descendant spanning the
// point range.
func (n Node) DescendantForPointRange(start, end Point) Node {
	return Node{n: ts.NodeDescendantForPointRange(n.n, toPoint(start), toPoint(end))}
}

// NamedDescendantForByteRange returns the smallest named descendant
// spanning the byte range.
func (n Node) NamedDescendantForByteRange(start, end uint32) Node {
	return Node{n: ts.NodeNamedDescendantForByteRange(n.n, start, end)}
}

// NamedDescendantForPointRange returns the smallest named descendant
// spanning the point range.
func (n Node) NamedDescendantForPointRange(start, end Point) Node {
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
