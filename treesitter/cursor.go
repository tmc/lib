package treesitter

import "github.com/tmc/lib/treesitter/internal/ts"

// TreeCursor walks a tree more efficiently than repeated [Node] calls. A
// cursor owns C memory and must be closed.
type TreeCursor struct {
	c    ts.TreeCursor
	open bool
}

// NewTreeCursor returns a cursor positioned at node.
func NewTreeCursor(node Node) *TreeCursor {
	return &TreeCursor{c: ts.TreeCursorNew(node.n), open: true}
}

// Close frees the cursor.
func (c *TreeCursor) Close() {
	if !c.open {
		return
	}
	ts.TreeCursorDelete(&c.c)
	c.open = false
}

// Reset repositions the cursor at node, which may be in another tree.
func (c *TreeCursor) Reset(node Node) {
	ts.TreeCursorReset(&c.c, node.n)
}

// ResetTo repositions the cursor at src's position.
func (c *TreeCursor) ResetTo(src *TreeCursor) {
	ts.TreeCursorResetTo(&c.c, &src.c)
}

// Copy returns a cursor at the same position.
func (c *TreeCursor) Copy() *TreeCursor {
	return &TreeCursor{c: ts.TreeCursorCopy(&c.c), open: true}
}

// CurrentNode returns the node the cursor is on.
func (c *TreeCursor) CurrentNode() Node {
	return Node{n: ts.TreeCursorCurrentNode(&c.c)}
}

// CurrentFieldName returns the field name of the current node, or "".
func (c *TreeCursor) CurrentFieldName() string {
	return ts.TreeCursorCurrentFieldName(&c.c)
}

// CurrentFieldID returns the field id of the current node, or 0.
func (c *TreeCursor) CurrentFieldID() FieldID {
	return FieldID(ts.TreeCursorCurrentFieldId(&c.c))
}

// CurrentDepth returns the depth of the current node below the node the
// cursor was created at.
func (c *TreeCursor) CurrentDepth() uint32 {
	return ts.TreeCursorCurrentDepth(&c.c)
}

// CurrentDescendantIndex returns the index of the current node among the
// descendants of the node the cursor was created at, in pre-order.
func (c *TreeCursor) CurrentDescendantIndex() uint32 {
	return ts.TreeCursorCurrentDescendantIndex(&c.c)
}

// GotoParent moves to the parent and reports whether there was one.
func (c *TreeCursor) GotoParent() bool { return ts.TreeCursorGotoParent(&c.c) }

// GotoNextSibling moves to the next sibling and reports whether there was
// one.
func (c *TreeCursor) GotoNextSibling() bool { return ts.TreeCursorGotoNextSibling(&c.c) }

// GotoPreviousSibling moves to the previous sibling and reports whether
// there was one.
func (c *TreeCursor) GotoPreviousSibling() bool { return ts.TreeCursorGotoPreviousSibling(&c.c) }

// GotoFirstChild moves to the first child and reports whether there was
// one.
func (c *TreeCursor) GotoFirstChild() bool { return ts.TreeCursorGotoFirstChild(&c.c) }

// GotoLastChild moves to the last child and reports whether there was one.
func (c *TreeCursor) GotoLastChild() bool { return ts.TreeCursorGotoLastChild(&c.c) }

// GotoDescendant moves to the descendant with the given pre-order index.
func (c *TreeCursor) GotoDescendant(index uint32) {
	ts.TreeCursorGotoDescendant(&c.c, index)
}

// GotoFirstChildForByte moves to the first child that extends past the
// byte offset and returns its index, or ok false if there is none.
func (c *TreeCursor) GotoFirstChildForByte(b uint32) (index uint32, ok bool) {
	i := ts.TreeCursorGotoFirstChildForByte(&c.c, b)
	if i < 0 {
		return 0, false
	}
	return uint32(i), true
}

// GotoFirstChildForPoint moves to the first child that extends past the
// point and returns its index, or ok false if there is none.
func (c *TreeCursor) GotoFirstChildForPoint(p Point) (index uint32, ok bool) {
	i := ts.TreeCursorGotoFirstChildForPoint(&c.c, toPoint(p))
	if i < 0 {
		return 0, false
	}
	return uint32(i), true
}
