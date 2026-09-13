package egraph

import "fmt"

// Node represents an expression node in an E-Graph AST, aliased to Expr.
type Node = Expr

// NewNodeExpr creates a new expression Node with the given operator and child subexpressions.
func NewNodeExpr(op string, children ...*Node) *Node {
	return &Node{
		Op:       op,
		Children: children,
	}
}

// NewLeafNode creates a leaf expression Node with the given symbol or literal value.
func NewLeafNode(val any) *Node {
	return &Node{
		Op:   fmt.Sprint(val),
		Data: val,
	}
}
