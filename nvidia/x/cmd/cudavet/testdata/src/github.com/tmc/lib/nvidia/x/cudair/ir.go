// Package cudair stands in for a hand-written package under x/ whose Set
// method validates Go values and is not a binding's setter.
package cudair

type Node struct{}

type Graph struct{}

func NewGraph() *Graph { return &Graph{} }

func (g *Graph) SetOutputs(nodes ...*Node) error { return nil }
