// Package cudaring stands in for a hand-written package under x/ whose type
// has a Close method but is not a handle of a binding.
package cudaring

type Ring struct{}

func NewRing() (*Ring, error) { return &Ring{}, nil }

func (r Ring) Close() error { return nil }
