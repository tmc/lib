// Package cuptirange is a declaration-only stand-in for a hand-written
// package under x/ whose type has a Close method but holds no device handle.
package cuptirange

type Catalog struct{}

func NewCatalog(chip string, availability []byte) (*Catalog, error) { return nil, nil }
func (c *Catalog) Close() error                                     { return nil }
