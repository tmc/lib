package xidcatalog

// Applicability distinguishes catalog coverage from product applicability.
type Applicability uint8

const (
	UnknownID Applicability = iota
	UncoveredProduct
	NotApplicable
	Applicable
)

// LookupRaw returns the catalog row without interpreting product coverage.
func LookupRaw(id uint64) (Entry, bool) {
	r, ok := catalog[id]
	return r.entry, ok
}

// LookupProduct returns the row and its documented product applicability.
// L4, GB10 and other products without a catalog column are UncoveredProduct;
// applicability is never inferred from an architecture family.
func LookupProduct(id uint64, product Architecture) (Entry, Applicability) {
	r, ok := catalog[id]
	if !ok {
		return Entry{}, UnknownID
	}
	var bit uint8
	switch product {
	case A100:
		bit = 1
	case H100:
		bit = 2
	case B100:
		bit = 4
	case GB200:
		bit = 8
	default:
		return r.entry, UncoveredProduct
	}
	if r.products&bit == 0 {
		return r.entry, NotApplicable
	}
	return r.entry, Applicable
}
