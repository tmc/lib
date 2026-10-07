package xidcatalog

//go:generate go run ./internal/gen -url https://docs.nvidia.com/deploy/xid-errors/_downloads/4586dadb59119a55d1e93a181caa4272/Xid-Catalog.xlsx -sha256 7f70ce9684be0c9d98a367770341ccc1d8bca4465cebe42dab7e23d57df5d5a5 -retrieved 2026-10-06 -out catalog.gen.go

// Architecture identifies a product applicability column in NVIDIA's catalog.
// These names are product-specific, not claims about an architecture family.
type Architecture string

const (
	A100  Architecture = "A100"
	H100  Architecture = "H100"
	B100  Architecture = "B100"
	GB200 Architecture = "GB200"
)

// Entry describes an XID. Action fields are NVIDIA's resolution bucket names,
// including workflow identifiers requiring further investigation. An empty
// action is unspecified; it is not evidence that no action is required.
type Entry struct {
	Code                uint64
	Name                string
	Description         string
	ImmediateAction     string
	InvestigatoryAction string
	RecoveryActionEvent bool
}

type row struct {
	entry    Entry
	products uint8
}

// Lookup returns a catalog entry only if it applies to architecture. XID154
// reports a change to the GPU's recovery action; inspect the actual event's
// recovery value rather than inferring a reset or reboot from its ID alone.
func Lookup(id uint64, architecture Architecture) (Entry, bool) {
	var bit uint8
	switch architecture {
	case A100:
		bit = 1
	case H100:
		bit = 2
	case B100:
		bit = 4
	case GB200:
		bit = 8
	default:
		return Entry{}, false
	}
	r, ok := catalog[id]
	if !ok || r.products&bit == 0 {
		return Entry{}, false
	}
	return r.entry, true
}
