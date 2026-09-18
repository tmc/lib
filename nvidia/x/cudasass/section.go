package cudasass

import "strings"

// A Section is the region of a disassembly listing that belongs to one kernel.
//
// Text is the listing region verbatim, from the kernel's banner comment (or,
// where nvdisasm printed none, from its first header line) through the line
// before the next kernel or trailing banner. It is a region of a listing, not a
// standalone file: the listing's preamble is not repeated into it, so Text
// re-parses through [ParseString] but is not by itself a complete nvdisasm
// output.
type Section struct {
	Name string
	Text []byte
}

// Sections returns one Section per kernel region of the parsed listing, in
// listing order.
//
// This is the shape callers reach for when they want one kernel's SASS, and it
// is deliberately not derived by matching a banner at the call site: Name comes
// from the same header the instructions were parsed under, so a section can
// never name a kernel the disassembly does not contain. nvdisasm prints several
// spellings of a kernel's header -- a banner comment, ".section .text.<name>",
// ".global <name>", ".type <name>,@function" -- and a call site that picks one
// of them is choosing which spelling to be wrong about.
//
// A Section is a region of the listing, not an occurrence of a name. nvdisasm
// spells one kernel's header more than once -- ".section .text.k4" and then
// ".text.k4:" as a label -- and those are one region, not two. Two regions for
// one name arise only where the listing genuinely separates them, with a banner
// or an intervening kernel, and then [Disassembly.Function] still merges the
// instructions while the regions stay distinct.
func (d *Disassembly) Sections() []Section {
	if d == nil {
		return nil
	}
	return d.sections
}

// sectionBanner reports whether trimmed is one of nvdisasm's banner comments,
//
//	//--------------------- .text.addk    --------------------------
//	//--------------------- SYMBOLS       --------------------------
//
// and, for a kernel's banner, the name it carries. The second return
// distinguishes "not a banner" from "a banner that opens no kernel": the
// trailing SYMBOLS banner ends the last kernel's region without starting
// another, and treating it as an ordinary line would append the symbol table to
// that kernel.
func sectionBanner(trimmed string) (name string, isBanner bool) {
	if !strings.HasPrefix(trimmed, "//--") {
		return "", false
	}
	body := strings.TrimSpace(strings.Trim(strings.TrimPrefix(trimmed, "//"), "-"))
	if after, ok := strings.CutPrefix(body, ".text."); ok {
		return strings.TrimSpace(after), true
	}
	return "", true
}
