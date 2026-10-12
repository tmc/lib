package cudasass

import (
	"errors"
	"regexp"
	"strings"
)

// ErrNotTextSection means a listing has an instruction under a code section
// other than a kernel's .text.<kernel>, such as the .nv.capmerc.text.<kernel>
// section ptxas emits beside it. [Parse] returns it, wrapped with the
// section's name, rather than attribute the instruction to the kernel.
var ErrNotTextSection = errors.New("code section is not a .text section")

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

	// ELFSection is the ELF section the region's instructions were read
	// from, as the listing names it: ".text.<Name>" for nvdisasm's
	// ".section" and label headers, and "" for a header that names no
	// section, such as cuobjdump's "Function : <Name>".
	ELFSection string
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

// headerELFSection returns the ELF section a kernel header line names:
// ".text.NAME" for the ".section .text.NAME" and ".text.NAME:" spellings, and
// "" for the spellings that name a kernel without its section.
func headerELFSection(trimmed string) string {
	name := extractFunctionName(trimmed)
	if name == "" {
		return ""
	}
	if strings.HasPrefix(trimmed, ".section") || strings.HasPrefix(trimmed, ".text.") {
		return TextSectionName(name)
	}
	return ""
}

// openedSection reports whether trimmed opens a section of the listing, the
// section's name, and whether that section is code other than
// .text.<kernel>. Only a ".section" directive or a banner comment opens a
// section; the section stays open until the next one does. A label or a
// kernel header names a position, not a section, and opens none.
func openedSection(trimmed string) (name string, nonText, opens bool) {
	var flags string
	switch {
	case strings.HasPrefix(trimmed, "//--"):
		name = strings.TrimSpace(strings.Trim(strings.TrimPrefix(trimmed, "//"), "-"))
	case isSectionDirective(trimmed):
		// A demangled name holds commas and spaces, so the flags are found
		// by their quotes: .section NAME,"FLAGS",@TYPE, with optional space
		// before the quote.
		rest := strings.TrimSpace(strings.TrimPrefix(trimmed, ".section"))
		if m := sectionFlags.FindStringSubmatchIndex(rest); m != nil {
			name = strings.TrimSpace(rest[:m[0]])
			flags = rest[m[2]:m[3]]
		} else {
			name = rest
		}
	default:
		return "", false, false
	}
	return name, isNonTextCode(name, flags), true
}

// sectionFlags matches the flags field of a .section directive.
var sectionFlags = regexp.MustCompile(`,[ \t]*"([^"]*)"`)

// nonTextLabel reports whether trimmed is the label of a code section other
// than .text.<kernel>, such as ".nv.capmerc.text.k:", and its name. Such a
// label places what follows in that section even where no directive opened
// it. The label of a .text section, or a code label such as ".L_x_0:", does
// not, so no label can close a section a directive opened.
func nonTextLabel(trimmed string) (string, bool) {
	label, ok := strings.CutSuffix(trimmed, ":")
	if !ok || strings.ContainsAny(label, " \t:,/") || !isNonTextCode(label, "") {
		return "", false
	}
	return label, true
}

// isSectionDirective reports whether trimmed is a ".section" directive, and
// not one of the directives that share its prefix, such as ".sectionflags".
func isSectionDirective(trimmed string) bool {
	rest, ok := strings.CutPrefix(trimmed, ".section")
	return ok && (rest == "" || rest[0] == ' ' || rest[0] == '\t')
}

// isNonTextCode reports whether the section name with flags may hold code
// other than .text.<kernel>: its flags mark it executable, or its name carries
// ".text" after a prefix, as .nv.capmerc.text.<kernel> does. Such a name may
// also be a relocation or other data section; that is harmless, because only
// an instruction under the section is refused.
func isNonTextCode(name, flags string) bool {
	if strings.HasPrefix(name, ".text.") && len(name) > len(".text.") {
		return false
	}
	return strings.Contains(flags, "x") || strings.Contains(name, ".text")
}
