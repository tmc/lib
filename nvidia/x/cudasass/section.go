package cudasass

import (
	"errors"
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
		rest := strings.TrimSpace(strings.TrimPrefix(trimmed, ".section"))
		name = rest
		if i, f := sectionFlags(rest); i >= 0 {
			name, flags = strings.TrimSpace(rest[:i]), f
		}
	default:
		return "", false, false
	}
	return name, isNonTextCode(name, flags), true
}

// sectionFlags returns the index in rest of the comma that begins the flags
// field of a .section directive, and the bytes between the field's quotes,
// where rest is the directive with ".section" and the blanks after it
// removed. It returns -1 and "" if rest has no flags field, and then all of
// rest is the section's name.
//
// LLVM spells the directive
//
//	.section NAME,"FLAGS",@TYPE,EXTRA
//
// where each field after NAME is optional. nvdisasm prints NAME unquoted, and
// a demangled NAME holds commas, blanks and string or character literals of
// its own, so the flags field is not simply the first comma before a quote.
// It is the first comma outside every literal (a literal read as literalEnd
// reads one) that is followed by optional spaces or tabs, a terminated
// double-quoted literal holding only bytes of sectionFlagBytes, and a
// metadata tail as sectionTail defines it. A comma and a quoted literal
// followed by more of a name, as in
// `decltype(fp, "<char const [5]>") s1<int>(int)`, or holding a byte no
// flags field holds, as in `k, "lit"`, belong to the name.
//
// A name that itself ends, outside every literal, in a comma and a quoted
// literal of flag bytes, such as `k, "a"`, cannot be told from a shorter
// name and its flags. sectionFlags reads it as the shorter name, and
// malformedSectionDirective refuses the directive when a second comma after
// the first also begins a flags field, as in `k, "a","ax",@progbits`.
func sectionFlags(rest string) (comma int, flags string) {
	for i := 0; i < len(rest); i++ {
		switch rest[i] {
		case '"', '\'':
			if end := literalEnd(rest, i); end >= 0 {
				i = end
			}
		case ',':
			j := i + 1
			for j < len(rest) && (rest[j] == ' ' || rest[j] == '\t') {
				j++
			}
			if j == len(rest) || rest[j] != '"' {
				continue
			}
			end := literalEnd(rest, j)
			if end >= 0 && isSectionFlags(rest[j+1:end]) && sectionTail(rest[end+1:]) {
				return i, rest[j+1 : end]
			}
		}
	}
	return -1, ""
}

// sectionFlagBytes are the bytes LLVM's ELF assembler reads in a .section
// directive's flags field: a, e, o, w, x, M, S, T, G, R and ? on every
// target, and c, d, l, s and y on some.
const sectionFlagBytes = "aeowxMSTGR?cdlsy"

// isSectionFlags reports whether flags, the bytes between a flags field's
// quotes, holds only bytes of sectionFlagBytes. The empty flags field ""
// does.
func isSectionFlags(flags string) bool {
	return strings.Trim(flags, sectionFlagBytes) == ""
}

// malformedSectionDirective reports whether trimmed is a ".section"
// directive Parse refuses: its name, the text before the flags field
// sectionFlags finds or all of it with none, holds a quote that opens no
// terminated literal, read as lineComment reads one; or a second comma after
// that flags field also begins one, so that the field's own literal could
// equally be the end of the name. Reading such a name would open a section
// named by a guess.
func malformedSectionDirective(trimmed string) bool {
	if !isSectionDirective(trimmed) {
		return false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, ".section"))
	name := rest
	if i, _ := sectionFlags(rest); i >= 0 {
		if j, _ := sectionFlags(rest[i+1:]); j >= 0 {
			return true
		}
		name = rest[:i]
	}
	return openLiteral(name)
}

// openLiteral reports whether s holds a quote that opens no terminated
// literal, reading s from left to right as lineComment does: a quote that
// opens a terminated literal is skipped through its closing quote.
func openLiteral(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '"' || s[i] == '\'' {
			end := literalEnd(s, i)
			if end < 0 {
				return true
			}
			i = end
		}
	}
	return false
}

// sectionTail reports whether s, the text after the closing quote of a
// .section directive's flags, is the rest of the directive's metadata:
// optional spaces or tabs, and then the end of s, a "//" comment, or a comma,
// optional spaces or tabs and a type. A type is '@' or '%' followed by one
// or more identifier bytes or by a terminated double-quoted literal, or such
// a literal alone, as in @progbits, @"SHT_CUDA_INFO" and "progbits". After
// the type come optional spaces or tabs and then the end of s, a "//"
// comment, or a comma and further fields such as an entry size or a group,
// which sectionTail does not read.
func sectionTail(s string) bool {
	s = strings.TrimLeft(s, " \t")
	if s == "" || strings.HasPrefix(s, "//") {
		return true
	}
	s, ok := strings.CutPrefix(s, ",")
	if !ok {
		return false
	}
	s = strings.TrimLeft(s, " \t")
	n := 0 // the length of the type, or 0 if s does not begin with one
	switch {
	case strings.HasPrefix(s, "\""):
		n = literalEnd(s, 0) + 1
	case strings.HasPrefix(s, "@"), strings.HasPrefix(s, "%"):
		if strings.HasPrefix(s[1:], "\"") {
			n = literalEnd(s, 1) + 1
			break
		}
		n = 1
		for n < len(s) && isIdentByte(s[n]) {
			n++
		}
		if n == 1 {
			n = 0
		}
	}
	if n == 0 {
		return false
	}
	s = strings.TrimLeft(s[n:], " \t")
	return s == "" || strings.HasPrefix(s, "//") || strings.HasPrefix(s, ",")
}

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

// malformedTextLabel reports whether trimmed is a ".text." line whose name
// Parse refuses. The name is the text before the colon labelColon finds or,
// with no such colon, before the "//" lineComment finds. Reading a malformed name, such
// as the "k:junk" of ".text.k:junk" or the "k /* c */" of ".text.k /* c */:",
// as a kernel's name would open a kernel named by a guess and move the
// instructions after it out of the kernel the listing declared.
func malformedTextLabel(trimmed string) bool {
	rest, ok := strings.CutPrefix(trimmed, ".text.")
	if !ok {
		return false
	}
	name := rest
	if i := labelColon(rest); i >= 0 {
		name = rest[:i]
	} else if i := lineComment(rest); i >= 0 {
		name = rest[:i]
	}
	return malformedName(name)
}

// malformedName reports whether name, a label's name, breaks the lexical
// rule Parse documents. malformedName reads name from left to right:
//
//  1. A double or single quote begins a literal that ends at the next
//     unescaped copy of the same quote; a backslash escapes the byte after
//     it. Nothing inside a literal counts below. An unterminated literal is
//     malformed.
//  2. Only '(', '[' and '{' open a bracket, and only ')', ']' and '}' close
//     one; '<' and '>' are ignored. A closing bracket that does not match
//     the innermost open bracket, or a bracket left open, is malformed.
//  3. Every "::" is well formed. A run of three or more colons is malformed.
//  4. A single colon is well formed only
//     (a) in "[abi:" followed by one or more identifier bytes (letters,
//     digits, '_' or '$') and then ']';
//     (b) in "auto:N", where "auto" begins name or follows a byte other
//     than an identifier byte, and N is one or more digits followed by a
//     byte other than an identifier byte;
//     (c) as a conditional's colon, paired with the latest '?' at the same
//     (), [] or {} depth that no colon has paired, with text other than
//     white space between them.
//     Every other single colon is malformed.
//  5. "/*", "*/" and "//" are malformed.
//
// The rule is lexical, not a C++ grammar, so some constructed names that are
// no C++ name are well formed: in "k ? f<a : b>" the colon pairs with the
// '?' before the '<', "f<1 >> 2, b ? 1 : 2>()" reads the same whether or
// not its ">>" closes the template arguments, and "k:: junk", "k ::junk",
// "f(a::)", "k ? junk : junk" and "k [abi:cxx11]" are well formed.
func malformedName(name string) bool {
	stack := []bracketDepth{{}} // stack[0] is the depth outside every bracket
	for i := 0; i < len(name); i++ {
		top := &stack[len(stack)-1]
		switch c := name[i]; c {
		case '"', '\'':
			end := literalEnd(name, i)
			if end < 0 {
				return true
			}
			i = end
		case '/':
			if i+1 < len(name) && (name[i+1] == '*' || name[i+1] == '/') {
				return true
			}
		case '*':
			if i+1 < len(name) && name[i+1] == '/' {
				return true
			}
		case '(', '[', '{':
			stack = append(stack, bracketDepth{open: c})
		case ')', ']', '}':
			if top.open != opens[c] {
				return true
			}
			stack = stack[:len(stack)-1]
		case '?':
			top.questions = append(top.questions, i)
		case ':':
			j := i + 1
			for j < len(name) && name[j] == ':' {
				j++
			}
			switch {
			case j-i == 2:
			case j-i > 2:
				return true
			case abiTag(name, i), autoPlaceholder(name, i), pairConditional(name, i, top):
			default:
				return true
			}
			i = j - 1
		}
	}
	return len(stack) > 1
}

// A bracketDepth is one bracket depth of a name malformedName is reading.
type bracketDepth struct {
	open      byte  // the opening bracket, or 0 outside every bracket
	questions []int // the indexes of the '?' at this depth no colon has paired
}

// opens maps a closing bracket to the bracket it closes.
var opens = map[byte]byte{')': '(', ']': '[', '}': '{'}

// literalEnd returns the index of the quote that ends the literal beginning
// with the quote at name[i], or -1 if the literal is unterminated. A
// backslash escapes the byte after it.
func literalEnd(name string, i int) int {
	for j := i + 1; j < len(name); j++ {
		switch name[j] {
		case '\\':
			j++
		case name[i]:
			return j
		}
	}
	return -1
}

// lineComment returns the index of the first "//" in s outside a string or
// character literal, read as literalEnd reads one, or -1 if there is none.
// A quote that opens no terminated literal is an ordinary byte, so the
// "//" after it is found and the name before it, holding the open quote,
// is malformed.
func lineComment(s string) int {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\'':
			if end := literalEnd(s, i); end >= 0 {
				i = end
			}
		case '/':
			if i+1 < len(s) && s[i+1] == '/' {
				return i
			}
		}
	}
	return -1
}

// pairConditional pairs the colon at name[i] with the latest unpaired '?' at
// depth d, if some text other than white space lies between them, and
// reports whether it did.
func pairConditional(name string, i int, d *bracketDepth) bool {
	q := d.questions
	if len(q) == 0 || strings.TrimSpace(name[q[len(q)-1]+1:i]) == "" {
		return false
	}
	d.questions = q[:len(q)-1]
	return true
}

// abiTag reports whether the colon at name[i] is an ABI tag's: name[:i] ends
// in "[abi", and the colon is followed by one or more identifier bytes and
// then ']'.
func abiTag(name string, i int) bool {
	if !strings.HasSuffix(name[:i], "[abi") {
		return false
	}
	j := i + 1
	for j < len(name) && isIdentByte(name[j]) {
		j++
	}
	return j > i+1 && j < len(name) && name[j] == ']'
}

// autoPlaceholder reports whether the colon at name[i] is a generic lambda
// placeholder's "auto:N": name[:i] ends in "auto", which begins name or
// follows a byte that cannot continue an identifier, and the colon is
// followed by one or more digits and then a byte that cannot continue an
// identifier.
func autoPlaceholder(name string, i int) bool {
	head, ok := strings.CutSuffix(name[:i], "auto")
	if !ok || head != "" && isIdentByte(head[len(head)-1]) {
		return false
	}
	j := i + 1
	for j < len(name) && '0' <= name[j] && name[j] <= '9' {
		j++
	}
	return j > i+1 && j < len(name) && !isIdentByte(name[j])
}

// isIdentByte reports whether c can continue an identifier: a letter, digit,
// '_' or '$'.
func isIdentByte(c byte) bool {
	return c == '_' || c == '$' || '0' <= c && c <= '9' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
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
