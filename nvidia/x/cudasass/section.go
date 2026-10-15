package cudasass

import (
	"errors"
	"iter"
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
// reads one) that begins a flags field as flagsField defines it. A comma and
// a quoted literal followed by more of a name, as in
// `decltype(fp, "<char const [5]>") s1<int>(int)`, or holding a byte no
// flags field holds, as in `k, "lit"`, belong to the name.
//
// A name that itself ends, outside every literal, in a comma and a quoted
// literal of flag bytes, such as `k, "a"`, cannot be told from a shorter
// name and its flags. sectionFlags reads it as the shorter name, and
// malformedSectionDirective refuses the directive when a later comma also
// begins a flags field and the text before that comma is a name it accepts,
// as in `k, "a","ax",@progbits`.
func sectionFlags(rest string) (comma int, flags string) {
	lits := readLiterals(rest)
	for i := 0; i < len(rest); i++ {
		switch rest[i] {
		case '"', '\'':
			if end := lits.end(i); end >= 0 {
				i = end
			}
		case ',':
			if f, ok := flagsField(rest, i); ok {
				return i, f
			}
		}
	}
	return -1, ""
}

// flagsField reports whether the comma at rest[i] begins a flags field, and
// returns the bytes between the field's quotes. A flags field is the comma,
// optional spaces or tabs, a terminated double-quoted literal holding only
// bytes of sectionFlagBytes, and a metadata tail as sectionTail defines it.
func flagsField(rest string, i int) (flags string, ok bool) {
	j := i + 1
	for j < len(rest) && (rest[j] == ' ' || rest[j] == '\t') {
		j++
	}
	if j == len(rest) || rest[j] != '"' {
		return "", false
	}
	end := literalEnd(rest, j)
	if end < 0 || !isSectionFlags(rest[j+1:end]) || !sectionTail(rest[end+1:]) {
		return "", false
	}
	return rest[j+1 : end], true
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
// sectionFlags finds or all of it with none, is a malformed section name as
// sectionNameRefusals defines one; or a later comma also begins a flags
// field and the text before that comma is a well-formed name, so that the
// first field's literal could equally be the end of the name. Reading such
// a name would open a section named by a guess.
//
// A later comma after the type, such as the one before the group "x" in
// `k,"axG",@progbits,"x"`, does not make a directive malformed: the text
// before it ends in a comma, a quoted literal and a type, a malformed name.
// Every later comma is checked, since the text before a later one can be a
// well-formed name where the text before an earlier one is not: in
// `k,"ax",@progbits,"a",@b,x(y),"ax",@progbits`, the text before the comma
// of "a" ends in a type, but the text before the last "ax" does not, since
// x(y) is no metadata field.
//
// malformedSectionDirective reads trimmed once, so its cost is linear in
// the length of the line.
func malformedSectionDirective(trimmed string) bool {
	if !isSectionDirective(trimmed) {
		return false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(trimmed, ".section"))
	flags, _ := sectionFlags(rest)
	for i, refused := range sectionNameRefusals(rest) {
		switch {
		case i == flags || flags < 0 && i == len(rest):
			if refused {
				return true
			}
		case flags >= 0 && i > flags && i < len(rest) && !refused:
			if _, ok := flagsField(rest, i); ok {
				return true
			}
		}
	}
	return false
}

// sectionNameRefusals returns an iterator over the indexes of the commas
// of s outside every literal, in order, and then len(s). With each index i
// it yields whether s[:i] is a malformed section name, one Parse refuses as
// the name of a ".section" directive. It reads s once, skipping its
// literals as a literals skips them, so iterating it costs time linear in
// the length of s.
//
// A name is malformed if it holds a quote that opens no terminated literal,
// read as lineComment reads one; or if, split at its commas outside every
// literal, it ends in a field holding a terminated double-quoted literal, a
// field holding a type as sectionType defines it, and then only metadata
// fields. A "//" comment outside every literal ends the fields, so the
// text after it decides nothing. Blanks around a field's text are ignored.
// A metadata field is a terminated double-quoted literal, or text holding
// no blank, quote or bracket, such as an entry size, a group or "comdat".
// The first literal is then a flags field holding a byte LLVM does not
// read, such as the q of `k,"axq",@progbits` or
// `k,"axq",@progbits,"grp",comdat`, and the text after it is the
// directive's type and metadata, not more of a kernel's name.
//
// A demangled name's string literals are arguments inside brackets, as in
// `decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int)`,
// and the bracket closing them lies outside every literal in a later field,
// which is no metadata field. So no such name is malformed for this reason.
func sectionNameRefusals(s string) iter.Seq2[int, bool] {
	return func(yield func(int, bool) bool) {
		var (
			open     bool // s[:i] holds a quote that opens no terminated literal
			typed    bool // s[:i] ends in a literal field, a type and metadata
			afterLit bool // the last field, after a comma, was a quoted literal
			comment  bool // a "//" comment began, so no field follows
		)
		var (
			start = 0  // the start of the current field
			slash = -1 // the index of the field's first "//", or -1
		)
		lits := readLiterals(s)
		for i := 0; i <= len(s); i++ {
			if i < len(s) {
				switch s[i] {
				case '"', '\'':
					if end := lits.end(i); end >= 0 {
						i = end
					} else {
						open = true
					}
					continue
				case '/':
					if slash < 0 && i+1 < len(s) && s[i+1] == '/' {
						slash = i
					}
					continue
				case ',':
				default:
					continue
				}
			}
			if !comment {
				text := s[start:i]
				if slash >= 0 {
					text = s[start:slash]
				}
				text = strings.Trim(text, " \t")
				typed = typed && metadataField(text) || afterLit && isSectionType(text)
				afterLit = start > 0 && quotedField(text)
				comment = slash >= 0
			}
			if !yield(i, open || typed) {
				return
			}
			start, slash = i+1, -1
		}
	}
}

// quotedField reports whether s is a terminated double-quoted literal and
// nothing else.
func quotedField(s string) bool {
	return strings.HasPrefix(s, "\"") && literalEnd(s, 0) == len(s)-1
}

// metadataField reports whether s, a field after a .section directive's
// type with its blanks removed, is a metadata field: a terminated
// double-quoted literal, or text holding no blank, quote or bracket.
func metadataField(s string) bool {
	return quotedField(s) || s != "" && !strings.ContainsAny(s, " \t\"'()<>[]{}")
}

// isSectionType reports whether s is a section type and nothing else.
func isSectionType(s string) bool {
	n := sectionType(s)
	return n > 0 && n == len(s)
}

// sectionTail reports whether s, the text after the closing quote of a
// .section directive's flags, is the rest of the directive's metadata:
// optional spaces or tabs, and then the end of s, a "//" comment, or a comma,
// optional spaces or tabs and a type as sectionType defines it. After the
// type come optional spaces or tabs and then the end of s, a "//" comment,
// or a comma and further fields such as an entry size or a group, which
// sectionTail does not read; a quoted group made only of flag bytes, as in
// `,@progbits,"x"`, is one of them.
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
	n := sectionType(s)
	if n == 0 {
		return false
	}
	s = strings.TrimLeft(s[n:], " \t")
	return s == "" || strings.HasPrefix(s, "//") || strings.HasPrefix(s, ",")
}

// sectionType returns the length of the section type that begins s, or 0 if
// s begins with none. A type is '@' or '%' followed by one or more
// identifier bytes or by a terminated double-quoted literal, or such a
// literal alone, as in @progbits, @"SHT_CUDA_INFO" and "progbits".
func sectionType(s string) int {
	switch {
	case strings.HasPrefix(s, "\""):
		return literalEnd(s, 0) + 1
	case strings.HasPrefix(s, "@"), strings.HasPrefix(s, "%"):
		if strings.HasPrefix(s[1:], "\"") {
			return literalEnd(s, 1) + 1
		}
		n := 1
		for n < len(s) && isIdentByte(s[n]) {
			n++
		}
		if n == 1 {
			return 0
		}
		return n
	}
	return 0
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

// literals reads the literals of one string, s, as literalEnd reads them,
// where a reader may skip each terminated literal and steps over the quote
// of an unterminated one.
//
// Once a literal of s is unterminated, so is every later literal opened by
// the same quote byte: literalEnd's scan from the first reads each later
// copy of that quote as an escaped byte, so it next reads the byte after
// that copy, where a scan from the copy begins, and from there both scans
// read the same bytes. So literals scans at most one unterminated literal
// for each quote byte, and a reader that skips each terminated literal
// reads all of s in time linear in its length, where calling literalEnd
// at each quote could take time quadratic in it.
type literals struct {
	s string
	// open holds, for '"' and then '\'', the least index known to begin
	// an unterminated literal opened by that quote, or len(s).
	open [2]int
}

// readLiterals returns a literals reading s.
func readLiterals(s string) *literals {
	return &literals{s: s, open: [2]int{len(s), len(s)}}
}

// end returns literalEnd(l.s, i) for the quote at l.s[i].
func (l *literals) end(i int) int {
	k := 0
	if l.s[i] == '\'' {
		k = 1
	}
	if i >= l.open[k] {
		return -1
	}
	end := literalEnd(l.s, i)
	if end < 0 {
		l.open[k] = i
	}
	return end
}

// lineComment returns the index of the first "//" in s outside a string or
// character literal, read as literalEnd reads one, or -1 if there is none.
// A quote that opens no terminated literal is an ordinary byte, so the
// "//" after it is found and the name before it, holding the open quote,
// is malformed.
func lineComment(s string) int {
	lits := readLiterals(s)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\'':
			if end := lits.end(i); end >= 0 {
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
