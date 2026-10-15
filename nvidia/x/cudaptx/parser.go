package cudaptx

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Statement termination in PTX is not uniform. Statements "begin with an
// optional label and end with a semicolon" (docs/ptx/04-syntax.md:39-40), but
// the module directives and the performance-tuning directives are defined with
// no terminator at all -- ".version  major.minor"
// (docs/ptx/11-directives.md:21-25), and likewise .target, .address_size,
// .file, .loc, .maxntid and friends. Real nvcc output relies on that: see
// testdata/lexer.sm90.ptx. A parser that splits only on ";" swallows the whole
// module preamble and the following kernel header into one statement and finds
// no kernels at all.
//
// So segmentation has three cases: newline-terminated directives, which are
// recognized by prefix before any accumulation; semicolon-terminated statements;
// and function headers, which have no terminator and end at the "{" that opens
// the body (or at the ";" of a bodiless declaration).
//
// Boundaries are found by scanning each line, not by testing how it ends. A
// statement may end anywhere in a line -- ".visible .entry k() { ret; }" holds
// three of them -- and nothing in the grammar puts at most one on a line. The
// only ambiguity a scan introduces is "{": it delimits a block, but it also
// delimits an operand group. The statement prefix distinguishes them.

// newlineTerminated lists the directives written without a terminator that may
// appear at module scope or inside a function body.
var newlineTerminated = []string{
	".version", ".target", ".address_size", ".file", ".loc", ".loc_intermediate",
}

// tuningDirectives lists the performance-tuning directives, which are written
// without a terminator between a function header's ")" and its "{"
// (docs/ptx/11-directives.md:925-1218, 11-directives.md:1830-1920).
var tuningDirectives = []string{
	".maxnreg", ".maxntid", ".reqntid", ".minnctapersm", ".maxnctapersm",
	".reqnctapercluster", ".maxclusterrank", ".explicitcluster",
}

// hasAnyPrefix reports whether s begins with one of prefixes followed by a
// delimiter, so that ".maxnt" does not match ".maxntid".
func hasAnyPrefix(s string, prefixes []string) (string, bool) {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			rest := s[len(p):]
			if rest == "" || rest[0] == ' ' || rest[0] == '\t' {
				return p, true
			}
		}
	}
	return "", false
}

// Parse reads PTX assembly and returns the module it describes.
//
// Labels are unique within a kernel or device function. Direct bra and bra.uni
// instructions, including predicated instructions, must name a label preceding
// an instruction in the same function. Targets not declared as same-function
// labels, including register operands and external symbols, are errors. Other branch forms, including brx.idx, are
// preserved without resolving their targets. Outside this label and direct
// branch contract, Parse remains permissive; it is not a PTX validator.
func Parse(r io.Reader) (*Module, error) {
	m := &Module{Files: make(map[int]string)}

	var (
		curKernel          *Kernel
		pendingTuning      Tuning
		depth              int
		operandDepth       int
		inString           bool
		curLoc             Location
		curIntermediateLoc Location
		curInlinedAt       []Location
		inBlockComment     bool
		inSection          bool // inside .nv_intermediate_source_section
		sectionDepth       int  // brace depth within it; it ends when this returns to 0
		sectionOpened      bool // its "{" has been seen; before that depth 0 is not the end
		inSectionSource    bool // between .source_begin and .source_end
		stmt               []string
		stmtLine           int
		lineNum            int
	)

	// resetLoc drops the sticky source position at a function boundary,
	// matching cudasass.Parse.
	resetLoc := func() {
		curLoc = Location{}
		curIntermediateLoc = Location{}
		curInlinedAt = nil
	}

	flush := func() string {
		s := strings.Join(stmt, " ")
		stmt = stmt[:0]
		return s
	}

	process := func(s string, line int) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}

		// Function and kernel headers: docs/ptx/11-directives.md:403-715.
		if isFunctionHeader(s) {
			k := parseFunctionHeader(s, line)
			if k == nil {
				return
			}
			k.Tuning = pendingTuning
			pendingTuning = Tuning{}
			m.Kernels = append(m.Kernels, k)
			resetLoc()
			if k.Body {
				curKernel = k
			}
			return
		}

		// Labels: an identifier followed by ":" (docs/ptx/04-syntax.md:39-40).
		if strings.HasSuffix(s, ":") && !strings.ContainsAny(s, " \t") {
			if curKernel != nil {
				curKernel.Labels = append(curKernel.Labels, Label{
					Name: strings.TrimSuffix(s, ":"), Line: line,
					Instruction: len(curKernel.Instructions),
				})
			}
			return
		}

		// Any other directive is not an instruction. Without this guard
		// ".reg .f32 %f<4>;" parses as an instruction with opcode ".reg".
		if strings.HasPrefix(s, ".") {
			return
		}

		if s == "{" || s == "}" {
			return
		}

		if curKernel != nil {
			inst := parseInstructionStatement(s, line, curLoc, curInlinedAt)
			inst.IntermediateLoc = curIntermediateLoc
			// A guard followed by a directive or a second guard, as in
			// "@%p1 .reg .b32 %r;", is not an instruction either.
			if inst.Opcode != "" && !strings.HasPrefix(inst.Opcode, ".") && !strings.HasPrefix(inst.Opcode, "@") {
				// .loc_intermediate applies to the immediately following
				// instruction and to at most one of them
				// (docs/ptx/11-directives.md:1810,1816), so it is consumed
				// here rather than left sticky like .loc. It is also kept out
				// of Loc: it names a position in a compiler intermediate file,
				// and the ISA does not merge it into the primary line table
				// (11-directives.md:1819-1821). An instruction with no .loc in
				// scope has no source position, and reporting the .ir path
				// instead would give a consumer a file the SASS tier cannot
				// agree with -- a wrong line, not a missing one.
				curIntermediateLoc = Location{}
				curKernel.Instructions = append(curKernel.Instructions, inst)
			}
		}
	}

	// add appends one segment of statement text, remembering the line the
	// statement started on so that Instruction.Line names its first token.
	add := func(seg string) {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			return
		}
		if len(stmt) == 0 {
			stmtLine = lineNum
		}
		stmt = append(stmt, seg)
	}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		lineNum++

		// A .nv_intermediate_source_section embeds intermediate IR source
		// text, which "may contain arbitrary text including newlines"
		// (docs/ptx/11-directives.md:1878-1879). It is skipped before comment
		// stripping and before the statement scanner: a brace in that text
		// would otherwise be counted as block structure and detach every
		// following kernel, and a "/*" in it would swallow the rest of the
		// module.
		if inSection {
			t := strings.TrimSpace(scanner.Text())
			switch {
			case inSectionSource:
				if t == ".source_end" {
					inSectionSource = false
				}
			case t == ".source_begin":
				inSectionSource = true
			default:
				if strings.Contains(t, "{") {
					sectionOpened = true
				}
				sectionDepth += strings.Count(t, "{") - strings.Count(t, "}")
				if sectionOpened && sectionDepth <= 0 {
					inSection = false
				}
			}
			continue
		}
		if !inString && !inBlockComment && len(stmt) == 0 && strings.HasPrefix(strings.TrimSpace(scanner.Text()), ".nv_intermediate_source_section") {
			inSection = true
			sectionDepth = strings.Count(scanner.Text(), "{") - strings.Count(scanner.Text(), "}")
			// The "{" may sit on the next line; until it is seen, a depth of
			// zero means the block has not opened yet, not that it has closed.
			sectionOpened = sectionDepth > 0
			continue
		}

		line, stillInComment := stripComments(scanner.Text(), inBlockComment, inString)
		inBlockComment = stillInComment
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		// cpp output: docs/ptx/04-syntax.md:15-18.
		if !inString && len(stmt) == 0 && strings.HasPrefix(trimmed, "#") {
			continue
		}

		// Tuning directives are intercepted before accumulation: they sit
		// between a header's ")" and its "{", so appending them would make
		// them part of the header statement.
		if _, ok := hasAnyPrefix(trimmed, tuningDirectives); !inString && ok {
			target := &pendingTuning
			if curKernel != nil && len(stmt) == 0 {
				target = &curKernel.Tuning
			}
			parseTuningDirective(trimmed, target)
			continue
		}

		// Newline-terminated directives, recognized only at a statement
		// boundary so that they cannot be confused with operand text.
		if !inString && len(stmt) == 0 {
			if _, ok := hasAnyPrefix(trimmed, newlineTerminated); ok {
				switch {
				case strings.HasPrefix(trimmed, ".version"):
					m.Version = directiveArg(trimmed, ".version")
				case strings.HasPrefix(trimmed, ".target"):
					m.Target = directiveArg(trimmed, ".target")
				case strings.HasPrefix(trimmed, ".address_size"):
					m.AddressSize, _ = strconv.Atoi(directiveArg(trimmed, ".address_size"))
				case strings.HasPrefix(trimmed, ".file"):
					parseFileDirective(trimmed, m)
				case strings.HasPrefix(trimmed, ".loc_intermediate"):
					curIntermediateLoc, _ = parseLocDirective(trimmed)
				case strings.HasPrefix(trimmed, ".loc"):
					loc, inlinedAt := parseLocDirective(trimmed)
					curLoc, curInlinedAt = linkInlineChain(curLoc, curInlinedAt, loc, inlinedAt)
					// A new source position starts here; an intermediate
					// location that never reached an instruction does not
					// carry across it.
					curIntermediateLoc = Location{}
				}
				continue
			}
		}

		// Scan for statement boundaries. seg marks the start of the text not
		// yet handed to add; operandDepth counts the brace groups that belong
		// to an operand list rather than to the block structure. Quotes and
		// operand groups may continue on the next line; their contents cannot
		// declare labels or terminate the enclosing statement.
		seg := 0
		for i := 0; i < len(trimmed); i++ {
			if inString {
				switch trimmed[i] {
				case '\\':
					if i+1 < len(trimmed) {
						i++
					}
				case '"':
					inString = false
				}
				continue
			}
			if trimmed[i] == '"' {
				inString = true
				continue
			}
			switch trimmed[i] {
			case ';':
				if operandDepth > 0 {
					continue
				}
				add(trimmed[seg : i+1])
				seg = i + 1
				process(flush(), stmtLine)

			case '{':
				if operandDepth > 0 {
					operandDepth++
					continue
				}
				prefix := strings.Join(stmt, " ") + " " + trimmed[seg:i]
				if operandBrace(prefix) {
					operandDepth++
					continue
				}
				add(trimmed[seg : i+1])
				seg = i + 1
				process(flush(), stmtLine)
				depth++

			case '}':
				if operandDepth > 0 {
					operandDepth--
					continue
				}
				// A block close ends the block, and discards any
				// unterminated text accumulated before it.
				seg = i + 1
				stmt = stmt[:0]
				if depth > 0 {
					depth--
				}
				if depth == 0 && curKernel != nil {
					curKernel.EndLine = lineNum
					curKernel = nil
					resetLoc()
				}

			case ':':
				if operandDepth > 0 {
					continue
				}
				// A label is a lone identifier at the start of a
				// statement (docs/ptx/04-syntax.md:39-40); a ":" anywhere
				// else belongs to an operand or double-colon qualifier (e.g. .cta_group::1).
				if i+1 < len(trimmed) && trimmed[i+1] == ':' {
					i++
					continue
				}
				if i > 0 && trimmed[i-1] == ':' {
					continue
				}
				tok := strings.TrimSpace(strings.Join(stmt, " ") + " " + trimmed[seg:i])
				if strings.ContainsAny(tok, " \t") {
					continue
				}
				labelLine := lineNum
				if len(stmt) > 0 {
					labelLine = stmtLine
				}
				stmt = stmt[:0]
				seg = i + 1
				process(tok+":", labelLine)
			}
		}
		add(trimmed[seg:])
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("cudaptx: read: %w", err)
	}
	for _, k := range m.Kernels {
		if err := validateBranches(k); err != nil {
			return nil, err
		}
	}

	// .file directives may follow every .loc that references them: nvcc emits
	// the file table at the end of the module (testdata/lexer.sm90.ptx:240), so
	// filenames can only be resolved once the whole module has been read.
	m.resolveFiles()
	return m, nil
}

func validateBranches(k *Kernel) error {
	labels := make(map[string]Label, len(k.Labels))
	for _, label := range k.Labels {
		if !labelIdentifier(label.Name) {
			return fmt.Errorf("cudaptx: %s line %d: invalid label %q", k.Name, label.Line, label.Name)
		}
		if prev, ok := labels[label.Name]; ok {
			return fmt.Errorf("cudaptx: %s line %d: duplicate label %q (first at line %d)", k.Name, label.Line, label.Name, prev.Line)
		}
		labels[label.Name] = label
	}
	for _, inst := range k.Instructions {
		if inst.Opcode != "bra" && inst.Opcode != "bra.uni" {
			continue
		}
		if !labelIdentifier(inst.Operands) {
			return fmt.Errorf("cudaptx: %s line %d: %s requires one direct label target, got %q", k.Name, inst.Line, inst.Opcode, inst.Operands)
		}
		label, ok := labels[inst.Operands]
		if !ok {
			return fmt.Errorf("cudaptx: %s line %d: unresolved branch target %q", k.Name, inst.Line, inst.Operands)
		}
		if label.Instruction == len(k.Instructions) {
			return fmt.Errorf("cudaptx: %s line %d: branch target %q has no instruction", k.Name, inst.Line, inst.Operands)
		}
	}
	return nil
}

func labelIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '_' || c == '$' || i == 0 && c == '%' || 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || i > 0 && '0' <= c && c <= '9' {
			continue
		}
		return false
	}
	return true
}

// ParseString parses PTX assembly held in a string.
func ParseString(ptx string) (*Module, error) {
	return Parse(strings.NewReader(ptx))
}

// ParseBytes parses PTX assembly held in a byte slice.
func ParseBytes(b []byte) (*Module, error) {
	return Parse(bytes.NewReader(b))
}

func (m *Module) resolveFiles() {
	fix := func(l *Location) {
		if name, ok := m.Files[l.FileIndex]; ok {
			l.File = name
		}
	}
	for _, k := range m.Kernels {
		for i := range k.Instructions {
			inst := &k.Instructions[i]
			fix(&inst.Loc)
			fix(&inst.IntermediateLoc)
			for j := range inst.InlinedAt {
				fix(&inst.InlinedAt[j])
			}
		}
	}
}

// stripComments removes non-nested /* */ and // comments, which PTX treats as
// whitespace (docs/ptx/04-syntax.md:28-35). It reports whether a block comment
// is still open at end of line. A comment marker inside a string literal, as
// in .file 1 "/a//b.cu", is part of the string.
func stripComments(line string, inBlock, inString bool) (string, bool) {
	var b strings.Builder
	for i := 0; i < len(line); {
		if inBlock {
			if j := strings.Index(line[i:], "*/"); j >= 0 {
				i += j + 2
				inBlock = false
				b.WriteByte(' ')
				continue
			}
			return b.String(), true
		}
		if inString {
			switch line[i] {
			case '\\':
				if i+1 < len(line) {
					b.WriteByte(line[i])
					i++
				}
			case '"':
				inString = false
			}
			b.WriteByte(line[i])
			i++
			continue
		}
		if strings.HasPrefix(line[i:], "//") {
			break
		}
		if strings.HasPrefix(line[i:], "/*") {
			i += 2
			inBlock = true
			continue
		}
		if line[i] == '"' {
			inString = true
		}
		b.WriteByte(line[i])
		i++
	}
	return b.String(), inBlock
}

// operandBrace reports whether a brace follows an instruction or declaration
// prefix. A brace at a statement boundary, or after a function header, opens
// a block. Later statements after its closing brace do not change that role.
func operandBrace(prefix string) bool {
	return strings.TrimSpace(prefix) != "" && !isFunctionHeader(prefix)
}

func directiveArg(stmt, directive string) string {
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(strings.TrimPrefix(stmt, directive)), ";"))
}

// parseFileDirective records a ".file index "name"" entry
// (docs/ptx/11-directives.md:1630-1685). The optional timestamp and file size
// are ignored.
func parseFileDirective(stmt string, m *Module) {
	rest := directiveArg(stmt, ".file")
	fields := strings.Fields(rest)
	if len(fields) < 2 {
		return
	}
	idx, err := strconv.Atoi(fields[0])
	if err != nil {
		return
	}
	name := strings.TrimSpace(rest[len(fields[0]):])
	if i := strings.Index(name, `"`); i >= 0 {
		// A name without its closing quote is not a name; recording
		// the text after the quote would report a file that does not
		// exist.
		if j := strings.Index(name[i+1:], `"`); j >= 0 {
			m.Files[idx] = name[i+1 : i+1+j]
		}
		return
	}
	m.Files[idx] = strings.TrimSuffix(strings.Join(strings.Fields(name), " "), ",")
}

// parseLocDirective parses ".loc file line column" or ".loc_intermediate file line column"
// and their optional function_name and inlined_at suffixes (docs/ptx/11-directives.md:1686-1791).
// Filenames are left empty; Module.resolveFiles fills them in once the file
// table is known.
func parseLocDirective(stmt string) (Location, []Location) {
	directive := ".loc"
	intermediate := false
	if strings.HasPrefix(stmt, ".loc_intermediate") {
		directive = ".loc_intermediate"
		intermediate = true
	}
	rest := directiveArg(stmt, directive)
	parts := strings.Split(rest, ",")

	loc, ok := parseLocTriple(strings.Fields(parts[0]))
	if !ok {
		// A location that does not parse is unknown, not a location on line 0
		// of a real file: reporting the latter gives a consumer a source
		// position nothing produced.
		return Location{}, nil
	}
	loc.Intermediate = intermediate
	var inlinedAt []Location
	for _, p := range parts[1:] {
		p = strings.TrimSpace(p)
		switch {
		case strings.HasPrefix(p, "inlined_at"):
			inlinedLoc, ok := parseLocTriple(strings.Fields(strings.TrimPrefix(p, "inlined_at")))
			if !ok {
				continue
			}
			inlinedLoc.Intermediate = intermediate
			inlinedAt = append(inlinedAt, inlinedLoc)
		case strings.HasPrefix(p, "function_name"):
			// "function_name label" or "function_name label + immediate";
			// keep it as written, spaces squeezed.
			loc.FunctionName = strings.Join(strings.Fields(strings.TrimPrefix(p, "function_name")), "")
		}
	}
	return loc, inlinedAt
}

// parseLocTriple parses "file_index line_number column_position". The file
// index and line number are required; the column is optional, and a column of 0
// is a value real toolchains write (docs/ptx/11-directives.md:1735), not a
// failure. ok is false if the required fields are missing or unparsable, and
// the caller must then report no location rather than a partial one.
func parseLocTriple(f []string) (Location, bool) {
	if len(f) < 2 {
		return Location{}, false
	}
	var l Location
	var err error
	if l.FileIndex, err = strconv.Atoi(f[0]); err != nil || l.FileIndex <= 0 {
		return Location{}, false
	}
	if l.Line, err = strconv.Atoi(f[1]); err != nil || l.Line < 0 {
		return Location{}, false
	}
	if len(f) > 2 {
		if l.Column, err = strconv.Atoi(f[2]); err != nil {
			return Location{}, false
		}
	}
	return l, true
}

func parseTuningDirective(stmt string, t *Tuning) {
	name, ok := hasAnyPrefix(stmt, tuningDirectives)
	if !ok {
		return
	}
	arg := directiveArg(stmt, name)
	switch name {
	case ".maxnreg":
		t.MaxNReg, _ = strconv.Atoi(arg)
	case ".maxntid":
		t.MaxNTID = parseTuple3(arg)
	case ".reqntid":
		t.ReqNTID = parseTuple3(arg)
	case ".minnctapersm":
		t.MinNCTAPerSM, _ = strconv.Atoi(arg)
	case ".maxnctapersm":
		t.MaxNCTAPerSM, _ = strconv.Atoi(arg)
	case ".reqnctapercluster":
		t.ReqNCTAPerCluster = parseTuple3(arg)
	case ".maxclusterrank":
		t.MaxClusterRank, _ = strconv.Atoi(arg)
	case ".explicitcluster":
		t.ExplicitCluster = true
	}
}

// parseTuple3 parses the "nx, ny, nz" argument of .maxntid and .reqntid. The
// directives accept one, two or three dimensions; omitted dimensions are 1
// (docs/ptx/11-directives.md:1028-1032).
func parseTuple3(s string) [3]int {
	res := [3]int{1, 1, 1}
	for i, p := range strings.Split(s, ",") {
		if i >= 3 {
			break
		}
		res[i], _ = strconv.Atoi(strings.TrimSpace(p))
	}
	return res
}

func isFunctionHeader(stmt string) bool {
	for _, field := range strings.Fields(stmt) {
		switch field {
		case ".visible", ".extern", ".weak", ".common":
			continue
		case ".entry", ".func":
			return true
		default:
			return false
		}
	}
	return false
}

// parseFunctionHeader parses a .entry or .func header. The grammar is
//
//	linkage .entry name ( params ) { ... }
//	linkage .func ( returns ) name ( params ) { ... }
//
// (docs/ptx/11-directives.md:403-715); a declaration without a body ends in ";"
// instead of a block.
func parseFunctionHeader(stmt string, line int) *Kernel {
	k := &Kernel{StartLine: line, Body: strings.HasSuffix(stmt, "{")}

	for _, l := range []string{"visible", "extern", "weak", "common"} {
		if strings.Contains(stmt, "."+l+" ") {
			k.Linkage = l
			break
		}
	}

	kw := ".entry"
	if strings.Contains(stmt, ".entry") {
		k.Entry = true
	} else if strings.Contains(stmt, ".func") {
		kw = ".func"
	} else {
		return nil
	}
	rest := strings.TrimSpace(stmt[strings.Index(stmt, kw)+len(kw):])

	// A ".func" may declare return parameters before the name.
	if strings.HasPrefix(rest, "(") {
		inner, after, ok := splitParens(rest)
		if !ok {
			return nil
		}
		k.Returns = parseParameters(inner)
		rest = strings.TrimSpace(after)
	}

	open := strings.Index(rest, "(")
	if open < 0 {
		fields := strings.Fields(strings.TrimSuffix(strings.TrimSpace(rest), ";"))
		if len(fields) == 0 {
			return nil
		}
		k.Name = fields[0]
		return k
	}
	k.Name = strings.TrimSpace(rest[:open])
	inner, _, ok := splitParens(rest[open:])
	if !ok {
		return nil
	}
	k.Params = parseParameters(inner)
	return k
}

// splitParens returns the text inside the leading parenthesized group of s and
// the text that follows it.
func splitParens(s string) (inner, after string, ok bool) {
	if len(s) == 0 || s[0] != '(' {
		return "", s, false
	}
	depth := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return s[1:i], s[i+1:], true
			}
		}
	}
	return "", s, false
}

// parseParameters parses a comma-separated parameter list such as
//
//	.param .u64 guarded_param_0, .param .align 4 .b8 byvalue_param_0[36]
func parseParameters(list string) []Param {
	var params []Param
	for _, entry := range strings.Split(list, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		// An array extent is written "[36]"; a split on "," cannot land
		// inside one, so each entry is self-contained.
		var p Param
		fields := strings.Fields(entry)
		for i, f := range fields {
			switch {
			case f == ".param", f == ".reg":
				p.Space = f[1:]
			case f == ".align":
				if i+1 < len(fields) {
					p.Align, _ = strconv.Atoi(fields[i+1])
				}
			case strings.HasPrefix(f, "."):
				if p.Type == "" {
					p.Type = f[1:]
				}
			case i > 0 && fields[i-1] == ".align":
				// consumed by the .align case above
			default:
				if p.Name != "" {
					continue
				}
				if b := strings.Index(f, "["); b >= 0 {
					p.Name = f[:b]
					p.Size, _ = strconv.Atoi(strings.TrimSuffix(f[b+1:], "]"))
				} else {
					p.Name = f
				}
			}
		}
		if p.Name == "" && p.Type == "" {
			continue
		}
		params = append(params, p)
	}
	return params
}

// space is the whitespace that separates a guard, an opcode and its operands.
// A CRLF file leaves a carriage return inside a line, as in "ret\r;".
const space = " \t\r\v\f"

// parseInstructionStatement parses a guarded instruction statement
// (docs/ptx/04-syntax.md:78-85).
func parseInstructionStatement(stmt string, line int, loc Location, inlinedAt []Location) Instruction {
	raw := stmt
	stmt = strings.TrimSuffix(strings.TrimSpace(stmt), ";")

	var pred string
	if strings.HasPrefix(stmt, "@") {
		if i := strings.IndexAny(stmt, space); i >= 0 {
			pred, stmt = stmt[:i], strings.TrimSpace(stmt[i:])
		} else {
			pred, stmt = stmt, ""
		}
	}

	opcode := stmt
	operands := ""
	if i := strings.IndexAny(stmt, space); i >= 0 {
		opcode, operands = stmt[:i], strings.TrimSpace(stmt[i:])
	}

	mnemonic := opcode
	if i := strings.Index(opcode, "."); i > 0 {
		mnemonic = opcode[:i]
	}

	inst := Instruction{
		Line:      line,
		Predicate: pred,
		Opcode:    opcode,
		Mnemonic:  mnemonic,
		Operands:  operands,
		Loc:       loc,
		Raw:       raw,
	}
	if len(inlinedAt) > 0 {
		inst.InlinedAt = append([]Location(nil), inlinedAt...)
	}
	return inst
}

// linkInlineChain returns the location and inline chain in effect after a .loc
// directive parsed as (loc, inlinedAt), given the ones the preceding .loc left
// in effect.
//
// A nested inline is written as successive .loc directives, each naming in its
// inlined_at the location of the directive before it
// (docs/ptx/11-directives.md:1739-1748). Reading each directive in isolation
// keeps only the innermost call site and drops every outer frame, so a two-deep
// inline is reported as one deep.
//
// The frames are linked only when the inlined_at actually names the preceding
// location. Without that match there is no evidence the two directives belong
// to one chain, and only the frame written is reported.
func linkInlineChain(prevLoc Location, prevChain []Location, loc Location, inlinedAt []Location) (Location, []Location) {
	if len(inlinedAt) == 0 || prevLoc.FileIndex == 0 {
		return loc, inlinedAt
	}
	outermost := inlinedAt[len(inlinedAt)-1]
	if outermost.FileIndex != prevLoc.FileIndex || outermost.Line != prevLoc.Line || outermost.Column != prevLoc.Column {
		return loc, inlinedAt
	}
	return loc, append(inlinedAt, prevChain...)
}
