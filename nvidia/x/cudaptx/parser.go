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
// delimits a vector operand list, and operandBrace resolves which.

// newlineTerminated lists the directives written without a terminator that may
// appear at module scope or inside a function body.
var newlineTerminated = []string{
	".version", ".target", ".address_size", ".file", ".loc", ".loc_intermediate",
	".nv_intermediate_source_section",
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
func Parse(r io.Reader) (*Module, error) {
	m := &Module{Files: make(map[int]string)}

	var (
		curKernel      *Kernel
		pendingTuning  Tuning
		depth          int
		curLoc             Location
		curIntermediateLoc Location
		curInlinedAt       []Location
		inBlockComment     bool
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

		// Any other directive is not an instruction. Without this guard
		// ".reg .f32 %f<4>;" parses as an instruction with opcode ".reg".
		if strings.HasPrefix(s, ".") {
			return
		}

		// Labels: an identifier followed by ":" (docs/ptx/04-syntax.md:39-40).
		if strings.HasSuffix(s, ":") && !strings.ContainsAny(s, " \t") {
			return
		}

		if s == "{" || s == "}" {
			return
		}

		if curKernel != nil {
			inst := parseInstructionStatement(s, line, curLoc, curInlinedAt)
			inst.IntermediateLoc = curIntermediateLoc
			if inst.Loc.FileIndex == 0 && curIntermediateLoc.FileIndex != 0 {
				inst.Loc = curIntermediateLoc
			}
			if inst.Opcode != "" {
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
		line, stillInComment := stripComments(scanner.Text(), inBlockComment)
		inBlockComment = stillInComment
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}

		// cpp output: docs/ptx/04-syntax.md:15-18.
		if len(stmt) == 0 && strings.HasPrefix(trimmed, "#") {
			continue
		}

		// Tuning directives are intercepted before accumulation: they sit
		// between a header's ")" and its "{", so appending them would make
		// them part of the header statement.
		if _, ok := hasAnyPrefix(trimmed, tuningDirectives); ok {
			target := &pendingTuning
			if curKernel != nil && len(stmt) == 0 {
				target = &curKernel.Tuning
			}
			parseTuningDirective(trimmed, target)
			continue
		}

		// Newline-terminated directives, recognized only at a statement
		// boundary so that they cannot be confused with operand text.
		if len(stmt) == 0 {
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
					curLoc, curInlinedAt = parseLocDirective(trimmed)
				}
				continue
			}
		}

		// Scan for statement boundaries. seg marks the start of the text not
		// yet handed to add; operandDepth counts the brace groups that belong
		// to an operand list rather than to the block structure. A group and
		// its statement always close on one line (see operandBrace), so the
		// count never has to survive to the next one.
		seg, operandDepth := 0, 0
		for i := 0; i < len(trimmed); i++ {
			switch trimmed[i] {
			case ';':
				add(trimmed[seg : i+1])
				seg = i + 1
				process(flush(), stmtLine)

			case '{':
				if operandDepth > 0 || operandBrace(trimmed, i) {
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
				tok := strings.TrimSpace(trimmed[seg:i])
				if len(stmt) > 0 || tok == "" || strings.ContainsAny(tok, " \t.") {
					continue
				}
				add(trimmed[seg : i+1])
				seg = i + 1
				process(flush(), stmtLine)
			}
		}
		add(trimmed[seg:])
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("cudaptx: read: %w", err)
	}

	// .file directives may follow every .loc that references them: nvcc emits
	// the file table at the end of the module (testdata/lexer.sm90.ptx:240), so
	// filenames can only be resolved once the whole module has been read.
	m.resolveFiles()
	return m, nil
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
// is still open at end of line.
func stripComments(line string, inBlock bool) (string, bool) {
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
		if strings.HasPrefix(line[i:], "//") {
			break
		}
		if strings.HasPrefix(line[i:], "/*") {
			i += 2
			inBlock = true
			continue
		}
		b.WriteByte(line[i])
		i++
	}
	return b.String(), inBlock
}

// operandBrace reports whether the "{" at s[i] opens a vector operand list
// rather than a block. An operand group closes on the same line and its
// statement is terminated after it -- "ld.global.v4.f32 {%f1, %f2, %f3, %f4},
// [%rd5];" (testdata/vector.sm90.ptx:45) -- whereas a block-opening "{" is the
// last thing in its own statement, with no ";" following it.
func operandBrace(s string, i int) bool {
	depth := 0
	for ; i < len(s); i++ {
		switch s[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				j := strings.IndexAny(s[i+1:], ";{")
				return j >= 0 && s[i+1+j] == ';'
			}
		}
	}
	return false
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
	name := strings.Join(fields[1:], " ")
	if i := strings.Index(name, `"`); i >= 0 {
		if j := strings.Index(name[i+1:], `"`); j >= 0 {
			m.Files[idx] = name[i+1 : i+1+j]
			return
		}
	}
	m.Files[idx] = strings.TrimSuffix(name, ",")
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

	loc := parseLocTriple(strings.Fields(parts[0]))
	loc.Intermediate = intermediate
	var inlinedAt []Location
	for _, p := range parts[1:] {
		p = strings.TrimSpace(p)
		if !strings.HasPrefix(p, "inlined_at") {
			continue
		}
		inlinedLoc := parseLocTriple(strings.Fields(strings.TrimPrefix(p, "inlined_at")))
		inlinedLoc.Intermediate = intermediate
		inlinedAt = append(inlinedAt, inlinedLoc)
	}
	return loc, inlinedAt
}

func parseLocTriple(f []string) Location {
	var l Location
	if len(f) > 0 {
		l.FileIndex, _ = strconv.Atoi(f[0])
	}
	if len(f) > 1 {
		l.Line, _ = strconv.Atoi(f[1])
	}
	if len(f) > 2 {
		l.Column, _ = strconv.Atoi(f[2])
	}
	return l
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
	for _, kw := range []string{".entry", ".func"} {
		if stmt == kw || strings.HasPrefix(stmt, kw+" ") || strings.HasPrefix(stmt, kw+"\t") ||
			strings.Contains(stmt, " "+kw+" ") || strings.HasSuffix(stmt, " "+kw) {
			return true
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

// parseInstructionStatement parses a guarded instruction statement
// (docs/ptx/04-syntax.md:78-85).
func parseInstructionStatement(stmt string, line int, loc Location, inlinedAt []Location) Instruction {
	raw := stmt
	stmt = strings.TrimSuffix(strings.TrimSpace(stmt), ";")

	var pred string
	if strings.HasPrefix(stmt, "@") {
		if i := strings.IndexAny(stmt, " \t"); i >= 0 {
			pred, stmt = stmt[:i], strings.TrimSpace(stmt[i:])
		} else {
			pred, stmt = stmt, ""
		}
	}

	opcode := stmt
	operands := ""
	if i := strings.IndexAny(stmt, " \t"); i >= 0 {
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
