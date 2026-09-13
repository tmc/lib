package tileirdisasm

import (
	"bufio"
	"bytes"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var (
	// Matches function header: entry @name(...) or func @name(...)
	funcHeaderRe = regexp.MustCompile(`^(?:entry|func\.func|func)\s+@([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*(.*))?\s*\{?`)
	// Matches basic SSA assignment: %result = opcode ...
	ssaAssignRe  = regexp.MustCompile(`^(%[a-zA-Z0-9_:.]+)\s*=\s*(.*)`)
	// Matches inline loc: loc(#loc1) or loc("file.cu":10:20)
	locRefRe     = regexp.MustCompile(`loc\((#[a-zA-Z0-9_]+)\)`)
	inlineLocRe  = regexp.MustCompile(`loc\("([^"]+)":([0-9]+):([0-9]+)\)`)
	// Matches location definition: #loc1 = ...
	locDefRe     = regexp.MustCompile(`^(#[a-zA-Z0-9_]+)\s*=\s*(.*)`)
)

// Parse reads textual Tile IR and returns the parsed Module.
func Parse(r io.Reader) (*Module, error) {
	mod := &Module{
		Locations: make(map[string]Location),
	}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	var (
		curFn    *Function
		curBlock *Block
		lineNum  int
		depth    int
	)

	// Temporary table for resolving DI definitions
	files := make(map[string]string)         // e.g. "#di_file" -> "/path/to/file.cu"
	subprograms := make(map[string]string)   // e.g. "#di_subprogram" -> "functionName"
	rawLocDefs := make(map[string]string)    // raw RHS of #locX = ...

	for scanner.Scan() {
		lineNum++
		rawLine := scanner.Text()
		trimmed := strings.TrimSpace(rawLine)

		if trimmed == "" {
			continue
		}

		// Comment or Producer info
		if strings.HasPrefix(trimmed, "//") {
			comment := strings.TrimSpace(strings.TrimPrefix(trimmed, "//"))
			if strings.HasPrefix(comment, "Producer:") {
				mod.Producer = strings.TrimSpace(strings.TrimPrefix(comment, "Producer:"))
			}
			mod.Comments = append(mod.Comments, comment)
			continue
		}

		// Location definition lines: #loc = ... or #di_... = ...
		if strings.HasPrefix(trimmed, "#") {
			if m := locDefRe.FindStringSubmatch(trimmed); len(m) > 2 {
				sym := m[1]
				rhs := strings.TrimSpace(m[2])
				rawLocDefs[sym] = rhs

				if strings.Contains(rhs, "#cuda_tile.di_file") {
					files[sym] = parseDIFile(rhs)
				} else if strings.Contains(rhs, "#cuda_tile.di_subprogram") {
					subprograms[sym] = parseDISubprogram(rhs)
				}
			}
			continue
		}

		// Function entry header
		if curFn == nil {
			if m := funcHeaderRe.FindStringSubmatch(trimmed); len(m) > 1 {
				name := m[1]
				paramsRaw := m[2]
				curFn = &Function{
					Name:      name,
					Entry:     strings.HasPrefix(trimmed, "entry"),
					StartLine: lineNum,
				}
				curFn.Params = parseParams(paramsRaw)
				if len(m) > 3 && m[3] != "" {
					curFn.Returns = parseReturns(m[3])
				}
				curBlock = &Block{Name: "entry"}
				curFn.Blocks = append(curFn.Blocks, curBlock)
				if strings.Contains(trimmed, "{") {
					depth++
				}
				continue
			}
		}

		// Track block open/close
		if strings.Contains(trimmed, "{") {
			depth++
		}
		if strings.Contains(trimmed, "}") {
			depth--
			if depth <= 0 && curFn != nil {
				curFn.EndLine = lineNum
				mod.Functions = append(mod.Functions, curFn)
				curFn = nil
				curBlock = nil
				depth = 0
				continue
			}
		}

		// Inside function body: parse operations
		if curFn != nil {
			op := parseOperation(trimmed, lineNum)
			op.Raw = rawLine
			curFn.Operations = append(curFn.Operations, op)
			if curBlock != nil {
				curBlock.Operations = append(curBlock.Operations, op)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// Resolve location symbols across all definitions
	resolveLocations(mod, rawLocDefs, files, subprograms)

	return mod, nil
}

// ParseString parses Tile IR text from a string.
func ParseString(s string) (*Module, error) {
	return Parse(strings.NewReader(s))
}

// ParseBytes parses Tile IR text from a byte slice.
func ParseBytes(b []byte) (*Module, error) {
	return Parse(bytes.NewReader(b))
}

func parseParams(raw string) []*Param {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var params []*Param
	parts := splitTopLevel(raw, ',')
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		colon := strings.Index(p, ":")
		if colon >= 0 {
			params = append(params, &Param{
				Name: strings.TrimSpace(p[:colon]),
				Type: strings.TrimSpace(p[colon+1:]),
			})
		} else {
			params = append(params, &Param{Name: p})
		}
	}
	return params
}

func parseReturns(raw string) []string {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimSuffix(raw, "{")
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "(")
	raw = strings.TrimSuffix(raw, ")")
	var rets []string
	for _, r := range splitTopLevel(raw, ',') {
		r = strings.TrimSpace(r)
		if r != "" {
			rets = append(rets, r)
		}
	}
	return rets
}

func parseOperation(stmt string, line int) *Operation {
	op := &Operation{Line: line}

	// Extract loc(...) if present at end of statement
	if idx := strings.LastIndex(stmt, " loc("); idx >= 0 {
		locPart := stmt[idx+1:]
		if m := locRefRe.FindStringSubmatch(locPart); len(m) > 1 {
			op.LocRef = m[1]
		} else if m := inlineLocRe.FindStringSubmatch(locPart); len(m) > 3 {
			l, _ := strconv.Atoi(m[2])
			c, _ := strconv.Atoi(m[3])
			op.Loc = Location{File: m[1], Line: l, Column: c}
		}
		stmt = strings.TrimSpace(stmt[:idx])
	}

	// Check if this is an assignment: %res = ...
	if m := ssaAssignRe.FindStringSubmatch(stmt); len(m) > 2 {
		op.Results = []string{m[1]}
		stmt = strings.TrimSpace(m[2])
	}

	// Extract type suffix after " : " (careful with attributes like <i32: 16>)
	if idx := findTopLevelColon(stmt); idx >= 0 {
		op.Type = strings.TrimSpace(stmt[idx+1:])
		stmt = strings.TrimSpace(stmt[:idx])
	}

	// Opcode is the first token of the remaining statement
	fields := strings.Fields(stmt)
	if len(fields) > 0 {
		op.OpCode = fields[0]
		if len(fields) > 1 {
			// Remaining text holds operands and attributes
			rest := strings.TrimSpace(stmt[len(op.OpCode):])
			op.Operands, op.Attributes = extractOperandsAndAttributes(rest)
		}
	}

	return op
}

func extractOperandsAndAttributes(s string) (operands []string, attributes string) {
	for _, tok := range strings.Split(s, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if strings.HasPrefix(tok, "%") {
			operands = append(operands, tok)
		} else if strings.HasPrefix(tok, "<") || strings.HasPrefix(tok, "{") {
			if attributes == "" {
				attributes = tok
			} else {
				attributes += ", " + tok
			}
		} else {
			// Sub-tokens inside operands
			for _, sub := range strings.Fields(tok) {
				if strings.HasPrefix(sub, "%") {
					operands = append(operands, sub)
				}
			}
		}
	}
	return operands, attributes
}

func findTopLevelColon(s string) int {
	angle, brace, paren := 0, 0, 0
	for i := len(s) - 1; i >= 0; i-- {
		switch s[i] {
		case '>':
			angle++
		case '<':
			if angle > 0 {
				angle--
			}
		case '}':
			brace++
		case '{':
			if brace > 0 {
				brace--
			}
		case ')':
			paren++
		case '(':
			if paren > 0 {
				paren--
			}
		case ':':
			if angle == 0 && brace == 0 && paren == 0 {
				// Avoid colon inside namespace or label
				if i > 0 && s[i-1] == ':' {
					continue
				}
				if i < len(s)-1 && s[i+1] == ':' {
					continue
				}
				return i
			}
		}
	}
	return -1
}

func splitTopLevel(s string, delim rune) []string {
	var parts []string
	angle, brace, paren := 0, 0, 0
	start := 0
	for i, r := range s {
		switch r {
		case '<':
			angle++
		case '>':
			if angle > 0 {
				angle--
			}
		case '{':
			brace++
		case '}':
			if brace > 0 {
				brace--
			}
		case '(':
			paren++
		case ')':
			if paren > 0 {
				paren--
			}
		default:
			if r == delim && angle == 0 && brace == 0 && paren == 0 {
				parts = append(parts, s[start:i])
				start = i + 1
			}
		}
	}
	if start < len(s) {
		parts = append(parts, s[start:])
	}
	return parts
}

func parseDIFile(s string) string {
	// #cuda_tile.di_file<"file.cu" in "/path/">
	idx := strings.Index(s, "<\"")
	if idx < 0 {
		return ""
	}
	content := s[idx+2:]
	quoteEnd := strings.Index(content, "\"")
	if quoteEnd < 0 {
		return ""
	}
	filename := content[:quoteEnd]
	inIdx := strings.Index(content, "in \"")
	dir := ""
	if inIdx >= 0 {
		dirContent := content[inIdx+4:]
		if dirEnd := strings.Index(dirContent, "\""); dirEnd >= 0 {
			dir = dirContent[:dirEnd]
		}
	}
	if dir != "" {
		if strings.HasSuffix(dir, "/") {
			return dir + filename
		}
		return dir + "/" + filename
	}
	return filename
}

func parseDISubprogram(s string) string {
	// name = "matrix"
	idx := strings.Index(s, "name = \"")
	if idx < 0 {
		return ""
	}
	content := s[idx+8:]
	end := strings.Index(content, "\"")
	if end >= 0 {
		return content[:end]
	}
	return ""
}

func resolveLocations(mod *Module, rawDefs, files, subprograms map[string]string) {
	// Pass 1: resolve base loc(...) definitions: #loc = loc("foo.cu":10:5)
	for sym, rhs := range rawDefs {
		if m := inlineLocRe.FindStringSubmatch(rhs); len(m) > 3 {
			l, _ := strconv.Atoi(m[2])
			c, _ := strconv.Atoi(m[3])
			mod.Locations[sym] = Location{File: m[1], Line: l, Column: c}
		}
	}

	// Pass 2: resolve fused di_loc definitions: #loc30 = #cuda_tile.di_loc<#loc in #di_subprogram>
	diLocRe := regexp.MustCompile(`#cuda_tile\.di_loc<(#[a-zA-Z0-9_]+)\s+in\s+(#[a-zA-Z0-9_]+)>`)
	for sym, rhs := range rawDefs {
		if m := diLocRe.FindStringSubmatch(rhs); len(m) > 2 {
			baseSym := m[1]
			subSym := m[2]
			baseLoc := mod.Locations[baseSym]
			baseLoc.Scope = subprograms[subSym]
			mod.Locations[sym] = baseLoc
		}
	}

	// Pass 3: Attach resolved locations to operations and functions
	for _, fn := range mod.Functions {
		for _, op := range fn.Operations {
			if op.LocRef != "" && op.Loc.File == "" {
				if loc, ok := mod.Locations[op.LocRef]; ok {
					op.Loc = loc
				}
			}
		}
	}
}
