package tileir

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var (
	moduleRegex  = regexp.MustCompile(`cuda_tile\.module\s+@([a-zA-Z0-9_]+)`)
	funcHeaderRe = regexp.MustCompile(`^(?:cuda_tile\.)?(entry|func)\s+@([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*(.*?))?\s*(?:loc\(.*?\))?\s*\{?`)
	locDefRe     = regexp.MustCompile(`^#([a-zA-Z0-9_]+)\s*=\s*(.*)`)
	locRefRe     = regexp.MustCompile(`loc\(#([a-zA-Z0-9_]+)\)`)
	inlineLocRe  = regexp.MustCompile(`loc\("([^"]+)"(?::(\d+)(?::(\d+))?)?\)`)
	fusedLocRe   = regexp.MustCompile(`"([^"]+)"(?::(\d+)(?::(\d+))?)?`)
	ssaAssignRe  = regexp.MustCompile(`^((?:%[a-zA-Z0-9_]+(?::\d+)?(?:\s*,\s*)?)+)\s*=\s*(.*)`)
)

// Parse parses textual Tile IR from an io.Reader into an AST Module.
func Parse(r io.Reader) (*Module, error) {
	scanner := bufio.NewScanner(r)
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading tileir source: %w", err)
	}

	return parseLines(lines)
}

// ParseString parses textual Tile IR from a string.
func ParseString(s string) (*Module, error) {
	return Parse(strings.NewReader(s))
}

// ParseBytes parses textual Tile IR from a byte slice.
func ParseBytes(b []byte) (*Module, error) {
	return Parse(bytes.NewReader(b))
}

func parseLines(lines []string) (*Module, error) {
	mod := &Module{
		Locations: make(map[string]Location),
	}

	rawLocDefs := make(map[string]string)
	files := make(map[string]string)
	subprograms := make(map[string]string)

	var curFn *Function
	var curBlock *Block
	depth := 0

	for lineIdx, rawLine := range lines {
		lineNum := lineIdx + 1
		trimmed := strings.TrimSpace(rawLine)

		if trimmed == "" {
			continue
		}

		// Comment line
		if strings.HasPrefix(trimmed, "//") {
			comment := strings.TrimSpace(strings.TrimPrefix(trimmed, "//"))
			if strings.HasPrefix(comment, "Producer:") {
				mod.Producer = strings.TrimSpace(strings.TrimPrefix(comment, "Producer:"))
			}
			mod.Comments = append(mod.Comments, comment)
			continue
		}

		// Module definition
		if m := moduleRegex.FindStringSubmatch(trimmed); len(m) == 2 {
			mod.Name = m[1]
			continue
		}

		// Location definition lines: #loc = ...
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
			if m := funcHeaderRe.FindStringSubmatch(trimmed); len(m) > 2 {
				kind := m[1]
				name := m[2]
				paramsRaw := m[3]
				retRaw := ""
				if len(m) > 4 {
					retRaw = m[4]
				}

				curFn = &Function{
					Name:      name,
					Entry:     kind == "entry",
					StartLine: lineNum,
				}
				curFn.Params = parseParams(paramsRaw)
				if retRaw != "" {
					curFn.Returns = parseReturns(retRaw)
				}

				if locRef := locRefRe.FindStringSubmatch(trimmed); len(locRef) > 1 {
					curFn.LocRef = locRef[1]
					curFn.Loc = mod.Locations[locRef[1]]
				}

				curBlock = &Block{Name: "^bb0"}
				curFn.Blocks = append(curFn.Blocks, curBlock)
				if strings.Contains(trimmed, "{") {
					depth++
				}
				continue
			}
		}

		// Block labels
		if strings.HasPrefix(trimmed, "^") && strings.HasSuffix(trimmed, ":") {
			if curFn != nil {
				curBlock = &Block{Name: strings.TrimSuffix(trimmed, ":")}
				curFn.Blocks = append(curFn.Blocks, curBlock)
			}
			continue
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

	// Resolve location symbols across all definitions
	resolveLocations(mod, rawLocDefs, files, subprograms)

	return mod, nil
}

func parseParams(raw string) []*Param {
	raw = strings.TrimSpace(raw)
	if raw == "" {
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
			name := strings.TrimSpace(p[:colon])
			typStr := strings.TrimSpace(p[colon+1:])
			pm := &Param{
				Name: name,
				Type: typStr,
			}
			if t, err := ParseType(typStr); err == nil {
				pm.TypeVal = t
			}
			params = append(params, pm)
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
		resPart := strings.TrimSpace(m[1])
		stmt = strings.TrimSpace(m[2])
		for _, p := range strings.Split(resPart, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				op.Results = append(op.Results, p)
			}
		}
	}

	// Extract type suffix after " : "
	if idx := findTopLevelColon(stmt); idx >= 0 {
		op.Type = strings.TrimSpace(stmt[idx+1:])
		if typ, err := ParseType(op.Type); err == nil {
			op.TypeVal = typ
		}
		stmt = strings.TrimSpace(stmt[:idx])
	}

	// Opcode is the first token of the remaining statement
	fields := strings.Fields(stmt)
	if len(fields) > 0 {
		op.OpCode = fields[0]
		if len(fields) > 1 {
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
	for sym, rhs := range rawDefs {
		if m := inlineLocRe.FindStringSubmatch(rhs); len(m) > 3 {
			l, _ := strconv.Atoi(m[2])
			c, _ := strconv.Atoi(m[3])
			mod.Locations[sym] = Location{File: m[1], Line: l, Column: c}
		} else if strings.HasPrefix(rhs, "loc(") && strings.HasSuffix(rhs, ")") {
			inner := rhs[4 : len(rhs)-1]
			if m := fusedLocRe.FindStringSubmatch(inner); len(m) >= 2 {
				loc := Location{File: m[1]}
				if len(m) >= 3 && m[2] != "" {
					l, _ := strconv.Atoi(m[2])
					loc.Line = l
				}
				if len(m) >= 4 && m[3] != "" {
					c, _ := strconv.Atoi(m[3])
					loc.Column = c
				}
				mod.Locations[sym] = loc
			}
		}
	}

	diLocRe := regexp.MustCompile(`#cuda_tile\.di_loc<(#[a-zA-Z0-9_]+)\s+in\s+(#[a-zA-Z0-9_]+)>`)
	for sym, rhs := range rawDefs {
		if m := diLocRe.FindStringSubmatch(rhs); len(m) > 2 {
			baseSym := strings.TrimPrefix(m[1], "#")
			subSym := strings.TrimPrefix(m[2], "#")
			baseLoc := mod.Locations[baseSym]
			baseLoc.Scope = subprograms[subSym]
			mod.Locations[sym] = baseLoc
		}
	}

	for _, fn := range mod.Functions {
		if fn.LocRef != "" && fn.Loc.File == "" {
			if loc, ok := mod.Locations[fn.LocRef]; ok {
				fn.Loc = loc
			}
		}
		for _, op := range fn.Operations {
			if op.LocRef != "" && op.Loc.File == "" {
				if loc, ok := mod.Locations[op.LocRef]; ok {
					op.Loc = loc
				}
			}
		}
	}
}
