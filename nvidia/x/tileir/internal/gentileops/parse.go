// Package main generates the Tile IR operation encoding table from NVIDIA's
// TableGen sources.
//
// The bytecode encoding of an operation is not ad hoc: cuda-tile's
// BytecodeGen.cpp derives a writer for every operation from its ODS
// definition, emitting opcode, result types, attributes, operands and regions
// in that order (tools/cuda-tile-tblgen/BytecodeGen.cpp:653-665). What varies
// per operation is only shape: how many results, whether they are variadic,
// which arguments are attributes rather than operands, and whether an operand
// count precedes the operands.
//
// This program recovers that shape from the TableGen sources and writes it as
// a Go table, so the encoder is derived from NVIDIA's definitions rather than
// transcribed from a reading of them.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// ArgKind classifies one entry of an ODS arguments list.
type ArgKind int

const (
	// ArgOperand is a plain SSA operand.
	ArgOperand ArgKind = iota
	// ArgVariadicOperand is an operand group of run-time length.
	ArgVariadicOperand
	// ArgOptionalOperand is an operand that may be absent.
	ArgOptionalOperand
	// ArgAttribute is an attribute, serialized before operands.
	ArgAttribute
	// ArgOptionalAttribute is an attribute that may be absent.
	ArgOptionalAttribute
)

func (k ArgKind) String() string {
	switch k {
	case ArgOperand:
		return "ArgOperand"
	case ArgVariadicOperand:
		return "ArgVariadicOperand"
	case ArgOptionalOperand:
		return "ArgOptionalOperand"
	case ArgAttribute:
		return "ArgAttribute"
	case ArgOptionalAttribute:
		return "ArgOptionalAttribute"
	}
	return "ArgKind(?)"
}

// Arg is one entry of an ODS arguments or results list.
type Arg struct {
	Name string
	Type string
	Kind ArgKind
}

// Op is the encoding shape of one Tile IR operation.
type Op struct {
	Def       string // TableGen def name, e.g. CudaTile_MakeTokenOp
	Mnemonic  string // dialect mnemonic, e.g. make_token
	Opcode    uint32
	HasOpcode bool
	Version   string
	Traits    []string
	Args      []Arg
	Results   []Arg
	Regions   int
}

// Operands returns the operand entries in order, excluding attributes.
func (o Op) Operands() []Arg {
	var res []Arg
	for _, a := range o.Args {
		switch a.Kind {
		case ArgOperand, ArgVariadicOperand, ArgOptionalOperand:
			res = append(res, a)
		}
	}
	return res
}

// Attributes returns the attribute entries in order.
func (o Op) Attributes() []Arg {
	var res []Arg
	for _, a := range o.Args {
		switch a.Kind {
		case ArgAttribute, ArgOptionalAttribute:
			res = append(res, a)
		}
	}
	return res
}

// AttrSizedOperandSegments reports whether operands are written one group at a
// time with their sizes in an attribute, rather than as one flat list.
func (o Op) AttrSizedOperandSegments() bool {
	for _, t := range o.Traits {
		if t == "AttrSizedOperandSegments" {
			return true
		}
	}
	return false
}

// EncodeOperandCount reports whether a count varint precedes the operands.
// BytecodeGen writes one only when some operand is variadic or optional
// (BytecodeGen.cpp:521-531).
func (o Op) EncodeOperandCount() bool {
	for _, a := range o.Operands() {
		if a.Kind == ArgVariadicOperand || a.Kind == ArgOptionalOperand {
			return true
		}
	}
	return false
}

// VariadicResults reports whether a result count varint precedes the result
// types (BytecodeGen.cpp:558-560).
func (o Op) VariadicResults() bool {
	for _, r := range o.Results {
		if r.Kind == ArgVariadicOperand {
			return true
		}
	}
	return false
}

var (
	opcodeRe = regexp.MustCompile(`def\s*:\s*(?:Public|Internal)Opcode<\s*(\w+)\s*,\s*(0x[0-9A-Fa-f]+|\d+)\s*>`)
	defRe    = regexp.MustCompile(`(?m)^def\s+(CudaTile_\w+)\s*\n?\s*:\s*(\w+)<`)
)

// parseOpcodes reads the opcode assignments from BytecodeOpcodes.td.
func parseOpcodes(path string) (map[string]uint32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	res := make(map[string]uint32)
	for _, m := range opcodeRe.FindAllStringSubmatch(string(b), -1) {
		v, err := strconv.ParseUint(strings.TrimPrefix(m[2], "0x"), 16, 32)
		if strings.HasPrefix(m[2], "0x") {
			if err != nil {
				return nil, fmt.Errorf("opcode %s for %s: %v", m[2], m[1], err)
			}
		} else {
			d, derr := strconv.ParseUint(m[2], 10, 32)
			if derr != nil {
				return nil, fmt.Errorf("opcode %s for %s: %v", m[2], m[1], derr)
			}
			v = d
		}
		res[m[1]] = uint32(v)
	}
	if len(res) == 0 {
		return nil, fmt.Errorf("no opcodes found in %s", path)
	}
	return res, nil
}

// splitTopLevel splits an ODS argument list on commas that are not nested
// inside angle brackets, parentheses or brackets.
func splitTopLevel(s string) []string {
	var res []string
	depth := 0
	start := 0
	for i, r := range s {
		switch r {
		case '<', '(', '[', '{':
			depth++
		case '>', ')', ']', '}':
			depth--
		case ',':
			if depth == 0 {
				res = append(res, strings.TrimSpace(s[start:i]))
				start = i + 1
			}
		}
	}
	if rest := strings.TrimSpace(s[start:]); rest != "" {
		res = append(res, rest)
	}
	return res
}

// innerType strips the CudaTileArg wrapper, returning the constraint that
// decides whether an entry is an operand or an attribute.
func innerType(entry string) string {
	e := strings.TrimSpace(entry)
	if i := strings.LastIndex(e, ":$"); i >= 0 {
		e = e[:i]
	}
	if strings.HasPrefix(e, "CudaTileArg<") {
		inner := e[len("CudaTileArg<") : len(e)-0]
		inner = strings.TrimSuffix(strings.TrimSpace(inner), ">")
		if parts := splitTopLevel(inner); len(parts) > 0 {
			e = strings.TrimSpace(parts[0])
		}
	}
	return e
}

// argName returns the ODS name of an entry, without the leading dollar.
func argName(entry string) string {
	if i := strings.LastIndex(entry, ":$"); i >= 0 {
		return strings.TrimSpace(entry[i+2:])
	}
	return ""
}

// classify decides what one arguments-list entry encodes as. MLIR spells
// attribute constraints with an Attr suffix, which is what separates an
// attribute from an operand here.
func classify(typ string) ArgKind {
	switch {
	case strings.HasPrefix(typ, "OptionalAttr<"):
		return ArgOptionalAttribute
	case strings.HasPrefix(typ, "DefaultValuedAttr<"), strings.HasPrefix(typ, "DefaultValuedOptionalAttr<"):
		return ArgOptionalAttribute
	case strings.HasPrefix(typ, "Variadic<"):
		return ArgVariadicOperand
	case strings.HasPrefix(typ, "Optional<"):
		return ArgOptionalOperand
	case strings.HasSuffix(typ, "Attr"), strings.Contains(typ, "Attr<"):
		return ArgAttribute
	default:
		return ArgOperand
	}
}

// parseArgs parses an ODS (ins ...) or (outs ...) list body.
func parseArgs(body string) []Arg {
	var res []Arg
	for _, entry := range splitTopLevel(body) {
		if entry == "" {
			continue
		}
		typ := innerType(entry)
		res = append(res, Arg{Name: argName(entry), Type: typ, Kind: classify(typ)})
	}
	return res
}

// extractList returns the body of `let <field> = (<head> ... );`. TableGen
// allows a line break between the equals sign and the list, which several
// operations use, so the opening is matched with a pattern rather than a
// literal.
func extractList(body, field, head string) (string, bool) {
	re := regexp.MustCompile(`let\s+` + field + `\s*=\s*\(\s*` + head + `\b`)
	loc := re.FindStringIndex(body)
	if loc == nil {
		return "", false
	}
	rest := body[loc[1]:]
	depth := 1
	for j, r := range rest {
		switch r {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return strings.TrimSpace(rest[:j]), true
			}
		}
	}
	return "", false
}

// parseOps reads Ops.td and returns every operation definition in it.
func parseOps(path string, opcodes map[string]uint32) ([]Op, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	src := string(b)
	locs := defRe.FindAllStringSubmatchIndex(src, -1)
	var ops []Op
	for i, loc := range locs {
		end := len(src)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		chunk := src[loc[0]:end]
		def := src[loc[2]:loc[3]]

		op := Op{Def: def}
		op.Opcode, op.HasOpcode = opcodes[def]

		// The template arguments carry the mnemonic, the version and the
		// trait list: Base<"mnemonic", "13.1", [Trait, ...]>.
		if open := strings.Index(chunk, "<"); open >= 0 {
			depth := 0
			var argsrc string
			for j := open; j < len(chunk); j++ {
				switch chunk[j] {
				case '<':
					depth++
				case '>':
					depth--
					if depth == 0 {
						argsrc = chunk[open+1 : j]
						j = len(chunk)
					}
				}
			}
			parts := splitTopLevel(argsrc)
			if len(parts) > 0 {
				op.Mnemonic = strings.Trim(parts[0], `"`)
			}
			if len(parts) > 1 {
				op.Version = strings.Trim(parts[1], `"`)
			}
			for _, p := range parts {
				if strings.HasPrefix(p, "[") {
					for _, t := range splitTopLevel(strings.Trim(p, "[]")) {
						if t = strings.TrimSpace(t); t != "" {
							op.Traits = append(op.Traits, t)
						}
					}
				}
			}
		}
		if body, ok := extractList(chunk, "arguments", "ins"); ok {
			op.Args = parseArgs(body)
		}
		if body, ok := extractList(chunk, "results", "outs"); ok {
			op.Results = parseArgs(body)
		}
		if m := regexp.MustCompile(`let regions = \(region ([^)]*)\)`).FindStringSubmatch(chunk); m != nil {
			op.Regions = len(splitTopLevel(m[1]))
		}
		ops = append(ops, op)
	}
	if len(ops) == 0 {
		return nil, fmt.Errorf("no operation definitions found in %s", path)
	}
	return ops, nil
}

// load parses both TableGen sources out of a cuda-tile checkout.
func load(root string) ([]Op, error) {
	dir := filepath.Join(root, "include", "cuda_tile", "Dialect", "CudaTile", "IR")
	opcodes, err := parseOpcodes(filepath.Join(dir, "BytecodeOpcodes.td"))
	if err != nil {
		return nil, fmt.Errorf("parsing opcodes: %w", err)
	}
	ops, err := parseOps(filepath.Join(dir, "Ops.td"), opcodes)
	if err != nil {
		return nil, fmt.Errorf("parsing ops: %w", err)
	}
	return ops, nil
}
