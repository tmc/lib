// Package main generates the Tile IR operation encoding table from NVIDIA's
// TableGen sources.
//
// The bytecode encoding of an operation is not ad hoc: cuda-tile's
// BytecodeGen.cpp derives a writer for every operation from its ODS
// definition, emitting result types, a flags field, attributes, operands and
// regions in that order (tools/cuda-tile-tblgen/BytecodeGen.cpp:660-666,
// after the caller has written the opcode). What varies per operation is only
// shape: how many results, whether they are variadic, which arguments are
// attributes rather than operands, which of those are optional and so carry a
// flags bit instead of or alongside a payload, and where counts appear.
//
// The flags field is the part a reading of the emission order misses. It is a
// single varint bitmask covering every optional attribute and, under
// AttrSizedOperandSegments, every optional operand; a UnitAttr has no payload
// at all and exists only as its bit. An encoder that writes attributes and
// operands without it produces a stream that is wrong from the first
// operation that has one.
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
	"sort"
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
	// ArgOptionalAttribute is an attribute that may be absent. It carries a
	// flags bit, and its payload follows only when the bit is set.
	ArgOptionalAttribute
	// ArgUnitAttribute is a UnitAttr. It is optional, and its flags bit is
	// its entire encoding: the writer emits no payload for it
	// (BytecodeGen.cpp:445-447).
	ArgUnitAttribute
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
	case ArgUnitAttribute:
		return "ArgUnitAttribute"
	}
	return "ArgKind(?)"
}

// Arg is one entry of an ODS arguments or results list.
type Arg struct {
	Name string
	Type string
	Kind ArgKind
	// Version is the CudaTileArg sinceVersion, e.g. "13.4". It orders the
	// flags bits; see [Op.FlagBits].
	Version string
}

// Optional reports whether the argument carries a flags bit.
func (a Arg) Optional() bool {
	switch a.Kind {
	case ArgOptionalAttribute, ArgUnitAttribute, ArgOptionalOperand:
		return true
	}
	return false
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
		case ArgAttribute, ArgOptionalAttribute, ArgUnitAttribute:
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

// EncodeOperandCount reports whether one count varint precedes the whole
// operand list.
//
// It is false under AttrSizedOperandSegments, and not because such an
// operation has no counts: it has one per group and no list-wide count at all.
// The writer takes the two branches in generateOperandSerialization
// (BytecodeGen.cpp:510-535) -- a flat writeOperands over op->getOperands()
// with encodeSize set when any operand is variadic or optional, or one
// writeOperands per group where only a variadic group carries a size and an
// optional group is written only when its flags bit is set. Reporting a
// list-wide count for the segmented form puts one extra varint in the stream
// and none of the per-group ones.
func (o Op) EncodeOperandCount() bool {
	if o.AttrSizedOperandSegments() {
		return false
	}
	for _, a := range o.Operands() {
		if a.Kind == ArgVariadicOperand || a.Kind == ArgOptionalOperand {
			return true
		}
	}
	return false
}

// FlagBit is one bit of an operation's flags varint.
type FlagBit struct {
	Name string
	// Operand says the bit stands for an optional operand group rather than
	// an optional attribute.
	Operand bool
	// Index is the position within [Op.Operands] or [Op.Attributes].
	Index int
	// Version is the sinceVersion that placed the bit where it is.
	Version string
}

// FlagBits returns an operation's flags bits in bit order, low bit first. An
// empty result means the operation writes no flags varint at all.
//
// The order is NVIDIA's, from getVersionOrderedBitAssignments
// (BytecodeGenUtilities.cpp:151-214): bits are grouped by the version that
// introduced the field, ascending, and within one version the optional
// attributes come before the optional operands, each in declaration order.
// Ordering by declaration alone would agree on most operations and disagree
// exactly on the ones that grew a field later, which is where a reader on a
// different toolkit version is already most at risk.
//
// Optional operands take bits only under AttrSizedOperandSegments. Without
// that trait the writer emits one flat operand list and an absent optional
// operand is not recoverable from a bit, so it assigns none.
func (o Op) FlagBits() []FlagBit {
	type entry struct {
		major, minor int
		seq          int
		bit          FlagBit
	}
	var entries []entry
	add := func(a Arg, operand bool, index int) {
		major, minor := parseVersion(a.Version)
		entries = append(entries, entry{
			major: major, minor: minor, seq: len(entries),
			bit: FlagBit{Name: a.Name, Operand: operand, Index: index, Version: a.Version},
		})
	}
	for i, a := range o.Attributes() {
		if a.Optional() {
			add(a, false, i)
		}
	}
	if o.AttrSizedOperandSegments() {
		for i, a := range o.Operands() {
			if a.Kind == ArgOptionalOperand {
				add(a, true, i)
			}
		}
	}
	// A stable sort on the version alone reproduces the C++ grouping: every
	// attribute was appended before every operand, so equal versions keep
	// attributes first.
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].major != entries[j].major {
			return entries[i].major < entries[j].major
		}
		return entries[i].minor < entries[j].minor
	})
	var bits []FlagBit
	for _, e := range entries {
		bits = append(bits, e.bit)
	}
	return bits
}

// parseVersion splits a "major.minor" sinceVersion. An unparseable version
// sorts first, which keeps a table entry rather than dropping one, and the
// generator reports it separately.
func parseVersion(v string) (major, minor int) {
	maj, min, ok := strings.Cut(v, ".")
	if !ok {
		return 0, 0
	}
	major, _ = strconv.Atoi(maj)
	minor, _ = strconv.Atoi(min)
	return major, minor
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

// entryVersion returns the sinceVersion an entry was introduced with. It is
// the third template argument of CudaTileArg<constraint, desc, version>
// (Dialect.td:257-258), and it orders the flags bits.
func entryVersion(entry string) string {
	e := strings.TrimSpace(entry)
	if i := strings.LastIndex(e, ":$"); i >= 0 {
		e = e[:i]
	}
	for _, wrapper := range []string{"CudaTileArg<", "CudaTileUnusedArg<"} {
		if !strings.HasPrefix(e, wrapper) {
			continue
		}
		inner := strings.TrimSuffix(strings.TrimSpace(e[len(wrapper):]), ">")
		if parts := splitTopLevel(inner); len(parts) > 2 {
			return strings.Trim(strings.TrimSpace(parts[2]), `"`)
		}
	}
	return ""
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
//
// Optional means what MLIR's isOptional means, because that is the predicate
// the writer branches on: OptionalAttr and UnitAttr set it,
// DefaultValuedOptionalAttr sets it, and plain DefaultValuedAttr does not.
// A DefaultValuedAttr is always written, with no flags bit, so classifying it
// as optional both invents a bit and drops a payload.
func classify(typ string) ArgKind {
	switch {
	case strings.HasPrefix(typ, "OptionalAttr<UnitAttr"), typ == "UnitAttr":
		return ArgUnitAttribute
	case strings.HasPrefix(typ, "OptionalAttr<"), strings.HasPrefix(typ, "DefaultValuedOptionalAttr<"):
		return ArgOptionalAttribute
	case strings.HasPrefix(typ, "DefaultValuedAttr<"):
		return ArgAttribute
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
		res = append(res, Arg{
			Name:    argName(entry),
			Type:    typ,
			Kind:    classify(typ),
			Version: entryVersion(entry),
		})
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
