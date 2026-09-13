package cudaptx

import (
	"bufio"
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

//go:embed ptxdoc.jsonl.gz
var ptxDocJSONLGz []byte

// DocKind describes the kind of PTX entity documented.
type DocKind string

const (
	DocKindInstruction DocKind = "instruction"
	DocKindDirective   DocKind = "directive"
	DocKindRegister    DocKind = "register"
	DocKindType        DocKind = "type"
	DocKindModifier    DocKind = "modifier"
)

// Doc contains structured reference documentation for a PTX instruction, directive,
// special register, data type, or modifier extracted from the PTX ISA documentation.
type Doc struct {
	Name        string   `json:"name"`                  // e.g. "fma", ".maxnreg", "%tid", ".f16x2", ".rn"
	Kind        DocKind  `json:"kind"`                  // instruction, directive, register, type, modifier
	Title       string   `json:"title,omitempty"`       // Descriptive section title
	Summary     string   `json:"summary,omitempty"`     // 1-2 sentence overview
	Syntax      []string `json:"syntax,omitempty"`      // Syntax prototypes
	Description string   `json:"description,omitempty"`  // Detailed description prose
	Semantics   string   `json:"semantics,omitempty"`   // Algorithmic semantics if available
	Category    string   `json:"category,omitempty"`    // Category name (e.g. "Floating-Point Instructions")
	MinPTX      string   `json:"min_ptx,omitempty"`     // Earliest PTX ISA version
	MinTarget   string   `json:"min_target,omitempty"`  // Minimum SM architecture (e.g. "sm_20", "sm_90a", "all")
	Notes       string   `json:"notes,omitempty"`       // Additional usage notes or target restrictions
}

var getDocDatabase = sync.OnceValues(func() (map[string]*Doc, error) {
	gr, err := gzip.NewReader(bytes.NewReader(ptxDocJSONLGz))
	if err != nil {
		return nil, fmt.Errorf("read ptxdoc.jsonl.gz: %w", err)
	}
	defer gr.Close()

	db := make(map[string]*Doc)
	scanner := bufio.NewScanner(gr)
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var d Doc
		if err := json.Unmarshal(line, &d); err != nil {
			return nil, fmt.Errorf("decode doc line: %w", err)
		}
		db[strings.ToLower(d.Name)] = &d
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan ptxdoc.jsonl.gz: %w", err)
	}
	return db, nil
})

// LookupDoc looks up reference documentation for a PTX entity (instruction mnemonic,
// directive, special register, data type, or modifier).
//
// The lookup is case-insensitive. Prefixes (such as "." for directives, types,
// and modifiers, or "%" for special registers) are optional when unambiguous:
// LookupDoc("fma"), LookupDoc("%tid"), LookupDoc("tid"), LookupDoc(".maxnreg"),
// LookupDoc("maxnreg"), LookupDoc(".f16x2"), and LookupDoc(".rn") all resolve.
// Returns nil if no matching documentation is found.
func LookupDoc(name string) *Doc {
	db, err := getDocDatabase()
	if err != nil || db == nil {
		return nil
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return nil
	}
	lower := strings.ToLower(name)
	if d, ok := db[lower]; ok {
		return d
	}
	if !strings.HasPrefix(lower, ".") && !strings.HasPrefix(lower, "%") {
		if d, ok := db["."+lower]; ok {
			return d
		}
		if d, ok := db["%"+lower]; ok {
			return d
		}
	}
	if strings.HasPrefix(lower, ".") {
		if d, ok := db[strings.TrimPrefix(lower, ".")]; ok {
			return d
		}
	}
	if strings.HasPrefix(lower, "%") {
		if d, ok := db[strings.TrimPrefix(lower, "%")]; ok {
			return d
		}
	}
	return nil
}

// AllDocs returns all reference documentation records sorted alphabetically by name.
func AllDocs() []*Doc {
	db, err := getDocDatabase()
	if err != nil || db == nil {
		return nil
	}
	out := make([]*Doc, 0, len(db))
	for _, d := range db {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

// HoverInfo holds structured documentation for a single PTX instruction statement
// or directive, suitable for interactive tooltips, side-by-side assembly views,
// or terminal inspection.
type HoverInfo struct {
	Opcode    string         `json:"opcode"`               // The instruction opcode or directive (e.g. "fma.rn.ftz.f32")
	Mnemonic  string         `json:"mnemonic"`             // Base mnemonic (e.g. "fma")
	Doc       *Doc           `json:"doc,omitempty"`        // Reference doc for the primary opcode or directive
	Predicate string         `json:"predicate,omitempty"`  // Guard predicate if present (e.g. "@%p1")
	Modifiers []ModifierInfo `json:"modifiers,omitempty"`  // Explanations for all parsed modifiers/qualifiers
	Operands  []OperandInfo  `json:"operands,omitempty"`   // Breakdown of operands (identifying special registers)
	MinPTX    string         `json:"min_ptx,omitempty"`    // Minimum PTX version required
	MinTarget string         `json:"min_target,omitempty"` // Minimum GPU architecture required
}

// ModifierInfo describes an individual modifier on an instruction.
type ModifierInfo struct {
	Name    string `json:"name"`              // e.g. ".rn", ".global", ".cta_group::1"
	Meaning string `json:"meaning,omitempty"` // concise meaning e.g. "Round to nearest even"
	Doc     *Doc   `json:"doc,omitempty"`     // full reference doc if available
}

// OperandInfo describes an individual operand in an instruction statement.
type OperandInfo struct {
	Text    string `json:"text"`              // raw operand text e.g. "%tid.x", "[%r1+4]"
	Doc     *Doc   `json:"doc,omitempty"`     // reference doc if operand references a known special register
	Summary string `json:"summary,omitempty"` // short summary if known special register
}

// Markdown formats the hover information into a GitHub-flavored markdown string
// suitable for hover tooltips, documentation sidebars, or terminal display.
func (h *HoverInfo) Markdown() string {
	if h == nil {
		return ""
	}
	var b strings.Builder
	if h.Doc != nil {
		cat := ""
		if h.Doc.Category != "" {
			cat = fmt.Sprintf(" *(%s)*", h.Doc.Category)
		}
		fmt.Fprintf(&b, "### `%s`%s\n\n", h.Doc.Name, cat)
		if h.Doc.Summary != "" {
			fmt.Fprintf(&b, "**%s**\n\n", h.Doc.Summary)
		}
	} else if h.Opcode != "" {
		fmt.Fprintf(&b, "### `%s`\n\n", h.Opcode)
	}

	if h.Predicate != "" {
		fmt.Fprintf(&b, "- **Predicate:** `%s`\n", h.Predicate)
	}
	if len(h.Modifiers) > 0 {
		fmt.Fprintf(&b, "- **Modifiers:**\n")
		for _, m := range h.Modifiers {
			if m.Meaning != "" {
				fmt.Fprintf(&b, "  - `%s`: %s\n", m.Name, m.Meaning)
			} else {
				fmt.Fprintf(&b, "  - `%s`\n", m.Name)
			}
		}
	}
	if len(h.Operands) > 0 {
		var specialOps []OperandInfo
		for _, op := range h.Operands {
			if op.Doc != nil {
				specialOps = append(specialOps, op)
			}
		}
		if len(specialOps) > 0 {
			fmt.Fprintf(&b, "- **Special Registers:**\n")
			for _, op := range specialOps {
				fmt.Fprintf(&b, "  - `%s`: %s\n", op.Text, op.Summary)
			}
		}
	}

	var meta []string
	if h.MinPTX != "" {
		meta = append(meta, fmt.Sprintf("**Minimum PTX:** %s", h.MinPTX))
	}
	if h.MinTarget != "" {
		meta = append(meta, fmt.Sprintf("**Minimum Target:** %s", h.MinTarget))
	}
	if len(meta) > 0 {
		fmt.Fprintf(&b, "\n%s\n", strings.Join(meta, " | "))
	}

	if h.Doc != nil && len(h.Doc.Syntax) > 0 {
		fmt.Fprintf(&b, "\n**Syntax:**\n```ptx\n%s\n```\n", strings.Join(h.Doc.Syntax, "\n"))
	}
	if h.Doc != nil && h.Doc.Description != "" {
		paragraphs := strings.Split(h.Doc.Description, "\n\n")
		limit := 2
		if len(paragraphs) < limit {
			limit = len(paragraphs)
		}
		desc := strings.Join(paragraphs[:limit], "\n\n")
		fmt.Fprintf(&b, "\n%s\n", desc)
	}
	return b.String()
}

// Explain returns structured hover information for the parsed instruction.
func (in Instruction) Explain() *HoverInfo {
	return explainInstruction(in.Predicate, in.Opcode, in.Mnemonic, in.Operands)
}

// Explain parses a PTX statement string (instruction or directive) and returns
// structured hover and reference information.
func Explain(ptxStmt string) *HoverInfo {
	s := strings.TrimSpace(ptxStmt)
	s = strings.TrimSuffix(s, ";")

	// If directive (e.g. ".maxnreg 128" or ".entry foo(...)")
	if strings.HasPrefix(s, ".") {
		fields := strings.Fields(s)
		dirName := fields[0]
		operands := strings.TrimSpace(strings.TrimPrefix(s, dirName))
		doc := LookupDoc(dirName)
		minPTX := ""
		minTarget := ""
		if doc != nil {
			minPTX = doc.MinPTX
			minTarget = doc.MinTarget
		}
		var ops []OperandInfo
		if operands != "" {
			ops = append(ops, OperandInfo{Text: operands})
		}
		return &HoverInfo{
			Opcode:    dirName,
			Mnemonic:  dirName,
			Doc:       doc,
			Operands:  ops,
			MinPTX:    minPTX,
			MinTarget: minTarget,
		}
	}

	inst := parseInstructionStatement(ptxStmt, 0, Location{}, nil)
	return explainInstruction(inst.Predicate, inst.Opcode, inst.Mnemonic, inst.Operands)
}

func explainInstruction(predicate, opcode, mnemonic, operands string) *HoverInfo {
	parts := strings.Split(opcode, ".")
	var baseMnemonic string
	var modParts []string

	if len(parts) > 0 {
		baseMnemonic = parts[0]
		modParts = parts[1:]

		// Check if compound mnemonic like tcgen05.alloc exists in docs
		if len(modParts) > 0 {
			compound := baseMnemonic + "." + modParts[0]
			if d := LookupDoc(compound); d != nil {
				baseMnemonic = compound
				modParts = modParts[1:]
			}
		}
	} else {
		baseMnemonic = mnemonic
	}

	doc := LookupDoc(baseMnemonic)
	if doc == nil && mnemonic != baseMnemonic {
		doc = LookupDoc(mnemonic)
	}

	minPTX := ""
	minTarget := ""
	if doc != nil {
		minPTX = doc.MinPTX
		minTarget = doc.MinTarget
	} else {
		minPTX = InstructionMinPTX(baseMnemonic)
		minTarget = InstructionMinTarget(baseMnemonic)
	}

	info := &HoverInfo{
		Opcode:    opcode,
		Mnemonic:  baseMnemonic,
		Doc:       doc,
		Predicate: predicate,
		MinPTX:    minPTX,
		MinTarget: minTarget,
	}

	// Resolve modifiers
	for _, mp := range modParts {
		modName := "." + mp
		mDoc := LookupDoc(modName)
		meaning := ""
		if mDoc != nil {
			meaning = mDoc.Summary
		}
		info.Modifiers = append(info.Modifiers, ModifierInfo{
			Name:    modName,
			Meaning: meaning,
			Doc:     mDoc,
		})
	}

	// Resolve operands and identify special registers
	if operands != "" {
		for _, op := range strings.Split(operands, ",") {
			op = strings.TrimSpace(op)
			opDoc := LookupDoc(op)
			summary := ""
			if opDoc != nil && opDoc.Kind == DocKindRegister {
				summary = opDoc.Summary
			} else {
				// Check for embedded registers (e.g. %tid.x or inside [%r1])
				tokens := strings.FieldsFunc(op, func(r rune) bool {
					return r == '[' || r == ']' || r == '+' || r == '-' || r == '*' || r == ' ' || r == '\t'
				})
				for _, tok := range tokens {
					if strings.HasPrefix(tok, "%") {
						if rDoc := LookupDoc(tok); rDoc != nil && rDoc.Kind == DocKindRegister {
							opDoc = rDoc
							summary = rDoc.Summary
							break
						}
					}
				}
			}
			info.Operands = append(info.Operands, OperandInfo{
				Text:    op,
				Doc:     opDoc,
				Summary: summary,
			})
		}
	}

	return info
}
