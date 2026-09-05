package cudasass

import (
	"encoding/json"
	"fmt"
	"io"
)

// JSONDisassembly retains metadata and function records from nvdisasm's JSON
// output. Unlike Disassembly it preserves gaps and does not infer instruction
// addresses for architectures with a different instruction width.
type JSONDisassembly struct {
	Metadata  map[string]json.RawMessage
	Functions []JSONFunction
}

// JSONFunction is a CUDA function in the nvdisasm JSON schema.
type JSONFunction struct {
	Name         string            `json:"function-name"`
	Start        uint64            `json:"start"`
	Length       uint64            `json:"length"`
	Attributes   []string          `json:"other-attributes"`
	Instructions []JSONInstruction `json:"sass-instructions"`
}

// JSONInstruction preserves empty opcodes representing instruction-slot gaps.
type JSONInstruction struct {
	Predicate  string            `json:"predicate"`
	Opcode     string            `json:"opcode"`
	Operands   string            `json:"operands"`
	Extra      string            `json:"extra"`
	Attributes map[string]string `json:"other-attributes"`
	Flags      []string          `json:"other-flags"`
}

// ParseJSON reads nvdisasm's [metadata, functions] format. It rejects missing
// required fields and trailing data; unknown fields are allowed for toolkit
// evolution. Metadata is retained verbatim, not interpreted as a version gate.
func ParseJSON(r io.Reader) (*JSONDisassembly, error) {
	dec := json.NewDecoder(r)
	var parts []json.RawMessage
	if err := dec.Decode(&parts); err != nil {
		return nil, fmt.Errorf("decode disassembly: %w", err)
	}
	if len(parts) != 2 {
		return nil, fmt.Errorf("disassembly must contain metadata and functions")
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return nil, fmt.Errorf("trailing disassembly data")
	}
	d := new(JSONDisassembly)
	if err := json.Unmarshal(parts[0], &d.Metadata); err != nil {
		return nil, err
	}
	for _, name := range []string{"ELF", "SM", "SchemaVersion", "Producer", "Description"} {
		if raw, ok := d.Metadata[name]; !ok || string(raw) == "null" {
			return nil, fmt.Errorf("missing metadata %s", name)
		}
	}
	var functions []map[string]json.RawMessage
	if err := json.Unmarshal(parts[1], &functions); err != nil {
		return nil, err
	}
	if functions == nil {
		return nil, fmt.Errorf("missing functions array")
	}
	for i, fn := range functions {
		for _, name := range []string{"function-name", "start", "length", "sass-instructions"} {
			if raw, ok := fn[name]; !ok || string(raw) == "null" {
				return nil, fmt.Errorf("function %d missing %s", i, name)
			}
		}
		var instructions []map[string]json.RawMessage
		if err := json.Unmarshal(fn["sass-instructions"], &instructions); err != nil {
			return nil, err
		}
		for j, inst := range instructions {
			if raw, ok := inst["opcode"]; !ok || string(raw) == "null" {
				return nil, fmt.Errorf("function %d instruction %d missing opcode", i, j)
			}
		}
	}
	if err := json.Unmarshal(parts[1], &d.Functions); err != nil {
		return nil, err
	}
	return d, nil
}
