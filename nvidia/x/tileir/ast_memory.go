package tileir

import (
	"fmt"
	"strings"
)

func memoryOp(name string) bool {
	return name == "cuda_tile.load_ptr_tko" || name == "cuda_tile.store_ptr_tko"
}

// The generated table misses LoadOpBase's inherited AttrSizedOperandSegments.
// Ops.td:2978-3055/4763-4800 and BytecodeGen.cpp:250-396/510-521
// give optional attributes first, then optional operand groups, no group count.
func astOpSpec(name string) (opSpec, bool) {
	spec, ok := lookupOpSpec(name)
	if !memoryOp(name) {
		return spec, ok
	}
	spec.AttrSegments, spec.EncodeOperandCount = true, false
	spec.Flags = []flagBit{{Name: "memory_scope", Index: 1}, {Name: "optimization_hints", Index: 2}, {Name: "mask", Operand: true}}
	if name == "cuda_tile.load_ptr_tko" {
		spec.Flags = append(spec.Flags, flagBit{Name: "paddingValue", Operand: true}, flagBit{Name: "token", Operand: true})
	} else {
		spec.Flags = append(spec.Flags, flagBit{Name: "token", Operand: true})
	}
	return spec, ok
}

func checkMemoryPointer(t *TileType) error {
	p, ok := t.ElementType.(*PointerType)
	if !ok || p == nil || len(t.Shape) != 0 || p.Attr != "" {
		return astUnsupported("memory pointer must be scalar tile<ptr<f32>> without pointer classification")
	}
	if pointee, ok := p.Pointee.(PrimitiveType); !ok || pointee != Float32 {
		return astUnsupported("memory pointer pointee must be canonical f32")
	}
	return nil
}

// Decode the precise absent-ptr_attr 13.4 subset. The general structural
// decoder must not guess a default pointee for a malformed reference.
func decodeMemoryPointer(payload []byte, pool []Type, version BytecodeVersion) (Type, error) {
	c := astCursor{data: payload}
	if versionAtLeast(version, ptrAttrSinceVersion) {
		flags, err := c.uint()
		if err != nil {
			return nil, err
		}
		if flags != 0 {
			return nil, astUnsupported("pointer classification")
		}
	}
	index, err := c.uint()
	if err != nil || index >= uint64(len(pool)) || pool[index] == nil || c.off != len(payload) {
		return nil, fmt.Errorf("tileir: invalid pointer type reference or payload")
	}
	return NewPointer(pool[index]), nil
}

// Multiple result types use the AST's existing FunctionType tuple carrier:
// zero inputs, one type per result. This does not encode a function type as
// the operation result; each result gets its own type-pool reference.
func astResultTypes(op *Operation, count int) ([]Type, error) {
	if count == 1 {
		t, err := astResultType(op)
		if err != nil {
			return nil, err
		}
		return []Type{t}, nil
	}
	t, ok := op.TypeVal.(*FunctionType)
	if !ok && op.Type != "" {
		parsed, err := ParseType(op.Type)
		if err == nil {
			t, ok = parsed.(*FunctionType)
		}
	}
	if !ok || t == nil || len(t.Inputs) != 0 || len(t.Results) != count {
		return nil, astUnsupported("multiple result type tuple")
	}
	if op.Type != "" && op.Type != t.String() {
		return nil, astUnsupported("result tuple text mismatch")
	}
	for _, result := range t.Results {
		if err := checkASTType(result); err != nil {
			return nil, err
		}
	}
	return t.Results, nil
}

func resultTuple(types []Type) Type {
	if len(types) == 1 {
		return types[0]
	}
	return NewFunction(nil, types)
}

func scalarMemoryType(t Type, kind TypeKind) bool {
	tile, ok := t.(*TileType)
	return ok && tile != nil && len(tile.Shape) == 0 && tile.ElementType != nil && tile.ElementType.Kind() == kind
}

func memoryFlags(name string, operands []Type) (uint64, error) {
	required, groups := 1, []TypeKind{KindInt1, KindFloat32, KindToken}
	if name == "cuda_tile.store_ptr_tko" {
		required, groups = 2, []TypeKind{KindInt1, KindToken}
	}
	if len(operands) < required {
		return 0, astUnsupported("memory operand arity")
	}
	pointer, ok := operands[0].(*TileType)
	if !ok || pointer == nil {
		return 0, astUnsupported("memory pointer type")
	}
	if err := checkMemoryPointer(pointer); err != nil {
		return 0, err
	}
	if required == 2 && !scalarMemoryType(operands[1], KindFloat32) {
		return 0, astUnsupported("store value must be scalar tile<f32>")
	}
	var flags uint64
	next := 0
	for _, operand := range operands[required:] {
		matched := false
		for next < len(groups) {
			group := next
			next++
			matches := scalarMemoryType(operand, groups[group])
			if groups[group] == KindToken {
				matches = astTokenType(operand)
			}
			if matches {
				flags |= 1 << (group + 2)
				matched = true
				break
			}
		}
		if !matched {
			return 0, astUnsupported("invalid or out-of-order optional memory operand")
		}
	}
	return flags, nil
}

func memoryOperandCount(name string, flags uint64) uint64 {
	n, groups := uint64(1), 3
	if name == "cuda_tile.store_ptr_tko" {
		n, groups = 2, 2
	}
	for i := 0; i < groups; i++ {
		if flags&(1<<(i+2)) != 0 {
			n++
		}
	}
	return n
}

func memoryOrderingAllowed(name string, value uint64) bool {
	// CudaTile.cpp:3956-3984/6082-6109: weak forbids scope; every
	// non-weak ordering requires scope. This slice has no scope encoding.
	return memoryOp(name) && value == 0
}

func encodeMemoryAttrs(body []byte, op *Operation, attrs map[string]string, values map[string]int, types []Type) ([]byte, error) {
	if len(attrs) != 1 {
		return nil, astUnsupported("memory scope, hints or unknown attributes")
	}
	ordering, exists := attrs["memory_ordering_semantics"]
	if !exists {
		return nil, astUnsupported("missing memory ordering")
	}
	index := -1
	for i, value := range astEnums["memory_ordering_semantics"] {
		if ordering == value {
			index = i
		}
	}
	if index < 0 || !memoryOrderingAllowed(op.OpCode, uint64(index)) {
		return nil, astUnsupported("invalid memory ordering for %s", op.OpCode)
	}
	var operands []Type
	for _, name := range op.Operands {
		index, ok := values[name]
		if !ok {
			return nil, astUnsupported("undefined memory operand %s", name)
		}
		operands = append(operands, types[index])
	}
	flags, err := memoryFlags(op.OpCode, operands)
	if err != nil {
		return nil, err
	}
	body = appendVarInt(body, flags)
	return appendVarInt(body, uint64(index)), nil
}

func checkMemoryOperands(name string, result Type, operands []Type) error {
	if _, err := memoryFlags("cuda_tile."+name, operands); err != nil {
		return err
	}
	if name == "store_ptr_tko" {
		if !astTokenType(result) {
			return astUnsupported("store result must be token")
		}
		return nil
	}
	tuple, ok := result.(*FunctionType)
	if !ok || tuple == nil || len(tuple.Results) != 2 || !scalarMemoryType(tuple.Results[0], KindFloat32) || !astTokenType(tuple.Results[1]) {
		return astUnsupported("load results must be scalar tile<f32>, token")
	}
	return nil
}

// Memory entries admit a single complete token chain. Optional wire groups
// are represented faithfully by the frame helpers, but an executable AST
// here requires every memory access to consume the preceding token. Only
// the final store token may be unused at the void kernel return.
func checkMemoryChain(ops []*Operation, inputs, returns []Type, entry bool) (string, error) {
	hasMemory := false
	for _, op := range ops {
		hasMemory = hasMemory || memoryOp(op.OpCode)
	}
	for _, input := range inputs {
		if tile, ok := input.(*TileType); ok {
			if _, ptr := tile.ElementType.(*PointerType); ptr && !hasMemory {
				return "", astUnsupported("pointer parameter without memory entry")
			}
		}
	}
	if !hasMemory {
		return "", nil
	}
	if !entry || len(returns) != 0 {
		return "", astUnsupported("memory route requires void kernel entry")
	}
	current, lastStore := "", false
	for _, op := range ops {
		switch op.OpCode {
		case "cuda_tile.make_token":
			if current != "" {
				return "", astUnsupported("multiple initial memory tokens")
			}
			current = op.Results[0]
		case "cuda_tile.join_tokens":
			return "", astUnsupported("memory route requires one linear predecessor")
		case "cuda_tile.load_ptr_tko", "cuda_tile.store_ptr_tko":
			if current == "" || len(op.Operands) == 0 || op.Operands[len(op.Operands)-1] != current {
				return "", astUnsupported("broken memory predecessor token")
			}
			current = op.Results[len(op.Results)-1]
			lastStore = op.OpCode == "cuda_tile.store_ptr_tko"
		case "cuda_tile.return":
			if len(op.Operands) != 0 || !lastStore {
				return "", astUnsupported("memory entry must end in store and void return")
			}
		}
	}
	if !lastStore {
		return "", astUnsupported("memory entry without terminal store")
	}
	return current, nil
}

func hasMemoryOps(ops []*Operation) bool {
	for _, op := range ops {
		if op != nil && memoryOp(strings.TrimSpace(op.OpCode)) {
			return true
		}
	}
	return false
}
