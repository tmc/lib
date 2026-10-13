package tileir

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// astBodyOps is the straight-line subset whose complete wire representation
// is supported here. Scalar pointer memory frames use the separately pinned
// spec in ast_memory.go. All nested regions remain unsupported. An opcode
// being in opSpecs is not an encoding promise.
var astBodyOps = map[string]bool{
	"constant": true, "make_token": true, "join_tokens": true, "return": true,
	"absf": true, "absi": true, "negf": true, "negi": true,
	"addf": true, "subf": true, "mulf": true, "divf": true,
	"addi": true, "subi": true, "muli": true,
	"maxf": true, "minf": true, "maxi": true, "mini": true,
	"andi": true, "ori": true, "xori": true,
	"cmpf": true, "cmpi": true, "select": true, "mmaf": true,
	"load_ptr_tko": true, "store_ptr_tko": true,
}

// Values and ordering come from NVIDIA/cuda-tile AttrDefs.td at 7e8e2e68.
// Unit attributes have no payload: their presence is the flags bit itself.
var astEnums = map[string][]string{
	"rounding_mode":             {"nearest_even", "zero", "negative_inf", "positive_inf", "approx", "full", "nearest_int_to_zero", "nearest_away"},
	"overflow":                  {"none", "no_signed_wrap", "no_unsigned_wrap", "no_wrap"},
	"signedness":                {"unsigned", "signed"},
	"comparison_predicate":      {"equal", "not_equal", "less_than", "less_than_or_equal", "greater_than", "greater_than_or_equal"},
	"comparison_ordering":       {"unordered", "ordered"},
	"memory_ordering_semantics": {"weak", "relaxed", "acquire", "release", "acq_rel"},
}

func astUnsupported(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrUnsupportedASTEncoding, fmt.Sprintf(format, args...))
}

func astAttributes(text string) (map[string]string, error) {
	text = strings.TrimSpace(text)
	attrs := make(map[string]string)
	if text == "" {
		return attrs, nil
	}
	if !strings.HasPrefix(text, "{") || !strings.HasSuffix(text, "}") {
		return nil, astUnsupported("attributes must be a dictionary")
	}
	for _, field := range SplitOperands(text[1 : len(text)-1]) {
		name, value, assigned := strings.Cut(field, "=")
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if !assigned {
			value = "unit"
		}
		if name == "" || value == "" {
			return nil, astUnsupported("empty attribute name or value")
		}
		if _, exists := attrs[name]; exists {
			return nil, astUnsupported("duplicate attribute %q", name)
		}
		attrs[name] = value
	}
	return attrs, nil
}

func astType(t Type, text string) (Type, error) {
	if t != nil {
		if err := checkASTType(t); err != nil {
			return nil, err
		}
	}
	if t == nil {
		var err error
		t, err = ParseType(text)
		if err != nil {
			return nil, astUnsupported("type %q: %v", text, err)
		}
	} else if text != "" {
		parsed, err := ParseType(text)
		if err != nil || !typeEqual(t, parsed) {
			return nil, astUnsupported("type value disagrees with %q", text)
		}
	}
	if err := checkASTType(t); err != nil {
		return nil, err
	}
	return t, nil
}

func checkASTType(t Type) error {
	switch t := t.(type) {
	case PrimitiveType:
		if t.name != t.kind.String() {
			return astUnsupported("noncanonical primitive type")
		}
		if t.Kind() == KindFloat8E5M3 {
			return astUnsupported("nonpublic primitive type %s", t)
		}
		if !t.IsInt() && !t.IsFloat() {
			return astUnsupported("unknown primitive type %s", t)
		}
	case TokenType:
	case *TileType:
		if t == nil {
			return astUnsupported("nil tile type")
		}
		if _, ok := t.ElementType.(*PointerType); ok {
			return checkMemoryPointer(t)
		}
		if _, ok := t.ElementType.(PrimitiveType); !ok {
			return astUnsupported("tile element type must be a number")
		}
		if _, err := astElements(t); err != nil {
			return err
		}
		return checkASTType(t.ElementType)
	default:
		return astUnsupported("structural type %T", t)
	}
	return nil
}

func astElements(t *TileType) (int, error) {
	n := 1
	for _, dim := range t.Shape {
		if dim <= 0 || uint64(dim) > uint64(int(^uint(0)>>1)/n) {
			return 0, astUnsupported("nonpositive or unrepresentable tile shape")
		}
		n *= int(dim)
	}
	return n, nil
}

func astResultType(op *Operation) (Type, error) {
	t, err := astType(op.TypeVal, op.Type)
	if err != nil {
		return nil, err
	}
	if _, ok := t.(*FunctionType); ok {
		return nil, astUnsupported("multiple result types")
	}
	return t, nil
}

func astModule(m *Module) (*BytecodeModule, error) {
	if m == nil {
		return nil, fmt.Errorf("tileir: nil module")
	}
	if len(m.Globals) != 0 || len(m.Locations) != 0 {
		return nil, astUnsupported("global or debug sections")
	}
	bm := &BytecodeModule{Version: BytecodeVersion{Major: 13, Minor: 4}, Producer: m.Producer}
	if m.Producer != "" {
		registerString(&bm.Strings, m.Producer)
	}
	names := make(map[string]bool)
	for _, fn := range m.Functions {
		if fn == nil || fn.Name == "" || names[fn.Name] {
			return nil, astUnsupported("nil, unnamed, or duplicate function")
		}
		names[fn.Name] = true
		if fn.OptHints != "" || fn.LocRef != "" || fn.Loc != (Location{}) {
			return nil, astUnsupported("function optimization hints or debug location")
		}
		if len(fn.Blocks) > 1 {
			return nil, astUnsupported("multiple function blocks")
		}
		if len(fn.Blocks) == 1 {
			block := fn.Blocks[0]
			if block == nil || len(block.Operations) != len(fn.Operations) {
				return nil, astUnsupported("block and function operation lists disagree")
			}
			for i, op := range block.Operations {
				if op != fn.Operations[i] {
					return nil, astUnsupported("block and function operation lists disagree")
				}
			}
		}
		values := make(map[string]int)
		tokenUses := make(map[string]int)
		var valueTypes, inputs, returns []Type
		for _, p := range fn.Params {
			if p == nil || !strings.HasPrefix(p.Name, "%") {
				return nil, astUnsupported("nil or unnamed parameter")
			}
			if _, exists := values[p.Name]; exists {
				return nil, astUnsupported("duplicate parameter %s", p.Name)
			}
			t, err := astType(p.TypeVal, p.Type)
			if err != nil {
				return nil, err
			}
			values[p.Name] = len(valueTypes)
			valueTypes = append(valueTypes, t)
			inputs = append(inputs, t)
		}
		for _, text := range fn.Returns {
			t, err := astType(nil, text)
			if err != nil {
				return nil, err
			}
			returns = append(returns, t)
		}
		if fn.Entry && len(returns) != 0 {
			return nil, astUnsupported("kernel entry cannot return values")
		}
		sig := NewFunction(inputs, returns)
		bfn := &BytecodeFunction{Name: fn.Name, Signature: sig, SignatureIndex: registerType(&bm.Types, sig), IsEntry: fn.Entry}
		if fn.Entry {
			bfn.EntryFlag = FuncKindKernel
		}
		registerString(&bm.Strings, fn.Name)
		if len(fn.Operations) == 0 {
			return nil, astUnsupported("function %s has no return", fn.Name)
		}
		for i, op := range fn.Operations {
			if op == nil || op.LocRef != "" || op.Loc != (Location{}) {
				return nil, astUnsupported("nil operation or debug location")
			}
			name := strings.TrimPrefix(op.OpCode, "cuda_tile.")
			if !astBodyOps[name] || op.OpCode != "cuda_tile."+name {
				return nil, astUnsupported("operation %q", op.OpCode)
			}
			spec, ok := astOpSpec(op.OpCode)
			if !ok || spec.NumRegions != 0 || (spec.AttrSegments && !memoryOp(op.OpCode)) {
				return nil, astUnsupported("encoding shape for %q", op.OpCode)
			}
			if len(op.Results) != spec.NumResults || (name == "return") != (i == len(fn.Operations)-1) {
				return nil, astUnsupported("result arity or misplaced return in %s", op.OpCode)
			}
			if spec.NumResults == 0 && (op.TypeVal != nil || op.Type != "") {
				return nil, astUnsupported("result type on zero-result operation")
			}
			attrs, err := astAttributes(op.Attributes)
			if err != nil {
				return nil, err
			}
			var result Type
			var results []Type
			bfn.BodyBytes = appendVarInt(bfn.BodyBytes, uint64(spec.Opcode))
			if astResultCountField(spec) {
				bfn.BodyBytes = appendVarInt(bfn.BodyBytes, uint64(spec.NumResults))
			}
			if spec.NumResults != 0 {
				results, err = astResultTypes(op, spec.NumResults)
				if err != nil {
					return nil, err
				}
				result = resultTuple(results)
				for _, t := range results {
					bfn.BodyBytes = appendVarInt(bfn.BodyBytes, uint64(registerType(&bm.Types, t)))
				}
			}
			if memoryOp(op.OpCode) {
				bfn.BodyBytes, err = encodeMemoryAttrs(bfn.BodyBytes, op, attrs, values, valueTypes)
			} else {
				bfn.BodyBytes, err = encodeASTAttrs(bfn.BodyBytes, spec, attrs, result, bm)
			}
			if err != nil {
				return nil, fmt.Errorf("function %s op %d: %w", fn.Name, i, err)
			}
			if spec.EncodeOperandCount {
				bfn.BodyBytes = appendVarInt(bfn.BodyBytes, uint64(len(op.Operands)))
			} else if !memoryOp(op.OpCode) && len(op.Operands) != len(spec.Operands) {
				return nil, astUnsupported("operand arity in %s", op.OpCode)
			}
			var operandTypes []Type
			for _, operand := range op.Operands {
				index, exists := values[operand]
				if !exists {
					return nil, astUnsupported("undefined operand %s", operand)
				}
				bfn.BodyBytes = appendVarInt(bfn.BodyBytes, uint64(index))
				operandTypes = append(operandTypes, valueTypes[index])
				if astTokenType(valueTypes[index]) {
					tokenUses[operand]++
				}
			}
			if err := checkASTOperands(name, result, operandTypes, returns); err != nil {
				return nil, err
			}
			for j, res := range op.Results {
				if _, exists := values[res]; exists || !strings.HasPrefix(res, "%") {
					return nil, astUnsupported("duplicate or unnamed result %s", res)
				}
				values[res] = len(valueTypes)
				valueTypes = append(valueTypes, results[j])
			}
		}
		terminal, err := checkMemoryChain(fn.Operations, inputs, returns, fn.Entry)
		if err != nil {
			return nil, err
		}
		for name, index := range values {
			if astTokenType(valueTypes[index]) && tokenUses[name] != 1 && !(name == terminal && tokenUses[name] == 0) {
				return nil, astUnsupported("token %s must have exactly one consumer", name)
			}
		}
		bfn.Operations, err = decodeASTBody(bfn.BodyBytes, bm.Types, bm.Constants, sig)
		if err != nil {
			return nil, err
		}
		bm.Functions = append(bm.Functions, bfn)
	}
	return bm, nil
}

func checkASTOperands(name string, result Type, operands, returns []Type) error {
	if name == "load_ptr_tko" || name == "store_ptr_tko" {
		return checkMemoryOperands(name, result, operands)
	}
	if name == "return" {
		if len(operands) != len(returns) {
			return astUnsupported("return arity differs from function signature")
		}
		for i := range returns {
			if !typeEqual(operands[i], returns[i]) {
				return astUnsupported("return type mismatch")
			}
		}
		return nil
	}
	if name == "make_token" || name == "join_tokens" {
		if !astTokenType(result) || (name == "join_tokens" && len(operands) == 0) {
			return astUnsupported("invalid token operation")
		}
		for _, t := range operands {
			if !astTokenType(t) {
				return astUnsupported("non-token join operand")
			}
		}
		return nil
	}
	tile, ok := result.(*TileType)
	if !ok {
		return astUnsupported("numeric operation needs tile result")
	}
	if !tile.ElementType.IsNumber() {
		return astUnsupported("numeric operation has pointer element type")
	}
	if name == "constant" {
		return nil
	}
	if name == "mmaf" {
		if len(operands) != 3 || !typeEqual(result, operands[2]) || len(tile.Shape) != 2 {
			return astUnsupported("mmaf accumulator type or arity")
		}
		a, aok := operands[0].(*TileType)
		b, bok := operands[1].(*TileType)
		if !aok || !bok || len(a.Shape) != 2 || len(b.Shape) != 2 ||
			a.Shape[1] != b.Shape[0] || a.Shape[0] != tile.Shape[0] || b.Shape[1] != tile.Shape[1] ||
			!typeEqual(a.ElementType, b.ElementType) {
			return astUnsupported("mmaf tile shape or element type")
		}
		// NVIDIA Ops.td:1489-1534 requires equal input element types and
		// enumerates the accumulator/result types for each input type.
		input, output := a.ElementType.Kind(), tile.ElementType.Kind()
		supported := false
		switch input {
		case KindFloat8E4M3, KindFloat8E5M2, KindFloat16:
			supported = output == KindFloat16 || output == KindFloat32
		case KindBFloat16, KindTFloat32, KindFloat32:
			supported = output == KindFloat32
		case KindFloat64:
			supported = output == KindFloat64
		}
		if !supported {
			return astUnsupported("mmaf input/accumulator element type combination")
		}
		return nil
	}
	start := 0
	if name == "select" {
		mask, ok := operands[0].(*TileType)
		if !ok || mask.ElementType.Kind() != KindInt1 || !sameASTShape(mask, tile) {
			return astUnsupported("select predicate type")
		}
		start = 1
	}
	for _, operand := range operands[start:] {
		want := result
		if name == "cmpf" || name == "cmpi" {
			want = operands[0]
			source, ok := want.(*TileType)
			if !ok || tile.ElementType.Kind() != KindInt1 || !sameASTShape(source, tile) {
				return astUnsupported("comparison result type")
			}
		}
		if !typeEqual(operand, want) {
			return astUnsupported("operand/result type mismatch in %s", name)
		}
	}
	if strings.HasSuffix(name, "f") {
		source := tile
		if name == "cmpf" {
			source, _ = operands[0].(*TileType)
		}
		if !source.ElementType.IsFloat() {
			return astUnsupported("floating operation has integer type")
		}
	} else if strings.HasSuffix(name, "i") {
		source := tile
		if name == "cmpi" {
			source, _ = operands[0].(*TileType)
		}
		if !source.ElementType.IsInt() {
			return astUnsupported("integer operation has floating type")
		}
	}
	return nil
}

func sameASTShape(a, b *TileType) bool {
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

func astTokenType(t Type) bool {
	return t != nil && t.Kind() == KindToken
}

// BytecodeGen.cpp:558 tests Operator::isVariadic, including variadic operands,
// before writing the result count. The vendored return goldens independently
// pin its zero result count followed by its zero operand count. opSpec's
// VariadicResults alone is therefore insufficient for this field.
func astResultCountField(spec opSpec) bool {
	if spec.VariadicResults {
		return true
	}
	for _, operand := range spec.Operands {
		if operand == operandVariadic {
			return true
		}
	}
	return false
}

func encodeASTAttrs(body []byte, spec opSpec, attrs map[string]string, result Type, bm *BytecodeModule) ([]byte, error) {
	for name := range attrs {
		found := false
		for _, a := range spec.Attributes {
			found = found || a.Name == name
		}
		if !found {
			return nil, astUnsupported("unknown attribute %q", name)
		}
	}
	var flags uint64
	for i, flag := range spec.Flags {
		value, present := attrs[flag.Name]
		if present {
			if value != "unit" {
				return nil, astUnsupported("unit attribute %s has payload", flag.Name)
			}
			flags |= 1 << i
		}
	}
	if spec.HasFlags() {
		body = appendVarInt(body, flags)
	}
	for _, a := range spec.Attributes {
		value, present := attrs[a.Name]
		if a.Unit || (a.Optional && !present) {
			continue
		}
		if !present {
			return nil, astUnsupported("missing attribute %s", a.Name)
		}
		if a.Name == "value" {
			data, err := encodeASTConstant(result, value)
			if err != nil {
				return nil, err
			}
			index := len(bm.Constants)
			for i, old := range bm.Constants {
				if bytes.Equal(old, data) {
					index = i
					break
				}
			}
			if index == len(bm.Constants) {
				bm.Constants = append(bm.Constants, data)
			}
			body = appendVarInt(body, uint64(index))
			continue
		}
		if strings.HasPrefix(value, "\"") || strings.HasSuffix(value, "\"") {
			var err error
			value, err = strconv.Unquote(value)
			if err != nil {
				return nil, astUnsupported("invalid quoted attribute %s", a.Name)
			}
		}
		choices, known := astEnums[a.Name]
		if !known {
			return nil, astUnsupported("attribute representation %s", a.Name)
		}
		index := -1
		for i, choice := range choices {
			if value == choice {
				index = i
			}
		}
		if index < 0 {
			return nil, astUnsupported("invalid %s value %q", a.Name, value)
		}
		body = appendVarInt(body, uint64(index))
	}
	return body, nil
}

func constantWidth(kind TypeKind) int {
	switch kind {
	case KindInt1, KindInt8, KindFloat8E4M3, KindFloat8E5M2, KindFloat8E8M0:
		return 1
	case KindInt16, KindFloat16, KindBFloat16:
		return 2
	case KindTFloat32:
		return 3
	case KindInt32, KindFloat32:
		return 4
	case KindInt64, KindFloat64:
		return 8
	}
	return 0
}

func encodeASTConstant(typ Type, value string) ([]byte, error) {
	tile, ok := typ.(*TileType)
	if !ok {
		return nil, astUnsupported("constant result is not a tile")
	}
	n, err := astElements(tile)
	if err != nil {
		return nil, err
	}
	width := constantWidth(tile.ElementType.Kind())
	if width == 0 {
		return nil, astUnsupported("constant element type %s", tile.ElementType)
	}
	var data []byte
	if strings.HasPrefix(value, `dense<"0x`) && strings.HasSuffix(value, `">`) {
		data, err = hex.DecodeString(value[len(`dense<"0x`) : len(value)-2])
		if err != nil {
			return nil, astUnsupported("invalid dense hexadecimal payload")
		}
	} else {
		parts := []string{value}
		if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
			parts = SplitOperands(value[1 : len(value)-1])
			if len(parts) != n {
				return nil, astUnsupported("constant list length differs from tile extent")
			}
		}
		for _, part := range parts {
			var bits uint64
			switch tile.ElementType.Kind() {
			case KindFloat32:
				f, e := strconv.ParseFloat(part, 32)
				err = e
				bits = uint64(math.Float32bits(float32(f)))
			case KindFloat64:
				f, e := strconv.ParseFloat(part, 64)
				err = e
				bits = math.Float64bits(f)
			case KindInt32, KindInt64:
				v, e := strconv.ParseInt(part, 0, width*8)
				err = e
				bits = uint64(v)
			default:
				return nil, astUnsupported("use dense hexadecimal bits for %s constants", tile.ElementType)
			}
			if err != nil {
				return nil, astUnsupported("constant %q: %v", part, err)
			}
			var raw [8]byte
			binary.LittleEndian.PutUint64(raw[:], bits)
			data = append(data, raw[:width]...)
		}
	}
	if err := checkConstantBytes(tile, n, width, data); err != nil {
		return nil, err
	}
	return append(appendVarInt(nil, uint64(len(data))), data...), nil
}

func checkConstantBytes(tile *TileType, n, width int, data []byte) error {
	if tile.ElementType.Kind() == KindInt1 {
		packed := n / 8
		if n%8 != 0 {
			packed++
		}
		if len(data) != packed && !(len(data) == 1 && (data[0] == 0 || data[0] == 0xff)) {
			return astUnsupported("invalid packed i1 constant length")
		}
	} else if len(data) != width && (n > int(^uint(0)>>1)/width || len(data) != n*width) {
		return astUnsupported("constant bytes do not match tile extent")
	}
	return nil
}

type astCursor struct {
	data []byte
	off  int
}

// decodeASTPoolType checks references and lengths for the types used by the
// supported AST subset. Other structural types retain the existing inspection
// decoder; checkASTType forbids them at the AST admission boundary.
func decodeASTPoolType(tag uint64, payload []byte, pool []Type, version BytecodeVersion) (Type, error) {
	if TypeKind(tag) == KindPointer {
		return decodeMemoryPointer(payload, pool, version)
	}
	if TypeKind(tag) != KindTile && TypeKind(tag) != KindFunction {
		t := decodeType(tag, payload, pool, version)
		if _, primitive := t.(PrimitiveType); primitive || tag == uint64(KindToken) {
			if len(payload) != 0 {
				return nil, fmt.Errorf("tileir: trailing primitive type payload")
			}
		}
		return t, nil
	}
	c := astCursor{data: payload}
	readType := func() (Type, error) {
		index, err := c.uint()
		if err != nil || index >= uint64(len(pool)) || pool[index] == nil {
			return nil, fmt.Errorf("tileir: invalid type pool reference")
		}
		return pool[index], nil
	}
	var typ Type
	if TypeKind(tag) == KindTile {
		element, err := readType()
		if err != nil {
			return nil, err
		}
		count, err := c.uint()
		if err != nil || count > uint64((len(payload)-c.off)/8) {
			return nil, fmt.Errorf("tileir: invalid tile shape length")
		}
		var shape []int64
		for i := uint64(0); i < count; i++ {
			shape = append(shape, int64(binary.LittleEndian.Uint64(payload[c.off:c.off+8])))
			c.off += 8
		}
		typ = NewTile(shape, element)
	} else {
		var parts [2][]Type
		for i := range parts {
			count, err := c.uint()
			if err != nil || count > uint64(len(payload)-c.off) {
				return nil, fmt.Errorf("tileir: invalid function type arity")
			}
			for j := uint64(0); j < count; j++ {
				t, err := readType()
				if err != nil {
					return nil, err
				}
				parts[i] = append(parts[i], t)
			}
		}
		typ = NewFunction(parts[0], parts[1])
	}
	if c.off != len(payload) {
		return nil, fmt.Errorf("tileir: trailing type payload")
	}
	return typ, nil
}

func (c *astCursor) uint() (uint64, error) {
	value, n := binary.Uvarint(c.data[c.off:])
	if n <= 0 {
		return 0, fmt.Errorf("tileir: malformed varint at body offset %d", c.off)
	}
	c.off += n
	return value, nil
}

func decodeASTBody(body []byte, types []Type, constants [][]byte, signature Type) ([]*Operation, error) {
	sig, ok := signature.(*FunctionType)
	if !ok || sig == nil {
		return nil, astUnsupported("function signature type")
	}
	for _, t := range append(append([]Type(nil), sig.Inputs...), sig.Results...) {
		if err := checkASTType(t); err != nil {
			return nil, err
		}
	}
	values := append([]Type(nil), sig.Inputs...)
	names := make([]string, len(values))
	for i := range names {
		names[i] = fmt.Sprintf("%%arg%d", i)
	}
	var ops []*Operation
	tokenUses := make(map[uint64]int)
	c := astCursor{data: body}
	for c.off < len(body) {
		code, err := c.uint()
		if err != nil {
			return nil, err
		}
		name, known := OpcodeNames[uint32(code)]
		short := strings.TrimPrefix(name, "cuda_tile.")
		if code > math.MaxUint32 || !known || !astBodyOps[short] {
			return nil, astUnsupported("body opcode 0x%x at offset %d", code, c.off)
		}
		spec, _ := astOpSpec(name)
		if astResultCountField(spec) {
			count, err := c.uint()
			if err != nil || count != uint64(spec.NumResults) {
				return nil, fmt.Errorf("tileir: invalid result count")
			}
		}
		op := &Operation{OpCode: name}
		var result Type
		var results []Type
		for j := 0; j < spec.NumResults; j++ {
			index, err := c.uint()
			if err != nil || index >= uint64(len(types)) || types[index] == nil {
				return nil, fmt.Errorf("tileir: invalid result type index")
			}
			t := types[index]
			if err := checkASTType(t); err != nil {
				return nil, err
			}
			results = append(results, t)
			op.Results = append(op.Results, fmt.Sprintf("%%%d", len(values)-len(sig.Inputs)+j))
		}
		if len(results) != 0 {
			result = resultTuple(results)
			op.TypeVal, op.Type = result, result.String()
		}
		var flags uint64
		if spec.HasFlags() {
			flags, err = c.uint()
			if err != nil || flags>>len(spec.Flags) != 0 {
				return nil, fmt.Errorf("tileir: invalid operation flags")
			}
		}
		if memoryOp(name) && flags&3 != 0 {
			return nil, astUnsupported("memory scope or optimization hints")
		}
		var attrs []string
		for i, a := range spec.Attributes {
			if a.Optional && !a.Unit && memoryOp(name) {
				continue
			}
			if a.Unit {
				for bit, f := range spec.Flags {
					if f.Index == i && flags&(1<<bit) != 0 {
						attrs = append(attrs, a.Name)
					}
				}
				continue
			}
			index, err := c.uint()
			if err != nil {
				return nil, err
			}
			if a.Name == "value" {
				if index >= uint64(len(constants)) {
					return nil, fmt.Errorf("tileir: invalid constant pool index")
				}
				pool := constants[index]
				length, n := binary.Uvarint(pool)
				tile, ok := result.(*TileType)
				if !ok || n <= 0 || length != uint64(len(pool)-n) {
					return nil, fmt.Errorf("tileir: malformed dense constant")
				}
				count, err := astElements(tile)
				if err != nil {
					return nil, err
				}
				width := constantWidth(tile.ElementType.Kind())
				if width == 0 {
					return nil, astUnsupported("constant type %s", tile.ElementType)
				}
				if err := checkConstantBytes(tile, count, width, pool[n:]); err != nil {
					return nil, err
				}
				attrs = append(attrs, `value = dense<"0x`+hex.EncodeToString(pool[n:])+`">`)
			} else {
				choices := astEnums[a.Name]
				if index >= uint64(len(choices)) {
					return nil, fmt.Errorf("tileir: invalid %s enum", a.Name)
				}
				attrs = append(attrs, a.Name+" = "+choices[index])
				if memoryOp(name) && !memoryOrderingAllowed(name, index) {
					return nil, fmt.Errorf("tileir: invalid memory ordering for %s", name)
				}
			}
		}
		if len(attrs) != 0 {
			op.Attributes = "{" + strings.Join(attrs, ", ") + "}"
		}
		count := uint64(len(spec.Operands))
		if memoryOp(name) {
			count = memoryOperandCount(name, flags)
		}
		if spec.EncodeOperandCount {
			count, err = c.uint()
			if err != nil {
				return nil, err
			}
		}
		if count > uint64(len(body)-c.off) {
			return nil, fmt.Errorf("tileir: operand count exceeds body")
		}
		var operands []Type
		for i := uint64(0); i < count; i++ {
			index, err := c.uint()
			if err != nil || index >= uint64(len(values)) {
				return nil, fmt.Errorf("tileir: undefined operand index")
			}
			op.Operands = append(op.Operands, names[index])
			operands = append(operands, values[index])
			if astTokenType(values[index]) {
				tokenUses[index]++
			}
		}
		if err := checkASTOperands(short, result, operands, sig.Results); err != nil {
			return nil, err
		}
		if memoryOp(name) {
			expected, err := memoryFlags(name, operands)
			if err != nil || expected != flags {
				return nil, fmt.Errorf("tileir: memory operand groups disagree with flags")
			}
		}
		if short == "return" && c.off != len(body) {
			return nil, fmt.Errorf("tileir: operations after return")
		}
		if result != nil {
			values = append(values, results...)
			names = append(names, op.Results...)
		}
		ops = append(ops, op)
	}
	if len(ops) == 0 || ops[len(ops)-1].OpCode != "cuda_tile.return" {
		return nil, fmt.Errorf("tileir: body has no return")
	}
	terminal, err := checkMemoryChain(ops, sig.Inputs, sig.Results, true)
	if err != nil {
		return nil, err
	}
	for i, t := range values {
		if astTokenType(t) && tokenUses[uint64(i)] != 1 && !(names[i] == terminal && tokenUses[uint64(i)] == 0) {
			return nil, astUnsupported("token value %d must have exactly one consumer", i)
		}
	}
	return ops, nil
}
