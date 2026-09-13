package tileir

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Magic header bytes for CUDA Tile IR bytecode files.
var Magic = [8]byte{0x7f, 'T', 'i', 'l', 'e', 'I', 'R', 0x00}

// Section IDs defined by the Tile IR bytecode format specification.
const (
	SectionEndOfBytecode uint8 = 0x00
	SectionString        uint8 = 0x01
	SectionFunc          uint8 = 0x02
	SectionDebug         uint8 = 0x03
	SectionConstant      uint8 = 0x04
	SectionType          uint8 = 0x05
	SectionGlobal        uint8 = 0x06
	SectionProducer      uint8 = 0x07
)

// Function entry flags.
const (
	FuncVisibilityPrivate uint8 = 0x01
	FuncKindKernel        uint8 = 0x02
	FuncHasOptHints       uint8 = 0x04
)

// Known Tile IR operation opcodes.
var OpcodeNames = map[uint32]string{
	0x00: "cuda_tile.abs_f",
	0x01: "cuda_tile.abs_i",
	0x02: "cuda_tile.add_f",
	0x03: "cuda_tile.add_i",
	0x04: "cuda_tile.and_i",
	0x05: "cuda_tile.assert",
	0x06: "cuda_tile.assume",
	0x07: "cuda_tile.atomic_cas_tko",
	0x08: "cuda_tile.atomic_rmw_tko",
	0x09: "cuda_tile.bitcast",
	0x0A: "cuda_tile.break",
	0x0B: "cuda_tile.broadcast",
	0x0C: "cuda_tile.cat",
	0x0D: "cuda_tile.ceil",
	0x0E: "cuda_tile.cmp_f",
	0x0F: "cuda_tile.cmp_i",
	0x10: "cuda_tile.constant",
	0x11: "cuda_tile.continue",
	0x12: "cuda_tile.cos",
	0x13: "cuda_tile.cosh",
	0x14: "cuda_tile.div_f",
	0x15: "cuda_tile.div_i",
	0x16: "cuda_tile.entry",
	0x17: "cuda_tile.exp",
	0x18: "cuda_tile.exp2",
	0x25: "cuda_tile.ext_i",
	0x26: "cuda_tile.extract",
	0x27: "cuda_tile.floor",
	0x28: "cuda_tile.fma",
	0x29: "cuda_tile.for",
	0x2A: "cuda_tile.ftof",
	0x2B: "cuda_tile.ftoi",
	0x2C: "cuda_tile.get_global",
	0x2D: "cuda_tile.get_index_space_shape",
	0x2E: "cuda_tile.get_num_tile_blocks",
	0x2F: "cuda_tile.get_tensor_shape",
	0x30: "cuda_tile.get_tile_block_id",
	0x31: "cuda_tile.global",
	0x32: "cuda_tile.if",
	0x33: "cuda_tile.int_to_ptr",
	0x3A: "cuda_tile.iota",
	0x3B: "cuda_tile.itof",
	0x3C: "cuda_tile.join_tokens",
	0x3D: "cuda_tile.load_ptr_tko",
	0x3E: "cuda_tile.load_view_tko",
	0x3F: "cuda_tile.log",
	0x40: "cuda_tile.log2",
	0x41: "cuda_tile.loop",
	0x42: "cuda_tile.make_partition_view",
	0x43: "cuda_tile.make_tensor_view",
	0x44: "cuda_tile.make_token",
	0x45: "cuda_tile.max_f",
	0x46: "cuda_tile.max_i",
	0x47: "cuda_tile.min_f",
	0x48: "cuda_tile.min_i",
	0x49: "cuda_tile.mma_f",
	0x4A: "cuda_tile.mma_i",
	0x4B: "cuda_tile.module",
	0x4C: "cuda_tile.mul_f",
	0x4D: "cuda_tile.mulhi_i",
	0x4E: "cuda_tile.mul_i",
	0x4F: "cuda_tile.neg_f",
	0x50: "cuda_tile.neg_i",
	0x51: "cuda_tile.offset",
	0x52: "cuda_tile.or_i",
	0x53: "cuda_tile.permute",
	0x54: "cuda_tile.fpow_f",
	0x55: "cuda_tile.print_tko",
	0x56: "cuda_tile.ptr_to_int",
	0x57: "cuda_tile.ptr_to_ptr",
	0x58: "cuda_tile.reduce",
	0x59: "cuda_tile.rem_f",
	0x5A: "cuda_tile.rem_i",
	0x5B: "cuda_tile.reshape",
	0x5C: "cuda_tile.return",
	0x5D: "cuda_tile.rsqrt",
	0x5E: "cuda_tile.scan",
	0x5F: "cuda_tile.select",
	0x60: "cuda_tile.shl_i",
}

// BytecodeVersion captures the major, minor, and tag version components.
type BytecodeVersion struct {
	Major uint8
	Minor uint8
	Tag   uint16
}

func (v BytecodeVersion) String() string {
	if v.Tag != 0 {
		return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Tag)
	}
	return fmt.Sprintf("%d.%d", v.Major, v.Minor)
}

// BytecodeModule represents a decoded binary Tile IR module.
type BytecodeModule struct {
	Version   BytecodeVersion
	Producer  string
	Strings   []string
	Types     []Type
	Constants [][]byte
	Functions []*BytecodeFunction
	Globals   []*BytecodeGlobal
}

// BytecodeFunction represents a decoded function record from the function table.
type BytecodeFunction struct {
	Name           string
	SignatureIndex int
	Signature      Type
	EntryFlag      uint8
	IsEntry        bool
	IsPrivate      bool
	HasOptHints    bool
	LocIndex       int
	BodyBytes      []byte
	Operations     []*Operation
}

// BytecodeGlobal represents a decoded global variable declaration.
type BytecodeGlobal struct {
	SymbolName    string
	TypeIndex     int
	Type          Type
	ConstantIndex int
	LocIndex      int
}

// IsTileIRBytecode checks if data begins with the Tile IR magic header.
func IsTileIRBytecode(data []byte) bool {
	if len(data) < len(Magic) {
		return false
	}
	return bytes.Equal(data[:len(Magic)], Magic[:])
}

// DecodeBytecode decodes a binary Tile IR bytecode stream from an io.Reader.
func DecodeBytecode(r io.Reader) (*BytecodeModule, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("reading tileir bytecode: %w", err)
	}
	return DecodeBytecodeBytes(data)
}

// DecodeBytecodeBytes decodes a binary Tile IR bytecode byte slice.
func DecodeBytecodeBytes(data []byte) (*BytecodeModule, error) {
	if !IsTileIRBytecode(data) {
		return nil, errors.New("invalid tileir bytecode magic header")
	}
	if len(data) < 12 {
		return nil, errors.New("truncated tileir bytecode header")
	}

	bm := &BytecodeModule{
		Version: BytecodeVersion{
			Major: data[8],
			Minor: data[9],
			Tag:   binary.LittleEndian.Uint16(data[10:12]),
		},
	}

	// Read sections
	offset := 12
	payloads := make(map[uint8][]byte)

	for offset < len(data) {
		idAndAlign := data[offset]
		offset++
		secID := idAndAlign & 0x7F
		hasAlign := (idAndAlign & 0x80) != 0

		if secID == SectionEndOfBytecode {
			break
		}

		length, n := readVarInt(data[offset:])
		if n <= 0 {
			return nil, errors.New("failed to read section length")
		}
		offset += n

		if hasAlign {
			align, an := readVarInt(data[offset:])
			if an <= 0 {
				return nil, errors.New("failed to read section alignment")
			}
			offset += an

			if align > 1 {
				rem := offset % int(align)
				pad := (int(align) - rem) % int(align)
				offset += pad
			}
		}

		if offset+int(length) > len(data) {
			return nil, fmt.Errorf("section %d length %d exceeds remaining buffer", secID, length)
		}

		payloads[secID] = data[offset : offset+int(length)]
		offset += int(length)
	}

	// 1. String Table
	if strPayload, ok := payloads[SectionString]; ok {
		strs, err := parseStringSection(strPayload)
		if err != nil {
			return nil, fmt.Errorf("parsing string section: %w", err)
		}
		bm.Strings = strs
	}

	// 2. Producer Section (optional)
	if prodPayload, ok := payloads[SectionProducer]; ok {
		idx, _ := readVarInt(prodPayload)
		if int(idx) < len(bm.Strings) {
			bm.Producer = bm.Strings[idx]
		}
	}

	// 3. Constant Section
	if constPayload, ok := payloads[SectionConstant]; ok {
		constants, err := parseConstantSection(constPayload)
		if err != nil {
			return nil, fmt.Errorf("parsing constant section: %w", err)
		}
		bm.Constants = constants
	}

	// 4. Type Section
	if typePayload, ok := payloads[SectionType]; ok {
		types, err := parseTypeSection(typePayload)
		if err != nil {
			return nil, fmt.Errorf("parsing type section: %w", err)
		}
		bm.Types = types
	}

	// 5. Function Table Section
	if funcPayload, ok := payloads[SectionFunc]; ok {
		fns, err := parseFunctionSection(funcPayload, bm.Strings, bm.Types)
		if err != nil {
			return nil, fmt.Errorf("parsing function section: %w", err)
		}
		bm.Functions = fns
	}

	return bm, nil
}

// ToModule converts the decoded BytecodeModule into an AST Module.
func (bm *BytecodeModule) ToModule() *Module {
	if bm == nil {
		return nil
	}
	mod := &Module{
		Name:      "kernels",
		Producer:  bm.Producer,
		Locations: make(map[string]Location),
	}

	for _, bfn := range bm.Functions {
		fn := &Function{
			Name:       bfn.Name,
			Entry:      bfn.IsEntry,
			Operations: bfn.Operations,
		}

		// Reconstruct signature
		if sig, ok := bfn.Signature.(*FunctionType); ok {
			for i, in := range sig.Inputs {
				fn.Params = append(fn.Params, &Param{
					Name:    fmt.Sprintf("%%arg%d", i),
					Type:    in.String(),
					TypeVal: in,
				})
			}
			for _, res := range sig.Results {
				fn.Returns = append(fn.Returns, res.String())
			}
		}

		if len(fn.Operations) > 0 {
			blk := &Block{
				Name:       "^bb0",
				Operations: fn.Operations,
			}
			fn.Blocks = append(fn.Blocks, blk)
		}

		mod.Functions = append(mod.Functions, fn)
	}

	return mod
}

func parseStringSection(payload []byte) ([]string, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	numStrings, n := readVarInt(payload)
	if n <= 0 {
		return nil, errors.New("failed to read string table size")
	}
	off := n
	// Align to 4 bytes
	rem := off % 4
	pad := (4 - rem) % 4
	off += pad

	if off+int(numStrings)*4 > len(payload) {
		return nil, errors.New("string offset array exceeds payload")
	}

	offsets := make([]uint32, numStrings)
	for i := uint64(0); i < numStrings; i++ {
		offsets[i] = binary.LittleEndian.Uint32(payload[off : off+4])
		off += 4
	}

	strData := payload[off:]
	strs := make([]string, numStrings)
	for i := 0; i < int(numStrings); i++ {
		start := int(offsets[i])
		end := len(strData)
		if i+1 < len(offsets) {
			end = int(offsets[i+1])
		}
		if start <= len(strData) && end <= len(strData) && start <= end {
			strs[i] = string(strData[start:end])
		}
	}
	return strs, nil
}

func parseConstantSection(payload []byte) ([][]byte, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	numConstants, n := readVarInt(payload)
	if n <= 0 {
		return nil, errors.New("failed to read constant pool size")
	}
	off := n
	// Align to 8 bytes
	rem := off % 8
	pad := (8 - rem) % 8
	off += pad

	if off+int(numConstants)*8 > len(payload) {
		return nil, errors.New("constant offset array exceeds payload")
	}

	offsets := make([]uint64, numConstants)
	for i := uint64(0); i < numConstants; i++ {
		offsets[i] = binary.LittleEndian.Uint64(payload[off : off+8])
		off += 8
	}

	data := payload[off:]
	constants := make([][]byte, numConstants)
	for i := 0; i < int(numConstants); i++ {
		start := int(offsets[i])
		end := len(data)
		if i+1 < len(offsets) {
			end = int(offsets[i+1])
		}
		if start <= len(data) && end <= len(data) && start <= end {
			constants[i] = data[start:end]
		}
	}
	return constants, nil
}

func parseTypeSection(payload []byte) ([]Type, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	numTypes, n := readVarInt(payload)
	if n <= 0 {
		return nil, errors.New("failed to read type table size")
	}
	off := n
	// Align to 4 bytes
	rem := off % 4
	pad := (4 - rem) % 4
	off += pad

	if off+int(numTypes)*4 > len(payload) {
		return nil, errors.New("type offset array exceeds payload")
	}

	offsets := make([]uint32, numTypes)
	for i := uint64(0); i < numTypes; i++ {
		offsets[i] = binary.LittleEndian.Uint32(payload[off : off+4])
		off += 4
	}

	typeData := payload[off:]
	types := make([]Type, numTypes)

	// Two passes: first allocate placeholders, then decode parameters
	for i := 0; i < int(numTypes); i++ {
		start := int(offsets[i])
		end := len(typeData)
		if i+1 < len(offsets) {
			end = int(offsets[i+1])
		}
		if start >= len(typeData) || end > len(typeData) || start > end {
			continue
		}
		raw := typeData[start:end]
		if len(raw) == 0 {
			continue
		}
		tag, tn := readVarInt(raw)
		types[i] = decodeType(tag, raw[tn:], types)
	}

	return types, nil
}

func decodeType(tag uint64, payload []byte, typePool []Type) Type {
	switch TypeKind(tag) {
	case KindInt1:
		return Int1
	case KindInt4:
		return Int4
	case KindInt8:
		return Int8
	case KindInt16:
		return Int16
	case KindInt32:
		return Int32
	case KindInt64:
		return Int64
	case KindFloat16:
		return Float16
	case KindBFloat16:
		return BFloat16
	case KindFloat32:
		return Float32
	case KindTFloat32:
		return TFloat32
	case KindFloat64:
		return Float64
	case KindFloat8E4M3:
		return Float8E4M3
	case KindFloat8E5M2:
		return Float8E5M2
	case KindFloat8E8M0:
		return Float8E8M0
	case KindFloat4E2M1:
		return Float4E2M1
	case KindFloat8E5M3:
		return Float8E5M3
	case KindToken:
		return Token
	case KindPointer:
		if len(payload) > 0 {
			pointeeIdx, _ := readVarInt(payload)
			var pointee Type = Float32
			if int(pointeeIdx) < len(typePool) && typePool[pointeeIdx] != nil {
				pointee = typePool[pointeeIdx]
			}
			return &PointerType{Pointee: pointee}
		}
		return &PointerType{Pointee: Float32}
	case KindTile:
		// TileType has needsReverseOrder=true:
		// elementType (type index varint), then shape (rank varint + rank int64 LE values)
		if len(payload) == 0 {
			return &TileType{ElementType: Float32}
		}
		elemIdx, n := readVarInt(payload)
		var elem Type = Float32
		if int(elemIdx) < len(typePool) && typePool[elemIdx] != nil {
			elem = typePool[elemIdx]
		}
		off := n
		if off < len(payload) {
			rank, rn := readVarInt(payload[off:])
			off += rn
			var shape []int64
			for r := uint64(0); r < rank && off+8 <= len(payload); r++ {
				dim := int64(binary.LittleEndian.Uint64(payload[off : off+8]))
				shape = append(shape, dim)
				off += 8
			}
			return &TileType{Shape: shape, ElementType: elem}
		}
		return &TileType{ElementType: elem}
	case KindTensorView:
		if len(payload) == 0 {
			return &TensorViewType{ElementType: Float32}
		}
		elemIdx, _ := readVarInt(payload)
		var elem Type = Float32
		if int(elemIdx) < len(typePool) && typePool[elemIdx] != nil {
			elem = typePool[elemIdx]
		}
		return &TensorViewType{ElementType: elem}
	case KindFunction:
		// Function type: inputs array, results array
		off := 0
		var inputs []Type
		var results []Type
		if off < len(payload) {
			numIn, n := readVarInt(payload[off:])
			off += n
			for j := uint64(0); j < numIn && off < len(payload); j++ {
				inIdx, inN := readVarInt(payload[off:])
				off += inN
				if int(inIdx) < len(typePool) && typePool[inIdx] != nil {
					inputs = append(inputs, typePool[inIdx])
				}
			}
		}
		if off < len(payload) {
			numRes, n := readVarInt(payload[off:])
			off += n
			for j := uint64(0); j < numRes && off < len(payload); j++ {
				resIdx, resN := readVarInt(payload[off:])
				off += resN
				if int(resIdx) < len(typePool) && typePool[resIdx] != nil {
					results = append(results, typePool[resIdx])
				}
			}
		}
		return &FunctionType{Inputs: inputs, Results: results}
	}
	return PrimitiveType{kind: TypeKind(tag), name: TypeKind(tag).String()}
}

func parseFunctionSection(payload []byte, strTable []string, typeTable []Type) ([]*BytecodeFunction, error) {
	if len(payload) == 0 {
		return nil, nil
	}
	numFunctions, n := readVarInt(payload)
	if n <= 0 {
		return nil, errors.New("failed to read function count")
	}
	off := n

	var fns []*BytecodeFunction
	for i := uint64(0); i < numFunctions; i++ {
		if off >= len(payload) {
			break
		}
		nameIdx, nn := readVarInt(payload[off:])
		off += nn

		sigIdx, sn := readVarInt(payload[off:])
		off += sn

		if off >= len(payload) {
			return nil, errors.New("truncated function entry flag")
		}
		entryFlag := payload[off]
		off++

		locIdx, ln := readVarInt(payload[off:])
		off += ln

		isKernel := (entryFlag & FuncKindKernel) != 0
		hasOptHints := (entryFlag & FuncHasOptHints) != 0
		if isKernel && hasOptHints {
			// Skip optimization hints attribute
			if off < len(payload) {
				_, hn := readVarInt(payload[off:])
				off += hn
			}
		}

		bodyLen, bn := readVarInt(payload[off:])
		off += bn

		if off+int(bodyLen) > len(payload) {
			return nil, errors.New("function body exceeds section payload")
		}
		bodyBytes := payload[off : off+int(bodyLen)]
		off += int(bodyLen)

		fnName := fmt.Sprintf("func_%d", i)
		if int(nameIdx) < len(strTable) {
			fnName = strTable[nameIdx]
		}

		var sig Type = NewFunction(nil, nil)
		if int(sigIdx) < len(typeTable) && typeTable[sigIdx] != nil {
			sig = typeTable[sigIdx]
		}

		bfn := &BytecodeFunction{
			Name:           fnName,
			SignatureIndex: int(sigIdx),
			Signature:      sig,
			EntryFlag:      entryFlag,
			IsEntry:        isKernel,
			IsPrivate:      (entryFlag & FuncVisibilityPrivate) != 0,
			HasOptHints:    hasOptHints,
			LocIndex:       int(locIdx),
			BodyBytes:      bodyBytes,
			Operations:     parseBodyOperations(bodyBytes, typeTable),
		}
		fns = append(fns, bfn)
	}

	return fns, nil
}

func parseBodyOperations(body []byte, typeTable []Type) []*Operation {
	var ops []*Operation
	off := 0
	valCounter := 0

	for off < len(body) {
		opCodeNum, on := readVarInt(body[off:])
		off += on

		opName := fmt.Sprintf("op_0x%x", opCodeNum)
		if name, ok := OpcodeNames[uint32(opCodeNum)]; ok {
			opName = name
		}

		op := &Operation{
			OpCode: opName,
		}

		// Synthesize result SSA values for non-terminator operations
		if opName != "cuda_tile.return" && opName != "cuda_tile.break" && opName != "cuda_tile.continue" {
			res := fmt.Sprintf("%%%d", valCounter)
			op.Results = append(op.Results, res)
			valCounter++
		}

		// Advance past operation operands / attributes in stream
		// Each remaining byte is safely consumed until end of body or next recognized opcode
		if opCodeNum == 0x10 { // ConstantOp
			if off < len(body) {
				typeIdx, tn := readVarInt(body[off:])
				off += tn
				if int(typeIdx) < len(typeTable) && typeTable[typeIdx] != nil {
					op.Type = typeTable[typeIdx].String()
					op.TypeVal = typeTable[typeIdx]
				}
				// Skip constant payload index/flags
				if off < len(body) {
					_, cn := readVarInt(body[off:])
					off += cn
				}
			}
		} else if opCodeNum == 0x5c { // ReturnOp
			// Return has no operands or type
		}

		ops = append(ops, op)
	}

	return ops
}

func readVarInt(b []byte) (uint64, int) {
	var result uint64
	var shift uint
	for i, val := range b {
		result |= uint64(val&0x7f) << shift
		shift += 7
		if val&0x80 == 0 {
			return result, i + 1
		}
		if shift >= 64 {
			return 0, -1
		}
	}
	return 0, -1
}
