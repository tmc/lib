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

// OpcodeNumbers maps known Tile IR operation mnemonics to their numeric opcodes.
var OpcodeNumbers = func() map[string]uint32 {
	m := make(map[string]uint32, len(OpcodeNames))
	for k, v := range OpcodeNames {
		m[v] = k
	}
	return m
}()

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

func appendVarInt(b []byte, v uint64) []byte {
	for v >= 0x80 {
		b = append(b, byte(v)|0x80)
		v >>= 7
	}
	return append(b, byte(v))
}

// EncodeBytecode encodes a BytecodeModule into binary Tile IR bytecode.
func EncodeBytecode(bm *BytecodeModule) ([]byte, error) {
	if bm == nil {
		return nil, errors.New("tileir: nil bytecode module")
	}

	var buf bytes.Buffer

	// 1. Magic header (8 bytes)
	buf.Write(Magic[:])

	// 2. Version (4 bytes: major, minor, tag uint16 LE)
	major := bm.Version.Major
	minor := bm.Version.Minor
	tag := bm.Version.Tag
	if major == 0 && minor == 0 {
		major = 13
		minor = 4
	}
	buf.WriteByte(major)
	buf.WriteByte(minor)
	var tagBytes [2]byte
	binary.LittleEndian.PutUint16(tagBytes[:], tag)
	buf.Write(tagBytes[:])

	// Helper to write a section
	writeSection := func(secID uint8, payload []byte) {
		buf.WriteByte(secID) // no alignment flag
		buf.Write(appendVarInt(nil, uint64(len(payload))))
		buf.Write(payload)
	}

	// 3. String section (SectionString = 0x01)
	if len(bm.Strings) > 0 {
		strPayload := encodeStringSection(bm.Strings)
		writeSection(SectionString, strPayload)
	}

	// 4. Producer section (SectionProducer = 0x07)
	if bm.Producer != "" {
		prodIdx := -1
		for i, s := range bm.Strings {
			if s == bm.Producer {
				prodIdx = i
				break
			}
		}
		if prodIdx >= 0 {
			prodPayload := appendVarInt(nil, uint64(prodIdx))
			writeSection(SectionProducer, prodPayload)
		}
	}

	// 5. Constant section (SectionConstant = 0x04)
	if len(bm.Constants) > 0 {
		constPayload := encodeConstantSection(bm.Constants)
		writeSection(SectionConstant, constPayload)
	}

	// 6. Type section (SectionType = 0x05)
	if len(bm.Types) > 0 {
		typePayload, err := encodeTypeSection(bm.Types)
		if err != nil {
			return nil, fmt.Errorf("encoding type section: %w", err)
		}
		writeSection(SectionType, typePayload)
	}

	// 7. Function section (SectionFunc = 0x02)
	if len(bm.Functions) > 0 {
		funcPayload := encodeFunctionSection(bm.Functions, bm.Strings, bm.Types)
		writeSection(SectionFunc, funcPayload)
	}

	// 8. End of bytecode (SectionEndOfBytecode = 0x00)
	buf.WriteByte(SectionEndOfBytecode)

	return buf.Bytes(), nil
}

func encodeStringSection(strs []string) []byte {
	numStrings := uint64(len(strs))
	payload := appendVarInt(nil, numStrings)

	// Align to 4 bytes
	rem := len(payload) % 4
	pad := (4 - rem) % 4
	for i := 0; i < pad; i++ {
		payload = append(payload, 0)
	}

	offsets := make([]byte, numStrings*4)
	var strData []byte
	for i, s := range strs {
		binary.LittleEndian.PutUint32(offsets[i*4:], uint32(len(strData)))
		strData = append(strData, []byte(s)...)
	}

	payload = append(payload, offsets...)
	payload = append(payload, strData...)
	return payload
}

func encodeConstantSection(constants [][]byte) []byte {
	numConstants := uint64(len(constants))
	payload := appendVarInt(nil, numConstants)

	// Align to 8 bytes
	rem := len(payload) % 8
	pad := (8 - rem) % 8
	for i := 0; i < pad; i++ {
		payload = append(payload, 0)
	}

	offsets := make([]byte, numConstants*8)
	var data []byte
	for i, c := range constants {
		binary.LittleEndian.PutUint64(offsets[i*8:], uint64(len(data)))
		data = append(data, c...)
	}

	payload = append(payload, offsets...)
	payload = append(payload, data...)
	return payload
}

func encodeTypeSection(types []Type) ([]byte, error) {
	numTypes := uint64(len(types))
	payload := appendVarInt(nil, numTypes)

	// Align to 4 bytes
	rem := len(payload) % 4
	pad := (4 - rem) % 4
	for i := 0; i < pad; i++ {
		payload = append(payload, 0)
	}

	offsets := make([]byte, numTypes*4)
	var typeData []byte
	for i, t := range types {
		binary.LittleEndian.PutUint32(offsets[i*4:], uint32(len(typeData)))
		raw := encodeSingleType(t, types)
		typeData = append(typeData, raw...)
	}

	payload = append(payload, offsets...)
	payload = append(payload, typeData...)
	return payload, nil
}

func encodeSingleType(t Type, pool []Type) []byte {
	if t == nil {
		return appendVarInt(nil, uint64(KindInt32))
	}
	switch pt := t.(type) {
	case PrimitiveType:
		return appendVarInt(nil, uint64(pt.Kind()))
	case TokenType:
		return appendVarInt(nil, uint64(KindToken))
	case *PointerType:
		pIdx := findTypeIndex(pool, pt.Pointee)
		b := appendVarInt(nil, uint64(KindPointer))
		return appendVarInt(b, uint64(pIdx))
	case *TileType:
		eIdx := findTypeIndex(pool, pt.ElementType)
		b := appendVarInt(nil, uint64(KindTile))
		b = appendVarInt(b, uint64(eIdx))
		b = appendVarInt(b, uint64(len(pt.Shape)))
		for _, dim := range pt.Shape {
			var dimBytes [8]byte
			binary.LittleEndian.PutUint64(dimBytes[:], uint64(dim))
			b = append(b, dimBytes[:]...)
		}
		return b
	case *TensorViewType:
		eIdx := findTypeIndex(pool, pt.ElementType)
		b := appendVarInt(nil, uint64(KindTensorView))
		return appendVarInt(b, uint64(eIdx))
	case *FunctionType:
		b := appendVarInt(nil, uint64(KindFunction))
		b = appendVarInt(b, uint64(len(pt.Inputs)))
		for _, in := range pt.Inputs {
			b = appendVarInt(b, uint64(findTypeIndex(pool, in)))
		}
		b = appendVarInt(b, uint64(len(pt.Results)))
		for _, res := range pt.Results {
			b = appendVarInt(b, uint64(findTypeIndex(pool, res)))
		}
		return b
	default:
		return appendVarInt(nil, uint64(t.Kind()))
	}
}

func findTypeIndex(pool []Type, t Type) int {
	if t == nil {
		return 0
	}
	for i, existing := range pool {
		if typeEqual(existing, t) {
			return i
		}
	}
	return 0
}

func typeEqual(a, b Type) bool {
	if a == nil || b == nil {
		return a == b
	}
	if a.Kind() != b.Kind() {
		return false
	}
	switch ta := a.(type) {
	case PrimitiveType:
		tb, ok := b.(PrimitiveType)
		return ok && ta.kind == tb.kind
	case *PointerType:
		tb, ok := b.(*PointerType)
		return ok && typeEqual(ta.Pointee, tb.Pointee)
	case *TileType:
		tb, ok := b.(*TileType)
		if !ok || len(ta.Shape) != len(tb.Shape) || !typeEqual(ta.ElementType, tb.ElementType) {
			return false
		}
		for i := range ta.Shape {
			if ta.Shape[i] != tb.Shape[i] {
				return false
			}
		}
		return true
	case *TensorViewType:
		tb, ok := b.(*TensorViewType)
		if !ok || len(ta.Shape) != len(tb.Shape) || !typeEqual(ta.ElementType, tb.ElementType) {
			return false
		}
		for i := range ta.Shape {
			if ta.Shape[i] != tb.Shape[i] {
				return false
			}
		}
		return true
	case *FunctionType:
		tb, ok := b.(*FunctionType)
		if !ok || len(ta.Inputs) != len(tb.Inputs) || len(ta.Results) != len(tb.Results) {
			return false
		}
		for i := range ta.Inputs {
			if !typeEqual(ta.Inputs[i], tb.Inputs[i]) {
				return false
			}
		}
		for i := range ta.Results {
			if !typeEqual(ta.Results[i], tb.Results[i]) {
				return false
			}
		}
		return true
	case TokenType:
		_, ok := b.(TokenType)
		return ok
	}
	return a.String() == b.String()
}

func encodeFunctionSection(fns []*BytecodeFunction, strTable []string, typeTable []Type) []byte {
	payload := appendVarInt(nil, uint64(len(fns)))
	for _, fn := range fns {
		nameIdx := 0
		for i, s := range strTable {
			if s == fn.Name {
				nameIdx = i
				break
			}
		}
		payload = appendVarInt(payload, uint64(nameIdx))
		payload = appendVarInt(payload, uint64(fn.SignatureIndex))
		payload = append(payload, fn.EntryFlag)
		payload = appendVarInt(payload, uint64(fn.LocIndex))

		bodyBytes := fn.BodyBytes
		if len(bodyBytes) == 0 && len(fn.Operations) > 0 {
			for _, op := range fn.Operations {
				opName := op.OpCode
				opCodeNum, ok := OpcodeNumbers[opName]
				if !ok {
					opCodeNum = 0x00
				}
				bodyBytes = appendVarInt(bodyBytes, uint64(opCodeNum))
				if opCodeNum == 0x10 { // constant
					tIdx := 0
					if op.TypeVal != nil {
						tIdx = findTypeIndex(typeTable, op.TypeVal)
					}
					bodyBytes = appendVarInt(bodyBytes, uint64(tIdx))
					bodyBytes = appendVarInt(bodyBytes, 0)
				}
			}
		}

		payload = appendVarInt(payload, uint64(len(bodyBytes)))
		payload = append(payload, bodyBytes...)
	}
	return payload
}

func registerType(pool *[]Type, t Type) int {
	if t == nil {
		return 0
	}
	// Register dependencies first
	switch pt := t.(type) {
	case *PointerType:
		registerType(pool, pt.Pointee)
	case *TileType:
		registerType(pool, pt.ElementType)
	case *TensorViewType:
		registerType(pool, pt.ElementType)
	case *FunctionType:
		for _, in := range pt.Inputs {
			registerType(pool, in)
		}
		for _, res := range pt.Results {
			registerType(pool, res)
		}
	}

	for i, existing := range *pool {
		if typeEqual(existing, t) {
			return i
		}
	}
	idx := len(*pool)
	*pool = append(*pool, t)
	return idx
}

func registerString(pool *[]string, s string) int {
	for i, existing := range *pool {
		if existing == s {
			return i
		}
	}
	idx := len(*pool)
	*pool = append(*pool, s)
	return idx
}

// ToBytecode converts an AST Module into a BytecodeModule suitable for binary serialization.
func (m *Module) ToBytecode() (*BytecodeModule, error) {
	if m == nil {
		return nil, errors.New("tileir: nil module")
	}

	bm := &BytecodeModule{
		Version:  BytecodeVersion{Major: 13, Minor: 4, Tag: 0},
		Producer: m.Producer,
	}

	var strPool []string
	var typePool []Type

	if m.Producer != "" {
		registerString(&strPool, m.Producer)
	}
	if m.Name != "" {
		registerString(&strPool, m.Name)
	}

	for _, fn := range m.Functions {
		registerString(&strPool, fn.Name)

		var inTypes []Type
		for _, p := range fn.Params {
			t := p.TypeVal
			if t == nil && p.Type != "" {
				t, _ = ParseType(p.Type)
			}
			if t == nil {
				t = Float32
			}
			registerType(&typePool, t)
			inTypes = append(inTypes, t)
		}

		var resTypes []Type
		for _, r := range fn.Returns {
			t, _ := ParseType(r)
			if t != nil {
				registerType(&typePool, t)
				resTypes = append(resTypes, t)
			}
		}

		fnSig := NewFunction(inTypes, resTypes)
		sigIdx := registerType(&typePool, fnSig)

		entryFlag := uint8(0)
		if fn.Entry {
			entryFlag |= FuncKindKernel
		}

		var bodyBytes []byte
		for _, op := range fn.Operations {
			opName := op.OpCode
			opCodeNum, ok := OpcodeNumbers[opName]
			if !ok {
				opCodeNum = 0x00
			}
			bodyBytes = appendVarInt(bodyBytes, uint64(opCodeNum))
			if opCodeNum == 0x10 { // constant
				t := op.TypeVal
				if t == nil && op.Type != "" {
					t, _ = ParseType(op.Type)
				}
				if t == nil {
					t = Float32
				}
				tIdx := registerType(&typePool, t)
				bodyBytes = appendVarInt(bodyBytes, uint64(tIdx))
				bodyBytes = appendVarInt(bodyBytes, 0)
			}
		}

		bfn := &BytecodeFunction{
			Name:           fn.Name,
			SignatureIndex: sigIdx,
			Signature:      fnSig,
			EntryFlag:      entryFlag,
			IsEntry:        fn.Entry,
			BodyBytes:      bodyBytes,
			Operations:     fn.Operations,
		}
		bm.Functions = append(bm.Functions, bfn)
	}

	bm.Strings = strPool
	bm.Types = typePool

	return bm, nil
}

// EncodeBytecode serializes the Module directly to binary Tile IR bytecode.
func (m *Module) EncodeBytecode() ([]byte, error) {
	bm, err := m.ToBytecode()
	if err != nil {
		return nil, err
	}
	return EncodeBytecode(bm)
}

