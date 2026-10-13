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

// sectionAlignFlag marks a section header whose length is followed by a
// varint alignment. alignByte is the filler the writer emits to reach it.
const (
	sectionAlignFlag uint8 = 0x80
	alignByte        byte  = 0xCB
)

// sectionAlign gives the alignment each section is written with. Sections
// absent from the map carry no alignment flag.
var sectionAlign = map[uint8]uint64{
	SectionString:   4,
	SectionFunc:     8,
	SectionDebug:    8,
	SectionConstant: 8,
	SectionType:     4,
}

// Function entry flags.
const (
	FuncVisibilityPrivate uint8 = 0x01
	FuncKindKernel        uint8 = 0x02
	FuncHasOptHints       uint8 = 0x04
)

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
	// ASTDecodeError records unsupported module metadata. Such a module cannot
	// be converted to an AST or re-encoded by this bounded codec.
	ASTDecodeError error
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
	// BodyDecodeError is nonnil when the body is preserved as opaque bytes.
	// Operations is then nil; no partial or guessed AST is published.
	BodyDecodeError error
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
	ended := false
	payloads := make(map[uint8][]byte)

	for offset < len(data) {
		idAndAlign := data[offset]
		offset++
		secID := idAndAlign & 0x7F
		hasAlign := (idAndAlign & sectionAlignFlag) != 0

		if secID == SectionEndOfBytecode {
			if hasAlign || offset != len(data) {
				return nil, errors.New("invalid bytecode terminator")
			}
			ended = true
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
			if align == 0 || align > uint64(len(data)) || align&(align-1) != 0 {
				return nil, errors.New("invalid section alignment")
			}

			if align > 1 {
				rem := offset % int(align)
				pad := (int(align) - rem) % int(align)
				if pad > len(data)-offset {
					return nil, errors.New("section padding exceeds buffer")
				}
				for _, b := range data[offset : offset+pad] {
					if b != alignByte {
						return nil, errors.New("invalid section padding")
					}
				}
				offset += pad
			}
		}

		if offset > len(data) || length > uint64(len(data)-offset) {
			return nil, fmt.Errorf("section %d length %d exceeds remaining buffer", secID, length)
		}

		if _, exists := payloads[secID]; exists {
			return nil, errors.New("duplicate bytecode section")
		}
		if secID == SectionDebug || secID == SectionGlobal || secID > SectionProducer {
			bm.ASTDecodeError = astUnsupported("module section %d", secID)
		}
		payloads[secID] = data[offset : offset+int(length)]
		offset += int(length)
	}
	if !ended {
		return nil, errors.New("missing bytecode terminator")
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
		idx, n := readVarInt(prodPayload)
		if n <= 0 || n != len(prodPayload) || idx >= uint64(len(bm.Strings)) {
			return nil, errors.New("invalid producer string index")
		}
		bm.Producer = bm.Strings[idx]
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
		types, err := parseTypeSection(typePayload, bm.Version)
		if err != nil {
			return nil, fmt.Errorf("parsing type section: %w", err)
		}
		bm.Types = types
	}

	// 5. Function Table Section
	if funcPayload, ok := payloads[SectionFunc]; ok {
		fns, err := parseFunctionSection(funcPayload, bm.Strings, bm.Types, bm.Constants, bm.Version)
		if err != nil {
			return nil, fmt.Errorf("parsing function section: %w", err)
		}
		bm.Functions = fns
	}

	return bm, nil
}

// ToModule converts a fully decoded supported body subset into an AST Module.
// It returns nil if any function has BodyDecodeError; raw bodies remain
// available for byte-preserving inspection and re-encoding.
func (bm *BytecodeModule) ToModule() *Module {
	if bm == nil || bm.ASTDecodeError != nil {
		return nil
	}
	mod := &Module{
		Name:      "kernels",
		Producer:  bm.Producer,
		Locations: make(map[string]Location),
	}

	for _, bfn := range bm.Functions {
		if bfn == nil || bfn.BodyDecodeError != nil {
			return nil
		}
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

	if off > len(payload) || numStrings > uint64((len(payload)-off)/4) {
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
		if start < 0 || start > len(strData) || end < start || end > len(strData) {
			return nil, errors.New("invalid string table offset")
		}
		strs[i] = string(strData[start:end])
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

	if off > len(payload) || numConstants > uint64((len(payload)-off)/8) {
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
		if start < 0 || start > len(data) || end < start || end > len(data) {
			return nil, errors.New("invalid constant table offset")
		}
		constants[i] = data[start:end]
	}
	return constants, nil
}

func parseTypeSection(payload []byte, ver BytecodeVersion) ([]Type, error) {
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

	if off > len(payload) || numTypes > uint64((len(payload)-off)/4) {
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
		if start < 0 || start >= len(typeData) || end > len(typeData) || start >= end {
			return nil, errors.New("invalid type table offset")
		}
		raw := typeData[start:end]
		if len(raw) == 0 {
			continue
		}
		tag, tn := readVarInt(raw)
		if tn <= 0 || tag > 255 {
			return nil, errors.New("invalid type tag")
		}
		var err error
		types[i], err = decodeASTPoolType(tag, raw[tn:], types, ver)
		if err != nil {
			return nil, err
		}
	}

	return types, nil
}

// skipOptionalParamFlags consumes the unified optional-parameter bitfield that
// precedes the parameters of a type introduced at version since, when a file of
// version ver carries one.
func skipOptionalParamFlags(payload []byte, ver, since BytecodeVersion) []byte {
	if !hasOptionalParamFlags(ver, since) {
		return payload
	}
	_, n := readVarInt(payload)
	if n <= 0 || n > len(payload) {
		return nil
	}
	return payload[n:]
}

// readLEVarSize reads a varint element count followed by that many
// little-endian int64s, returning the values and the unread remainder.
func readLEVarSize(payload []byte) ([]int64, []byte) {
	count, n := readVarInt(payload)
	if n <= 0 || n > len(payload) {
		return nil, nil
	}
	off := n
	var vals []int64
	for i := uint64(0); i < count && off+8 <= len(payload); i++ {
		vals = append(vals, int64(binary.LittleEndian.Uint64(payload[off:off+8])))
		off += 8
	}
	return vals, payload[off:]
}

func decodeType(tag uint64, payload []byte, typePool []Type, ver BytecodeVersion) Type {
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
		payload = skipOptionalParamFlags(payload, ver, ptrAttrSinceVersion)
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
		shape, _ := readLEVarSize(payload[n:])
		return &TileType{Shape: shape, ElementType: elem}
	case KindTensorView:
		payload = skipOptionalParamFlags(payload, ver, ptrAttrSinceVersion)
		if len(payload) == 0 {
			return &TensorViewType{ElementType: Float32}
		}
		elemIdx, n := readVarInt(payload)
		var elem Type = Float32
		if int(elemIdx) < len(typePool) && typePool[elemIdx] != nil {
			elem = typePool[elemIdx]
		}
		shape, rest := readLEVarSize(payload[n:])
		strides, _ := readLEVarSize(rest)
		return &TensorViewType{Shape: shape, Strides: strides, ElementType: elem}
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

func parseFunctionSection(payload []byte, strTable []string, typeTable []Type, constants [][]byte, version BytecodeVersion) ([]*BytecodeFunction, error) {
	c := astCursor{data: payload}
	count, err := c.uint()
	if err != nil || count > uint64(len(payload)) {
		return nil, errors.New("invalid function count")
	}
	var fns []*BytecodeFunction
	for i := uint64(0); i < count; i++ {
		nameIndex, err := c.uint()
		if err != nil || nameIndex >= uint64(len(strTable)) {
			return nil, errors.New("invalid function name index")
		}
		signatureIndex, err := c.uint()
		if err != nil || signatureIndex >= uint64(len(typeTable)) {
			return nil, errors.New("invalid function signature index")
		}
		sig, ok := typeTable[signatureIndex].(*FunctionType)
		if !ok || sig == nil {
			return nil, errors.New("function signature is not a function type")
		}
		if c.off >= len(payload) {
			return nil, errors.New("truncated function flags")
		}
		flags := payload[c.off]
		c.off++
		if flags & ^uint8(FuncVisibilityPrivate|FuncKindKernel|FuncHasOptHints) != 0 {
			return nil, errors.New("invalid function flags")
		}
		if flags&FuncKindKernel != 0 && len(sig.Results) != 0 {
			return nil, errors.New("kernel function signature cannot return values")
		}
		location, err := c.uint()
		if err != nil || location > uint64(int(^uint(0)>>1)) {
			return nil, errors.New("invalid function location index")
		}
		if flags&FuncHasOptHints != 0 {
			return nil, astUnsupported("function optimization hints")
		}
		length, err := c.uint()
		if err != nil || length > uint64(len(payload)-c.off) {
			return nil, errors.New("function body exceeds section")
		}
		body := payload[c.off : c.off+int(length)]
		c.off += int(length)
		bfn := &BytecodeFunction{
			Name: strTable[nameIndex], SignatureIndex: int(signatureIndex), Signature: sig,
			EntryFlag: flags, IsEntry: flags&FuncKindKernel != 0,
			IsPrivate: flags&FuncVisibilityPrivate != 0,
			LocIndex:  int(location), BodyBytes: body,
		}
		if version.Major != 13 || version.Minor != 4 || version.Tag != 0 {
			bfn.BodyDecodeError = astUnsupported("body version %s", version)
		} else if bfn.IsPrivate || location != 0 {
			bfn.BodyDecodeError = astUnsupported("private function or debug location")
		} else {
			bfn.Operations, bfn.BodyDecodeError = decodeASTBody(body, typeTable, constants, sig)
			if bfn.BodyDecodeError == nil && hasMemoryOps(bfn.Operations) && !bfn.IsEntry {
				return nil, astUnsupported("memory route requires kernel entry")
			}
			if bfn.BodyDecodeError != nil && !errors.Is(bfn.BodyDecodeError, ErrUnsupportedASTEncoding) {
				return nil, bfn.BodyDecodeError
			}
		}
		fns = append(fns, bfn)
	}
	if len(payload)-c.off > 7 {
		return nil, errors.New("excess function section padding")
	}
	for _, b := range payload[c.off:] {
		if b != alignByte {
			return nil, errors.New("invalid function section padding")
		}
	}
	return fns, nil
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
	if bm.ASTDecodeError != nil || len(bm.Globals) != 0 {
		return nil, astUnsupported("module metadata cannot be preserved")
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
	// fileVersion is the version actually written to the header; the type
	// encoder must gate optional-parameter bitfields on it, not on the
	// possibly-zero version the caller supplied.
	fileVersion := BytecodeVersion{Major: major, Minor: minor, Tag: tag}
	buf.WriteByte(major)
	buf.WriteByte(minor)
	var tagBytes [2]byte
	binary.LittleEndian.PutUint16(tagBytes[:], tag)
	buf.Write(tagBytes[:])

	// Helper to write a section
	writeSection := func(secID uint8, payload []byte) {
		align, aligned := sectionAlign[secID]
		if !aligned {
			buf.WriteByte(secID)
			buf.Write(appendVarInt(nil, uint64(len(payload))))
			buf.Write(payload)
			return
		}
		buf.WriteByte(secID | sectionAlignFlag)
		buf.Write(appendVarInt(nil, uint64(len(payload))))
		buf.Write(appendVarInt(nil, align))
		for pad := (align - uint64(buf.Len())%align) % align; pad > 0; pad-- {
			buf.WriteByte(alignByte)
		}
		buf.Write(payload)
	}

	// 3. Function section (SectionFunc = 0x02)
	if len(bm.Functions) > 0 {
		funcPayload, err := encodeFunctionSection(bm.Functions, bm.Strings, bm.Types)
		if err != nil {
			return nil, err
		}
		writeSection(SectionFunc, funcPayload)
	}

	// 4. Constant section (SectionConstant = 0x04)
	if len(bm.Constants) > 0 {
		constPayload := encodeConstantSection(bm.Constants)
		writeSection(SectionConstant, constPayload)
	}

	// 5. Type section (SectionType = 0x05)
	if len(bm.Types) > 0 {
		typePayload, err := encodeTypeSection(bm.Types, fileVersion)
		if err != nil {
			return nil, fmt.Errorf("encoding type section: %w", err)
		}
		writeSection(SectionType, typePayload)
	}

	// 6. Producer section (SectionProducer = 0x07)
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

	// 7. String section (SectionString = 0x01)
	if len(bm.Strings) > 0 {
		strPayload := encodeStringSection(bm.Strings)
		writeSection(SectionString, strPayload)
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
		payload = append(payload, alignByte)
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
		payload = append(payload, alignByte)
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

func encodeTypeSection(types []Type, ver BytecodeVersion) ([]byte, error) {
	numTypes := uint64(len(types))
	payload := appendVarInt(nil, numTypes)

	// Align to 4 bytes
	rem := len(payload) % 4
	pad := (4 - rem) % 4
	for i := 0; i < pad; i++ {
		payload = append(payload, alignByte)
	}

	offsets := make([]byte, numTypes*4)
	var typeData []byte
	for i, t := range types {
		binary.LittleEndian.PutUint32(offsets[i*4:], uint32(len(typeData)))
		raw := encodeSingleType(t, types, ver)
		typeData = append(typeData, raw...)
	}

	payload = append(payload, offsets...)
	payload = append(payload, typeData...)
	return payload, nil
}

// Versions at which a parametric type first gained an optional parameter.
// A type carrying optional parameters writes a unified presence bitfield as a
// varint immediately after its type tag, ahead of every parameter. The
// bitfield exists only from the unified-bitfield version (13.3) onward, and
// only once the file version reaches the version that introduced the type's
// first optional parameter.
//
// Derived from NVIDIA/cuda-tile
// tools/cuda-tile-tblgen/BytecodeTypeCodeGen.cpp:536 (generateOptionalParamFlags,
// emitted between the type tag and the parameters) and the parameter versions in
// include/cuda_tile/Dialect/CudaTile/IR/Types.td:119 and :295.
var (
	unifiedBitfieldVersion = BytecodeVersion{Major: 13, Minor: 3}
	ptrAttrSinceVersion    = BytecodeVersion{Major: 13, Minor: 4}
)

// versionAtLeast reports whether v is at or past want, ignoring the tag.
func versionAtLeast(v, want BytecodeVersion) bool {
	if v.Major != want.Major {
		return v.Major > want.Major
	}
	return v.Minor >= want.Minor
}

// hasOptionalParamFlags reports whether a type whose first optional parameter
// arrived in since writes its presence bitfield in a file of version ver.
func hasOptionalParamFlags(ver, since BytecodeVersion) bool {
	return versionAtLeast(ver, unifiedBitfieldVersion) && versionAtLeast(ver, since)
}

// appendOptionalParamFlags appends the unified optional-parameter bitfield, if
// a file of version ver carries one for a type introduced at version since.
func appendOptionalParamFlags(b []byte, ver, since BytecodeVersion, flags uint64) []byte {
	if !hasOptionalParamFlags(ver, since) {
		return b
	}
	return appendVarInt(b, flags)
}

// appendLEVarSize appends an int64 array in the vendor's writeLEVarSize form:
// an element count as a varint followed by that many little-endian int64s.
func appendLEVarSize(b []byte, vals []int64) []byte {
	b = appendVarInt(b, uint64(len(vals)))
	for _, v := range vals {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], uint64(v))
		b = append(b, buf[:]...)
	}
	return b
}

func encodeSingleType(t Type, pool []Type, ver BytecodeVersion) []byte {
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
		b = appendOptionalParamFlags(b, ver, ptrAttrSinceVersion, 0)
		return appendVarInt(b, uint64(pIdx))
	case *TileType:
		eIdx := findTypeIndex(pool, pt.ElementType)
		b := appendVarInt(nil, uint64(KindTile))
		b = appendVarInt(b, uint64(eIdx))
		return appendLEVarSize(b, pt.Shape)
	case *TensorViewType:
		eIdx := findTypeIndex(pool, pt.ElementType)
		b := appendVarInt(nil, uint64(KindTensorView))
		b = appendOptionalParamFlags(b, ver, ptrAttrSinceVersion, 0)
		b = appendVarInt(b, uint64(eIdx))
		b = appendLEVarSize(b, pt.Shape)
		return appendLEVarSize(b, pt.Strides)
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

func encodeFunctionSection(fns []*BytecodeFunction, strTable []string, typeTable []Type) ([]byte, error) {
	payload := appendVarInt(nil, uint64(len(fns)))
	for _, fn := range fns {
		if fn == nil {
			return nil, fmt.Errorf("tileir: nil bytecode function")
		}
		nameIdx := -1
		for i, s := range strTable {
			if s == fn.Name {
				nameIdx = i
				break
			}
		}
		if nameIdx < 0 || fn.SignatureIndex < 0 || fn.SignatureIndex >= len(typeTable) {
			return nil, fmt.Errorf("tileir: invalid name or signature reference for %s", fn.Name)
		}
		signature, ok := typeTable[fn.SignatureIndex].(*FunctionType)
		if !ok || signature == nil {
			return nil, fmt.Errorf("tileir: non-function signature for %s", fn.Name)
		}
		if fn.EntryFlag&FuncKindKernel != 0 && len(signature.Results) != 0 {
			return nil, fmt.Errorf("tileir: kernel function %s cannot return values", fn.Name)
		}
		if fn.EntryFlag&^byte(3) != 0 || fn.LocIndex != 0 {
			return nil, astUnsupported("function flags or location for %s", fn.Name)
		}
		payload = appendVarInt(payload, uint64(nameIdx))
		payload = appendVarInt(payload, uint64(fn.SignatureIndex))
		payload = append(payload, fn.EntryFlag)
		payload = appendVarInt(payload, uint64(fn.LocIndex))

		bodyBytes := fn.BodyBytes
		if len(bodyBytes) == 0 {
			return nil, astUnsupported("missing encoded body for %s; use Module.ToBytecode", fn.Name)
		}

		payload = appendVarInt(payload, uint64(len(bodyBytes)))
		payload = append(payload, bodyBytes...)
	}
	// The function section payload is padded to its section alignment.
	for pad := (8 - len(payload)%8) % 8; pad > 0; pad-- {
		payload = append(payload, alignByte)
	}
	return payload, nil
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

// ErrUnsupportedASTEncoding reports that the AST does not preserve the NVIDIA
// operation encoding needed to assemble an executable module.
var ErrUnsupportedASTEncoding = errors.New("tileir: AST bytecode encoding is unsupported")

// ToBytecode serializes the supported straight-line numeric and token subset
// at bytecode version 13.4. It preserves typed SSA operands, constant bits,
// numerical attributes, and function return types. It refuses memory operand
// segments, nested regions, attention, globals, debug locations, and unknown
// attributes. Module names, comments and SSA spelling are not wire identities.
func (m *Module) ToBytecode() (*BytecodeModule, error) {
	return astModule(m)
}

// EncodeBytecode encodes the complete supported AST subset, or returns no bytes
// and an error. The package function separately preserves decoded raw bodies.
func (m *Module) EncodeBytecode() ([]byte, error) {
	bm, err := m.ToBytecode()
	if err != nil {
		return nil, err
	}
	return EncodeBytecode(bm)
}
