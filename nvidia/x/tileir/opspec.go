package tileir

// operandKind says how one operand position is encoded.
type operandKind uint8

const (
	// operandPlain is always present and contributes exactly one value index.
	operandPlain operandKind = iota
	// operandVariadic is a group of run-time length.
	operandVariadic
	// operandOptional may be absent.
	operandOptional
)

// opSpec is the bytecode encoding shape of one operation.
//
// NVIDIA's writer emits an operation as opcode, result types, attributes,
// operands and regions, in that order
// (cuda-tile tools/cuda-tile-tblgen/BytecodeGen.cpp:653-665). Everything that
// varies between operations is shape, and this is that shape. The table is
// generated from NVIDIA's TableGen sources; see internal/gentileops.
type opSpec struct {
	Opcode uint32

	// NumResults is the count of results, and VariadicResults says whether a
	// count varint precedes the result type indices.
	NumResults      int
	VariadicResults bool

	// Operands lists the operand positions in order.
	Operands []operandKind

	// EncodeOperandCount says whether a count varint precedes the operands.
	// The writer emits one only when some operand is variadic or optional.
	EncodeOperandCount bool

	// AttrSegments says the operand groups are written with their sizes in an
	// attribute rather than as one flat list (AttrSizedOperandSegments).
	AttrSegments bool

	NumAttributes int
	NumRegions    int
}

// lookupOpSpec returns the encoding shape for a dialect-qualified mnemonic.
func lookupOpSpec(mnemonic string) (opSpec, bool) {
	s, ok := opSpecs[mnemonic]
	return s, ok
}

//go:generate go run ./internal/gentileops -root ../../../../NVIDIA/cuda-tile -out opspec_gen.go
