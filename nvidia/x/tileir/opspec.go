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

// attrSpec is one attribute of an operation, in the order the writer emits
// attribute payloads.
type attrSpec struct {
	Name string

	// Optional says the attribute owns a bit of the flags varint and its
	// payload is written only when that bit is set.
	Optional bool

	// Unit says the attribute is a UnitAttr, whose flags bit is its entire
	// encoding: no payload follows, set or clear
	// (BytecodeGen.cpp:445-447). A Unit attribute is always Optional.
	Unit bool
}

// flagBit is one bit of an operation's flags varint.
type flagBit struct {
	Name string

	// Operand says the bit stands for an optional operand group rather than
	// an optional attribute.
	Operand bool

	// Index is the position within opSpec.Operands or opSpec.Attributes.
	Index int
}

// opSpec is the bytecode encoding shape of one operation.
//
// NVIDIA's writer emits an operation as opcode, result types, a flags field,
// attributes, operands and regions, in that order
// (cuda-tile tools/cuda-tile-tblgen/BytecodeGen.cpp:660-666; the opcode comes
// from the caller). Everything that varies between operations is shape, and
// this is that shape. The table is generated from NVIDIA's TableGen sources;
// see internal/gentileops.
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

	// AttrSegments says the operand groups are written one at a time rather
	// than as one flat list (AttrSizedOperandSegments). Under it a variadic
	// group carries its own count, a plain group carries none, and an
	// optional group appears only when its flags bit is set --
	// EncodeOperandCount is false, because there is no list-wide count.
	AttrSegments bool

	// Attributes lists the attribute positions in declaration order, which
	// is payload order.
	Attributes []attrSpec

	// Flags is the flags varint's bit assignment, low bit first. An empty
	// Flags means the operation writes no flags varint at all.
	//
	// The order is NVIDIA's, not declaration order: bits are grouped by the
	// version that introduced the field, ascending, attributes before
	// operands within a version
	// (BytecodeGenUtilities.cpp:151-214). The two agree except on
	// operations that grew a field after their own introduction.
	Flags []flagBit

	NumRegions int
}

// HasFlags reports whether the operation writes a flags varint.
func (s opSpec) HasFlags() bool { return len(s.Flags) > 0 }

// lookupOpSpec returns the encoding shape for a dialect-qualified mnemonic.
func lookupOpSpec(mnemonic string) (opSpec, bool) {
	s, ok := opSpecs[mnemonic]
	return s, ok
}

//go:generate go run ./internal/gentileops -root ../../../../../NVIDIA/cuda-tile -out opspec_gen.go
