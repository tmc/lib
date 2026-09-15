package tileir

import (
	"fmt"
	"strconv"
	"strings"
)

// TypeKind identifies the variant of a Tile IR type.
type TypeKind uint8

const (
	// Primitive integer types (tags 0-4, 22).
	KindInt1  TypeKind = 0
	KindInt8  TypeKind = 1
	KindInt16 TypeKind = 2
	KindInt32 TypeKind = 3
	KindInt64 TypeKind = 4
	KindInt4  TypeKind = 22

	// Floating point types (tags 5-11, 18, 19, 0x82).
	KindFloat16    TypeKind = 5
	KindBFloat16   TypeKind = 6
	KindFloat32    TypeKind = 7
	KindTFloat32   TypeKind = 8
	KindFloat64    TypeKind = 9
	KindFloat8E4M3 TypeKind = 10
	KindFloat8E5M2 TypeKind = 11
	KindFloat8E8M0 TypeKind = 18
	KindFloat4E2M1 TypeKind = 19
	KindFloat8E5M3 TypeKind = 0x82

	// Structural / composite types (tags 12-17, 20-21).
	KindPointer       TypeKind = 12
	KindTile          TypeKind = 13
	KindTensorView    TypeKind = 14
	KindPartitionView TypeKind = 15
	KindFunction      TypeKind = 16
	KindToken         TypeKind = 17
	KindGatherScatter TypeKind = 20
	KindStridedView   TypeKind = 21
)

// String returns a human-readable name for the TypeKind.
func (k TypeKind) String() string {
	switch k {
	case KindInt1:
		return "i1"
	case KindInt4:
		return "i4"
	case KindInt8:
		return "i8"
	case KindInt16:
		return "i16"
	case KindInt32:
		return "i32"
	case KindInt64:
		return "i64"
	case KindFloat16:
		return "f16"
	case KindBFloat16:
		return "bf16"
	case KindFloat32:
		return "f32"
	case KindTFloat32:
		return "tf32"
	case KindFloat64:
		return "f64"
	case KindFloat8E4M3:
		return "f8E4M3FN"
	case KindFloat8E5M2:
		return "f8E5M2"
	case KindFloat8E8M0:
		return "f8E8M0FNU"
	case KindFloat4E2M1:
		return "f4E2M1FN"
	case KindFloat8E5M3:
		return "f8E5M3FNU"
	case KindPointer:
		return "ptr"
	case KindTile:
		return "tile"
	case KindTensorView:
		return "tensor_view"
	case KindPartitionView:
		return "partition_view"
	case KindFunction:
		return "function"
	case KindToken:
		return "token"
	case KindGatherScatter:
		return "gather_scatter_view"
	case KindStridedView:
		return "strided_view"
	default:
		return fmt.Sprintf("unknown(%d)", uint8(k))
	}
}

// Type represents any valid Tile IR data or structural type.
type Type interface {
	Kind() TypeKind
	String() string
	IsTile() bool
	IsNumber() bool
	IsFloat() bool
	IsInt() bool
}

// PrimitiveType represents scalar integer and floating-point types.
type PrimitiveType struct {
	kind TypeKind
	name string
}

func (p PrimitiveType) Kind() TypeKind { return p.kind }
func (p PrimitiveType) String() string { return p.name }
func (p PrimitiveType) IsTile() bool   { return false }
func (p PrimitiveType) IsNumber() bool { return true }
func (p PrimitiveType) IsFloat() bool {
	switch p.kind {
	case KindFloat16, KindBFloat16, KindFloat32, KindTFloat32, KindFloat64,
		KindFloat8E4M3, KindFloat8E5M2, KindFloat8E8M0, KindFloat4E2M1, KindFloat8E5M3:
		return true
	}
	return false
}
func (p PrimitiveType) IsInt() bool {
	switch p.kind {
	case KindInt1, KindInt4, KindInt8, KindInt16, KindInt32, KindInt64:
		return true
	}
	return false
}

// Standard primitive types.
var (
	Int1  = PrimitiveType{kind: KindInt1, name: "i1"}
	Int4  = PrimitiveType{kind: KindInt4, name: "i4"}
	Int8  = PrimitiveType{kind: KindInt8, name: "i8"}
	Int16 = PrimitiveType{kind: KindInt16, name: "i16"}
	Int32 = PrimitiveType{kind: KindInt32, name: "i32"}
	Int64 = PrimitiveType{kind: KindInt64, name: "i64"}

	Float16    = PrimitiveType{kind: KindFloat16, name: "f16"}
	BFloat16   = PrimitiveType{kind: KindBFloat16, name: "bf16"}
	Float32    = PrimitiveType{kind: KindFloat32, name: "f32"}
	TFloat32   = PrimitiveType{kind: KindTFloat32, name: "tf32"}
	Float64    = PrimitiveType{kind: KindFloat64, name: "f64"}
	Float8E4M3 = PrimitiveType{kind: KindFloat8E4M3, name: "f8E4M3FN"}
	Float8E5M2 = PrimitiveType{kind: KindFloat8E5M2, name: "f8E5M2"}
	Float8E8M0 = PrimitiveType{kind: KindFloat8E8M0, name: "f8E8M0FNU"}
	Float4E2M1 = PrimitiveType{kind: KindFloat4E2M1, name: "f4E2M1FN"}
	Float8E5M3 = PrimitiveType{kind: KindFloat8E5M3, name: "f8E5M3FNU"}
)

// PointerType represents a typed global device pointer.
type PointerType struct {
	Pointee Type
	Attr    string // optional pointer classification (e.g. none)
}

func (p *PointerType) Kind() TypeKind { return KindPointer }
func (p *PointerType) IsTile() bool   { return false }
func (p *PointerType) IsNumber() bool { return false }
func (p *PointerType) IsFloat() bool  { return false }
func (p *PointerType) IsInt() bool    { return false }
func (p *PointerType) String() string {
	if p == nil {
		return "!cuda_tile.ptr<unknown>"
	}
	if p.Attr != "" {
		return fmt.Sprintf("!cuda_tile.ptr<%s, %s>", p.Pointee, p.Attr)
	}
	return fmt.Sprintf("!cuda_tile.ptr<%s>", p.Pointee)
}

// TileType represents a statically shaped multidimensional tile.
type TileType struct {
	Shape       []int64
	ElementType Type
}

func (t *TileType) Kind() TypeKind { return KindTile }
func (t *TileType) IsTile() bool   { return true }
func (t *TileType) IsNumber() bool { return false }
func (t *TileType) IsFloat() bool  { return false }
func (t *TileType) IsInt() bool    { return false }

// Rank returns the dimensionality of the tile.
func (t *TileType) Rank() int {
	if t == nil {
		return 0
	}
	return len(t.Shape)
}

// NumElements returns the total element count in the tile, or 1 for rank 0.
func (t *TileType) NumElements() int64 {
	if t == nil || len(t.Shape) == 0 {
		return 1
	}
	total := int64(1)
	for _, dim := range t.Shape {
		total *= dim
	}
	return total
}

func (t *TileType) String() string {
	if t == nil {
		return "!cuda_tile.tile<unknown>"
	}
	if len(t.Shape) == 0 {
		return fmt.Sprintf("!cuda_tile.tile<%s>", t.ElementType)
	}
	var b strings.Builder
	b.WriteString("!cuda_tile.tile<")
	for _, dim := range t.Shape {
		b.WriteString(strconv.FormatInt(dim, 10))
		b.WriteByte('x')
	}
	b.WriteString(t.ElementType.String())
	b.WriteByte('>')
	return b.String()
}

// TokenType represents a synchronization token for ordering asynchronous TMA transfers.
type TokenType struct{}

func (TokenType) Kind() TypeKind { return KindToken }
func (TokenType) String() string { return "!cuda_tile.token" }
func (TokenType) IsTile() bool   { return false }
func (TokenType) IsNumber() bool { return false }
func (TokenType) IsFloat() bool  { return false }
func (TokenType) IsInt() bool    { return false }

// Token is the canonical instance of TokenType.
var Token = TokenType{}

// FunctionType represents a function signature with input parameter types and result types.
type FunctionType struct {
	Inputs  []Type
	Results []Type
}

func (f *FunctionType) Kind() TypeKind { return KindFunction }
func (f *FunctionType) IsTile() bool   { return false }
func (f *FunctionType) IsNumber() bool { return false }
func (f *FunctionType) IsFloat() bool  { return false }
func (f *FunctionType) IsInt() bool    { return false }
func (f *FunctionType) String() string {
	if f == nil {
		return "() -> ()"
	}
	var inStrs []string
	for _, in := range f.Inputs {
		inStrs = append(inStrs, in.String())
	}
	var resStrs []string
	for _, res := range f.Results {
		resStrs = append(resStrs, res.String())
	}
	inPart := strings.Join(inStrs, ", ")
	if len(f.Results) == 0 {
		return fmt.Sprintf("(%s) -> ()", inPart)
	}
	if len(f.Results) == 1 {
		return fmt.Sprintf("(%s) -> %s", inPart, resStrs[0])
	}
	return fmt.Sprintf("(%s) -> (%s)", inPart, strings.Join(resStrs, ", "))
}

// DynamicDim is the shape and stride extent standing for a dynamic ("?")
// dimension. It matches MLIR's ShapedType::kDynamic, which the Tile IR
// bytecode stores verbatim as a little-endian int64.
const DynamicDim int64 = -1 << 63

// TensorViewType models an N-dimensional view into a global memory tensor.
//
// Strides has the same length as Shape when present; a nil Strides encodes as
// an empty stride list, which the assembler accepts only for a rank-0 view.
// Dynamic extents are represented by DynamicDim.
type TensorViewType struct {
	Shape       []int64
	Strides     []int64
	ElementType Type
}

func (v *TensorViewType) Kind() TypeKind { return KindTensorView }
func (v *TensorViewType) IsTile() bool   { return false }
func (v *TensorViewType) IsNumber() bool { return false }
func (v *TensorViewType) IsFloat() bool  { return false }
func (v *TensorViewType) IsInt() bool    { return false }
func (v *TensorViewType) String() string {
	if v == nil {
		return "!cuda_tile.tensor_view<unknown>"
	}
	var b strings.Builder
	b.WriteString("!cuda_tile.tensor_view<")
	for _, dim := range v.Shape {
		b.WriteString(strconv.FormatInt(dim, 10))
		b.WriteByte('x')
	}
	b.WriteString(v.ElementType.String())
	b.WriteByte('>')
	return b.String()
}

// PartitionViewType represents a partitioned view over tile coordinates.
type PartitionViewType struct {
	Shape       []int64
	ElementType Type
}

func (p *PartitionViewType) Kind() TypeKind { return KindPartitionView }
func (p *PartitionViewType) IsTile() bool   { return false }
func (p *PartitionViewType) IsNumber() bool { return false }
func (p *PartitionViewType) IsFloat() bool  { return false }
func (p *PartitionViewType) IsInt() bool    { return false }
func (p *PartitionViewType) String() string {
	return "!cuda_tile.partition_view"
}

// StridedViewType represents a strided tensor memory access view.
type StridedViewType struct {
	ElementType Type
}

func (s *StridedViewType) Kind() TypeKind { return KindStridedView }
func (s *StridedViewType) IsTile() bool   { return false }
func (s *StridedViewType) IsNumber() bool { return false }
func (s *StridedViewType) IsFloat() bool  { return false }
func (s *StridedViewType) IsInt() bool    { return false }
func (s *StridedViewType) String() string {
	return "!cuda_tile.strided_view"
}

// GatherScatterViewType represents an indexed non-contiguous tensor view.
type GatherScatterViewType struct {
	ElementType Type
}

func (g *GatherScatterViewType) Kind() TypeKind { return KindGatherScatter }
func (g *GatherScatterViewType) IsTile() bool   { return false }
func (g *GatherScatterViewType) IsNumber() bool { return false }
func (g *GatherScatterViewType) IsFloat() bool  { return false }
func (g *GatherScatterViewType) IsInt() bool    { return false }
func (g *GatherScatterViewType) String() string {
	return "!cuda_tile.gather_scatter_view"
}

// NewTile constructs a new TileType.
func NewTile(shape []int64, elem Type) *TileType {
	cp := make([]int64, len(shape))
	copy(cp, shape)
	return &TileType{Shape: cp, ElementType: elem}
}

// NewPointer constructs a new PointerType.
func NewPointer(pointee Type) *PointerType {
	return &PointerType{Pointee: pointee}
}

// NewFunction constructs a new FunctionType.
func NewFunction(inputs, results []Type) *FunctionType {
	in := make([]Type, len(inputs))
	copy(in, inputs)
	res := make([]Type, len(results))
	copy(res, results)
	return &FunctionType{Inputs: in, Results: res}
}

// ParseType parses a Tile IR textual type specification into a Type.
func ParseType(s string) (Type, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "!cuda_tile.")

	switch s {
	case "i1":
		return Int1, nil
	case "i4":
		return Int4, nil
	case "i8":
		return Int8, nil
	case "i16":
		return Int16, nil
	case "i32":
		return Int32, nil
	case "i64":
		return Int64, nil
	case "f16":
		return Float16, nil
	case "bf16":
		return BFloat16, nil
	case "f32":
		return Float32, nil
	case "tf32":
		return TFloat32, nil
	case "f64":
		return Float64, nil
	case "f8E4M3FN", "f8e4m3fn", "fp8", "fp8_e4m3", "e4m3", "e4m3fn":
		return Float8E4M3, nil
	case "f8E5M2", "f8e5m2", "fp8_e5m2", "e5m2":
		return Float8E5M2, nil
	case "f8E8M0FNU", "f8e8m0fnu", "fp8_e8m0", "e8m0":
		return Float8E8M0, nil
	case "f4E2M1FN", "f4e2m1fn", "f4", "fp4", "e2m1", "f4e2m1":
		return Float4E2M1, nil
	case "f8E5M3FNU", "f8e5m3fnu", "fp8_e5m3", "e5m3":
		return Float8E5M3, nil
	case "token":
		return Token, nil
	case "none", "()":
		return NewFunction(nil, nil), nil
	}

	if strings.HasPrefix(s, "tile<") && strings.HasSuffix(s, ">") {
		inner := s[5 : len(s)-1]
		return parseTileTypeString(inner)
	}

	if strings.HasPrefix(s, "ptr<") && strings.HasSuffix(s, ">") {
		inner := s[4 : len(s)-1]
		parts := strings.Split(inner, ",")
		pointee, err := ParseType(strings.TrimSpace(parts[0]))
		if err != nil {
			return nil, fmt.Errorf("invalid pointer target %q: %w", parts[0], err)
		}
		var attr string
		if len(parts) > 1 {
			attr = strings.TrimSpace(parts[1])
		}
		return &PointerType{Pointee: pointee, Attr: attr}, nil
	}

	if strings.HasPrefix(s, "tensor_view<") && strings.HasSuffix(s, ">") {
		inner := s[12 : len(s)-1]
		t, err := parseTileTypeString(inner)
		if err != nil {
			return nil, err
		}
		return &TensorViewType{Shape: t.Shape, ElementType: t.ElementType}, nil
	}

	if strings.Contains(s, "->") {
		arrowIdx := strings.Index(s, "->")
		inPart := strings.TrimSpace(s[:arrowIdx])
		resPart := strings.TrimSpace(s[arrowIdx+2:])

		inPart = strings.TrimPrefix(inPart, "(")
		inPart = strings.TrimSuffix(inPart, ")")
		var inputs []Type
		for _, inStr := range SplitOperands(inPart) {
			inStr = strings.TrimSpace(inStr)
			if inStr != "" {
				inType, err := ParseType(inStr)
				if err != nil {
					return nil, err
				}
				inputs = append(inputs, inType)
			}
		}

		resPart = strings.TrimPrefix(resPart, "(")
		resPart = strings.TrimSuffix(resPart, ")")
		var results []Type
		for _, resStr := range SplitOperands(resPart) {
			resStr = strings.TrimSpace(resStr)
			if resStr != "" {
				resType, err := ParseType(resStr)
				if err != nil {
					return nil, err
				}
				results = append(results, resType)
			}
		}
		return NewFunction(inputs, results), nil
	}

	return nil, fmt.Errorf("unrecognized tileir type: %q", s)
}

func parseTileTypeString(s string) (*TileType, error) {
	// Look for dimension separator 'x'
	parts := strings.Split(s, "x")
	if len(parts) == 1 {
		elem, err := ParseType(parts[0])
		if err != nil {
			return nil, err
		}
		return &TileType{Shape: nil, ElementType: elem}, nil
	}

	var shape []int64
	for i := 0; i < len(parts)-1; i++ {
		dim, err := strconv.ParseInt(parts[i], 10, 64)
		if err != nil {
			// In case the element type itself had an 'x' (nested), handle fallback
			elem, errElem := ParseType(strings.Join(parts[i:], "x"))
			if errElem == nil {
				return &TileType{Shape: shape, ElementType: elem}, nil
			}
			return nil, fmt.Errorf("invalid tile dimension %q: %w", parts[i], err)
		}
		shape = append(shape, dim)
	}
	elem, err := ParseType(parts[len(parts)-1])
	if err != nil {
		return nil, err
	}
	return &TileType{Shape: shape, ElementType: elem}, nil
}
