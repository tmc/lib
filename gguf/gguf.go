package gguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"unicode/utf8"
)

// Parsing bounds. They keep a malformed file from driving a large allocation.
const (
	maxMetadata = 1 << 20
	maxTensors  = 1 << 20
	maxString   = 1 << 24
	maxArray    = 1 << 20
)

// Errors reported by [Read]. Use [errors.Is] to distinguish a malformed file
// from one this package cannot decode.
var (
	ErrInvalid     = errors.New("invalid GGUF")
	ErrUnsupported = errors.New("unsupported GGUF encoding")
)

// Metadata value types, as stored in [Value.TypeID].
const (
	TypeUint8   = 0
	TypeInt8    = 1
	TypeUint16  = 2
	TypeInt16   = 3
	TypeUint32  = 4
	TypeInt32   = 5
	TypeFloat32 = 6
	TypeBool    = 7
	TypeString  = 8
	TypeArray   = 9
	TypeUint64  = 10
	TypeInt64   = 11
	TypeFloat64 = 12
)

// A File is the parsed header, metadata, and tensor directory of a GGUF file.
// The tensor data itself is not read; it begins at DataOffset.
type File struct {
	Metadata   map[string]Value
	Tensors    []Tensor
	DataOffset uint64 // byte offset of the tensor data block
	Size       uint64 // size of the whole file
}

// A Value is one metadata value. Value holds the Go value matching TypeID:
// uint8, int8, uint16, int16, uint32, int32, float32, bool, string, uint64,
// int64, float64, or []any for an array.
type Value struct {
	TypeID uint32
	Value  any
}

// Uint reports v as an unsigned integer. It returns false if v is not one of
// the unsigned integer types.
func (v Value) Uint() (uint64, bool) {
	switch n := v.Value.(type) {
	case uint8:
		return uint64(n), true
	case uint16:
		return uint64(n), true
	case uint32:
		return uint64(n), true
	case uint64:
		return n, true
	}
	return 0, false
}

// Int reports v as a signed integer. It returns false if v is not an integer
// type, or is an unsigned value too large for int64.
func (v Value) Int() (int64, bool) {
	switch n := v.Value.(type) {
	case int8:
		return int64(n), true
	case int16:
		return int64(n), true
	case int32:
		return int64(n), true
	case int64:
		return n, true
	case uint8:
		return int64(n), true
	case uint16:
		return int64(n), true
	case uint32:
		return int64(n), true
	case uint64:
		if n > math.MaxInt64 {
			return 0, false
		}
		return int64(n), true
	}
	return 0, false
}

// A Tensor describes one tensor in the tensor directory. Its data lives at
// [File.DataOffset]+Offset and runs for [Tensor.Bytes] bytes.
type Tensor struct {
	Name       string
	Dimensions []uint64
	TypeID     uint32
	Offset     uint64
}

// Read parses the GGUF file of the given size read through r.
func Read(r io.ReaderAt, size int64) (*File, error) {
	if size < 0 {
		return nil, fmt.Errorf("%w: negative file size", ErrInvalid)
	}
	c := cursor{r: r, size: uint64(size)}
	magic, err := c.bytes(4)
	if err != nil {
		return nil, err
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("%w: bad magic", ErrInvalid)
	}
	version, err := c.u32()
	if err != nil {
		return nil, err
	}
	if version != 3 {
		return nil, fmt.Errorf("%w: version %d", ErrUnsupported, version)
	}
	tensorCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	metadataCount, err := c.u64()
	if err != nil {
		return nil, err
	}
	if tensorCount > maxTensors || metadataCount > maxMetadata {
		return nil, fmt.Errorf("%w: unreasonable counts", ErrInvalid)
	}
	metadataBytes, ok := multiply(metadataCount, 13) // empty key, type, and smallest value
	if !ok || metadataBytes > c.size-c.off {
		return nil, fmt.Errorf("%w: metadata count exceeds file size", ErrInvalid)
	}
	g := &File{Metadata: make(map[string]Value, metadataCount), Size: uint64(size)}
	for range metadataCount {
		key, err := c.string()
		if err != nil {
			return nil, fmt.Errorf("metadata key: %w", err)
		}
		if _, ok := g.Metadata[key]; ok {
			return nil, fmt.Errorf("%w: duplicate metadata key %q", ErrInvalid, key)
		}
		if len(key) > 1<<16-1 {
			return nil, fmt.Errorf("%w: metadata key too long", ErrInvalid)
		}
		value, err := c.value(0)
		if err != nil {
			return nil, fmt.Errorf("metadata %q: %w", key, err)
		}
		g.Metadata[key] = value
	}
	minimumTensorBytes, ok := multiply(tensorCount, 32) // empty name and one dimension
	if !ok || minimumTensorBytes > c.size-c.off {
		return nil, fmt.Errorf("%w: tensor count exceeds file size", ErrInvalid)
	}
	g.Tensors = make([]Tensor, 0, tensorCount)
	names := make(map[string]bool, tensorCount)
	for range tensorCount {
		name, err := c.string()
		if err != nil {
			return nil, fmt.Errorf("tensor name: %w", err)
		}
		if names[name] {
			return nil, fmt.Errorf("%w: duplicate tensor %q", ErrInvalid, name)
		}
		if len(name) > 64 {
			return nil, fmt.Errorf("%w: tensor name too long", ErrInvalid)
		}
		names[name] = true
		n, err := c.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q dimensions: %w", name, err)
		}
		if n == 0 || n > 4 {
			return nil, fmt.Errorf("%w: tensor %q has %d dimensions", ErrInvalid, name, n)
		}
		t := Tensor{Name: name, Dimensions: make([]uint64, n)}
		for i := range t.Dimensions {
			t.Dimensions[i], err = c.u64()
			if err != nil || t.Dimensions[i] == 0 {
				return nil, fmt.Errorf("%w: tensor %q dimension %d", ErrInvalid, name, i)
			}
		}
		t.TypeID, err = c.u32()
		if err != nil {
			return nil, fmt.Errorf("tensor %q type: %w", name, err)
		}
		t.Offset, err = c.u64()
		if err != nil {
			return nil, fmt.Errorf("tensor %q offset: %w", name, err)
		}
		g.Tensors = append(g.Tensors, t)
	}
	alignment := uint64(32)
	if v, ok := g.Metadata["general.alignment"]; ok {
		if v.TypeID != TypeUint32 {
			return nil, fmt.Errorf("%w: alignment is not uint32", ErrInvalid)
		}
		n, ok := v.Uint()
		if !ok || n == 0 || n > 1<<20 || n%8 != 0 {
			return nil, fmt.Errorf("%w: invalid alignment", ErrInvalid)
		}
		alignment = n
	}
	g.DataOffset = c.off
	if tensorCount != 0 {
		g.DataOffset, err = align(c.off, alignment)
	}
	if err != nil || g.DataOffset > g.Size {
		return nil, fmt.Errorf("%w: tensor data offset", ErrInvalid)
	}
	expectedOffset := uint64(0)
	for _, t := range g.Tensors {
		if t.Offset%alignment != 0 {
			return nil, fmt.Errorf("%w: tensor %q offset is not aligned", ErrInvalid, t.Name)
		}
		if t.Offset != expectedOffset {
			return nil, fmt.Errorf("%w: tensor %q offset %d, want %d", ErrInvalid, t.Name, t.Offset, expectedOffset)
		}
		length, err := t.Bytes()
		if err != nil {
			return nil, err
		}
		end, ok := add(t.Offset, length)
		if !ok || end > g.Size-g.DataOffset {
			return nil, fmt.Errorf("%w: tensor %q offset outside file", ErrInvalid, t.Name)
		}
		expectedOffset, err = align(end, alignment)
		if err != nil {
			return nil, fmt.Errorf("%w: tensor data size overflow", ErrInvalid)
		}
	}
	return g, nil
}

// CheckSHA256 reports whether the size bytes read through r hash to want,
// which is a lowercase hexadecimal SHA-256 digest.
func CheckSHA256(r io.ReaderAt, size int64, want string) error {
	if size < 0 {
		return fmt.Errorf("%w: negative file size", ErrInvalid)
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(r, 0, size)); err != nil {
		return fmt.Errorf("hash model: %w", err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("model sha256 %s, want %s", got, want)
	}
	return nil
}

// blockType is the block layout of one quantization type: a block covers
// block elements of a row and occupies bytes bytes.
type blockType struct{ block, bytes uint64 }

var blockTypes = map[uint32]blockType{
	0: {1, 4}, 1: {1, 2}, 2: {32, 18}, 3: {32, 20},
	6: {32, 22}, 7: {32, 24}, 8: {32, 34}, 9: {32, 40},
	10: {256, 84}, 11: {256, 110}, 12: {256, 144}, 13: {256, 176},
	14: {256, 210}, 15: {256, 292}, 16: {256, 66}, 17: {256, 74},
	18: {256, 98}, 19: {256, 110}, 20: {256, 50}, 21: {256, 110},
	22: {256, 82}, 23: {256, 136}, 24: {1, 1}, 25: {1, 2},
	26: {1, 4}, 27: {1, 8}, 28: {1, 8}, 29: {256, 56},
	30: {1, 2}, 39: {32, 17},
}

// Bytes reports how many bytes t occupies in the tensor data block.
func (t Tensor) Bytes() (uint64, error) {
	typ, ok := blockTypes[t.TypeID]
	if !ok {
		return 0, fmt.Errorf("%w: tensor %q type %d", ErrUnsupported, t.Name, t.TypeID)
	}
	if t.Dimensions[0]%typ.block != 0 {
		return 0, fmt.Errorf("%w: tensor %q row is not block aligned", ErrInvalid, t.Name)
	}
	n, ok := multiply(t.Dimensions[0]/typ.block, typ.bytes)
	if !ok {
		return 0, fmt.Errorf("%w: tensor %q byte count overflow", ErrInvalid, t.Name)
	}
	for _, d := range t.Dimensions[1:] {
		var ok bool
		n, ok = multiply(n, d)
		if !ok {
			return 0, fmt.Errorf("%w: tensor %q byte count overflow", ErrInvalid, t.Name)
		}
	}
	return n, nil
}

// A cursor reads little-endian GGUF fields forward through a file.
type cursor struct {
	r    io.ReaderAt
	size uint64
	off  uint64
}

func (c *cursor) bytes(n uint64) ([]byte, error) {
	end, ok := add(c.off, n)
	if !ok || end > c.size || n > uint64(math.MaxInt) {
		return nil, io.ErrUnexpectedEOF
	}
	b := make([]byte, int(n))
	if _, err := c.r.ReadAt(b, int64(c.off)); err != nil {
		return nil, err
	}
	c.off = end
	return b, nil
}

func (c *cursor) u32() (uint32, error) {
	b, err := c.bytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (c *cursor) u64() (uint64, error) {
	b, err := c.bytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (c *cursor) string() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if n > maxString {
		return "", fmt.Errorf("%w: string too long", ErrInvalid)
	}
	b, err := c.bytes(n)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", fmt.Errorf("%w: string is not UTF-8", ErrInvalid)
	}
	return string(b), nil
}

func (c *cursor) value(depth int) (Value, error) {
	if depth > 8 {
		return Value{}, fmt.Errorf("%w: nested array", ErrUnsupported)
	}
	t, err := c.u32()
	if err != nil {
		return Value{}, err
	}
	return c.valueOfType(t, depth)
}

func (c *cursor) valueOfType(t uint32, depth int) (Value, error) {
	if depth > 8 {
		return Value{}, fmt.Errorf("%w: metadata array nesting is too deep", ErrInvalid)
	}
	v := Value{TypeID: t}
	var err error
	switch t {
	case TypeUint8:
		b, err := c.bytes(1)
		if err != nil {
			return v, err
		}
		v.Value = b[0]
	case TypeInt8:
		b, err := c.bytes(1)
		if err != nil {
			return v, err
		}
		v.Value = int8(b[0])
	case TypeUint16:
		b, err := c.bytes(2)
		if err != nil {
			return v, err
		}
		v.Value = binary.LittleEndian.Uint16(b)
	case TypeInt16:
		b, err := c.bytes(2)
		if err != nil {
			return v, err
		}
		v.Value = int16(binary.LittleEndian.Uint16(b))
	case TypeUint32:
		v.Value, err = c.u32()
	case TypeInt32:
		n, e := c.u32()
		err = e
		v.Value = int32(n)
	case TypeFloat32:
		n, e := c.u32()
		err = e
		v.Value = math.Float32frombits(n)
	case TypeBool:
		b, e := c.bytes(1)
		err = e
		if e == nil {
			if b[0] > 1 {
				return v, fmt.Errorf("%w: invalid boolean", ErrInvalid)
			}
			v.Value = b[0] == 1
		}
	case TypeString:
		v.Value, err = c.string()
	case TypeArray:
		v.Value, err = c.array(depth + 1)
	case TypeUint64:
		v.Value, err = c.u64()
	case TypeInt64:
		n, e := c.u64()
		err = e
		v.Value = int64(n)
	case TypeFloat64:
		n, e := c.u64()
		err = e
		v.Value = math.Float64frombits(n)
	default:
		return v, fmt.Errorf("%w: value type %d", ErrUnsupported, t)
	}
	return v, err
}

func (c *cursor) array(depth int) (any, error) {
	t, err := c.u32()
	if err != nil {
		return nil, err
	}
	n, err := c.u64()
	if err != nil {
		return nil, err
	}
	if n > maxArray {
		return nil, fmt.Errorf("%w: array too long", ErrInvalid)
	}
	minimum, ok := minimumValueBytes(t)
	if !ok {
		return nil, fmt.Errorf("%w: value type %d", ErrUnsupported, t)
	}
	need, ok := multiply(n, minimum)
	if !ok || need > c.size-c.off {
		return nil, fmt.Errorf("%w: array exceeds file size", ErrInvalid)
	}
	values := make([]any, 0, n)
	for range n {
		v, err := c.valueOfType(t, depth)
		if err != nil {
			return nil, err
		}
		values = append(values, v.Value)
	}
	return values, nil
}

func minimumValueBytes(t uint32) (uint64, bool) {
	switch t {
	case TypeUint8, TypeInt8, TypeBool:
		return 1, true
	case TypeUint16, TypeInt16:
		return 2, true
	case TypeUint32, TypeInt32, TypeFloat32:
		return 4, true
	case TypeString, TypeUint64, TypeInt64, TypeFloat64:
		return 8, true
	case TypeArray:
		return 12, true
	default:
		return 0, false
	}
}

func add(a, b uint64) (uint64, bool) {
	if a > math.MaxUint64-b {
		return 0, false
	}
	return a + b, true
}

func multiply(a, b uint64) (uint64, bool) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, false
	}
	return a * b, true
}

func align(n, alignment uint64) (uint64, error) {
	rem := n % alignment
	if rem == 0 {
		return n, nil
	}
	v, ok := add(n, alignment-rem)
	if !ok {
		return 0, errors.New("offset overflow")
	}
	return v, nil
}
