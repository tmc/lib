package cudakernel

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"strings"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// DevicePtrProvider represents any type that exposes a raw GPU device pointer.
type DevicePtrProvider interface {
	Ptr() driver.DevicePtr
}

// Param describes a kernel parameter in the ABI schema.
type Param struct {
	Name      string `json:"name"`
	Type      string `json:"type"`       // PTX data type, e.g. "u32", "u64", "f32", "s32", "b8", "ptr"
	ByteWidth int    `json:"byte_width"` // Width in bytes: 1, 2, 4, 8, 16
	Alignment int    `json:"alignment"`  // Alignment in bytes: 1, 2, 4, 8, 16
	IsPointer bool   `json:"is_pointer"` // True if parameter is a device memory pointer
	Offset    int    `json:"offset"`     // Byte offset in parameter buffer
}

func (p Param) String() string {
	if p.IsPointer {
		return fmt.Sprintf("*%s %s (offset %d, align %d)", p.Type, p.Name, p.Offset, p.Alignment)
	}
	return fmt.Sprintf("%s %s (offset %d, size %d, align %d)", p.Type, p.Name, p.Offset, p.ByteWidth, p.Alignment)
}

// Schema defines the fixed-width parameter layout and ABI requirements for a kernel.
type Schema struct {
	Params          []Param `json:"params"`
	TotalParamBytes int     `json:"total_param_bytes"`
}

// NewSchema computes aligned offsets and an eight-byte-minimum bank alignment.
// Zero widths and alignments select defaults; negative values are invalid.
// Known PTX types use their fixed widths. Other types require an explicit width
// for opaque ABI metadata, which does not imply that Pack supports the type.
func NewSchema(params ...Param) (Schema, error) {
	if len(params) == 0 {
		return Schema{}, nil
	}
	offset, maxAlign := 0, 8
	computed := make([]Param, len(params))
	for i, p := range params {
		if p.ByteWidth == 0 {
			p.ByteWidth = defaultByteWidth(p.Type, p.IsPointer)
		}
		if p.Alignment == 0 {
			p.Alignment = p.ByteWidth
		}
		if err := validateParam(p); err != nil {
			return Schema{}, fmt.Errorf("cudakernel: param %d: %w", i, err)
		}
		if p.Alignment > maxAlign {
			maxAlign = p.Alignment
		}
		var err error
		offset, err = alignedOffset(offset, p.Alignment)
		if err != nil {
			return Schema{}, fmt.Errorf("cudakernel: param %q: %w", p.Name, err)
		}
		p.Offset = offset
		if p.ByteWidth > math.MaxInt-offset {
			return Schema{}, fmt.Errorf("cudakernel: param %q layout overflows int", p.Name)
		}
		offset += p.ByteWidth
		computed[i] = p
	}
	total, err := alignedOffset(offset, maxAlign)
	if err != nil {
		return Schema{}, fmt.Errorf("cudakernel: total parameter layout: %w", err)
	}
	schema := Schema{Params: computed, TotalParamBytes: total}
	if err := schema.Validate(); err != nil {
		return Schema{}, err
	}
	return schema, nil
}

// alignedOffset checks padding before adding it to a nonnegative offset.
func alignedOffset(offset, alignment int) (int, error) {
	padding := (alignment - offset%alignment) % alignment
	if padding > math.MaxInt-offset {
		return 0, fmt.Errorf("alignment padding overflows int")
	}
	return offset + padding, nil
}

func validateParam(p Param) error {
	if p.Name == "" || strings.ContainsRune(p.Name, 0) {
		return fmt.Errorf("invalid parameter name %q", p.Name)
	}
	if p.Type == "" && !p.IsPointer || strings.ContainsRune(p.Type, 0) {
		return fmt.Errorf("param %q has invalid type %q", p.Name, p.Type)
	}
	if p.ByteWidth <= 0 {
		return fmt.Errorf("param %q has invalid byte width %d", p.Name, p.ByteWidth)
	}
	if width := defaultByteWidth(p.Type, p.IsPointer); width != 0 && p.ByteWidth != width {
		return fmt.Errorf("param %q type %q requires byte width %d, got %d", p.Name, p.Type, width, p.ByteWidth)
	}
	if p.Type == "ptr" && !p.IsPointer {
		return fmt.Errorf("param %q type ptr requires IsPointer", p.Name)
	}
	if p.Alignment > 16 || !isPowerOfTwo(p.Alignment) {
		return fmt.Errorf("param %q has invalid alignment %d", p.Name, p.Alignment)
	}
	if p.Offset < 0 {
		return fmt.Errorf("param %q has negative offset %d", p.Name, p.Offset)
	}
	return nil
}

func defaultByteWidth(t string, isPtr bool) int {
	if isPtr {
		return 8
	}
	switch t {
	case "b8", "u8", "s8", "bool":
		return 1
	case "b16", "u16", "s16", "f16", "bf16":
		return 2
	case "b32", "u32", "s32", "f32", "tf32":
		return 4
	case "b64", "u64", "s64", "f64", "ptr":
		return 8
	case "b128":
		return 16
	default:
		return 0
	}
}

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// Validate checks ABI metadata, widths, names and aligned nonoverlapping ranges.
// Opaque types and unsupported scalar kinds may describe valid metadata without
// being supported by Pack.
func (s Schema) Validate() error {
	if s.TotalParamBytes < 0 {
		return fmt.Errorf("cudakernel: negative total parameter bytes %d", s.TotalParamBytes)
	}
	offset, maxAlign := 0, 8
	names := make(map[string]bool, len(s.Params))
	for i, p := range s.Params {
		if err := validateParam(p); err != nil {
			return fmt.Errorf("cudakernel: param %d: %w", i, err)
		}
		if names[p.Name] {
			return fmt.Errorf("cudakernel: duplicate parameter name %q", p.Name)
		}
		names[p.Name] = true
		if p.Alignment > maxAlign {
			maxAlign = p.Alignment
		}
		if p.Offset < offset {
			return fmt.Errorf("cudakernel: param %q offset %d overlaps prior end %d", p.Name, p.Offset, offset)
		}
		if p.Offset%p.Alignment != 0 {
			return fmt.Errorf("cudakernel: param %q offset %d not aligned to %d", p.Name, p.Offset, p.Alignment)
		}
		if p.ByteWidth > math.MaxInt-p.Offset {
			return fmt.Errorf("cudakernel: param %q layout overflows int", p.Name)
		}
		offset = p.Offset + p.ByteWidth
	}
	required, err := alignedOffset(offset, maxAlign)
	if err != nil {
		return fmt.Errorf("cudakernel: total parameter layout: %w", err)
	}
	if s.TotalParamBytes < required || s.TotalParamBytes%maxAlign != 0 {
		return fmt.Errorf("cudakernel: total parameter bytes %d require at least %d and alignment %d", s.TotalParamBytes, required, maxAlign)
	}
	return nil
}

func packableScalar(t string) bool {
	switch t {
	case "bool", "u8", "b8", "s8", "u16", "b16", "s16", "u32", "b32", "s32", "u64", "b64", "s64", "f32", "f64":
		return true
	}
	return false
}

// Pack marshals Go arguments into a contiguous little-endian parameter byte buffer.
// It verifies argument count, enforces scalar widths and overflow constraints,
// rejects unresolved or Go heap pointers, and validates the parameter bank layout.
// Numeric device addresses are serialized unchanged. Pointer slot alignment does
// not specify pointee alignment, extent or allocation validity; the caller and
// typed operation own those checks.
func (s Schema) Pack(args ...any) ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, fmt.Errorf("cudakernel: invalid schema: %w", err)
	}
	for _, p := range s.Params {
		if !p.IsPointer && !packableScalar(p.Type) {
			return nil, fmt.Errorf("cudakernel: unsupported parameter type %q", p.Type)
		}
	}
	if len(args) != len(s.Params) {
		return nil, fmt.Errorf("cudakernel: argument count mismatch: schema expects %d arguments, got %d",
			len(s.Params), len(args))
	}

	buf := make([]byte, s.TotalParamBytes)

	for i, arg := range args {
		p := s.Params[i]
		if arg == nil {
			return nil, fmt.Errorf("cudakernel: argument %d (%s) is nil", i, p.Name)
		}

		if p.IsPointer {
			// Pointer parameter handling:
			// Strictly reject Go host heap pointers!
			rv := reflect.ValueOf(arg)
			if rv.Kind() == reflect.Pointer || rv.Kind() == reflect.UnsafePointer {
				return nil, fmt.Errorf("cudakernel: argument %d (%s): cannot pass Go heap pointer %T as device pointer; never persist Go heap addresses", i, p.Name, arg)
			}

			var devPtr uint64
			switch v := arg.(type) {
			case driver.DevicePtr:
				devPtr = uint64(v)
			case DevicePtrProvider:
				devPtr = uint64(v.Ptr())
			case uint64:
				devPtr = v
			case uintptr:
				devPtr = uint64(v)
			default:
				return nil, fmt.Errorf("cudakernel: argument %d (%s): unsupported device pointer type %T", i, p.Name, arg)
			}

			if p.Offset+8 > len(buf) {
				return nil, fmt.Errorf("cudakernel: argument %d (%s): buffer overflow", i, p.Name)
			}
			binary.LittleEndian.PutUint64(buf[p.Offset:], devPtr)
			continue
		}

		// Scalar parameter handling:
		// Reject any pointer passed for a scalar parameter.
		rv := reflect.ValueOf(arg)
		if rv.Kind() == reflect.Pointer || rv.Kind() == reflect.UnsafePointer {
			return nil, fmt.Errorf("cudakernel: argument %d (%s): scalar parameter expects value, got pointer %T", i, p.Name, arg)
		}

		if err := packScalar(buf[p.Offset:p.Offset+p.ByteWidth], p, arg); err != nil {
			return nil, fmt.Errorf("cudakernel: argument %d (%s): %w", i, p.Name, err)
		}
	}

	return buf, nil
}

func packScalar(dest []byte, p Param, arg any) error {
	switch p.Type {
	case "bool":
		b, ok := arg.(bool)
		if !ok {
			return fmt.Errorf("type mismatch: expected bool, got %T", arg)
		}
		if b {
			dest[0] = 1
		} else {
			dest[0] = 0
		}
		return nil

	case "u8", "b8":
		v, err := toUint64Exact(arg, 1)
		if err != nil {
			return err
		}
		if v > math.MaxUint8 {
			return fmt.Errorf("value %d overflows uint8", v)
		}
		dest[0] = byte(v)
		return nil

	case "s8":
		v, err := toInt64Exact(arg, 1)
		if err != nil {
			return err
		}
		if v < math.MinInt8 || v > math.MaxInt8 {
			return fmt.Errorf("value %d overflows int8", v)
		}
		dest[0] = byte(v)
		return nil

	case "u16", "b16":
		v, err := toUint64Exact(arg, 2)
		if err != nil {
			return err
		}
		if v > math.MaxUint16 {
			return fmt.Errorf("value %d overflows uint16", v)
		}
		binary.LittleEndian.PutUint16(dest, uint16(v))
		return nil

	case "s16":
		v, err := toInt64Exact(arg, 2)
		if err != nil {
			return err
		}
		if v < math.MinInt16 || v > math.MaxInt16 {
			return fmt.Errorf("value %d overflows int16", v)
		}
		binary.LittleEndian.PutUint16(dest, uint16(v))
		return nil

	case "u32", "b32":
		v, err := toUint64Exact(arg, 4)
		if err != nil {
			return err
		}
		if v > math.MaxUint32 {
			return fmt.Errorf("value %d overflows uint32", v)
		}
		binary.LittleEndian.PutUint32(dest, uint32(v))
		return nil

	case "s32":
		v, err := toInt64Exact(arg, 4)
		if err != nil {
			return err
		}
		if v < math.MinInt32 || v > math.MaxInt32 {
			return fmt.Errorf("value %d overflows int32", v)
		}
		binary.LittleEndian.PutUint32(dest, uint32(v))
		return nil

	case "u64", "b64":
		v, err := toUint64Exact(arg, 8)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(dest, v)
		return nil

	case "s64":
		v, err := toInt64Exact(arg, 8)
		if err != nil {
			return err
		}
		binary.LittleEndian.PutUint64(dest, uint64(v))
		return nil

	case "f32":
		switch v := arg.(type) {
		case float32:
			binary.LittleEndian.PutUint32(dest, math.Float32bits(v))
			return nil
		case float64:
			return fmt.Errorf("scalar width mismatch: expected 4-byte float32, got 8-byte float64")
		default:
			return fmt.Errorf("type mismatch: expected float32, got %T", arg)
		}

	case "f64":
		switch v := arg.(type) {
		case float64:
			binary.LittleEndian.PutUint64(dest, math.Float64bits(v))
			return nil
		case float32:
			return fmt.Errorf("scalar width mismatch: expected 8-byte float64, got 4-byte float32")
		default:
			return fmt.Errorf("type mismatch: expected float64, got %T", arg)
		}

	default:
		return fmt.Errorf("unsupported parameter type %q", p.Type)
	}
}

func toUint64Exact(arg any, targetWidth int) (uint64, error) {
	switch v := arg.(type) {
	case uint8:
		return uint64(v), nil
	case uint16:
		if targetWidth < 2 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (2 bytes) exceeds target width %d", arg, targetWidth)
		}
		return uint64(v), nil
	case uint32:
		if targetWidth < 4 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (4 bytes) exceeds target width %d", arg, targetWidth)
		}
		return uint64(v), nil
	case uint64:
		if targetWidth < 8 {
			if v > uint64((1<<(targetWidth*8))-1) {
				return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows %d-byte target", arg, v, targetWidth)
			}
			return 0, fmt.Errorf("scalar width mismatch: argument %T (8 bytes) exceeds target width %d", arg, targetWidth)
		}
		return v, nil
	case uint:
		val := uint64(v)
		if targetWidth < int(unsafe.Sizeof(v)) && val > uint64((1<<(targetWidth*8))-1) {
			return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows %d-byte target", arg, val, targetWidth)
		}
		return val, nil
	case int8:
		if v < 0 {
			return 0, fmt.Errorf("scalar underflow: negative signed value %d for unsigned parameter", v)
		}
		return uint64(v), nil
	case int16:
		if v < 0 {
			return 0, fmt.Errorf("scalar underflow: negative signed value %d for unsigned parameter", v)
		}
		if targetWidth < 2 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (2 bytes) exceeds target width %d", arg, targetWidth)
		}
		return uint64(v), nil
	case int32:
		if v < 0 {
			return 0, fmt.Errorf("scalar underflow: negative signed value %d for unsigned parameter", v)
		}
		if targetWidth < 4 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (4 bytes) exceeds target width %d", arg, targetWidth)
		}
		return uint64(v), nil
	case int64:
		if v < 0 {
			return 0, fmt.Errorf("scalar underflow: negative signed value %d for unsigned parameter", v)
		}
		if targetWidth < 8 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (8 bytes) exceeds target width %d", arg, targetWidth)
		}
		return uint64(v), nil
	case int:
		if v < 0 {
			return 0, fmt.Errorf("scalar underflow: negative signed value %d for unsigned parameter", v)
		}
		val := uint64(v)
		if targetWidth < 8 && val > uint64((1<<(targetWidth*8))-1) {
			return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows %d-byte target", arg, val, targetWidth)
		}
		return val, nil
	default:
		return 0, fmt.Errorf("type mismatch: expected unsigned integer, got %T", arg)
	}
}

func toInt64Exact(arg any, targetWidth int) (int64, error) {
	switch v := arg.(type) {
	case int8:
		return int64(v), nil
	case int16:
		if targetWidth < 2 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (2 bytes) exceeds target width %d", arg, targetWidth)
		}
		return int64(v), nil
	case int32:
		if targetWidth < 4 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (4 bytes) exceeds target width %d", arg, targetWidth)
		}
		return int64(v), nil
	case int64:
		if targetWidth < 8 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (8 bytes) exceeds target width %d", arg, targetWidth)
		}
		return v, nil
	case int:
		val := int64(v)
		if targetWidth == 4 && (val < math.MinInt32 || val > math.MaxInt32) {
			return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows 4-byte int32", arg, val)
		}
		if targetWidth == 2 && (val < math.MinInt16 || val > math.MaxInt16) {
			return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows 2-byte int16", arg, val)
		}
		if targetWidth == 1 && (val < math.MinInt8 || val > math.MaxInt8) {
			return 0, fmt.Errorf("scalar overflow: argument %T value %d overflows 1-byte int8", arg, val)
		}
		return val, nil
	case uint8:
		return int64(v), nil
	case uint16:
		if targetWidth < 2 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (2 bytes) exceeds target width %d", arg, targetWidth)
		}
		return int64(v), nil
	case uint32:
		if targetWidth < 4 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (4 bytes) exceeds target width %d", arg, targetWidth)
		}
		return int64(v), nil
	case uint64:
		if targetWidth < 8 {
			return 0, fmt.Errorf("scalar width mismatch: argument %T (8 bytes) exceeds target width %d", arg, targetWidth)
		}
		if v > math.MaxInt64 {
			return 0, fmt.Errorf("scalar overflow: uint64 value %d overflows int64", v)
		}
		return int64(v), nil
	default:
		return 0, fmt.Errorf("type mismatch: expected integer, got %T", arg)
	}
}

// PackPointers prepares an array of unsafe.Pointer elements for cuLaunchKernel.
// The returned pointers reference values in an allocated, pinned buffer.
func (s Schema) PackPointers(args ...any) ([]unsafe.Pointer, error) {
	packed, err := s.Pack(args...)
	if err != nil {
		return nil, err
	}
	ptrs := make([]unsafe.Pointer, len(s.Params))
	for i, p := range s.Params {
		ptrs[i] = unsafe.Pointer(&packed[p.Offset])
	}
	return ptrs, nil
}
