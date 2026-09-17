package cudakernel

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
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

// NewSchema creates a Schema by computing natural alignment offsets for the given parameters.
func NewSchema(params ...Param) (Schema, error) {
	if len(params) == 0 {
		return Schema{}, nil
	}

	offset := 0
	maxAlign := 1
	computed := make([]Param, len(params))

	for i, p := range params {
		if p.Name == "" {
			return Schema{}, fmt.Errorf("cudakernel: param %d has empty name", i)
		}
		if p.ByteWidth <= 0 {
			p.ByteWidth = defaultByteWidth(p.Type, p.IsPointer)
		}
		if p.ByteWidth <= 0 {
			return Schema{}, fmt.Errorf("cudakernel: param %q has invalid byte width %d", p.Name, p.ByteWidth)
		}
		if p.Alignment <= 0 {
			p.Alignment = p.ByteWidth
		}
		if p.Alignment > 16 || !isPowerOfTwo(p.Alignment) {
			return Schema{}, fmt.Errorf("cudakernel: param %q has invalid alignment %d", p.Name, p.Alignment)
		}
		if p.Alignment > maxAlign {
			maxAlign = p.Alignment
		}

		// Pad current offset to parameter alignment.
		rem := offset % p.Alignment
		if rem != 0 {
			offset += p.Alignment - rem
		}

		p.Offset = offset
		offset += p.ByteWidth
		computed[i] = p
	}

	// Pad total struct size to max alignment (minimum 8 bytes for GPU parameter bank).
	if maxAlign < 8 {
		maxAlign = 8
	}
	rem := offset % maxAlign
	if rem != 0 {
		offset += maxAlign - rem
	}

	return Schema{
		Params:          computed,
		TotalParamBytes: offset,
	}, nil
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

// Validate checks internal consistency of the Schema offsets and widths.
func (s Schema) Validate() error {
	offset := 0
	for i, p := range s.Params {
		if p.ByteWidth <= 0 {
			return fmt.Errorf("param %d (%s) has non-positive byte width %d", i, p.Name, p.ByteWidth)
		}
		if p.Alignment <= 0 || !isPowerOfTwo(p.Alignment) {
			return fmt.Errorf("param %d (%s) has invalid alignment %d", i, p.Name, p.Alignment)
		}
		if p.Offset < offset {
			return fmt.Errorf("param %d (%s) offset %d overlaps prior offset %d", i, p.Name, p.Offset, offset)
		}
		if p.Offset%p.Alignment != 0 {
			return fmt.Errorf("param %d (%s) offset %d not aligned to %d", i, p.Name, p.Offset, p.Alignment)
		}
		offset = p.Offset + p.ByteWidth
	}
	if offset > s.TotalParamBytes {
		return fmt.Errorf("total param bytes %d smaller than required offset %d", s.TotalParamBytes, offset)
	}
	return nil
}

// Pack marshals Go arguments into a contiguous little-endian parameter byte buffer.
// It verifies argument count, enforces scalar widths and overflow constraints,
// rejects unresolved or Go heap pointers, and checks device pointer alignment.
func (s Schema) Pack(args ...any) ([]byte, error) {
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

			if devPtr%8 != 0 {
				return nil, fmt.Errorf("cudakernel: argument %d (%s): device pointer 0x%x is not 8-byte aligned", i, p.Name, devPtr)
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
