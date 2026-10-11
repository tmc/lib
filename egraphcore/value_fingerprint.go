package egraphcore

import (
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"
	"math"
	"reflect"
)

// ValueFingerprint returns a conservative FNV-1a fingerprint for values made
// from booleans, numbers, strings, arrays, slices, structs, and interfaces.
// The result is a bucket key, not proof of equality; callers must still compare
// values when fingerprints match. It returns false for unsupported kinds,
// cyclic slices, or values nested more than 128 levels deep.
func ValueFingerprint(value any) (string, bool) {
	if value == nil {
		return "nil", true
	}
	h := fnv.New64a()
	if !writeValueFingerprint(h, reflect.ValueOf(value), 0, make(map[uintptr]bool)) {
		return "", false
	}
	return fmt.Sprintf("%016x", h.Sum64()), true
}

// StructuralFingerprint combines operator fields, payload data, and ordered
// child fingerprints into a bucket key. Unsupported payloads use a type-only
// key. Collisions are possible; callers must still confirm structural equality.
func StructuralFingerprint(op, sort string, data any, children []string) string {
	h := fnv.New64a()
	writeField := func(value string) {
		writeValueFingerprintUint64(h, uint64(len(value)))
		_, _ = h.Write([]byte(value))
	}
	writeField(op)
	writeField(sort)
	payload, ok := ValueFingerprint(data)
	if !ok {
		payload = fmt.Sprintf("type:%T", data)
	}
	writeField(payload)
	writeValueFingerprintUint64(h, uint64(len(children)))
	for _, child := range children {
		writeField(child)
	}
	return fmt.Sprintf("%016x", h.Sum64())
}

func writeValueFingerprint(h hash.Hash64, value reflect.Value, depth int, activeSlices map[uintptr]bool) bool {
	if depth > 128 {
		return false
	}
	if !value.IsValid() {
		h.Write([]byte{0})
		return true
	}
	h.Write([]byte{byte(value.Kind()) + 1})
	typeName := value.Type().PkgPath() + "/" + value.Type().String()
	writeValueFingerprintBytes(h, []byte(typeName))
	switch value.Kind() {
	case reflect.Bool:
		if value.Bool() {
			h.Write([]byte{1})
		} else {
			h.Write([]byte{0})
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		var bits [8]byte
		binary.LittleEndian.PutUint64(bits[:], uint64(value.Int()))
		h.Write(bits[:])
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		var bits [8]byte
		binary.LittleEndian.PutUint64(bits[:], value.Uint())
		h.Write(bits[:])
	case reflect.Float32:
		f := float32(value.Float())
		bits := math.Float32bits(f)
		if f == 0 {
			bits = 0
		}
		var encoded [4]byte
		binary.LittleEndian.PutUint32(encoded[:], bits)
		h.Write(encoded[:])
	case reflect.Float64:
		f := value.Float()
		bits := math.Float64bits(f)
		if f == 0 {
			bits = 0
		}
		var encoded [8]byte
		binary.LittleEndian.PutUint64(encoded[:], bits)
		h.Write(encoded[:])
	case reflect.Complex64:
		c := complex64(value.Complex())
		re, im := math.Float32bits(real(c)), math.Float32bits(imag(c))
		if real(c) == 0 {
			re = 0
		}
		if imag(c) == 0 {
			im = 0
		}
		var encoded [8]byte
		binary.LittleEndian.PutUint32(encoded[:4], re)
		binary.LittleEndian.PutUint32(encoded[4:], im)
		h.Write(encoded[:])
	case reflect.Complex128:
		c := value.Complex()
		re, im := math.Float64bits(real(c)), math.Float64bits(imag(c))
		if real(c) == 0 {
			re = 0
		}
		if imag(c) == 0 {
			im = 0
		}
		var encoded [16]byte
		binary.LittleEndian.PutUint64(encoded[:8], re)
		binary.LittleEndian.PutUint64(encoded[8:], im)
		h.Write(encoded[:])
	case reflect.String:
		writeValueFingerprintBytes(h, []byte(value.String()))
	case reflect.Array:
		writeValueFingerprintUint64(h, uint64(value.Len()))
		for i := 0; i < value.Len(); i++ {
			if !writeValueFingerprint(h, value.Index(i), depth+1, activeSlices) {
				return false
			}
		}
	case reflect.Slice:
		if value.IsNil() {
			h.Write([]byte{0})
			return true
		}
		writeValueFingerprintUint64(h, uint64(value.Len()))
		if value.Len() == 0 {
			return true
		}
		ptr := value.Pointer()
		if ptr != 0 && activeSlices[ptr] {
			return false
		}
		if ptr != 0 {
			activeSlices[ptr] = true
			defer delete(activeSlices, ptr)
		}
		for i := 0; i < value.Len(); i++ {
			if !writeValueFingerprint(h, value.Index(i), depth+1, activeSlices) {
				return false
			}
		}
	case reflect.Struct:
		writeValueFingerprintUint64(h, uint64(value.NumField()))
		for i := 0; i < value.NumField(); i++ {
			if !writeValueFingerprint(h, value.Field(i), depth+1, activeSlices) {
				return false
			}
		}
	case reflect.Interface:
		if value.IsNil() {
			h.Write([]byte{0})
			return true
		}
		return writeValueFingerprint(h, value.Elem(), depth+1, activeSlices)
	default:
		return false
	}
	return true
}

func writeValueFingerprintBytes(h hash.Hash64, value []byte) {
	writeValueFingerprintUint64(h, uint64(len(value)))
	h.Write(value)
}

func writeValueFingerprintUint64(h hash.Hash64, value uint64) {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	h.Write(encoded[:])
}
