package nvml_test

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/nvml"
)

func ExampleValue_Float64() {
	v := nvml.Value{Data: [1]uint64{math.Float64bits(1.5)}}
	fmt.Println(v.Float64())
	// Output: 1.5
}

func ExampleValue_Int32() {
	var data [8]byte
	binary.NativeEndian.PutUint32(data[:], 0xfffffffe)
	v := nvml.Value{Data: [1]uint64{binary.NativeEndian.Uint64(data[:])}}
	fmt.Println(v.Int32())
	// Output: -2
}

func ExampleValue_Uint32() {
	var data [8]byte
	binary.NativeEndian.PutUint32(data[:], 42)
	v := nvml.Value{Data: [1]uint64{binary.NativeEndian.Uint64(data[:])}}
	fmt.Println(v.Uint32())
	// Output: 42
}

func ExampleValue_UnsignedLong() {
	// A zero-initialized union is zero for either native unsigned-long width.
	var v nvml.Value
	fmt.Println(v.UnsignedLong())
	// Output: 0
}

func ExampleValue_Uint64() {
	v := nvml.Value{Data: [1]uint64{1 << 40}}
	fmt.Println(v.Uint64())
	// Output: 1099511627776
}

func ExampleValue_Int64() {
	v := nvml.Value{Data: [1]uint64{0xfffffffffffffffe}}
	fmt.Println(v.Int64())
	// Output: -2
}

func ExampleValue_Uint16() {
	var data [8]byte
	binary.NativeEndian.PutUint16(data[:], 42)
	v := nvml.Value{Data: [1]uint64{binary.NativeEndian.Uint64(data[:])}}
	fmt.Println(v.Uint16())
	// Output: 42
}
