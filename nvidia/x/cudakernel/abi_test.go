package cudakernel_test

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestABIWidthsAndAlignment(t *testing.T) {
	// Define a mixed ABI schema
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "b8_val", Type: "u8"},
		cudakernel.Param{Name: "u32_val", Type: "u32"},
		cudakernel.Param{Name: "u16_val", Type: "u16"},
		cudakernel.Param{Name: "ptr_val", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "f32_val", Type: "f32"},
		cudakernel.Param{Name: "f64_val", Type: "f64"},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}

	// Verify alignment offsets
	// b8_val: offset 0 (size 1, align 1)
	// u32_val: offset 4 (size 4, align 4) -> pad 3 bytes
	// u16_val: offset 8 (size 2, align 2)
	// ptr_val: offset 16 (size 8, align 8) -> pad 6 bytes
	// f32_val: offset 24 (size 4, align 4)
	// f64_val: offset 32 (size 8, align 8) -> pad 4 bytes
	// Total: 40 bytes (aligned to 8)
	expectedOffsets := map[string]int{
		"b8_val":  0,
		"u32_val": 4,
		"u16_val": 8,
		"ptr_val": 16,
		"f32_val": 24,
		"f64_val": 32,
	}

	for _, p := range schema.Params {
		expected, ok := expectedOffsets[p.Name]
		if !ok {
			t.Fatalf("unexpected param %q", p.Name)
		}
		if p.Offset != expected {
			t.Errorf("param %q offset = %d, want %d", p.Name, p.Offset, expected)
		}
	}

	if schema.TotalParamBytes != 40 {
		t.Errorf("TotalParamBytes = %d, want 40", schema.TotalParamBytes)
	}

	// Pack valid arguments
	packed, err := schema.Pack(
		uint8(42),
		uint32(100000),
		uint16(500),
		driver.DevicePtr(0x8000),
		float32(3.14),
		float64(2.718281828),
	)
	if err != nil {
		t.Fatalf("Pack failed: %v", err)
	}
	if len(packed) != 40 {
		t.Fatalf("packed len = %d, want 40", len(packed))
	}

	// Verify packed values byte-by-byte
	if packed[0] != 42 {
		t.Errorf("packed[0] = %d, want 42", packed[0])
	}
	if val := binary.LittleEndian.Uint32(packed[4:8]); val != 100000 {
		t.Errorf("packed u32 = %d, want 100000", val)
	}
	if val := binary.LittleEndian.Uint16(packed[8:10]); val != 500 {
		t.Errorf("packed u16 = %d, want 500", val)
	}
	if val := binary.LittleEndian.Uint64(packed[16:24]); val != 0x8000 {
		t.Errorf("packed ptr = 0x%x, want 0x8000", val)
	}
	if val := math.Float32frombits(binary.LittleEndian.Uint32(packed[24:28])); val != float32(3.14) {
		t.Errorf("packed f32 = %f, want 3.14", val)
	}

	// PackPointers verification
	ptrs, err := schema.PackPointers(
		uint8(42),
		uint32(100000),
		uint16(500),
		driver.DevicePtr(0x8000),
		float32(3.14),
		float64(2.718281828),
	)
	if err != nil {
		t.Fatalf("PackPointers failed: %v", err)
	}
	if len(ptrs) != 6 {
		t.Fatalf("PackPointers len = %d, want 6", len(ptrs))
	}
	for i, ptr := range ptrs {
		if ptr == nil {
			t.Errorf("ptrs[%d] is nil", i)
		}
	}

	// Negative control 1: Changed scalar width (passing 64-bit int to 32-bit slot)
	_, err = schema.Pack(
		uint8(42),
		uint64(100000), // Changed scalar width: uint64 instead of uint32!
		uint16(500),
		driver.DevicePtr(0x8000),
		float32(3.14),
		float64(2.718281828),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted changed scalar width (uint64 for u32)")
	} else if !strings.Contains(err.Error(), "scalar width mismatch") {
		t.Fatalf("expected scalar width mismatch error, got: %v", err)
	}

	// Negative control 2: Changed float width (float64 for float32)
	_, err = schema.Pack(
		uint8(42),
		uint32(100000),
		uint16(500),
		driver.DevicePtr(0x8000),
		float64(3.14), // Changed scalar width: float64 for f32!
		float64(2.718281828),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted changed float scalar width")
	} else if !strings.Contains(err.Error(), "scalar width mismatch") {
		t.Fatalf("expected scalar width mismatch error, got: %v", err)
	}

	// Negative control 3: Integer overflow (value exceeds uint8)
	_, err = schema.Pack(
		uint8(255),     // ok
		uint32(100000), // ok
		uint16(500),
		driver.DevicePtr(0x8000),
		float32(3.14),
		float64(2.718281828),
	)
	if err != nil {
		t.Fatalf("unexpected error for uint8(255): %v", err)
	}
	_, err = schema.Pack(
		int32(-1), // underflow for u8!
		uint32(100000),
		uint16(500),
		driver.DevicePtr(0x8000),
		float32(3.14),
		float64(2.718281828),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted negative signed value for unsigned u8")
	}

	// Negative control 4: Go heap pointer rejected!
	heapInt := 1234
	_, err = schema.Pack(
		uint8(42),
		uint32(100000),
		uint16(500),
		&heapInt, // Go host heap pointer!
		float32(3.14),
		float64(2.718281828),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted Go host heap pointer")
	} else if !strings.Contains(err.Error(), "cannot pass Go heap pointer") {
		t.Fatalf("expected heap pointer rejection error, got: %v", err)
	}

	// Numeric device addresses are preserved; this schema does not type the pointee.
	packed, err = schema.Pack(
		uint8(42),
		uint32(100000),
		uint16(500),
		driver.DevicePtr(0x8003), // An unknown byte pointee may have an odd address.
		float32(3.14),
		float64(2.718281828),
	)
	if err != nil {
		t.Fatalf("numeric device address rejected: %v", err)
	}
	if got := binary.LittleEndian.Uint64(packed[16:24]); got != 0x8003 {
		t.Fatalf("numeric device address = 0x%x, want 0x8003", got)
	}

	// Unsafe pointer host heap rejected
	unsafePtr := unsafe.Pointer(&heapInt)
	_, err = schema.Pack(
		uint8(42),
		uint32(100000),
		uint16(500),
		unsafePtr,
		float32(3.14),
		float64(2.718281828),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted unsafe.Pointer host heap address")
	}
}
