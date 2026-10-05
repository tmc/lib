package cudakernel

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestCheckPointerAlignment(t *testing.T) {
	for _, alignment := range []int{1, 2, 4, 8, 16} {
		t.Run(fmt.Sprint(alignment), func(t *testing.T) {
			for _, ptr := range []driver.DevicePtr{0, 0x1000, driver.DevicePtr(0x1000 + alignment)} {
				if err := CheckPointerAlignment("output", ptr, alignment); err != nil {
					t.Fatal(err)
				}
			}
			if alignment == 1 {
				return
			}
			for _, ptr := range []driver.DevicePtr{0x1001, ^driver.DevicePtr(0)} {
				err := CheckPointerAlignment("output", ptr, alignment)
				if err == nil || !strings.Contains(err.Error(), "output") || !strings.Contains(err.Error(), fmt.Sprintf("0x%x", ptr)) || !strings.Contains(err.Error(), fmt.Sprintf("%d-byte aligned", alignment)) {
					t.Fatalf("pointer %x alignment %d: %v", ptr, alignment, err)
				}
			}
		})
	}
	for _, alignment := range []int{-1, 0, 3, 32} {
		if err := CheckPointerAlignment("output", 0x1000, alignment); err == nil {
			t.Fatalf("accepted alignment %d", alignment)
		}
	}
	if err := CheckPointerAlignment("", 0x1000, 4); err == nil {
		t.Fatal("accepted unnamed buffer")
	}
}

func FuzzCheckPointerAlignment(f *testing.F) {
	f.Add(uint64(0x1004), uint8(4))
	f.Add(^uint64(0), uint8(16))
	f.Fuzz(func(t *testing.T, ptr uint64, alignment uint8) {
		want := false
		if alignment != 0 {
			_, err := NewSchema(Param{Name: "input", Type: "u8", ByteWidth: 1, Alignment: int(alignment)})
			want = err == nil && ptr%uint64(alignment) == 0
		}
		got := CheckPointerAlignment("input", driver.DevicePtr(ptr), int(alignment)) == nil
		if got != want {
			t.Fatalf("pointer %x alignment %d accepted %v, want %v", ptr, alignment, got, want)
		}
	})
}

func ExampleCheckPointerAlignment() {
	fmt.Println(CheckPointerAlignment("output", driver.DevicePtr(0x1000), 16))
	fmt.Println(CheckPointerAlignment("output", driver.DevicePtr(0x1004), 16))
	// Output:
	// <nil>
	// cudakernel: buffer "output": device pointer 0x1004 is not 16-byte aligned
}
