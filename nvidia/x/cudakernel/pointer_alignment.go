package cudakernel

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
)

// CheckPointerAlignment checks that ptr is aligned to alignment bytes using
// the alignment rule of [NewSchema]. Alignment must be explicitly specified
// as 1, 2, 4, 8 or 16. name identifies the buffer in an error.
//
// This checks only alignment. A null pointer is aligned; the caller must check
// buffer extents, ownership and context separately.
func CheckPointerAlignment(name string, ptr driver.DevicePtr, alignment int) error {
	if alignment <= 0 {
		return fmt.Errorf("cudakernel: buffer %q has invalid alignment %d", name, alignment)
	}
	schema, err := NewSchema(Param{Name: name, Type: "u8", ByteWidth: 1, Alignment: alignment})
	if err != nil {
		return err
	}
	if uint64(ptr)%uint64(schema.Params[0].Alignment) != 0 {
		return fmt.Errorf("cudakernel: buffer %q: device pointer 0x%x is not %d-byte aligned", name, uint64(ptr), alignment)
	}
	return nil
}
