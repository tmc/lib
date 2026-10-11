package cudasass

import (
	"crypto/sha256"
	"fmt"
)

func ExampleCubinImage_Symbols() {
	// This constructed ELF is a host example, not a measured compiler image.
	image, _ := OpenCubin(symbolImage(0x21, 0, 0))
	rows, _ := image.Symbols("raw")
	fmt.Println(len(rows), rows[0].Value, rows[0].Size, rows[0].Section)
	// Output: 1 0 4 SHN_UNDEF
}

func ExampleCubinImage_SHA256() {
	raw := symbolImage(0x21, 0, 0)
	original := sha256.Sum256(raw)
	image, _ := OpenCubin(raw)
	// Mutating the caller's input does not change the copied image identity.
	raw[len(raw)-1] ^= 1
	fmt.Println(image.SHA256() == original)
	// Output: true
}
