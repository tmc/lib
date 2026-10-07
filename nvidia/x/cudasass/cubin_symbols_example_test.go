package cudasass

import "fmt"

func ExampleCubinImage_Symbols() {
	// This constructed ELF is a host example, not a measured compiler image.
	image, _ := OpenCubin(symbolImage(0x21, 0, 0))
	rows, _ := image.Symbols("raw")
	fmt.Println(len(rows), rows[0].Value, rows[0].Size, rows[0].Section)
	// Output: 1 0 4 SHN_UNDEF
}

func ExampleCubinImage_SHA256() {
	image, _ := OpenCubin(symbolImage(0x21, 0, 0))
	fmt.Println(image.SHA256() == image.SHA256())
	// Output: true
}
