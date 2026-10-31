package elfinspect

import (
	"bytes"
	"fmt"
)

func ExampleRead() {
	image, err := Read(bytes.NewReader(fixture()))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(image.TableState, len(image.Tables), len(image.Segments))
	symbol := image.Tables[0].Symbols[0]
	fmt.Printf("%s binding=%d section=%d\n", symbol.Name, symbol.Binding, symbol.Section.Index)
	// Output:
	// present 2 1
	// cache binding=10 section=1
}
