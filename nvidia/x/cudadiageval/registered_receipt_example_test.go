package cudadiageval

import (
	"context"
	"fmt"
	"io"
	"strings"
)

func ExampleReadRegisteredReceipt() {
	_, err := ReadRegisteredReceipt(strings.NewReader("{}"))
	fmt.Println(err)
	// Output: paired receipt: missing field version
}

func ExampleWriteRegisteredReceipt() {
	err := WriteRegisteredReceipt(io.Discard, RegisteredReceipt{})
	fmt.Println(err)
	// Output: registered receipt: invalid identity
}

func ExampleValidateRegisteredPair() {
	_, err := ValidateRegisteredPair(context.Background(), Registration{}, QualificationLedger{}, RegisteredReceipt{}, RegisteredReceipt{}, "unused")
	fmt.Println(err)
	// Output: registration: invalid version or identity
}

func ExampleValidatedRegisteredPair_Identity() {
	var pair ValidatedRegisteredPair
	fmt.Println(pair.Identity().Version)
	// Output: 0
}
