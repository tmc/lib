package ncclras_test

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/ncclras"
)

func ExampleReadText() {
	_, err := ncclras.ReadText(strings.NewReader("unknown format\n"), ncclras.Limits{})
	var unsupported *ncclras.UnsupportedError
	fmt.Println(errors.As(err, &unsupported), unsupported.Version == "")
	// Output: true true
}
