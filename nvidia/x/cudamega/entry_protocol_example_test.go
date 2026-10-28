package cudamega_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudamega"
)

func ExampleStartOrderWords() {
	words, err := cudamega.StartOrderWords(2, 2)
	// One counter, four task tickets, and two fresh startup words.
	fmt.Println(words, err)
	// Output: 7 <nil>
}
