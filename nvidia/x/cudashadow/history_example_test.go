package cudashadow_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

func ExampleWriteHistory() {
	var data bytes.Buffer
	if err := cudashadow.WriteHistory(&data, nil); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(data.String())
	// Output: {"version":1,"snapshots":null}
}

func ExampleReadHistory() {
	history, err := cudashadow.ReadHistory(strings.NewReader(`{"version":1,"snapshots":[]}`))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(history))
	// Output: 0
}
