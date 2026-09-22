package cudaobjdump_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaobjdump"
)

func ExampleTool_ExtractWithReceipts() {
	// Invalid kinds fail before invoking an executable, so there are no commands.
	_, commands, err := (cudaobjdump.Tool{}).ExtractWithReceipts(context.Background(), "input.fatbin", cudaobjdump.Kind(-1))
	fmt.Println(len(commands), err != nil)
	// Output: 0 true
}
