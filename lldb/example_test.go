package lldb_test

import (
	"fmt"

	"github.com/tmc/lib/lldb"
)

func ExampleRequire() {
	err := lldb.Require("a_symbol_no_lldb_bridge_exports")
	fmt.Println(err != nil)
	// Output: true
}
