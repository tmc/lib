package dcgmjson_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/dcgmjson"
)

func ExampleRead() {
	report, err := dcgmjson.Read(strings.NewReader(`{"metadata":{"version":"4.6.1"},"DCGM Diagnostic":{}}`), dcgmjson.Limits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(report.Version, report.CategoriesAvailable)
	// Output: 4.6.1 false
}
