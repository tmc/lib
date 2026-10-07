package xidcatalog_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/xidcatalog"
)

func ExampleLookup() {
	e, ok := xidcatalog.Lookup(13, xidcatalog.H100)
	fmt.Println(ok, e.Description, e.ImmediateAction)
	_, ok = xidcatalog.Lookup(999, xidcatalog.H100)
	fmt.Println(ok)
	// Output:
	// true Graphics Engine Exception RESTART_APP
	// false
}
