package xidlog_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
	"github.com/tmc/lib/nvidia/x/xidlog"
)

func inputs() xidlog.Options {
	return xidlog.Options{Source: xidlog.Source{BootID: "supplied-boot", InputID: "constructed example"}, Product: "L4", Driver: "580"}
}
func Example() {
	r, _ := xidlog.NewReader(strings.NewReader("NVRM: Xid (PCI:0000:65:00): 31, constructed\n"), xidlog.Text, inputs())
	v, _ := r.Next()
	fmt.Println(v.Kind, v.PCI, v.Code, v.Applicability == xidcatalog.UncoveredProduct)
	// Output: xid 0000:65:00 31 true
}
func ExampleNewReader() {
	r, _ := xidlog.NewReader(strings.NewReader("unrelated\n"), xidlog.Text, inputs())
	v, _ := r.Next()
	fmt.Println(v.Kind)
	// Output: other
}
func ExampleReader_Next() {
	r, _ := xidlog.NewReader(strings.NewReader("truncated"), xidlog.Text, inputs())
	v, _ := r.Next()
	fmt.Println(v.Kind, v.Partial)
	// Output: malformed true
}
func ExampleParseText() {
	v, _ := xidlog.ParseText("NVRM: Xid (PCI:0000:65:00): 9999, future", inputs())
	fmt.Println(v.Kind, v.Applicability == xidcatalog.UnknownID)
	// Output: xid true
}
func ExampleParseKmsg() {
	v, _ := xidlog.ParseKmsg("6,10,123,-;unrelated\n", inputs())
	fmt.Println(v.Header.Sequence, v.Header.MonotonicMicroseconds)
	// Output: 10 123
}
func ExampleKind_String() {
	fmt.Println(xidlog.Gap.String())
	// Output: gap
}
func ExampleSource() {
	s := xidlog.Source{BootID: "previous", InputID: "caller supplied text", PreviousBoot: true}
	fmt.Println(s.PreviousBoot)
	// Output: true
}
func ExampleOptions() {
	fmt.Println(inputs().Product)
	// Output: L4
}
func ExampleRecord() {
	var r xidlog.Record
	fmt.Println(r.Recovery == nil)
	// Output: true
}
func ExampleKmsg() {
	h := xidlog.Kmsg{Sequence: 10}
	fmt.Println(h.Sequence)
	// Output: 10
}
func ExampleFormat() {
	fmt.Println(xidlog.Text != xidlog.KernelRecords)
	// Output: true
}
