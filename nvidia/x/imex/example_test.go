package imex_test

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/imex"
)

func Example() {
	f, err := os.Open("testdata/dcgm-spec-580.30.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	snapshot, err := imex.Read(f, imex.DCGMSpec58030, imex.Limits{})
	if err != nil {
		panic(err)
	}
	fmt.Println(snapshot.Status, len(snapshot.Nodes))
	// Output: DEGRADED 2
}
func ExampleRead() {
	f, err := os.Open("testdata/dcgm-spec-580.30.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	s, err := imex.Read(f, imex.DCGMSpec58030, imex.Limits{Nodes: 8})
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Nodes["3"].Connections["6"].Status)
	_, listed := s.Nodes["6"]
	fmt.Println(listed)
	// Output:
	// CONNECTED
	// false
}
func ExampleProfile() {
	fmt.Println(imex.DCGMSpec58030) // Output: dcgm-64df9f89-spec-580.30
}
func ExampleLimits() {
	fmt.Println(imex.Limits{Nodes: 8}.Nodes) // Output: 8
}
func ExampleSnapshot() {
	s := imex.Snapshot{Status: "DEGRADED"}
	fmt.Println(s.Status) // Output: DEGRADED
}
func ExampleNode() {
	n := imex.Node{}
	fmt.Println(n.Version == nil, n.Hostname == nil) // Output: true true
}
func ExampleConnection() {
	c := imex.Connection{Status: "CONNECTED", Changed: true}
	fmt.Println(c.Status, c.Changed) // Output: CONNECTED true
}

func ExampleUnsupportedProfileError() {
	_, err := imex.Read(strings.NewReader(`{}`), imex.Profile("unknown"), imex.Limits{})
	var unsupported *imex.UnsupportedProfileError
	fmt.Println(errors.As(err, &unsupported), unsupported.Profile)
	// Output: true unknown
}
