package cudainspect_test

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudainspect"
)

func ExampleInspector_Report() {
	raw, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		log.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		log.Fatal(err)
	}
	i, err := cudainspect.NewInspector(os.TempDir())
	if err != nil {
		log.Fatal(err)
	}
	r, err := i.Report(context.Background(), cubin, cudainspect.Request{
		Symbol:   "addk",
		Required: []string{"target"},
		Optional: []string{"feature_suffix"},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(r.Result)
	for _, name := range []string{"target", "feature_suffix", "dynamic_shared_bytes"} {
		f := r.Feature(name)
		if f.Known() {
			fmt.Println(f.Name, f.State, string(f.Value))
		} else {
			fmt.Println(f.Name, f.State)
		}
	}
	// Output:
	// partial
	// target observed {"architecture":"sm_121","major":12,"minor":1}
	// feature_suffix unknown
	// dynamic_shared_bytes unsupported
}
