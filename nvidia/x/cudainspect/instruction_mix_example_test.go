package cudainspect_test

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudainspect"
)

func ExampleReport_InstructionMix() {
	raw, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		log.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		log.Fatal(err)
	}
	sass, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.sass")
	if err != nil {
		log.Fatal(err)
	}
	i, err := cudainspect.NewInspector(os.TempDir())
	if err != nil {
		log.Fatal(err)
	}
	r, err := i.Report(context.Background(), cubin, cudainspect.Request{
		Symbol:   "addk",
		Required: []string{"precision_attribution"},
		SASS:     string(sass),
	})
	if err != nil {
		log.Fatal(err)
	}
	mix, state, err := r.InstructionMix()
	if err != nil {
		log.Fatal(err)
	}
	attr, err := cudacensus.AttributeFromMix(mix)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("state=%s precision=%s method=%s share=%.2f total=%d\n",
		state, attr.Precision, attr.Method, attr.Share, mix.Total())
	// Output:
	// state=observed precision=fp32 method=instruction_mix share=1.00 total=1
}
