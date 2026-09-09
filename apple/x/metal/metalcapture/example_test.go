package metalcapture_test

import (
	"bytes"
	"fmt"
	"log"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

func Example() {
	// Create a minimal valid pprof profile
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "gpu_time", Unit: "nanoseconds"},
		},
		PeriodType: &profile.ValueType{
			Type: "gpu_time", Unit: "nanoseconds",
		},
	}

	cap := &metalcapture.Capture{
		Profile: p,
		Files: map[string][]byte{
			"kernel.metal": []byte("#include <metal_stdlib>\nkernel void add() {}\n"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, cap); err != nil {
		log.Fatal(err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Source file:", readBack.Code.Source)
	fmt.Println("Source exists:", len(readBack.Files["kernel.metal"]) > 0)
	// Output:
	// Source file: kernel.metal
	// Source exists: true
}
