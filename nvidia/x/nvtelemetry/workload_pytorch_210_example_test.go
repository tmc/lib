package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadPyTorchAnnotations_cpu210() {
	raw, err := os.ReadFile("testdata/pytorch-2.10.0-cpu-eager.chrome.json")
	if err != nil {
		fmt.Println(err)
		return
	}
	source := nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "retained-cpu-run"}
	export, annotations, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	// These are CPU annotation records, not exact CUDA operation identities.
	fmt.Println(len(annotations), len(export.Raw))
	fmt.Println(export.RawSHA256 == "da43e72ecac5387566568d1f139af9b4aaefa3cb5861ce11ff5028e6e96cad47")
	// Output:
	// 2 680712
	// true
}
