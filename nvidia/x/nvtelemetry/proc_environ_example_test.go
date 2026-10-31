package nvtelemetry_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadProcEnviron() {
	facts, err := nvtelemetry.ReadProcEnviron(strings.NewReader("SLURM_JOB_ID=999\x00TOKEN=not-retained\x00"))
	fmt.Println(facts.State, facts.SlurmJobID, err)
	// Output: self_reported 999 <nil>
}
