package nvtelemetry_test

import (
	"bytes"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadServingContextV2() {
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&b, servingContextV2Fixture(), nvtelemetry.WorkloadLimits{}); err != nil {
		fmt.Println(err)
		return
	}
	raw := b.Bytes()
	_, c, err := nvtelemetry.ReadServingContextV2(bytes.NewReader(raw), servingSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(c.Attempts), c.Batches[0].ID, len(c.Batches[0].Members), c.Transfers[0].TransferID)
	_, _, err = nvtelemetry.ReadServingContext(bytes.NewReader(raw), servingSource(), nvtelemetry.WorkloadLimits{})
	fmt.Println(err)
	// Output:
	// 2 b1 2 t2
	// decode serving context: json: unknown field "routes"
}

func ExampleServingContextV2_V1() {
	s, err := nvtelemetry.SummarizeServingContext(servingContextV2Fixture().V1(), nvtelemetry.WorkloadLimits{})
	fmt.Println(s.Accepted, s.Failed, s.AcceptedOutputTokens, err)
	// Output: 1 1 9007199254740993 <nil>
}
