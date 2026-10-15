package cudapipeline_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func ExamplePipelineSchedule_AttentionSource() {
	layout := cudapipeline.AttentionLayout{Keys: 5, HeadDim: 3, TileSize: 2}
	payload, err := layout.PayloadBytes()
	if err != nil {
		panic(err)
	}
	schedule, err := cudapipeline.BuildWarpSpecializedSchedule(2, payload, 3)
	if err != nil {
		panic(err)
	}
	source, schema, err := schedule.AttentionSource("attention", layout)
	if err != nil {
		panic(err)
	}
	fmt.Println(payload, len(schema.Params), strings.Contains(source, "u[d]/l"))
	// Output: 32 2 true
}
