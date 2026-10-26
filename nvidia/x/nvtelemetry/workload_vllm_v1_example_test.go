package nvtelemetry_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadServingExposition_vllmV1() {
	// Constructed source-derived control, not a deployment capture.
	raw := "# TYPE vllm:prompt_tokens_total counter\nvllm:prompt_tokens_total{model_name=\"m\",engine=\"0\"} 7\n"
	source := nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", Engine: "v1", RunID: "example"}
	export, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(export.Source.Engine, export.Metrics[0].Semantic, export.Metrics[0].Labels["engine"])
	// Output: v1 processed_prefill_tokens 0
}
