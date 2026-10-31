package nvtelemetry_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// Ray 2.44.0 reports GPU memory as integer MiB. The input below is a
// constructed Prometheus text control in Ray's metric shape, not a Ray export.
func ExampleReadServingExposition_rayMemoryUnits() {
	source := nvtelemetry.WorkloadSource{Provider: "ray", Version: "2.44.0", RunID: "cluster1"}
	raw := `# HELP ray_node_gram_used Total GPU RAM usage on a ray node
# TYPE ray_node_gram_used gauge
ray_node_gram_used{GpuIndex="0",ip="10.0.0.1"} 1234.0
# HELP ray_node_gram_available Total GPU RAM available on a ray node
# TYPE ray_node_gram_available gauge
ray_node_gram_available{GpuIndex="0",ip="10.0.0.1"} 80686.0
`
	export, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range export.Metrics {
		fmt.Println(p.Semantic, p.RawValue, p.Unit)
	}
	// Output:
	// node_gpu_memory_used 1234.0 MiB
	// node_gpu_memory_available 80686.0 MiB
}
