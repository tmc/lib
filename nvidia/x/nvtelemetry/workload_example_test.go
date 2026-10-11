package nvtelemetry_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadServingExposition() {
	source := nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "serving1"}
	raw := `# TYPE nv_inference_exec_count counter
nv_inference_exec_count{model="batching",version="1"} 2
`
	export, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	p := export.Metrics[0]
	fmt.Println(p.Semantic, p.RawValue, p.Unit, p.HasTime)
	// Output: uncached_batch_executions 2 executions false
}

func ExampleCounterWorkloadWindow() {
	source := nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "serving1"}
	read := func(count string, when int64) nvtelemetry.WorkloadExport {
		raw := "# TYPE nv_inference_request_success counter\nnv_inference_request_success{model=\"example\",version=\"1\"} " + count + "\n"
		e, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
		if err != nil {
			fmt.Println(err)
			return nvtelemetry.WorkloadExport{}
		}
		e, err = nvtelemetry.StampWorkloadSnapshot(e, when, "replay", "epoch1")
		if err != nil {
			fmt.Println(err)
		}
		return e
	}
	a, b := read("10", 0), read("13", 100)
	delta, err := nvtelemetry.CounterWorkloadWindow(a, b, 0, 100, "replay", "epoch1")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(delta[0].Semantic, delta[0].Delta)
	// Output: successful_requests 3
}

func ExampleReadMLflowHistory() {
	source := nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "train1", Units: map[string]string{"loss": "1"}}
	raw := `{"metrics":[{"key":"loss","value":0.5,"timestamp":1700000000000,"step":0}]}`
	export, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	p := export.Metrics[0]
	fmt.Println(p.Name, p.RawValue, p.RawTime, p.Step, p.HasStep)
	// Output: loss 0.5 1700000000000 0 true
}

func ExampleReadWandBHistory() {
	source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "train1"}
	export, err := nvtelemetry.ReadWandBHistory(strings.NewReader("{\"loss\":0.5,\"_step\":0}\n"), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(export.Metrics[0].Unit, export.Metrics[0].HasTime)
	// Output: unknown false
}

func ExampleReadClearMLScalars() {
	source := nvtelemetry.WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "train1"}
	raw := `{"events":[{"type":"training_stats_scalar","task":"train1","metric":"loss","variant":"train","value":0.5,"iter":0}]}`
	export, err := nvtelemetry.ReadClearMLScalars(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(export.Metrics[0].Labels["variant"], export.Metrics[0].HasTime)
	// Output: train false
}

func ExampleReadPyTorchAnnotations() {
	source := nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.7.0", RunID: "train1"}
	raw := `{"traceEvents":[{"name":"train/step","cat":"user_annotation","ph":"X","ts":10,"dur":2,"pid":1,"tid":1}]}`
	_, annotations, err := nvtelemetry.ReadPyTorchAnnotations(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(annotations[0].Name, annotations[0].DurationNanos)
	// Output: train/step 2000
}

func ExampleInspectTrainingSteps() {
	rank := 0
	source := nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "train1", Rank: &rank}
	raw := `{"metrics":[{"key":"loss","value":0.5,"step":0},{"key":"loss","value":0.2,"step":2}]}`
	export, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	report, err := nvtelemetry.InspectTrainingSteps([]nvtelemetry.WorkloadExport{export}, "loss", []int{0, 1})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(report.Steps, report.GapAfter, report.MissingRanks)
	// Output: [0 2] [0] [1]
}

func ExampleExactWorkloadJoin() {
	fmt.Println(nvtelemetry.ExactWorkloadJoin(nvtelemetry.WorkloadExport{}))
	// Output: aggregate workload exports cannot establish exact request or kernel joins
}

func ExampleStampWorkloadSnapshot() {
	source := nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "serving1"}
	raw := "# TYPE nv_inference_request_success counter\nnv_inference_request_success{model=\"example\",version=\"1\"} 1\n"
	e, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	e, err = nvtelemetry.StampWorkloadSnapshot(e, 0, "replay", "epoch1")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(e.Metrics[0].HasTime, e.Metrics[0].TimeNanos, e.Metrics[0].TimeSource, e.Metrics[0].RawTime == "")
	// Output: true 0 collector_observation true
}

func ExampleHistogramWorkloadWindow() {
	source := nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "serving1"}
	raw := `# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="example",le="+Inf"} 1
vllm:time_to_first_token_seconds_count{model_name="example"} 1
vllm:time_to_first_token_seconds_sum{model_name="example"} 0.5
`
	e, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	a, err := nvtelemetry.StampWorkloadSnapshot(e, 0, "replay", "epoch1")
	if err != nil {
		fmt.Println(err)
		return
	}
	b, err := nvtelemetry.StampWorkloadSnapshot(e, 10, "replay", "epoch1")
	if err != nil {
		fmt.Println(err)
		return
	}
	delta, err := nvtelemetry.HistogramWorkloadWindow(a, b, 0, 10, "replay", "epoch1")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(delta), delta[0].Delta)
	// Output: 3 0
}
