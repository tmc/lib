package shape_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func Example() {
	report := shape.Compare(timeline.Records{}, timeline.Records{}, "driver")
	fmt.Println(report.Version, report.Pass, len(report.Groups))
	// Output: 1 false 0
}

func ExampleCompare() {
	records := timeline.Records{Activities: []timeline.Activity{{Name: "gemv<float>", Start: 10, End: 20}}}
	report := shape.Compare(records, records, "driver")
	fmt.Println(report.Pass, report.BaseUnmatched[0].RawSymbol)
	// Output: false gemv<float>
}

func ExampleIdentity() {
	id := shape.Identity{Phase: "decode", Token: 0, Operation: "gemv", DType: "float32", CacheLength: 32}
	fmt.Println(id.Phase, id.Token, id.Layer == nil)
	// Output: decode 0 true
}

func ExampleLaunch() {
	launch := shape.Launch{RawSymbol: "gemv<float>", Reason: "required evidence unavailable"}
	fmt.Println(launch.RawSymbol, launch.Reason)
	// Output: gemv<float> required evidence unavailable
}

func ExampleDistribution() {
	d := shape.Distribution{Count: 1, Min: 10, Median: 10, Max: 10, Sum: 10}
	fmt.Println(d.Count, d.Median)
	// Output: 1 10
}

func ExampleGroup() {
	group := shape.Group{Identity: shape.Identity{Operation: "gemv"}}
	fmt.Println(group.Identity.Operation)
	// Output: gemv
}

func ExampleReport() {
	report := shape.Compare(timeline.Records{}, timeline.Records{}, "driver")
	fmt.Println(report.Sources["grid"])
	// Output: observed CUPTI activity
}

func ExampleRequestPair() {
	pair := shape.RequestPair{BaseJournal: "go", BaseProcess: 1, BaseID: "request-1", TargetJournal: "python", TargetProcess: 2, TargetID: "run-a", Role: "measured"}
	fmt.Println(pair.BaseID, pair.TargetID, pair.Role)
	// Output: request-1 run-a measured
}

func ExampleCompareScoped() {
	report := shape.CompareScoped(timeline.Records{}, timeline.Records{}, "driver", nil)
	fmt.Println(report.Pass)
	fmt.Println(report.WorkScope)
	// Output:
	// false
	// unavailable: legacy metadata or successful execution counters missing
}
