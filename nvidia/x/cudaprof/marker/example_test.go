package marker_test

import (
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/nvtx"
)

func Example() {
	message, err := marker.Encode(marker.Metadata{Phase: "decode", Operation: "gemv"})
	if err != nil {
		panic(err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	nvtx.RangePush(message)
	// Submit the operation on this thread.
	nvtx.RangePop()
	fmt.Println(message)
	// Output: cudaprof:v1 {"phase":"decode","operation":"gemv"}
}

func ExampleMetadata() {
	token := 0
	m := marker.Metadata{Phase: "decode", Token: &token}
	fmt.Println(m.Phase, *m.Token, m.Layer == nil)
	// Output: decode 0 true
}

func ExampleEncode() {
	message, err := marker.Encode(marker.Metadata{Phase: "prefill"})
	fmt.Println(message, err)
	// Output: cudaprof:v1 {"phase":"prefill"} <nil>
}

func ExampleParse() {
	m, err := marker.Parse(`cudaprof:v1 {"phase":"decode","token":3}`)
	fmt.Println(m.Phase, *m.Token, err)
	// Output: decode 3 <nil>
}

func ExampleParseRange() {
	m, err := marker.ParseRange(`inference/cudaprof:v1 {"phase":"decode"}`)
	fmt.Println(m.Phase, err)
	// Output: decode <nil>
}

func ExampleEncode_request() {
	executions := 32
	message, err := marker.Encode(marker.Metadata{Version: 2, RequestID: "request-1", Role: "measured", Phase: "cleanup", Event: "request_end", Boundary: "completion", ExecutionUnit: "logical_forward", Executions: &executions})
	fmt.Println(err)
	m, _ := marker.Parse(message)
	fmt.Println(m.RequestID, m.Role, m.Phase, *m.Executions)
	// Output:
	// <nil>
	// request-1 measured cleanup 32
}
