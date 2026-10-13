package stack_test

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof/stack"
)

func Example() {
	var output bytes.Buffer
	journal := stack.NewJournal(&output, stack.Limits{})
	r := stack.Record{Version: 1, Process: "producer", PID: 1, Language: "go", Attribution: "submission", ExternalKind: 5, ExternalID: 1, Status: "attached", Frames: []stack.Frame{{Function: "main.launch"}}}
	journal.Record(r)
	records, _ := stack.Read(&output, stack.Limits{})
	fmt.Println(records[0].Frames[0].Function)
	// Output: main.launch
}

func ExampleNewJournal() {
	var output bytes.Buffer
	journal := stack.NewJournal(&output, stack.Limits{Records: 10})
	fmt.Println(journal != nil)
	// Output: true
}

func ExampleRead() {
	records, err := stack.Read(strings.NewReader(""), stack.Limits{})
	fmt.Println(len(records), err)
	// Output: 0 <nil>
}

func ExampleReadSamples() {
	input := `{"version":1,"type":"sample","process":"generation","pid":1,"session":"session","generation":2,"language":"python","tid":3,"time":100,"clock":"monotonic","status":"clock_unavailable","provenance":"wall_stack_snapshot","frames":[{"function":"work"}]}` + "\n"
	samples, _ := stack.ReadSamples(strings.NewReader(input), stack.Limits{})
	fmt.Println(samples[0].Provenance)
	// Output: wall_stack_snapshot
}

func ExampleReadGoTraceDebug() {
	input := "M=1 P=0 G=2 StackSample Time=100\nStack=\n\tmain.work @ 0x123\n\t\twork.go:7\n"
	trace, _ := stack.ReadGoTraceDebug(strings.NewReader(input), stack.Record{Process: "generation", PID: 1}, stack.Limits{})
	fmt.Println(trace.Samples[0].TraceTime)
	// Output: 100
}

func ExampleGoTraceDecoder_Validate() {
	_, err := new(stack.GoTraceDecoder).Validate(context.Background(), nil, stack.GoProducer{})
	fmt.Println(err)
	// Output: go trace decoder not selected
}

func ExampleQualifyGoMonotonic() {
	trace := &stack.GoTrace{ClockSnapshots: []stack.ClockSnapshot{{Trace: 100, Monotonic: 110}}, Samples: []stack.Sample{{TraceTime: 150}}}
	stack.QualifyGoMonotonic(trace, "linux", "go1.27.1")
	fmt.Println(trace.Samples[0].Time, trace.ClockUncertaintyNs)
	// Output: 150 64
}

func ExampleWriteCoverage() {
	var b bytes.Buffer
	stack.WriteCoverage(&b, stack.Record{Process: "generation", PID: 1, Language: "go"})
	fmt.Println(bytes.Contains(b.Bytes(), []byte(`"final":true`)))
	// Output: true
}

func ExampleGoTraceDecoder_Analyze() {
	_, err := new(stack.GoTraceDecoder).Analyze(context.Background(), nil, stack.GoProducer{}, stack.Record{}, stack.Limits{})
	fmt.Println(err)
	// Output: go trace decoder not selected
}
