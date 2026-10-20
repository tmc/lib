package ncclinspector_test

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

const exampleRecord = `{"header":{"id":"0x1234567890abcdef","name":"constructed-comm","rank":1,"n_ranks":2,"nnodes":2},"metadata":{"inspector_output_format_version":"v4.1","git_rev":"constructed-producer-r1","rec_mechanism":"event_trace","dump_timestamp_us":9000,"hostname":"constructed-node-r1","pid":2101},"coll_perf":{"coll":"AllReduce","coll_sn":7,"coll_msg_size_bytes":4096,"coll_exec_time_us":351,"coll_timing_source":"event_trace","coll_algobw_gbs":0,"coll_busbw_gbs":0,"event_trace_ts":{"coll_start_ts":1001,"coll_stop_ts":1051,"kernel_events":[{"channel_id":0,"kernel_start_ts":1101,"kernel_stop_ts":1301,"kernel_record_ts":1401}]}}}
`

func ExampleBindProfiler() {
	sum := sha256.Sum256([]byte(exampleTrace))
	trace, err := ncclinspector.ReadProfilerTrace(strings.NewReader(exampleTrace), ncclinspector.ProfilerCapture{
		Profile: ncclinspector.ProfilerNCCL2323,
		Build:   "constructed-build",
		SHA256:  hex.EncodeToString(sum[:]),
	}, ncclinspector.Limits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	records, err := ncclinspector.Read(strings.NewReader(exampleRecord), ncclinspector.Limits{})
	if err != nil {
		fmt.Println(err)
		return
	}
	binding := ncclinspector.ProfilerBinding{
		Communicator: "0x1234567890abcdef",
		Rank:         1,
		Incarnation:  "constructed-incarnation-17",
		Host:         "constructed-node-r1",
		PID:          2101,
		Operation:    "AllReduce",
		Sequence:     7,
		Channels:     []int{0},
		Evidence:     "constructed-binding",
	}
	clock := ncclinspector.ClockMapping{Domain: ncclinspector.ClockDomainHostRealtime}
	bound, err := ncclinspector.BindProfiler(trace, binding, records[0], clock)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range bound.Channels[0].Phases {
		fmt.Printf("%s (%s to %s): %d ns\n", p.Name, p.From, p.To, p.ReportedNS)
	}
	fmt.Println("absent:", bound.Channels[0].Absent)

	// Requiring a fact the example profiler does not export is refused.
	binding.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedPhases}
	_, err = ncclinspector.BindProfiler(trace, binding, records[0], clock)
	fmt.Println(errors.Is(err, ncclinspector.ErrProfilerUnavailable))
	binding.Require = []ncclinspector.ProfilerNeed{ncclinspector.NeedLoss}
	_, err = ncclinspector.BindProfiler(trace, binding, records[0], clock)
	fmt.Println(errors.Is(err, ncclinspector.ErrProfilerUnavailable))
	// Output:
	// initial_sync (begin to after_open): 200 ns
	// compute (after_open to before_close): 700 ns
	// absent: [final_sync]
	// true
	// true
}
