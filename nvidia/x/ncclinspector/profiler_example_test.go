package ncclinspector_test

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// exampleTrace is a constructed trace in the print_event.cc v7 format.
// final_sync is absent because the kernel stamped before_close at its end.
const exampleTrace = `[
{"name": "AllReduce", "cat": "COLL", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 10.500000, "args": {"SeqNum": 7, "CommHash": 1311768467294899695, "Rank": 1, "Count": 1024, "Datatype": "ncclFloat32", "Algorithm": "Ring", "Protocol": "Simple", "nChannels": 1, "KernelVariant": "", "IsSymColl": 1}},
{"name": "KernelCh", "cat": "GPU", "ph": "b", "id": 0, "pid": 2101, "tid": 1, "ts": 11.250000, "args": {"Channel": 0, "StartGpuClk": 5000, "StopGpuClk": 5900}},
{"name": "initial_sync", "cat": "GPU", "ph": "X", "id": 0, "pid": 2101, "tid": 1, "ts": 11.300000, "dur": 0.200000, "args": {"Channel": 0, "PhaseId": 0, "StartGpuClk": 5000, "StopGpuClk": 5200}},
{"name": "compute", "cat": "GPU", "ph": "X", "id": 0, "pid": 2101, "tid": 1, "ts": 11.400000, "dur": 0.700000, "args": {"Channel": 0, "PhaseId": 1, "StartGpuClk": 5200, "StopGpuClk": 5900}},
{"name": "KernelCh", "cat": "GPU", "ph": "e", "id": 0, "pid": 2101, "tid": 1, "ts": 13.000000},
{"name": "AllReduce", "cat": "COLL", "ph": "e", "id": 0, "pid": 2101, "tid": 1, "ts": 14.750000},
{}]
`

func ExampleReadProfilerTrace() {
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
	c := trace.Collectives[0]
	fmt.Println(c.Communicator(), c.Name, c.Sequence, "rank", c.Rank)
	for _, p := range c.KernelChannels[0].Phases {
		ns, _ := p.ReportedNS()
		fmt.Printf("channel %d %s: %d ns on the GPU timer, host callback at %s us\n", p.Channel, p.Name, ns, p.Micros)
	}
	// Output:
	// 0x1234567890abcdef AllReduce 7 rank 1
	// channel 0 initial_sync: 200 ns on the GPU timer, host callback at 11.300000 us
	// channel 0 compute: 700 ns on the GPU timer, host callback at 11.400000 us
}

func ExampleProfilerProfile_Source() {
	for _, p := range []ncclinspector.ProfilerProfile{ncclinspector.ProfilerNCCL2323, ncclinspector.ProfilerNCCL2312, "future"} {
		s, err := p.Source()
		fmt.Println(p, s.MaxChannels, s.RegularPhasesGated, s.APIRanks, err != nil)
	}
	// Output:
	// nccl-v2.32.3-1/example-profiler-v7 64 true true false
	// nccl-v2.31.2-1/example-profiler-v7 32 false false false
	// future 0 false false true
}
