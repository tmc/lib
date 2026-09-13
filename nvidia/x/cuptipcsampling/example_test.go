package cuptipcsampling_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

func Example() {
	_, err := cuptipcsampling.StallReasons(0, 128)
	fmt.Println(err)
	// Output: cuptipcsampling: invalid reason inventory request
}

func ExampleStallReasons() {
	_, err := cuptipcsampling.StallReasons(0, 128)
	fmt.Println(err)
	// Output: cuptipcsampling: invalid reason inventory request
}

func ExampleStallReason() {
	r := cuptipcsampling.StallReason{Index: 7, Name: "native_reason"}
	fmt.Println(r.Index, r.Name)
	// Output: 7 native_reason
}

func ExampleReasonInventory() {
	r := cuptipcsampling.ReasonInventory{CUPTIVersion: 28}
	fmt.Println(r.CUPTIVersion, len(r.Reasons))
	// Output: 28 0
}

func ExampleIdentifyCubin() {
	_, err := cuptipcsampling.IdentifyCubin(nil, 1024)
	fmt.Println(err)
	// Output: cuptipcsampling: invalid cubin size or limit
}

func ExampleCubinIdentity() {
	var identity cuptipcsampling.CubinIdentity
	fmt.Println(identity.CUPTIVersion)
	// Output: 0
}

func ExampleConfig() {
	// Period is an exponent: the native sampling period is 2^Period cycles.
	// MaxOutputBytes covers both output buffers, so each receives half.
	c := cuptipcsampling.Config{Period: 10, MaxOutputBytes: 304}
	fmt.Println(uint64(1)<<c.Period, c.MaxOutputBytes/2)
	// Output: 1024 152
}

func ExampleStart() {
	// Start validates every budget before it makes a native call, so an
	// unowned context is rejected without touching the collector.
	config := cuptipcsampling.Config{
		Period: 10, WorkerSleepMS: 1,
		ReasonIDs: []uint32{7}, MaxReasons: 8,
		ScratchBytes: 1 << 20, HardwareBytes: 1 << 20, MaxNativeBytes: 1 << 21,
		PCsPerBuffer: 1, MaxOutputBytes: 304, MaxReads: 2,
		MaxNames: 4, MaxNameBytes: 20, MaxTotalNameBytes: 80,
	}
	_, err := cuptipcsampling.Start(0, config)
	fmt.Println(err)
	// Output: cuptipcsampling: invalid sampling configuration or limits
}

func ExampleSession_Read() {
	// A zero Session owns nothing; only a Session from Start can read.
	var s *cuptipcsampling.Session
	_, err := s.Read()
	fmt.Println(err)
	// Output: cuptipcsampling: invalid session
}

func ExampleSession_Stop() {
	var s *cuptipcsampling.Session
	fmt.Println(s.Stop())
	// Output: cuptipcsampling: invalid session
}

func ExampleSession_Close() {
	var s *cuptipcsampling.Session
	fmt.Println(s.Close())
	// Output: cuptipcsampling: invalid session
}

func ExampleSession_Final() {
	// Final reports availability, not capture success, and is unavailable
	// until a successful native disable and local finalization.
	var s *cuptipcsampling.Session
	_, ok := s.Final()
	fmt.Println(ok)
	// Output: false
}

func ExampleSession_Info() {
	var s *cuptipcsampling.Session
	info := s.Info()
	fmt.Println(info.CUPTIVersion, info.Verified, info.Closed)
	// Output: 0 false false
}

func ExampleStallSamples() {
	s := cuptipcsampling.StallSamples{Index: 7, Samples: 128}
	fmt.Println(s.Index, s.Samples)
	// Output: 7 128
}

func ExamplePC() {
	pc := cuptipcsampling.PC{
		CRC: 0x1234, Offset: 0x40, FunctionIndex: 3, Name: "kernel",
		Reasons: []cuptipcsampling.StallSamples{{Index: 7, Samples: 2}},
	}
	fmt.Println(pc.Name, pc.Offset, pc.Reasons[0].Samples)
	// Output: kernel 64 2
}

func ExampleBatch() {
	// Valid reports decoder validity. Loss is reported separately and a valid
	// batch may still be incomplete.
	b := cuptipcsampling.Batch{Valid: true, TotalSamples: 10, DroppedSamples: 4}
	fmt.Println(b.Valid, b.TotalSamples, b.DroppedSamples)
	// Output: true 10 4
}

func ExampleInfo() {
	// Effective attributes stay diagnostic while Verified is false, and Failed
	// remains set even after a cleanup retry succeeds.
	var info cuptipcsampling.Info
	fmt.Println(info.Verified, info.Failed, info.Reads)
	// Output: false false 0
}

func ExampleProfileBridge() {
	var buf bytes.Buffer
	prof := cudaprof.NewSession(&buf)

	bridge := &cuptipcsampling.ProfileBridge{
		Session: prof,
		Inventory: cuptipcsampling.ReasonInventory{
			Reasons: []cuptipcsampling.StallReason{
				{Index: 1, Name: "stall_math_pipe_throttle"},
			},
		},
	}

	batch := cuptipcsampling.Batch{
		Valid: true,
		PCs: []cuptipcsampling.PC{
			{
				Name:   "compute_kernel",
				Offset: 0x40,
				Reasons: []cuptipcsampling.StallSamples{
					{Index: 1, Samples: 16},
				},
			},
		},
	}

	err := bridge.RecordBatch(context.Background(), batch)
	fmt.Println("Recorded:", err == nil)
	// Output:
	// Recorded: true
}

func ExampleCaptureResolver() {
	resolver := cuptipcsampling.NewCaptureResolver(nil, 0)
	loc := resolver.ResolvePC(cuptipcsampling.ModuleID{}, "compute", 0)
	fmt.Println(loc.KernelName, loc.Resolved)
	// Output:
	// compute false
}

func ExampleClassifyPipelineStall() {
	canonical, cat, desc, _ := cuptipcsampling.ClassifyPipelineStall("smsp__pcsamp_warps_issue_stalled_long_scoreboard")
	fmt.Printf("%s [%s]: %s\n", canonical, cat, desc[:22])
	// Output:
	// long_scoreboard [memory]: Waiting on scoreboard 
}

func ExampleReasonInventory_DecodeSASSMetric() {
	inv := cuptipcsampling.ReasonInventory{
		Reasons: []cuptipcsampling.StallReason{
			{Index: 0, Name: "selected"},
			{Index: 1, Name: "long_scoreboard"},
		},
	}
	pc := cuptipcsampling.PC{
		Offset: 0x200,
		Name:   "gemm",
		Reasons: []cuptipcsampling.StallSamples{
			{Index: 0, Samples: 50},
			{Index: 1, Samples: 150},
		},
	}
	metric := inv.DecodeSASSMetric(pc)
	fmt.Printf("PC=0x%x Total=%d Dominant=%s (%.1f%%)\n",
		metric.PC, metric.TotalSamples, metric.DominantReason(), metric.Dominant.Percent)
	// Output:
	// PC=0x200 Total=200 Dominant=long_scoreboard (100.0%)
}

func ExampleAttributePCSamples() {
	records := []cuptiactivity.Record{
		{
			Kind:     cupti.ActivityKindConcurrentKernel,
			Name:     "vec_add",
			Start:    1000,
			End:      3000,
			Complete: true,
		},
	}
	metrics := []cuptipcsampling.SASSMetric{
		{
			PC:           0x10,
			KernelName:   "vec_add",
			TotalSamples: 100,
			Selected:     20,
			StallSamples: 80,
			Stalls: []cuptipcsampling.PipelineStall{
				{Name: "long_scoreboard", Samples: 80, Percent: 100.0},
			},
		},
	}
	attributed := cuptipcsampling.AttributePCSamples(records, metrics)
	if len(attributed) > 0 {
		ak := attributed[0]
		fmt.Printf("%s: %dns dominant=%s (%.1f%%)\n",
			ak.KernelName, ak.DurationNs, ak.DominantStall, ak.DominantPct)
	}
	// Output:
	// vec_add: 2000ns dominant=long_scoreboard (100.0%)
}
