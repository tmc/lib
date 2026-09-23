package cudaprobe_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprobe"
)

func ExampleTargets() {
	for _, target := range cudaprobe.Targets() {
		fmt.Println(target.Library, target.Symbol)
	}
	// Output:
	// libcuda.so.1 cuLaunchKernel
	// libcuda.so.1 cuMemAlloc_v2
	// libcuda.so.1 cuMemFree_v2
	// libcudart.so cudaLaunchKernel
	// libcudart.so cudaMalloc
	// libcudart.so cudaFree
}

type exampleResolver struct{}

func (exampleResolver) Resolve(_ context.Context, target cudaprobe.Target) (cudaprobe.Resolution, error) {
	// A real collector resolves the ELF symbol; this is a synthetic receipt.
	return cudaprobe.Resolution{Target: target, BuildID: "synthetic", Offset: 64, FileSize: 128, Function: true}, nil
}

func ExampleResolve() {
	r, err := cudaprobe.Resolve(context.Background(), exampleResolver{}, cudaprobe.Target{Library: "fixture.so", Symbol: "cuLaunchKernel"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Target.Symbol, r.BuildID, r.Offset)
	// Output: cuLaunchKernel synthetic 64
}

func ExampleDemangler_Batch() {
	// This deliberately unavailable tool illustrates the offline fallback.
	d, err := cudaprobe.NewDemangler(os.DevNull+"/cu++filt", 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	names, receipt, err := d.Batch(context.Background(), []string{"_Z1fv"})
	fmt.Println(names[0].Display, names[0].Demangled, receipt.Failure != "", err != nil)
	// Output: _Z1fv false true true
}

func ExampleEvent_Record() {
	e := cudaprobe.Event{Identity: cudaprobe.Identity{Provenance: "synthetic", Symbol: "cuLaunchKernel"}, Timestamp: 123}
	r, err := e.Record(time.Unix(1, 0))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.TimestampNs, r.Labels["probe.timestamp_ns"], r.Labels["duration_status"])
	// Output: 1000000000 123 unavailable
}

func ExampleCompareCounts() {
	p := cudaprobe.Process{PID: 42, Generation: 123}
	a := cudaprobe.Count{Process: p, Interval: "fixture-1", Source: "uprobe", Value: 4}
	b := cudaprobe.Count{Process: p, Interval: "fixture-1", Source: "application", Value: 5}
	c, err := cudaprobe.CompareCounts(a, b)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(c.Probe.Value, c.Independent.Value, c.Missing)
	// Output: 4 5 1
}

// exampleSource stands in for a collector replaying a completed capture.
type exampleSource struct{ records []cudaprobe.Capture }

func (s *exampleSource) Next(ctx context.Context) (cudaprobe.Capture, error) {
	if err := ctx.Err(); err != nil {
		return cudaprobe.Capture{}, err
	}
	if len(s.records) == 0 {
		return cudaprobe.Capture{}, io.EOF
	}
	c := s.records[0]
	s.records = s.records[1:]
	return c, nil
}

func ExampleTrack() {
	c := cudaprobe.Capture{Identity: cudaprobe.Identity{
		Process: cudaprobe.Process{PID: 42, Generation: 123}, AddressSpace: 1,
		Library: "fixture.so", BuildID: "synthetic", Version: "fixture-v1",
		Symbol: "cudaMalloc", ABI: "linux-amd64-sysv-v1",
	}}
	binary.LittleEndian.PutUint16(c.Data[:], 1)
	binary.LittleEndian.PutUint16(c.Data[2:], 7)
	binary.LittleEndian.PutUint64(c.Data[8:], 1)
	binary.LittleEndian.PutUint64(c.Data[40:], 64)
	binary.LittleEndian.PutUint64(c.Data[120:], 0x8000)
	t, err := cudaprobe.NewTracker(c.Identity.Process, 100)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := cudaprobe.Track(context.Background(), &exampleSource{[]cudaprobe.Capture{c}}, t); err != nil {
		fmt.Println(err)
		return
	}
	unmatched, incomplete := t.Report()
	fmt.Println("unmatched at exit:", len(unmatched), "incomplete:", incomplete)
	// Output: unmatched at exit: 1 incomplete: false
}

func ExampleTracker_Observe() {
	p := cudaprobe.Process{PID: 42, Generation: 123}
	t, err := cudaprobe.NewTracker(p, 100)
	if err != nil {
		fmt.Println(err)
		return
	}
	e := cudaprobe.Event{
		Identity: cudaprobe.Identity{Process: p, AddressSpace: 1, Symbol: "cudaMalloc"},
		Sequence: 1, Decoded: true, ReturnKnown: true, Address: 0x8000, Bytes: 64,
	}
	if err := t.Observe(e); err != nil {
		fmt.Println(err)
		return
	}
	a, incomplete := t.Report()
	fmt.Println(a[0].Bytes, incomplete)
	// Output: 64 false
}

func ExampleDecode() {
	c := cudaprobe.Capture{Identity: cudaprobe.Identity{
		Process: cudaprobe.Process{PID: 42, Generation: 123},
		Library: "fixture.so", BuildID: "synthetic", Version: "fixture-v1",
		Symbol: "cuLaunchKernel", ABI: "linux-amd64-sysv-v1",
	}}
	binary.LittleEndian.PutUint16(c.Data[:], 1)
	binary.LittleEndian.PutUint16(c.Data[2:], 1)
	binary.LittleEndian.PutUint64(c.Data[8:], 1)
	for i, arg := range []uint64{0x100, 2, 3, 4, 32, 1, 1} {
		binary.LittleEndian.PutUint64(c.Data[32+8*i:], arg)
	}
	e, err := cudaprobe.Decode(c)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(e.Launch.Grid, e.Launch.Block, e.ReturnKnown)
	// Output: [2 3 4] [32 1 1] false
}

func ExampleNewTracker() {
	t, err := cudaprobe.NewTracker(cudaprobe.Process{PID: 42, Generation: 123}, 100)
	if err != nil {
		fmt.Println(err)
		return
	}
	t.Gap()
	t.Exit()
	unmatched, incomplete := t.Report()
	fmt.Println(len(unmatched), incomplete)
	// Output: 0 true
}
