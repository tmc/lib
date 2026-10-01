package main

import (
	"context"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

var (
	stubAgentOnce sync.Once
	stubAgentPath string
	stubAgentErr  error
)

// stubAgentTarget builds the agent's own control harness: the real producer
// linked against a stub CUPTI, so a capture-range scenario writes real edge
// records into a real session without a GPU.
func stubAgentTarget(t *testing.T) string {
	t.Helper()
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skipf("no C toolchain: %v", err)
	}
	stubAgentOnce.Do(func() {
		dir, err := os.MkdirTemp("", "cudaprof-stub-agent")
		if err != nil {
			stubAgentErr = err
			return
		}
		src := filepath.Join("..", "..", "..", "internal", "cudainject", "native")
		stubAgentPath = filepath.Join(dir, "cupti-agent-control")
		out, err := exec.Command(cc, "-std=c11", "-rdynamic", "-I", filepath.Join(src, "testcupti"),
			"-DCUDAPROF_TEST_HARNESS=1",
			filepath.Join(src, "cupti_agent.c"), filepath.Join(src, "testcupti", "stubcupti.c"), filepath.Join(src, "cupti_agent_control.c"),
			"-o", stubAgentPath, "-lpthread").CombinedOutput()
		if err != nil {
			stubAgentErr = errors.New(string(out))
		}
	})
	if stubAgentErr != nil {
		t.Fatalf("build stub agent: %v", stubAgentErr)
	}
	return stubAgentPath
}

func u32(v uint32) *uint32 { return &v }

var fakeDevices = []gpuReading{{Index: 0, UUID: "GPU-fake", Name: "fake", SMClockMHz: u32(1980), MemClockMHz: u32(8000), TemperatureC: u32(41), PowerMilliwatts: u32(73000)}}

func fakeGPU(opens *int) gpuOpener {
	return func() (func() []gpuReading, func(), error) {
		*opens++
		return func() []gpuReading { return fakeDevices }, func() {}, nil
	}
}

func TestCaptureSamplesGPUAtRangeEdges(t *testing.T) {
	target := stubAgentTarget(t)
	noNVML := gpuOpener(func() (func() []gpuReading, func(), error) {
		return nil, nil, errors.New("nvml init: no libnvidia-ml")
	})
	tests := []struct {
		name        string
		rangeMode   string
		open        func(*int) gpuOpener
		wantOpens   int
		unavailable string
	}{
		{name: "gated", rangeMode: captureRangeProfilerAPI, open: fakeGPU, wantOpens: 1},
		{name: "no nvml", rangeMode: captureRangeProfilerAPI, open: func(n *int) gpuOpener {
			return func() (func() []gpuReading, func(), error) { *n++; return noNVML() }
		}, wantOpens: 1, unavailable: "nvml init: no libnvidia-ml"},
		{name: "ungated", rangeMode: captureRangeNone, open: fakeGPU, wantOpens: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opens := 0
			result, err := runCapture(context.Background(), captureOptions{
				command:      []string{target, "capture-range", "1", "4", "0", "0", "2"},
				Env:          []string{"PATH=" + os.Getenv("PATH")},
				NoInject:     true,
				CaptureRange: tt.rangeMode,
				Stdout:       io.Discard,
				openGPU:      tt.open(&opens),
				DrainTimeout: time.Second,
			})
			if err != nil {
				t.Fatal(err)
			}
			if opens != tt.wantOpens {
				t.Fatalf("GPU opened %d times, want %d", opens, tt.wantOpens)
			}
			if tt.wantOpens == 0 {
				if result.PhaseReceipt != nil {
					t.Fatalf("ungated capture has edges: %+v", result.PhaseReceipt.Edges)
				}
				return
			}
			r := result.PhaseReceipt
			if r == nil || len(r.Edges) != 2 {
				t.Fatalf("phase receipt = %+v (capture error %q), want two edges", r, result.CaptureError)
			}
			for _, e := range r.Edges {
				g := e.GPU
				if g == nil {
					t.Fatalf("%s edge has no GPU sample", e.Edge)
				}
				if tt.unavailable != "" {
					if g.Unavailable != tt.unavailable || g.Devices != nil || g.SampledAt != 0 || g.LagNanos != 0 {
						t.Errorf("%s edge = %+v, want only unavailable %q", e.Edge, g, tt.unavailable)
					}
					continue
				}
				if g.Unavailable != "" || !reflect.DeepEqual(g.Devices, fakeDevices) {
					t.Errorf("%s edge = %+v, want the fake device", e.Edge, g)
				}
				if g.SampledAt <= e.Timestamp || g.LagNanos != int64(g.SampledAt-e.Timestamp) {
					t.Errorf("%s edge at %d sampled at %d with lag %d: the reading must follow the edge and say by how much", e.Edge, e.Timestamp, g.SampledAt, g.LagNanos)
				}
			}
		})
	}
}

// TestEdgeClocksSampleWhileTargetRuns writes an edge into a journal in two
// pieces, as a writer the controller races might, and holds the capture
// open well past it. The sample must be taken when the edge completes, not
// when the capture ends.
func TestEdgeClocksSampleWhileTargetRuns(t *testing.T) {
	if _, ok := monotonicNanos(); !ok {
		t.Skip("no CLOCK_MONOTONIC")
	}
	dir := t.TempDir()
	proc := filepath.Join(dir, "process-4242-7")
	if err := os.Mkdir(proc, 0o700); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(proc, "capture.journal"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var session [16]byte
	if _, err := hex.Decode(session[:], []byte("00112233445566778899aabbccddeeff")); err != nil {
		t.Fatal(err)
	}
	frame := func(seq uint64, ts uint64, payload string) []byte {
		b, err := wire.Record{Kind: wire.KindActivity, Identity: wire.Identity{Session: session, Process: 4242, Generation: 7}, Sequence: seq, Timestamp: ts, Payload: []byte(payload)}.MarshalBinary(wire.Limits{})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	opens := 0
	c := openEdgeClocks(fakeGPU(&opens))
	c.watch(dir, 4242)

	// A kernel record is not an edge and must not be sampled.
	f.Write(frame(1, 5, `{"kind":"activity","subtype":"kernel"}`))
	edgeAt, _ := monotonicNanos()
	edge := frame(2, edgeAt, `{"kind":"activity","subtype":"capture_range","edge":"start","api":"cudaProfilerStart","index":0}`)
	f.Write(edge[:50])
	time.Sleep(20 * time.Millisecond)
	completedAt, _ := monotonicNanos()
	f.Write(edge[50:])
	const held = 300 * time.Millisecond
	time.Sleep(held)
	samples := c.finish()

	if len(samples) != 1 {
		t.Fatalf("samples = %v, want the one edge", samples)
	}
	g := samples[edgeKey{2, edgeAt}]
	if g == nil {
		t.Fatalf("no sample for the edge: %v", samples)
	}
	if g.SampledAt < completedAt {
		t.Errorf("sampled at %d, before the edge's frame completed at %d", g.SampledAt, completedAt)
	}
	if lag := time.Duration(g.SampledAt - completedAt); lag >= held/2 {
		t.Errorf("sampled %v after the edge completed, with the capture held %v: the reading was taken at finish, not live", lag, held)
	}
}

func TestAttachEdgeClocksNamesAnUnseenEdge(t *testing.T) {
	seen := &gpuEdgeSample{SampledAt: 20, LagNanos: 10, Devices: fakeDevices}
	r := &capturePhaseReceipt{Edges: []capturePhaseEdge{{Sequence: 3, Timestamp: 10, Edge: "start"}, {Sequence: 9, Timestamp: 50, Edge: "stop"}}}
	attachEdgeClocks(r, map[edgeKey]*gpuEdgeSample{{3, 10}: seen, {9, 49}: seen})
	if r.Edges[0].GPU != seen {
		t.Errorf("start edge = %+v, want its sample", r.Edges[0].GPU)
	}
	if g := r.Edges[1].GPU; g == nil || g.Unavailable == "" || g.Devices != nil {
		t.Errorf("stop edge = %+v, want it named unseen: a sample keyed to another timestamp is not its sample", g)
	}
	attachEdgeClocks(nil, nil)
}
