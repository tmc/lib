//go:build linux

package cuptibench_test

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// This is a diagnostic for read-only file-backed inputs, not a claim that HMM
// causes any particular profiler failure. All inputs execute the same kernel.
// Input hmm uses MAP_PRIVATE; hmm-shared uses MAP_SHARED, as DS4 weights do.
func TestPhysicalReadOnlyHMMReplay(t *testing.T) {
	selected := os.Getenv("CUPTIBENCH_HMM_CASE")
	if selected == "" {
		t.Skip("set CUPTIBENCH_HMM_CASE to one input/mode; run each case in a fresh process")
	}
	matched := false
	for _, input := range []string{"device", "hmm", "hmm-shared"} {
		for _, mode := range []string{"direct-user", "direct-kernel", "graph-whole", "graph-kernel"} {
			if selected != input+"/"+mode {
				continue
			}
			matched = true
			t.Run(input+"/"+mode, func(t *testing.T) {
				p := newPhysicalPipeline(t, 4)
				ptr := p.state.Ptr()
				var mapped []byte
				mapping := "device-allocation"
				if input != "device" {
					mapping = "PROT_READ|MAP_PRIVATE"
					mapFlags := syscall.MAP_PRIVATE
					if input == "hmm-shared" {
						mapping = "PROT_READ|MAP_SHARED"
						mapFlags = syscall.MAP_SHARED
					}
					for _, attr := range []driver.DeviceAttribute{driver.DevAttrIntegrated, driver.DevAttrUnifiedAddressing, driver.DevAttrPageableMemoryAccess, driver.DevAttrPageableMemoryAccessUsesHostPageTables} {
						value, err := p.dev.Attribute(attr)
						physicalCheck(t, err)
						t.Logf("HPT capability %s=%d", attr, value)
						if value == 0 {
							t.Skipf("unqualified: required host-page-table capability %s absent", attr)
						}
					}
					home, err := os.UserHomeDir()
					physicalCheck(t, err)
					dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cuptibench-hmm-")
					physicalCheck(t, err)
					physicalClose(t, func() error { return os.RemoveAll(dir) })
					data := make([]byte, syscall.Getpagesize())
					for i := 0; i < physicalCapacity; i++ {
						binary.LittleEndian.PutUint32(data[4*i:], uint32(i))
					}
					path := filepath.Join(dir, "input.bin")
					physicalCheck(t, os.WriteFile(path, data, 0600))
					file, err := os.Open(path)
					physicalCheck(t, err)
					mapped, err = syscall.Mmap(int(file.Fd()), 0, len(data), syscall.PROT_READ, mapFlags)
					closeErr := file.Close()
					physicalCheck(t, err)
					physicalClose(t, func() error { return syscall.Munmap(mapped) })
					physicalCheck(t, closeErr)
					// Touch every mapped page before passing its ordinary virtual address.
					var touch byte
					for i := 0; i < len(mapped); i += syscall.Getpagesize() {
						touch ^= mapped[i]
					}
					runtime.KeepAlive(touch)
					ptr = driver.DevicePtr(uintptr(unsafe.Pointer(&mapped[0])))
				}
				// Registered after mmap cleanup so outstanding GPU work is drained first.
				physicalClose(t, p.synchronize)
				project := func() error {
					return cudalaunch.Kernel(p.project).Grid(p.grid).Block(physicalBlock).On(p.streams[0]).Launch(ptr, p.mid)
				}
				run := project
				var graphs []*cudagraph.Exec
				if strings.HasPrefix(mode, "graph") {
					var launchErr error
					graph, err := cudagraph.Capture(p.streams[0], func() { launchErr = project() })
					if graph != nil {
						physicalClose(t, graph.Destroy)
					}
					physicalCheck(t, err)
					physicalCheck(t, launchErr)
					executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: mode == "graph-kernel"})
					if executable != nil {
						physicalClose(t, executable.Destroy)
					}
					physicalCheck(t, err)
					physicalCheck(t, graph.Destroy())
					run = func() error { return executable.Launch(p.streams[0]) }
					if mode == "graph-kernel" {
						graphs = []*cudagraph.Exec{executable}
					}
				}
				captureMode, replayMode := cuptibench.Kernels, cuptirange.UserReplay
				if mode == "graph-whole" {
					captureMode = cuptibench.Whole
				}
				if strings.HasSuffix(mode, "-kernel") {
					replayMode = cuptirange.KernelReplay
				}
				plan, err := cuptibench.New(p.dev, cuptibench.Config{Mode: captureMode, ReplayMode: replayMode, Operations: 1, Metrics: []string{physicalMetric}, CachePolicy: cuptirange.CachePreserve, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 16, MaxCounterBytes: 32 << 20, MaxPasses: 8}})
				if plan != nil {
					physicalClose(t, plan.Close)
				}
				physicalCheck(t, err)
				w := cuptibench.Workload{View: "full", Graphs: graphs, Streams: p.streams[:1], Prepare: p.prepare, Synchronize: p.synchronize, Preparation: "input=" + mapping + "; reset device state/mid/out only; file mapping prefaulted before capture; cache preserve", Run: func(*cuptibench.Scope) error { return run() }}
				r, err := plan.Collect(w)
				if raw, jsonErr := json.Marshal(r); jsonErr == nil {
					t.Logf("input=%s mapping=%s mode=%s result=%s collect_error=%v", input, mapping, mode, raw, err)
				}
				physicalCheck(t, err)
				if len(r.Ranges) != 1 || physicalValue(t, r.Ranges[0]) != 4 {
					t.Fatalf("expected one range with four CTAs: %+v", r.Ranges)
				}
				if captureMode == cuptibench.Kernels && !strings.Contains(r.Ranges[0].Name, "fixture_project") {
					t.Fatalf("wrong kernel identity: %+v", r.Ranges[0])
				}
				if !r.Quality.Decoded || !r.Quality.AllPassesSubmitted || r.Quality.DroppedRanges != 0 {
					t.Fatalf("quality=%+v", r.Quality)
				}
				physicalCheck(t, p.synchronize())
				got := make([]uint32, physicalCapacity)
				physicalCheck(t, p.mid.CopyToHost(got))
				for i, value := range got {
					want := uint32(0)
					if i < int(p.grid)*physicalBlock {
						want = uint32(i + 3)
					}
					if value != want {
						t.Fatal(fmt.Errorf("mid[%d]=%d want %d", i, value, want))
					}
				}
				for i := 0; i < len(mapped)/4 && i < physicalCapacity; i++ {
					if binary.LittleEndian.Uint32(mapped[4*i:]) != uint32(i) {
						t.Fatalf("read-only input changed at %d", i)
					}
				}
				runtime.KeepAlive(mapped)
			})
		}
	}
	if !matched {
		t.Fatalf("unknown CUPTIBENCH_HMM_CASE %q", selected)
	}

}
