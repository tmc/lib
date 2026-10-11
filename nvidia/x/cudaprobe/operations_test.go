package cudaprobe

import (
	"encoding/binary"
	"fmt"
	"testing"
	"time"
)

func TestCopyDecode(t *testing.T) {
	for _, tt := range []struct {
		symbol, direction string
		async             bool
	}{
		{"cudaMemcpy", "runtime_kind", false}, {"cudaMemcpyAsync", "runtime_kind", true},
		{"cuMemcpyHtoD_v2", "host_to_device", false}, {"cuMemcpyDtoH_v2", "device_to_host", false}, {"cuMemcpyDtoD_v2", "device_to_device", false},
		{"cuMemcpyHtoDAsync_v2", "host_to_device", true}, {"cuMemcpyDtoHAsync_v2", "device_to_host", true}, {"cuMemcpyDtoDAsync_v2", "device_to_device", true},
	} {
		t.Run(tt.symbol, func(t *testing.T) {
			c := capture(tt.symbol, 1, 0xfedcba9876543210, 0x123456789abcdef0, 1<<54, 4, 9)
			e, err := Decode(c)
			if err != nil || !e.Decoded || e.Copy == nil {
				t.Fatalf("event=%+v err=%v", e, err)
			}
			want := Copy{Destination: 0xfedcba9876543210, Source: 0x123456789abcdef0, Bytes: 1 << 54, Direction: tt.direction, Async: tt.async}
			if tt.direction == "runtime_kind" {
				want.Kind = 4
				if tt.async {
					want.Stream = 9
				}
			} else if tt.async {
				want.Stream = 4
			}
			if *e.Copy != want {
				t.Fatalf("copy=%+v want=%+v", *e.Copy, want)
			}
			binary.LittleEndian.PutUint16(c.Data[2:], 1)
			e, err = Decode(c)
			if err != nil || !e.Decoded || e.ReturnKnown {
				t.Fatalf("entry=%+v err=%v", e, err)
			}
			r, err := e.Record(time.Unix(1, 0))
			if err != nil || r.Labels["duration_status"] != "unavailable" || r.Labels["probe.copy.source"] != "1311768467463790320" {
				t.Fatalf("record=%+v err=%v", r, err)
			}
		})
	}
}

func TestSynchronizationAndGraphDecode(t *testing.T) {
	for _, tt := range []struct {
		symbol, scope string
		stream        uint64
	}{
		{"cudaDeviceSynchronize", "device", 0}, {"cuCtxSynchronize", "context", 0}, {"cudaStreamSynchronize", "stream", 123}, {"cuStreamSynchronize", "stream", 123},
	} {
		t.Run(tt.symbol, func(t *testing.T) {
			c := capture(tt.symbol, 1, 123)
			binary.LittleEndian.PutUint32(c.Data[24:], 17)
			e, err := Decode(c)
			if err != nil || !e.Decoded || e.Sync == nil || *e.Sync != (Sync{Scope: tt.scope, Stream: tt.stream}) || e.Status != 17 {
				t.Fatalf("event=%+v err=%v", e, err)
			}
		})
	}
	for _, symbol := range []string{"cudaGraphLaunch", "cuGraphLaunch"} {
		e, err := Decode(capture(symbol, 1, 0xdeadbeef, 7))
		if err != nil || !e.Decoded || e.Graph == nil || *e.Graph != (Graph{Exec: 0xdeadbeef, Stream: 7}) || e.Launch != nil {
			t.Fatalf("event=%+v err=%v", e, err)
		}
	}
}

func TestUnsupportedSignatures(t *testing.T) {
	for _, symbol := range []string{"cuMemcpyHtoD", "cuMemcpy2D_v2", "cudaMemcpy2DAsync", "cuMemcpyPeerAsync", "cudaLaunchKernelExC", "cuLaunchKernel_ptsz", "cudaMemcpyAsync_ptsz", "cuGraphLaunch_ptsz"} {
		e, err := Decode(capture(symbol, 1, 1, 2, 3))
		if err != nil || e.Decoded || e.Reason != "unsupported CUDA symbol" {
			t.Fatalf("%s event=%+v err=%v", symbol, e, err)
		}
	}
}

func ExampleCopy() {
	e := Event{Decoded: true, Copy: &Copy{Bytes: 4096, Direction: "host_to_device", Async: true}}
	fmt.Println(e.Copy.Bytes, e.Copy.Direction, e.Copy.Async)
	// Output: 4096 host_to_device true
}

func ExampleSync() {
	e := Event{Decoded: true, Sync: &Sync{Scope: "stream", Stream: 7}}
	fmt.Println(e.Sync.Scope, e.Sync.Stream)
	// Output: stream 7
}

func ExampleGraph() {
	e := Event{Decoded: true, Graph: &Graph{Exec: 9, Stream: 7}}
	fmt.Println(e.Graph.Exec, e.Graph.Stream)
	// Output: 9 7
}

func TestAsyncMemoryDoesNotChangeTrackedLifetime(t *testing.T) {
	for _, symbol := range []string{"cudaMallocAsync", "cuMemAllocAsync", "cudaFreeAsync", "cuMemFreeAsync"} {
		t.Run(symbol, func(t *testing.T) {
			tr, _ := NewTracker(Process{PID: 7, Generation: 99}, 2)
			alloc := capture("cudaMalloc", 1, 1, 128)
			alloc.Identity.AddressSpace = 1
			binary.LittleEndian.PutUint64(alloc.Data[120:], 0x8000)
			e, _ := Decode(alloc)
			if err := tr.Observe(e); err != nil {
				t.Fatal(err)
			}
			c := capture(symbol, 2, 0x8000, 256, 9)
			c.Identity.AddressSpace = 1
			binary.LittleEndian.PutUint64(c.Data[120:], 0x9000)
			e, err := Decode(c)
			if err != nil || !e.Decoded || e.AsyncMemory == nil {
				t.Fatalf("%+v %v", e, err)
			}
			free := symbol == "cudaFreeAsync" || symbol == "cuMemFreeAsync"
			want := AsyncMemory{Address: 0x9000, Bytes: 256, Stream: 9}
			if free {
				want = AsyncMemory{Address: 0x8000, Stream: 256, Free: true}
			}
			if *e.AsyncMemory != want {
				t.Fatalf("got %+v want %+v", e.AsyncMemory, want)
			}
			if err := tr.Observe(e); err != nil {
				t.Fatal(err)
			}
			active, incomplete := tr.Report()
			if len(active) != 1 || active[0].Address != 0x8000 || !incomplete {
				t.Fatalf("async changed lifetime: %+v incomplete=%v", active, incomplete)
			}
			if !free {
				binary.LittleEndian.PutUint16(c.Data[2:], 3)
				e, _ = Decode(c)
				if e.Decoded || e.Reason != "allocation output unavailable" {
					t.Fatalf("accepted missing output %+v", e)
				}
				binary.LittleEndian.PutUint32(c.Data[24:], 1)
				e, _ = Decode(c)
				if !e.Decoded || e.AsyncMemory.Address != 0 {
					t.Fatalf("failed output admitted %+v", e)
				}
			}
		})
	}
}

func TestControlHandles(t *testing.T) {
	for _, tt := range []struct {
		symbol string
		want   Control
		output bool
	}{
		{"cuCtxCreate_v2", Control{Scope: "context", Operation: "create", Handle: 0x999, Flags: 3, Device: 2}, true},
		{"cuCtxGetCurrent", Control{Scope: "context", Operation: "get_current", Handle: 0x999}, true},
		{"cuStreamCreate", Control{Scope: "stream", Operation: "create", Handle: 0x999, Flags: 3}, true},
		{"cudaStreamCreate", Control{Scope: "stream", Operation: "create", Handle: 0x999}, true},
		{"cuCtxSetCurrent", Control{Scope: "context", Operation: "set_current", Handle: 123}, false},
		{"cuCtxDestroy_v2", Control{Scope: "context", Operation: "destroy", Handle: 123}, false},
		{"cuStreamDestroy_v2", Control{Scope: "stream", Operation: "destroy", Handle: 123}, false},
		{"cudaStreamDestroy", Control{Scope: "stream", Operation: "destroy", Handle: 123}, false},
	} {
		t.Run(tt.symbol, func(t *testing.T) {
			c := capture(tt.symbol, 1, 123, 3, 2)
			binary.LittleEndian.PutUint64(c.Data[120:], 0x999)
			e, err := Decode(c)
			if err != nil || !e.Decoded || e.Control == nil || *e.Control != tt.want || e.Identity.AddressSpace != 0 {
				t.Fatalf("event=%+v control=%+v err=%v", e, e.Control, err)
			}
			if tt.output {
				binary.LittleEndian.PutUint16(c.Data[2:], 3)
				e, _ = Decode(c)
				if e.Decoded || e.Reason != "control output unavailable" {
					t.Fatalf("output bypass %+v", e)
				}
				binary.LittleEndian.PutUint32(c.Data[24:], 1)
				e, _ = Decode(c)
				if !e.Decoded || e.Control.Handle != 0 {
					t.Fatalf("failed output admitted %+v", e)
				}
			}
		})
	}
}

func ExampleAsyncMemory() {
	m := AsyncMemory{Address: 4096, Stream: 7, Free: true}
	fmt.Println(m.Address, m.Stream, m.Free)
	// Output: 4096 7 true
}

func ExampleControl() {
	c := Control{Scope: "context", Operation: "set_current", Handle: 7}
	fmt.Println(c.Scope, c.Operation, c.Handle)
	// Output: context set_current 7
}
