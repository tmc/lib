package cudabpfdev

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// fakeRun records native operations and models queued work, not PTX execution.
// Its synthetic values establish snapshot ordering only.
type fakeRun struct {
	t      *testing.T
	calls  []string
	fail   string
	after  func(string)
	live   map[driver.DevicePtr][]byte
	next   driver.DevicePtr
	queued bool
	ptrs   [3]driver.DevicePtr
}

var errRun = errors.New("injected native failure")

func (f *fakeRun) call(name string) error {
	f.calls = append(f.calls, name)
	if f.after != nil {
		f.after(name)
	}
	if f.fail == name {
		return errRun
	}
	return nil
}

func (f *fakeRun) ops() runOps {
	f.live = make(map[driver.DevicePtr][]byte)
	return runOps{
		run: func(fn func() error) error {
			if err := f.call("enter"); err != nil {
				return err
			}
			return errors.Join(fn(), f.call("restore"))
		},
		load: func([]byte, cudakernel.JITOptions) (driver.Module, error) {
			if err := f.call("load"); err != nil {
				return 0, err
			}
			return 10, nil
		},
		function: func(driver.Module, string) (driver.Function, error) { return 20, f.call("function") },
		resources: func(driver.Function) (compiledResources, error) {
			return compiledResources{registers: 24, maxThreads: 32}, f.call("resources")
		},
		stream: func() (driver.Stream, error) {
			if err := f.call("stream"); err != nil {
				return 0, err
			}
			return 30, nil
		},
		alloc: func(size uint) (driver.DevicePtr, error) {
			f.next++
			if err := f.call(fmt.Sprintf("alloc%d", f.next)); err != nil {
				return 0, err
			}
			f.live[f.next] = make([]byte, size)
			return f.next, nil
		},
		zero: func(ptr driver.DevicePtr, size uint, stream driver.Stream) error {
			if stream != 30 || uint(len(f.live[ptr])) != size {
				f.t.Fatal("wrong zero stream or size")
			}
			f.queued = true
			return f.call(fmt.Sprintf("zero%d", ptr))
		},
		launch: func(fn driver.Function, stream driver.Stream, ptrs [3]driver.DevicePtr, active uint32) error {
			if fn != 20 || stream != 30 || active != 5 {
				f.t.Fatal("wrong launch arguments")
			}
			f.ptrs, f.queued = ptrs, true
			return f.call("launch")
		},
		synchronize: func(stream driver.Stream) error {
			if stream != 30 {
				f.t.Fatal("wrong wait stream")
			}
			if err := f.call("wait"); err != nil {
				return err
			}
			f.queued = false
			if f.ptrs[0] != 0 {
				binary.LittleEndian.PutUint32(f.live[f.ptrs[0]], 17)
				binary.LittleEndian.PutUint32(f.live[f.ptrs[1]], 23)
				binary.LittleEndian.PutUint64(f.live[f.ptrs[2]], 29)
			}
			return nil
		},
		read: func(ptr driver.DevicePtr, dst []byte) error {
			if f.queued {
				f.t.Fatal("read before completion boundary")
			}
			if err := f.call(fmt.Sprintf("read%d", ptr)); err != nil {
				return err
			}
			copy(dst, f.live[ptr])
			return nil
		},
		free: func(ptr driver.DevicePtr) error {
			if f.queued {
				f.t.Fatal("free before completion boundary")
			}
			if _, ok := f.live[ptr]; !ok {
				f.t.Fatal("double free")
			}
			if err := f.call(fmt.Sprintf("free%d", ptr)); err != nil {
				return err
			}
			delete(f.live, ptr)
			return nil
		},
		destroy: func(driver.Stream) error { return f.call("destroy") },
		unload:  func(driver.Module) error { return f.call("unload") },
	}
}

func runFixture(t *testing.T, f *fakeRun) (*Execution, *ArtifactSource) {
	t.Helper()
	s, err := NewArtifactSource(sourceFixture(t), Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	f.t = t
	return &Execution{ops: f.ops(), limits: resourceLimits{registers: 32}}, s
}

func TestExecutionSnapshot(t *testing.T) {
	f := &fakeRun{}
	e, s := runFixture(t, f)
	if err := e.execute(context.Background(), s, 2, 5); err != nil {
		t.Fatal(err)
	}
	snap, ok := e.Snapshot()
	if !ok || !reflect.DeepEqual(snap.Map, []uint32{17, 0}) || snap.Values[0] != 23 || snap.Misses != 29 {
		t.Fatalf("snapshot=%+v ready=%v", snap, ok)
	}
	snap.Map[0] = 100
	snap, _ = e.Snapshot()
	if snap.Map[0] != 17 {
		t.Fatal("mutable snapshot alias")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if len(f.live) != 0 {
		t.Fatal("leaked allocations")
	}
	n := len(f.calls)
	if err := e.Close(); err != nil || len(f.calls) != n {
		t.Fatal("close not idempotent")
	}
	if _, ok := e.Snapshot(); !ok {
		t.Fatal("close lost snapshot")
	}
}

func TestExecutionFailures(t *testing.T) {
	for _, stage := range []string{"enter", "load", "function", "resources", "stream", "alloc1", "zero1", "alloc2", "zero2", "alloc3", "zero3", "launch", "wait", "read1", "read2", "read3", "restore"} {
		t.Run(stage, func(t *testing.T) {
			f := &fakeRun{fail: stage}
			e, s := runFixture(t, f)
			if err := e.execute(context.Background(), s, 2, 5); !errors.Is(err, errRun) {
				t.Fatalf("error=%v", err)
			}
			if snap, ok := e.Snapshot(); ok || snap.Map != nil {
				t.Fatal("published failed execution")
			}
			if stage == "wait" {
				if err := e.Close(); !errors.Is(err, errRun) {
					t.Fatalf("failed wait cleanup: %v", err)
				}
				if len(f.live) != 3 {
					t.Fatal("freed buffers after failed wait")
				}
			}
			f.fail = ""
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if len(f.live) != 0 {
				t.Fatal("leaked failed invocation")
			}
			if _, ok := e.Snapshot(); ok {
				t.Fatal("cleanup invented snapshot")
			}
		})
	}
}

func TestExecutionCleanupRetry(t *testing.T) {
	for _, stage := range []string{"free1", "free2", "free3", "destroy", "unload", "restore"} {
		t.Run(stage, func(t *testing.T) {
			f := &fakeRun{}
			e, s := runFixture(t, f)
			if err := e.execute(context.Background(), s, 2, 5); err != nil {
				t.Fatal(err)
			}
			f.fail = stage
			if err := e.Close(); !errors.Is(err, errRun) {
				t.Fatalf("error=%v", err)
			}
			f.fail = ""
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
			if len(f.live) != 0 {
				t.Fatal("cleanup retry leaked buffers")
			}
		})
	}
}

func TestExecutionCancellation(t *testing.T) {
	for _, stage := range []string{"enter", "zero1", "launch", "wait", "read3", "restore"} {
		t.Run(stage, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			f := &fakeRun{after: func(name string) {
				if name == stage {
					cancel()
				}
			}}
			e, s := runFixture(t, f)
			if err := e.execute(ctx, s, 2, 5); !errors.Is(err, context.Canceled) {
				t.Fatalf("error=%v", err)
			}
			if _, ok := e.Snapshot(); ok {
				t.Fatal("cancelled snapshot published")
			}
			if err := e.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestExecutionConcurrentCloseAndSnapshot(t *testing.T) {
	f := &fakeRun{}
	e, s := runFixture(t, f)
	if err := e.execute(context.Background(), s, 2, 5); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			snap, ready := e.Snapshot()
			if !ready || snap.Map[0] != 17 {
				t.Error("lost snapshot")
			}
			snap.Map[0]++
			if err := e.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if len(f.live) != 0 {
		t.Fatal("leaked allocations")
	}
}
