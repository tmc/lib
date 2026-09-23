package cudabpfdev

import (
	"context"
	"encoding/binary"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type fakeInjectionInvocation struct {
	f        *fakeInjectionLoad
	queued   bool
	launches int
	fn       driver.Function
	packed   []byte
	geometry Geometry
	data     map[driver.DevicePtr][]byte
	value    uint32
}

func invocationFixture(t *testing.T, candidate bool) (*LoadedInjection, *fakeInjectionInvocation) {
	t.Helper()
	i, f := loadedFixture(t)
	if !candidate {
		i.candidate = nil
		i.reason = "injection disabled"
	}
	v := &fakeInjectionInvocation{f: f, value: 17, data: map[driver.DevicePtr][]byte{4096: make([]byte, 16), 8192: make([]byte, 8)}}
	ops := f.ops()
	ops.stream = func() (driver.Stream, error) {
		if err := f.call("stream"); err != nil {
			return 0, err
		}
		return 77, nil
	}
	ops.zero = func(ptr driver.DevicePtr, size uint, stream driver.Stream) error {
		if stream != 77 || uint(len(v.data[ptr])) != size {
			t.Fatal("wrong zero address/size/stream")
		}
		v.queued = true
		name := "zero_map"
		if ptr == 8192 {
			name = "zero_misses"
		}
		if err := f.call(name); err != nil {
			return err
		}
		clear(v.data[ptr])
		return nil
	}
	ops.launch = func(fn driver.Function, g Geometry, stream driver.Stream, schema cudakernel.Schema, packed []byte) error {
		if stream != 77 || len(schema.Params) != 1 || schema.Params[0].Name != "out" {
			t.Fatal("launch ABI or stream changed")
		}
		v.launches++
		v.fn = fn
		v.geometry = g
		v.packed = slices.Clone(packed)
		v.queued = true
		return f.call("launch")
	}
	ops.synchronize = func(stream driver.Stream) error {
		if stream != 77 {
			t.Fatal("wrong stream synchronized")
		}
		if err := f.call("wait"); err != nil {
			return err
		}
		v.queued = false
		if v.launches > 0 && v.fn == 2 {
			binary.LittleEndian.PutUint32(v.data[4096], v.value)
			binary.LittleEndian.PutUint64(v.data[8192], 2)
		}
		return nil
	}
	ops.read = func(ptr driver.DevicePtr, dst []byte) error {
		if v.queued {
			t.Fatal("map read before completion")
		}
		name := "read_map"
		if ptr == 8192 {
			name = "read_misses"
		}
		if err := f.call(name); err != nil {
			return err
		}
		copy(dst, v.data[ptr])
		return nil
	}
	ops.destroy = func(driver.Stream) error {
		if v.queued {
			t.Fatal("destroy before completion")
		}
		return f.call("destroy")
	}
	unload := ops.unload
	ops.unload = func(m driver.Module) error {
		if v.queued {
			t.Fatal("module unloaded with work pending")
		}
		return unload(m)
	}
	l, err := i.load(ops)
	if err != nil {
		t.Fatal(err)
	}
	return l, v
}

func TestInjectionLaunchAndSnapshot(t *testing.T) {
	for _, candidate := range []bool{false, true} {
		l, v := invocationFixture(t, candidate)
		if _, ok := l.Snapshot(); ok {
			t.Fatal("snapshot before launch")
		}
		if _, err := l.Record(); err == nil {
			t.Fatal("record before launch")
		}
		if err := l.Launch(context.Background(), uint64(0x1000)); err != nil {
			t.Fatal(err)
		}
		wantFn := driver.Function(1)
		if candidate {
			wantFn = 2
		}
		if v.launches != 1 || v.fn != wantFn || binary.LittleEndian.Uint64(v.packed) != 0x1000 || v.geometry != l.geometry {
			t.Fatal("selected launch or original ABI changed")
		}
		s, ok := l.Snapshot()
		if ok != candidate {
			t.Fatalf("snapshot available=%v candidate=%v", ok, candidate)
		}
		if candidate {
			if s.Values[0] != 17 || s.Misses != 2 {
				t.Fatalf("snapshot=%+v", s)
			}
			s.Values[0]++
			again, _ := l.Snapshot()
			if again.Values[0] != 17 {
				t.Fatal("snapshot aliases internal storage")
			}
		} else {
			for _, op := range []string{"zero_map", "zero_misses", "read_map", "read_misses"} {
				if slices.Contains(v.f.calls, op) {
					t.Fatalf("fallback touched instrumentation: %s", op)
				}
			}
		}
		r, err := l.Record()
		if err != nil || r.DurationNs != 0 || r.Labels["duration_status"] != "unavailable" {
			t.Fatalf("record=%+v error=%v", r, err)
		}
		if err := l.Launch(context.Background(), uint64(0x1000)); err == nil || v.launches != 1 {
			t.Fatal("owner allowed second invocation")
		}
		if err := l.Close(); err != nil {
			t.Fatal(err)
		}
		if _, err := l.Record(); err != nil {
			t.Fatal("close lost boundary record")
		}
		if err := l.Launch(context.Background(), uint64(0x1000)); err == nil {
			t.Fatal("launch after close")
		}
	}
}

func TestInjectionLaunchFailures(t *testing.T) {
	for _, stage := range []string{"enter", "stream", "zero_map", "zero_misses", "launch", "wait", "read_map", "read_misses", "restore"} {
		t.Run(stage, func(t *testing.T) {
			l, v := invocationFixture(t, true)
			v.f.fail = stage
			if err := l.Launch(context.Background(), uint64(0x1000)); !errors.Is(err, errRun) {
				t.Fatalf("error=%v", err)
			}
			if _, ok := l.Snapshot(); ok {
				t.Fatal("failed invocation published snapshot")
			}
			if _, err := l.Record(); err == nil {
				t.Fatal("failed invocation published success record")
			}
			if v.launches > 1 || v.fn == 1 {
				t.Fatal("failure retried original")
			}
			if stage == "wait" {
				if err := l.Close(); !errors.Is(err, errRun) {
					t.Fatal("cleanup ignored failed wait")
				}
				if len(v.f.live) != 2 {
					t.Fatal("failed wait released modules")
				}
			}
			v.f.fail = ""
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
			if len(v.f.live) != 0 {
				t.Fatal("failed invocation leaked modules")
			}
		})
	}
}

func TestInjectionLaunchCancellation(t *testing.T) {
	for _, stage := range []string{"enter", "zero_map", "launch", "wait", "read_misses", "restore"} {
		t.Run(stage, func(t *testing.T) {
			l, v := invocationFixture(t, true)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			v.f.after = func(name string) {
				if name == stage {
					cancel()
				}
			}
			if err := l.Launch(ctx, uint64(0x1000)); !errors.Is(err, context.Canceled) {
				t.Fatalf("error=%v", err)
			}
			if _, ok := l.Snapshot(); ok {
				t.Fatal("cancelled snapshot")
			}
			if err := l.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestInjectionInvalidArgumentsDoNotConsume(t *testing.T) {
	l, v := invocationFixture(t, true)
	if err := l.Launch(context.Background(), new(int)); err == nil {
		t.Fatal("accepted Go pointer")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := l.Launch(ctx, uint64(0x1000)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := l.Launch(context.Background(), uint64(0x1000)); err != nil {
		t.Fatal(err)
	}
	if v.launches != 1 {
		t.Fatal("invalid arguments caused a launch")
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestConcurrentInjectionLaunchIsSingleUse(t *testing.T) {
	l, v := invocationFixture(t, true)
	var successes atomic.Int32
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if l.Launch(context.Background(), uint64(0x1000)) == nil {
				successes.Add(1)
			}
		})
	}
	wg.Wait()
	if successes.Load() != 1 || v.launches != 1 {
		t.Fatal("concurrent invocation duplicated")
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestIndependentInjectionSnapshots(t *testing.T) {
	a, av := invocationFixture(t, true)
	b, bv := invocationFixture(t, true)
	av.value, bv.value = 101, 202
	var wg sync.WaitGroup
	for _, l := range []*LoadedInjection{a, b} {
		wg.Go(func() {
			if err := l.Launch(context.Background(), uint64(0x1000)); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	as, _ := a.Snapshot()
	bs, _ := b.Snapshot()
	if as.Values[0] != 101 || bs.Values[0] != 202 {
		t.Fatal("invocations shared snapshots")
	}
	as.Values[0]++
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}
