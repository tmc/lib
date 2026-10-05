package cudahotswap

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

func TestServiceGraphLaunchPinsAndSelection(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	exec, node, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	var installed driver.Function
	var launches, waits int
	s.graphs.ops.patch = func(_ *cudagraph.Exec, _ *cudagraph.Node, fn driver.Function, _ driver.Dim3, _ []any) error {
		installed = fn
		return nil
	}
	s.graphs.ops.launch = func(e *cudagraph.Exec, got *cuda.Stream) error {
		if e != exec || got != stream || len(s.graphWork) != 1 {
			t.Fatal("graph work or owning stream not retained before enqueue")
		}
		launches++
		return nil
	}
	s.graphs.ops.wait = func(_ context.Context, got *cuda.Stream) error {
		if got != stream {
			t.Fatal("wait does not use owning stream")
		}
		waits++
		return nil
	}
	state, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, driver.Dims(1, 1, 1))
	if err != nil || state.Version != 1 || installed != 1 || (*leases)[1].calls != 1 {
		t.Fatalf("register: %+v, %v, function=%d", state, err, installed)
	}
	p, err := s.LaunchGraph(context.Background(), exec)
	if err != nil || p == nil || len(s.graphWork) != 1 || (*leases)[2].calls != 0 {
		t.Fatalf("launch: %v, %v", p, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Wait(ctx); !errors.Is(err, context.Canceled) || len(s.graphWork) != 1 || waits != 0 || k.calls != 0 {
		t.Fatal("canceled observation completes work or unloads its module")
	}
	if q, err := s.LaunchGraph(context.Background(), exec); q != nil || err == nil || len(*leases) != 3 {
		t.Fatal("duplicate pending graph acquires another lease or enqueues")
	}
	if err := p.Wait(context.Background()); err != nil || waits != 1 || (*leases)[2].calls != 1 || len(s.graphWork) != 0 {
		t.Fatalf("wait: %v", err)
	}
	if err := p.Wait(context.Background()); err != nil || waits != 1 {
		t.Fatal("completed graph waited twice")
	}
	g, second := addSelectionVersion(t, s) // Disclosed direct test publication.
	if _, err := s.Select(context.Background(), "kernel", g.version, "graph B"); err != nil || installed != 2 {
		t.Fatalf("selection patch: %v, installed=%d", err, installed)
	}
	p, err = s.LaunchGraph(context.Background(), exec)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Wait(context.Background()); err != nil || launches != 2 || waits != 2 {
		t.Fatal("new selected graph not launched and drained")
	}
	if err := s.Close(context.Background()); err != nil || k.calls != 1 || second.calls != 1 {
		t.Fatalf("close graph before modules: %v, %d/%d", err, k.calls, second.calls)
	}
}

func TestServiceGraphPartialSubmissionAndCleanupRetry(t *testing.T) {
	s, leases := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	exec, node, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return nil }
	if _, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, driver.Dims(1, 1, 1)); err != nil {
		t.Fatal(err)
	}
	launchErr := errors.New("possibly submitted graph")
	s.graphs.ops.launch = func(*cudagraph.Exec, *cuda.Stream) error { return launchErr }
	waitErr := errors.New("completion not established")
	waits := 0
	s.graphs.ops.wait = func(context.Context, *cuda.Stream) error { waits++; return waitErr }
	p, err := s.LaunchGraph(context.Background(), exec)
	if p == nil || !errors.Is(err, launchErr) || len(s.graphWork) != 1 {
		t.Fatal("partial graph owner lost")
	}
	if err := p.Wait(context.Background()); !errors.Is(err, waitErr) || (*leases)[2].calls != 0 {
		t.Fatal("failed completion releases lease")
	}
	waitErr = nil
	(*leases)[2].fail = true
	if err := p.Wait(context.Background()); err == nil || waits != 2 || p.pending != nil || len(s.graphWork) != 1 {
		t.Fatal("failed lease release lost retry owner")
	}
	(*leases)[2].fail = false
	if err := p.Wait(context.Background()); err != nil || waits != 2 || len(s.graphWork) != 0 || (*leases)[2].calls != 2 {
		t.Fatal("cleanup retry repeats synchronization or loses lease")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceGraphShutdownDrainsBeforeUnloading(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, _ := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	exec, node, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error { return nil }
	s.graphs.ops.launch = func(*cudagraph.Exec, *cuda.Stream) error { return nil }
	entered, resume := make(chan struct{}), make(chan struct{})
	var waits atomic.Int64
	s.graphs.ops.wait = func(context.Context, *cuda.Stream) error {
		waits.Add(1)
		close(entered)
		<-resume
		return nil
	}
	if _, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, driver.Dims(1, 1, 1)); err != nil {
		t.Fatal(err)
	}
	p, err := s.LaunchGraph(context.Background(), exec)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- s.Close(context.Background()) }()
	select {
	case <-entered:
	case err := <-done:
		close(resume)
		t.Fatalf("shutdown returned before draining graph: %v", err)
	case <-time.After(2 * time.Second):
		close(resume)
		t.Fatal("shutdown did not begin draining graph")
	}
	if k.calls != 0 {
		close(resume)
		t.Fatal("shutdown unloads graph module before completion")
	}
	close(resume)
	select {
	case err := <-done:
		if err != nil || k.calls != 1 || waits.Load() != 1 {
			t.Fatalf("shutdown: %v, closes=%d waits=%d", err, k.calls, waits.Load())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("shutdown did not finish after graph completion")
	}
	if err := p.Wait(context.Background()); err != nil || waits.Load() != 1 {
		t.Fatal("old pending handle repeats shutdown's synchronization")
	}
}

func TestServiceGraphRegistrationPartialPatchAndRefusals(t *testing.T) {
	k := &serviceKernel{raw: 1}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	exec, node, stream := new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream)
	grid := driver.Dims(1, 1, 1)
	if _, err := s.RegisterGraph(nil, "kernel", exec, node, stream, grid); err == nil {
		t.Fatal("nil context admitted")
	}
	if _, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, driver.Dim3{}); err == nil {
		t.Fatal("invalid grid admitted")
	}
	if _, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, grid, uint32(1)); err == nil {
		t.Fatal("ABI argument count mismatch admitted")
	}
	if p, err := s.LaunchGraph(context.Background(), exec); p != nil || err == nil || len(*leases) != 1 {
		t.Fatal("unbound graph takes lease or launches")
	}
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error {
		return errors.New("possibly patched node")
	}
	state, err := s.RegisterGraph(context.Background(), "kernel", exec, node, stream, grid)
	if state.Name != "kernel" || err == nil || s.graphs.bindings[0].pin.gen.pins.Load() != 1 {
		t.Fatal("partial registration pin lost or adoption concealed")
	}
	if p, err := s.LaunchGraph(context.Background(), exec); p != nil || err == nil || len(*leases) != 2 {
		t.Fatal("failed binding is launchable")
	}
	if err := s.Close(context.Background()); err != nil || k.calls != 1 {
		t.Fatalf("partial registration cleanup: %v, %d", err, k.calls)
	}
	for _, p := range []*GraphLaunch{nil, new(GraphLaunch)} {
		if p.Wait(context.Background()) != nil || p.Wait(nil) == nil {
			t.Fatal("zero pending graph contract")
		}
	}
}

func TestServiceGraphRegistrationRefusedDuringApplicationCheck(t *testing.T) {
	l := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, leases := testService(t, l)
	registerService(t, s)
	s.graphs.ops.patch = func(*cudagraph.Exec, *cudagraph.Node, driver.Function, driver.Dim3, []any) error {
		t.Fatal("callback graph registration changed bindings")
		return nil
	}
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(ctx context.Context, _ State) error {
		before := len(*leases)
		state, err := s.RegisterGraph(ctx, "kernel", new(cudagraph.Exec), new(cudagraph.Node), new(cuda.Stream), driver.Dims(1, 1, 1))
		if err == nil || state.Generation != 0 || len(*leases) != before || len(s.graphs.bindings) != 0 {
			t.Fatal("graph mutation was admitted during active application check")
		}
		return nil
	}}, "check"); err != nil {
		t.Fatal(err)
	}
	l.kernel = &serviceKernel{raw: 2}
	if _, err := s.Push(context.Background(), "kernel", Source{Code: "candidate", Entry: "entry"}, "graph mutation control"); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
