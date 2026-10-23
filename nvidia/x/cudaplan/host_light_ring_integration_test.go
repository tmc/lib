package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamegacpu"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

const (
	// hostLightElements is the fixed extent every fixture task declares.
	hostLightElements = 4
	// hostLightGeneration is the ring generation; collective fixtures
	// derive their communicator generation from it.
	hostLightGeneration = 7
)

// hostLightAdmission is synthetic CPU worker geometry: one CTA of 32 OS
// threads. It is not a measured SM count or device residency.
var hostLightAdmission = cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 32, SharedBytes: 1}

// hostLightChain is the three-task chain (x+token)*3+5.
func hostLightChain(token int32) ([]cudataskgraph.Task, map[string]cudamega.Operation) {
	tasks := []cudataskgraph.Task{
		{ID: "token", Rank: 0, Scope: cudataskgraph.ScopeDevice, Extent: hostLightElements},
		{ID: "scale", Deps: []string{"token"}, Inputs: []string{"token"}, Rank: 1, Scope: cudataskgraph.ScopeDevice, Extent: hostLightElements},
		{ID: "bias", Deps: []string{"scale"}, Inputs: []string{"scale"}, Rank: 2, Scope: cudataskgraph.ScopeDevice, Extent: hostLightElements},
	}
	ops := map[string]cudamega.Operation{
		"token": {Kind: cudamega.Add, Scalar: uint32(token)},
		"scale": {Kind: cudamega.Multiply, Scalar: 3},
		"bias":  {Kind: cudamega.Add, Scalar: 5},
	}
	return tasks, ops
}

// hostLightRun builds a finite graph and program and runs it once on the
// cudamegacpu backend. This is finite CPU execution of the program, not a
// CUDA launch, persistent GPU worker or model decode step. It returns each
// task's output keyed by task ID, after checking that the backend's scratch
// agrees with cudamegacpu.Reference.
func hostLightRun(tasks []cudataskgraph.Task, ops map[string]cudamega.Operation, input []uint32) (map[string][]uint32, error) {
	graph, err := cudataskgraph.New(tasks)
	if err != nil {
		return nil, fmt.Errorf("graph: %w", err)
	}
	grid, err := cudamega.CTAGrid(hostLightAdmission, graph)
	if err != nil {
		return nil, fmt.Errorf("grid: %w", err)
	}
	grid.MaxElements = hostLightElements
	p, err := cudamega.NewProgram(graph, ops, grid, cudamega.SM80)
	if err != nil {
		return nil, fmt.Errorf("program: %w", err)
	}
	if _, err := p.Invocation(uint64(len(input))); err != nil {
		return nil, fmt.Errorf("invocation: %w", err)
	}
	exec, err := cudamegacpu.New(hostLightAdmission)
	if err != nil {
		return nil, err
	}
	result, err := exec.Run(p, input, 1, 0)
	if err != nil {
		return nil, err
	}
	if result.Abort != cudamega.AbortNone {
		return nil, fmt.Errorf("finite run aborted with code %d", result.Abort)
	}
	ref, err := cudamegacpu.Reference(p, input)
	if err != nil {
		return nil, err
	}
	if !reflect.DeepEqual(result.Scratch, ref) {
		return nil, fmt.Errorf("backend scratch %v disagrees with reference %v", result.Scratch, ref)
	}
	n := len(input)
	out := make(map[string][]uint32)
	for i, s := range p.Steps() {
		out[s.Task.ID] = append([]uint32(nil), result.Scratch[i*n:(i+1)*n]...)
	}
	return out, nil
}

// hostLightWant computes the chain directly, sharing no code with cudamega.
func hostLightWant(input []uint32, token int32) map[string][]uint32 {
	want := map[string][]uint32{}
	for _, x := range input {
		t := x + uint32(token)
		want["token"] = append(want["token"], t)
		want["scale"] = append(want["scale"], t*3)
		want["bias"] = append(want["bias"], t*3+5)
	}
	return want
}

func hostLightConsumer(t *testing.T, capacity uint64, inputs ...cudakv.HostCommandInput) (*cudakv.DeviceCommandRing, *cudakv.HostCommandConsumer) {
	t.Helper()
	ring, err := cudakv.NewDeviceCommandRing(capacity, hostLightGeneration)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{
		Mode:             "host-finite",
		Generation:       hostLightGeneration,
		VocabSize:        16,
		MaxEntries:       4,
		MaxRegistryWords: 16,
		MaxBatch:         4,
		MaxBatchWords:    16,
	}, inputs)
	if err != nil {
		t.Fatal(err)
	}
	return ring, c
}

// TestHostLightRingFiniteCPU drives the public ring and bounded consumer into
// the finite CPU program for each command and checks every task's output
// against direct arithmetic.
func TestHostLightRingFiniteCPU(t *testing.T) {
	a := []uint32{1, 4, 9, 12}
	b := []uint32{100, 200, 300, 400}
	ring, c := hostLightConsumer(t, 4, cudakv.HostCommandInput{SeqID: 1, Words: a}, cudakv.HostCommandInput{SeqID: 2, Words: b})
	a[0] = 999 // the consumer registry holds a copy
	type call struct {
		seq   uint64
		token int32
		out   map[string][]uint32
	}
	var calls []call
	run := func(_ context.Context, d cudakv.Descriptor, input []uint32) error {
		tasks, ops := hostLightChain(d.PromptToken)
		out, err := hostLightRun(tasks, ops, input)
		if err != nil {
			return err
		}
		calls = append(calls, call{d.SeqID, d.PromptToken, out})
		return nil
	}
	for _, d := range []cudakv.Descriptor{{SeqID: 1, PromptToken: 2}, {SeqID: 2, PromptToken: 5}, {SeqID: 1, PromptToken: 7}} {
		d.Generation = hostLightGeneration
		if err := ring.Publish(d); err != nil {
			t.Fatal(err)
		}
	}
	batch, err := c.ConsumeBatch(context.Background(), 2, run)
	if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 2, Completed: 2, Words: 8}) || ring.Len() != 1 {
		t.Fatalf("first batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	batch, err = c.ConsumeBatch(context.Background(), 2, run)
	if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) || ring.Len() != 0 {
		t.Fatalf("second batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	inputs := map[uint64][]uint32{1: {1, 4, 9, 12}, 2: b}
	want := []struct {
		seq   uint64
		token int32
	}{{1, 2}, {2, 5}, {1, 7}}
	if len(calls) != len(want) {
		t.Fatalf("%d commands ran, want %d", len(calls), len(want))
	}
	for i, w := range want {
		got := calls[i]
		if got.seq != w.seq || got.token != w.token {
			t.Fatalf("command %d = sequence %d token %d, want sequence %d token %d", i, got.seq, got.token, w.seq, w.token)
		}
		if wantOut := hostLightWant(inputs[w.seq], w.token); !reflect.DeepEqual(got.out, wantOut) {
			t.Fatalf("command %d output %v, want direct arithmetic %v", i, got.out, wantOut)
		}
	}
}

// TestHostLightRingFiniteCPURefuses checks that malformed graphs, operations
// and extents are refused by the public cudataskgraph and cudamega APIs, and
// that a refusal reaches the ring consumer's caller with later commands queued.
func TestHostLightRingFiniteCPURefuses(t *testing.T) {
	tests := []struct {
		name   string
		change func([]cudataskgraph.Task, map[string]cudamega.Operation) []cudataskgraph.Task
		words  int
		want   string
	}{
		{
			name: "missing dependency",
			change: func(tasks []cudataskgraph.Task, _ map[string]cudamega.Operation) []cudataskgraph.Task {
				tasks[2].Deps, tasks[2].Inputs = []string{"absent"}, []string{"absent"}
				return tasks
			},
			words: hostLightElements,
			want:  "graph:",
		},
		{
			name: "skipped task operation",
			change: func(tasks []cudataskgraph.Task, ops map[string]cudamega.Operation) []cudataskgraph.Task {
				delete(ops, "scale")
				return tasks
			},
			words: hostLightElements,
			want:  "program:",
		},
		{
			name:  "short input",
			words: hostLightElements - 1,
			want:  "invocation:",
		},
		{
			name:  "input beyond MaxElements",
			words: hostLightElements + 1,
			want:  "invocation:",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]uint32, tt.words)
			ring, c := hostLightConsumer(t, 2, cudakv.HostCommandInput{SeqID: 1, Words: input}, cudakv.HostCommandInput{SeqID: 2, Words: make([]uint32, hostLightElements)})
			for _, seq := range []uint64{1, 2} {
				if err := ring.Publish(cudakv.Descriptor{SeqID: seq, Generation: hostLightGeneration}); err != nil {
					t.Fatal(err)
				}
			}
			var runErr error
			batch, err := c.ConsumeBatch(context.Background(), 2, func(_ context.Context, d cudakv.Descriptor, input []uint32) error {
				tasks, ops := hostLightChain(d.PromptToken)
				if tt.change != nil {
					tasks = tt.change(tasks, ops)
				}
				_, runErr = hostLightRun(tasks, ops, input)
				return runErr
			})
			if runErr == nil || !errors.Is(err, runErr) || !strings.HasPrefix(runErr.Error(), tt.want) {
				t.Fatalf("batch error %v (run %v), want %q refusal", err, runErr, tt.want)
			}
			if batch != (cudakv.HostCommandBatch{Consumed: 1, Words: tt.words}) || ring.Len() != 1 {
				t.Fatalf("batch = %+v, ring %d; want one consumed, none completed, one queued", batch, ring.Len())
			}
		})
	}
}

var hostLightUUID = regexp.MustCompile(`^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// hostLightNativeAdmission checks the explicit two-device request. It makes
// no vendor or device call. It returns the two physical GPU UUIDs, in rank
// order.
func hostLightNativeAdmission(env func(string) string) ([]string, error) {
	if env == nil || env("LIB_HOST_LIGHT_NATIVE") != "1" {
		return nil, errors.New("LIB_HOST_LIGHT_NATIVE=1 required")
	}
	for _, key := range []string{"GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK", "CUDATEST_FORCE_NO_LOCK"} {
		if env(key) != "" {
			return nil, fmt.Errorf("lock bypass %s refused", key)
		}
	}
	ids := strings.Split(env("LIB_HOST_LIGHT_NATIVE_UUIDS"), ",")
	if len(ids) != 2 || ids[0] == ids[1] || !hostLightUUID.MatchString(ids[0]) || !hostLightUUID.MatchString(ids[1]) {
		return nil, errors.New("LIB_HOST_LIGHT_NATIVE_UUIDS must name two distinct physical GPU UUIDs")
	}
	if _, err := cudalock.Inherited(env); err != nil {
		return nil, err
	}
	return ids, nil
}

func TestHostLightNativeAdmission(t *testing.T) {
	dir := t.TempDir()
	lockdir := filepath.Join(dir, "lock")
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithDir(lockdir), cudalock.WithLabel("synthetic host admission control"), cudalock.WithNoReclaim())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	base := map[string]string{
		"LIB_HOST_LIGHT_NATIVE":       "1",
		"LIB_HOST_LIGHT_NATIVE_UUIDS": "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002",
		"GPU_LOCKDIR":                 lockdir,
		"GPU_LOCK_HELD":               strconv.Itoa(os.Getpid()),
	}
	tests := []struct {
		name  string
		key   string
		value string
		ok    bool
		lock  bool // want cudalock.ErrNotInherited
	}{
		{name: "admitted host control", ok: true},
		{name: "no opt-in", key: "LIB_HOST_LIGHT_NATIVE", value: ""},
		{name: "GPU_LOCK_DISABLED", key: "GPU_LOCK_DISABLED", value: "1"},
		{name: "CUDATEST_NO_LOCK", key: "CUDATEST_NO_LOCK", value: "1"},
		{name: "CUDATEST_FORCE_NO_LOCK", key: "CUDATEST_FORCE_NO_LOCK", value: "1"},
		{name: "one device", key: "LIB_HOST_LIGHT_NATIVE_UUIDS", value: "GPU-00000000-0000-0000-0000-000000000001"},
		{name: "same device twice", key: "LIB_HOST_LIGHT_NATIVE_UUIDS", value: "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000001"},
		{name: "ordinal not UUID", key: "LIB_HOST_LIGHT_NATIVE_UUIDS", value: "0,1"},
		{name: "presence not pid", key: "GPU_LOCK_HELD", value: "1", lock: true},
		{name: "no lock", key: "GPU_LOCK_HELD", value: "", lock: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := func(k string) string {
				if k == tt.key {
					return tt.value
				}
				return base[k]
			}
			ids, err := hostLightNativeAdmission(env)
			if tt.ok {
				if err != nil || len(ids) != 2 {
					t.Fatalf("admission = %v, %v", ids, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("admitted %v", ids)
			}
			if got := errors.Is(err, cudalock.ErrNotInherited); got != tt.lock {
				t.Fatalf("errors.Is(%v, cudalock.ErrNotInherited) = %v, want %v", err, got, tt.lock)
			}
		})
	}
	if _, err := hostLightNativeAdmission(nil); err == nil {
		t.Fatal("nil environment admitted")
	}
}

// errHostLightBinding reports a consumed command whose registered collective
// binding does not match the communicator or step it would drive.
var errHostLightBinding = errors.New("collective binding mismatch")

// hostLightBinding is the collective identity registered for one command:
// the communicator generation, rank and world size, and the step's element
// count and data type. Generation is the ring's, so a command cannot drive a
// communicator from another lifetime.
type hostLightBinding struct {
	Generation cudacollective.Generation
	Rank       int
	World      int
	Count      int
	DataType   cudair.DataType
}

func newHostLightBinding(rank, world int) hostLightBinding {
	return hostLightBinding{Generation: hostLightGeneration, Rank: rank, World: world, Count: hostLightElements, DataType: cudair.Int32}
}

// check refuses comm, op and a payload of n elements unless they all match b.
// It runs before any upload or collective dispatch.
func (b hostLightBinding) check(comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, n int) error {
	switch {
	case comm.Generation() != b.Generation:
		return fmt.Errorf("%w: communicator generation %d, registered %d", errHostLightBinding, comm.Generation(), b.Generation)
	case comm.Rank() != b.Rank || comm.Membership().WorldSize() != b.World:
		return fmt.Errorf("%w: communicator rank %d of %d, registered %d of %d", errHostLightBinding, comm.Rank(), comm.Membership().WorldSize(), b.Rank, b.World)
	case op.Count != b.Count || n != b.Count:
		return fmt.Errorf("%w: step count %d, payload %d, registered %d", errHostLightBinding, op.Count, n, b.Count)
	case op.DataType != b.DataType:
		return fmt.Errorf("%w: step data type %v, registered %v", errHostLightBinding, op.DataType, b.DataType)
	}
	return nil
}

// hostLightSyntheticTransport is a synthetic cudacollective.Transport for
// host tests. It records each dispatch and blocks until released. It moves
// no data and is not evidence of any collective or network transfer.
type hostLightSyntheticTransport struct {
	world   int
	entered chan string
	release chan struct{}

	mu    sync.Mutex
	calls []string
}

func newHostLightSyntheticTransport(world int) *hostLightSyntheticTransport {
	return &hostLightSyntheticTransport{world: world, entered: make(chan string, 16), release: make(chan struct{}, 16)}
}

func (t *hostLightSyntheticTransport) Type() cudacollective.TransportType {
	return cudacollective.TransportTypeControlPlaneTest
}

func (t *hostLightSyntheticTransport) Available(m cudacollective.Membership) bool {
	return m.WorldSize() == t.world
}

func (t *hostLightSyntheticTransport) Execute(ctx context.Context, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	call := fmt.Sprintf("rank %d %v %v count %d send %#x recv %#x", comm.Rank(), op.Kind, op.DataType, op.Count, send, recv)
	t.mu.Lock()
	t.calls = append(t.calls, call)
	t.mu.Unlock()
	t.entered <- call
	<-t.release
	return nil
}

func (t *hostLightSyntheticTransport) Close() error { return nil }

func (t *hostLightSyntheticTransport) dispatched() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.calls...)
}

// hostLightSyntheticBuffers are fake, nonoverlapping device addresses. The
// synthetic transport never dereferences them.
func hostLightSyntheticBuffers(rank int) (send, recv driver.DevicePtr) {
	base := driver.DevicePtr(0x10000 * (rank + 1))
	return base, base + 0x1000
}

// hostLightCollectiveCommand returns a ConsumeBatch function that computes
// the command's payload with the finite CPU program, checks the registered
// binding, and runs plan.
func hostLightCollectiveCommand(b hostLightBinding, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, plan *cudacollective.Plan) func(context.Context, cudakv.Descriptor, []uint32) error {
	return func(ctx context.Context, d cudakv.Descriptor, input []uint32) error {
		tasks, ops := hostLightChain(d.PromptToken)
		out, err := hostLightRun(tasks, ops, input)
		if err != nil {
			return err
		}
		if err := b.check(comm, op, len(out["bias"])); err != nil {
			return err
		}
		return plan.ExecuteSteps(ctx)
	}
}

func hostLightAllReduce(count int, dtype cudair.DataType) cudacollective.OperationDescriptor {
	return cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: dtype, Count: count, Reduction: cudacollective.ReductionSum, Topology: cudacollective.TopologyAny}
}

func hostLightMembership(t *testing.T, world int) cudacollective.Membership {
	t.Helper()
	ranks := make([]cudacollective.Rank, world)
	for i := range ranks {
		ranks[i] = cudacollective.Rank{ID: i, DeviceID: i}
	}
	m, err := cudacollective.NewMembership(ranks)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func hostLightClique(t *testing.T, world int) *cudacollective.Clique {
	t.Helper()
	q, err := cudacollective.NewClique(hostLightMembership(t, world))
	if err != nil {
		t.Fatal(err)
	}
	return q
}

// TestHostLightRingSyntheticCollective drives consumed commands into public
// cudacollective Plans over a synthetic transport. It checks dispatch order
// and that neither ExecuteSteps nor the consumer reports completion before
// the transport returns. It makes no data-movement claim.
func TestHostLightRingSyntheticCollective(t *testing.T) {
	const world = 2
	tr := newHostLightSyntheticTransport(world)
	q := hostLightClique(t, world)
	var want []string
	for rank := 0; rank < world; rank++ {
		comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: rank, Clique: q, Generation: hostLightGeneration, Transport: tr})
		if err != nil {
			t.Fatal(err)
		}
		plan, err := cudacollective.NewPlan(comm, nil)
		if err != nil {
			t.Fatal(err)
		}
		op := hostLightAllReduce(hostLightElements, cudair.Int32)
		send, recv := hostLightSyntheticBuffers(rank)
		for _, name := range []string{"first", "second"} {
			if err := plan.AddCollectiveStep(name, op, send, recv); err != nil {
				t.Fatal(err)
			}
			want = append(want, fmt.Sprintf("rank %d %v %v count %d send %#x recv %#x", rank, op.Kind, op.DataType, op.Count, send, recv))
		}
		ring, c := hostLightConsumer(t, 1, cudakv.HostCommandInput{SeqID: 1, Words: []uint32{1, 2, 3, 4}})
		if err := ring.Publish(cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration, PromptToken: int32(rank)}); err != nil {
			t.Fatal(err)
		}
		type result struct {
			batch cudakv.HostCommandBatch
			err   error
		}
		done := make(chan result, 1)
		go func() {
			b, err := c.ConsumeBatch(context.Background(), 1, hostLightCollectiveCommand(newHostLightBinding(rank, world), comm, op, plan))
			done <- result{b, err}
		}()
		for step := 0; step < 2; step++ {
			<-tr.entered
			select {
			case r := <-done:
				t.Fatalf("rank %d batch returned %+v, %v while step %d was still in the transport", rank, r.batch, r.err, step)
			default:
			}
			// The in-flight command still owns the consumer.
			if _, err := c.ConsumeBatch(context.Background(), 1, func(context.Context, cudakv.Descriptor, []uint32) error { return nil }); !errors.Is(err, cudakv.ErrHostCommandBusy) {
				t.Fatalf("rank %d concurrent batch during step %d: %v, want ErrHostCommandBusy", rank, step, err)
			}
			tr.release <- struct{}{}
		}
		r := <-done
		if r.err != nil || r.batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) {
			t.Fatalf("rank %d batch = %+v, %v", rank, r.batch, r.err)
		}
	}
	if got := tr.dispatched(); !reflect.DeepEqual(got, want) {
		t.Fatalf("dispatched\n%q\nwant\n%q", got, want)
	}
}

// TestHostLightRingSyntheticCollectiveRefuses checks that a command bound to
// another rank, generation, count or data type, or a step the public Plan
// refuses, reaches no transport dispatch.
func TestHostLightRingSyntheticCollectiveRefuses(t *testing.T) {
	const world = 2
	tests := []struct {
		name    string
		rank    int
		gen     cudacollective.Generation
		op      cudacollective.OperationDescriptor
		binding func(*hostLightBinding)
		zeroGen bool
		fail    bool // mark the communicator failed before running
		stage   string
		want    error
	}{
		{name: "wrong rank", binding: func(b *hostLightBinding) { b.Rank = 1 }, want: errHostLightBinding},
		{name: "wrong world", binding: func(b *hostLightBinding) { b.World = 3 }, want: errHostLightBinding},
		{name: "wrong generation", gen: hostLightGeneration + 1, want: errHostLightBinding},
		{name: "wrong count", op: hostLightAllReduce(hostLightElements-1, cudair.Int32), want: errHostLightBinding},
		{name: "wrong dtype", op: hostLightAllReduce(hostLightElements, cudair.Float32), want: errHostLightBinding},
		{name: "failed generation", fail: true, want: cudacollective.ErrCommunicatorGeneration},
		{name: "rank outside world", rank: world, stage: "communicator", want: cudacollective.ErrInvalidRank},
		{name: "zero generation", zeroGen: true, stage: "communicator", want: cudacollective.ErrCommunicatorGeneration},
		{name: "zero count", op: hostLightAllReduce(0, cudair.Int32), stage: "step", want: cudacollective.ErrCountMismatch},
		{name: "unsupported dtype", op: hostLightAllReduce(hostLightElements, cudair.DataType(-1)), stage: "step"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := newHostLightSyntheticTransport(world)
			for range cap(tr.release) {
				tr.release <- struct{}{} // a wrongly admitted dispatch returns and is reported below
			}
			gen := tt.gen
			if gen == 0 && !tt.zeroGen {
				gen = hostLightGeneration
			}
			comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: tt.rank, Clique: hostLightClique(t, world), Generation: gen, Transport: tr})
			if tt.stage == "communicator" {
				if !errors.Is(err, tt.want) {
					t.Fatalf("NewCommunicator: %v, want %v", err, tt.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			plan, err := cudacollective.NewPlan(comm, nil)
			if err != nil {
				t.Fatal(err)
			}
			op := tt.op
			if op == (cudacollective.OperationDescriptor{}) {
				op = hostLightAllReduce(hostLightElements, cudair.Int32)
			}
			send, recv := hostLightSyntheticBuffers(tt.rank)
			err = plan.AddCollectiveStep("step", op, send, recv)
			if tt.stage == "step" {
				if err == nil || tt.want != nil && !errors.Is(err, tt.want) {
					t.Fatalf("AddCollectiveStep: %v, want refusal %v", err, tt.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tt.fail {
				comm.MarkFailed(errors.New("synthetic peer failure"))
			}
			b := newHostLightBinding(tt.rank, world)
			if tt.binding != nil {
				tt.binding(&b)
			}
			ring, c := hostLightConsumer(t, 2, cudakv.HostCommandInput{SeqID: 1, Words: []uint32{1, 2, 3, 4}})
			for range 2 {
				if err := ring.Publish(cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration}); err != nil {
					t.Fatal(err)
				}
			}
			batch, err := c.ConsumeBatch(context.Background(), 2, hostLightCollectiveCommand(b, comm, op, plan))
			if !errors.Is(err, tt.want) {
				t.Fatalf("ConsumeBatch: %v, want %v", err, tt.want)
			}
			if batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 4}) || ring.Len() != 1 {
				t.Fatalf("batch = %+v, ring %d; want one consumed, none completed, one queued", batch, ring.Len())
			}
			if got := tr.dispatched(); len(got) != 0 {
				t.Fatalf("refused command dispatched %q", got)
			}
		})
	}
}
