package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// hlcCommGeneration is the communicator generation. It differs from the
// ring generations 7 and 8 so that neither can stand in for the other.
const hlcCommGeneration = 3

var (
	hlcSum4 = cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 4, Reduction: cudacollective.ReductionSum}
	hlcMax2 = cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 2, Reduction: cudacollective.ReductionMax}
)

// hlcLog records prepare and dispatch events in the order they happen.
type hlcLog struct {
	events []string
}

func (l *hlcLog) add(format string, args ...any) {
	l.events = append(l.events, fmt.Sprintf(format, args...))
}

// hlcCPUTransport is a labelled CPU control-plane test transport. Execute
// records the dispatch, calls during, and returns err. It moves no data and
// computes nothing: a recorded dispatch is not evidence of any collective,
// numerical, native or network result.
type hlcCPUTransport struct {
	world  int
	log    *hlcLog
	during func()
	err    error
	calls  int
	closed bool
}

func (t *hlcCPUTransport) Type() cudacollective.TransportType {
	return cudacollective.TransportTypeControlPlaneTest
}

func (t *hlcCPUTransport) Available(m cudacollective.Membership) bool {
	return m.WorldSize() == t.world
}

func (t *hlcCPUTransport) Execute(ctx context.Context, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
	if err := comm.CheckAlive(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	t.calls++
	t.log.add("dispatch rank %d %+v send %#x recv %#x", comm.Rank(), op, send, recv)
	if f := t.during; f != nil {
		t.during = nil
		f()
	}
	return t.err
}

func (t *hlcCPUTransport) Close() error {
	t.closed = true
	return nil
}

// hlcBuffers are fake, nonoverlapping device addresses for rank and slot.
// The CPU transport never dereferences them.
func hlcBuffers(rank, slot int) (send, recv driver.DevicePtr) {
	base := driver.DevicePtr(0x100000*(rank+1) + 0x10000*slot)
	return base, base + 0x1000
}

func hlcBinding(rank int, seq, gen uint64, op cudacollective.OperationDescriptor, slot int) cudaplan.HostLightCollectiveBinding {
	send, recv := hlcBuffers(rank, slot)
	return cudaplan.HostLightCollectiveBinding{
		SeqID:                  seq,
		Generation:             gen,
		Rank:                   rank,
		World:                  2,
		CommunicatorGeneration: hlcCommGeneration,
		Op:                     op,
		Send:                   send,
		Recv:                   recv,
	}
}

// hlcBindings are the two steps every rank registers: sequence 1 at ring
// generation 7 and sequence 1 at ring generation 8 sum and max different
// extents into different buffers, and sequence 2 is bound only at
// generation 8.
func hlcBindings(rank int) []cudaplan.HostLightCollectiveBinding {
	return []cudaplan.HostLightCollectiveBinding{
		hlcBinding(rank, 1, 7, hlcSum4, 0),
		hlcBinding(rank, 1, 8, hlcMax2, 1),
		hlcBinding(rank, 2, 8, hlcMax2, 2),
	}
}

func hlcDispatch(rank int, b cudaplan.HostLightCollectiveBinding) string {
	return fmt.Sprintf("dispatch rank %d %+v send %#x recv %#x", rank, b.Op, b.Send, b.Recv)
}

func hlcCommunicator(t *testing.T, q *cudacollective.Clique, tr cudacollective.Transport, rank int) *cudacollective.Communicator {
	t.Helper()
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: rank, Clique: q, Generation: hlcCommGeneration, Transport: tr})
	if err != nil {
		t.Fatal(err)
	}
	return comm
}

// hlcRegistry returns a registry whose prepare function records each call.
// If cancel is non-nil, prepare calls it.
func hlcRegistry(t *testing.T, comm *cudacollective.Communicator, log *hlcLog, cancel *context.CancelFunc) *cudaplan.HostLightCollective {
	t.Helper()
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings:     hlcBindings(comm.Rank()),
		Prepare: func(_ context.Context, c cudaplan.HostLightCollectiveCall) error {
			log.add("prepare rank %d sequence %d generation %d values %v", c.Rank, c.SeqID, c.Generation, c.Values)
			if cancel != nil && *cancel != nil {
				(*cancel)()
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func hlcConsumer(t *testing.T, gen uint64, inputs ...cudakv.HostCommandInput) (*cudakv.DeviceCommandRing, *cudakv.HostCommandConsumer) {
	t.Helper()
	ring, err := cudakv.NewDeviceCommandRing(4, gen)
	if err != nil {
		t.Fatal(err)
	}
	c, err := cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{
		Mode:             "host-finite",
		Generation:       gen,
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

func hlcPublish(t *testing.T, ring *cudakv.DeviceCommandRing, gen uint64, seqs ...uint64) {
	t.Helper()
	for _, seq := range seqs {
		if err := ring.Publish(cudakv.Descriptor{SeqID: seq, Generation: gen}); err != nil {
			t.Fatal(err)
		}
	}
}

// TestHostLightCollectiveDispatch drives two ranks through two steps each:
// public ring, HostCommandConsumer, the production binding, its private
// Plan and the public Transport API on a CPU control-plane transport. Each
// command is prepared with its own Int32 values and then dispatched with
// exactly its registered operation and buffers, in order. No data moves.
func TestHostLightCollectiveDispatch(t *testing.T) {
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log}
	q := hostLightClique(t, 2)
	inputs := map[uint64][2][]uint32{
		7: {{1, 2, 3, 0xffffffff}, {11, 12, 13, 14}},
		8: {{5, 0x80000000}, {6, 7}},
	}
	var comms [2]*cudacollective.Communicator
	var regs [2]*cudaplan.HostLightCollective
	consumers := map[uint64][2]*cudakv.HostCommandConsumer{}
	rings := map[uint64][2]*cudakv.DeviceCommandRing{}
	for rank := range regs {
		comms[rank] = hlcCommunicator(t, q, tr, rank)
		regs[rank] = hlcRegistry(t, comms[rank], log, nil)
	}
	for _, gen := range []uint64{7, 8} {
		var rs [2]*cudakv.DeviceCommandRing
		var cs [2]*cudakv.HostCommandConsumer
		for rank := range rs {
			rs[rank], cs[rank] = hlcConsumer(t, gen, cudakv.HostCommandInput{SeqID: 1, Words: inputs[gen][rank]})
			hlcPublish(t, rs[rank], gen, 1)
		}
		rings[gen], consumers[gen] = rs, cs
	}

	var want []string
	for i, gen := range []uint64{7, 8} {
		for rank := range regs {
			batch, err := consumers[gen][rank].ConsumeBatch(context.Background(), 2, regs[rank].Run)
			n := len(inputs[gen][rank])
			if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: n}) || rings[gen][rank].Len() != 0 {
				t.Fatalf("rank %d generation %d batch = %+v, %v; ring %d", rank, gen, batch, err, rings[gen][rank].Len())
			}
			values := make([]int32, n)
			for j, w := range inputs[gen][rank] {
				values[j] = int32(w)
			}
			want = append(want,
				fmt.Sprintf("prepare rank %d sequence 1 generation %d values %v", rank, gen, values),
				hlcDispatch(rank, hlcBindings(rank)[i]))
		}
	}
	if !reflect.DeepEqual(log.events, want) {
		t.Fatalf("events\n%q\nwant\n%q", log.events, want)
	}
	for rank := range regs {
		if err := regs[rank].Close(); err != nil {
			t.Fatalf("rank %d Close: %v", rank, err)
		}
		if err := comms[rank].CheckAlive(); err != nil {
			t.Fatalf("rank %d communicator after registry Close: %v", rank, err)
		}
	}
	if tr.closed {
		t.Fatal("registry Close closed the borrowed transport")
	}
	for rank := range comms {
		if err := comms[rank].Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// TestHostLightCollectiveRefuses checks that a consumed command whose
// identity, payload, communicator or registry state does not admit it is
// refused before prepare: zero uploads and zero dispatches, the command
// consumed, and the next one still queued.
func TestHostLightCollectiveRefuses(t *testing.T) {
	tests := []struct {
		name  string
		seq   uint64
		words int
		setup func(*testing.T, *cudacollective.Communicator, *cudaplan.HostLightCollective)
		want  error
	}{
		{name: "unregistered sequence", seq: 3, words: 4, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "sequence registered in another generation", seq: 2, words: 2, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "short payload", seq: 1, words: 3, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "long payload", seq: 1, words: 5, want: cudaplan.ErrHostLightCollectiveBinding},
		{
			name: "failed communicator", seq: 1, words: 4,
			setup: func(_ *testing.T, comm *cudacollective.Communicator, _ *cudaplan.HostLightCollective) {
				comm.MarkFailed(errors.New("synthetic peer failure"))
			},
			want: cudacollective.ErrCommunicatorGeneration,
		},
		{
			name: "closed registry", seq: 1, words: 4,
			setup: func(t *testing.T, _ *cudacollective.Communicator, reg *cudaplan.HostLightCollective) {
				if err := reg.Close(); err != nil {
					t.Fatal(err)
				}
			},
			want: cudaplan.ErrHostLightCollectiveClosed,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := new(hlcLog)
			tr := &hlcCPUTransport{world: 2, log: log}
			comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
			reg := hlcRegistry(t, comm, log, nil)
			if tt.setup != nil {
				tt.setup(t, comm, reg)
			}
			ring, c := hlcConsumer(t, 7, cudakv.HostCommandInput{SeqID: tt.seq, Words: make([]uint32, tt.words)})
			hlcPublish(t, ring, 7, tt.seq, tt.seq)
			batch, err := c.ConsumeBatch(context.Background(), 2, reg.Run)
			if !errors.Is(err, tt.want) {
				t.Fatalf("ConsumeBatch: %v, want %v", err, tt.want)
			}
			if batch != (cudakv.HostCommandBatch{Consumed: 1, Words: tt.words}) || ring.Len() != 1 {
				t.Fatalf("batch = %+v, ring %d; want one consumed, none completed, one queued", batch, ring.Len())
			}
			if len(log.events) != 0 || tr.calls != 0 {
				t.Fatalf("refused command reached %q, %d dispatches", log.events, tr.calls)
			}
			if err := reg.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestHostLightCollectiveRunRefuses checks direct Run refusals and a failing
// prepare function: none dispatches.
func TestHostLightCollectiveRunRefuses(t *testing.T) {
	errPrepare := errors.New("synthetic upload failure")
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log}
	comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings:     hlcBindings(0),
		Prepare: func(_ context.Context, c cudaplan.HostLightCollectiveCall) error {
			log.add("prepare sequence %d", c.SeqID)
			return errPrepare
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	words := make([]uint32, 4)
	var zero cudaplan.HostLightCollective
	tests := []struct {
		name string
		reg  *cudaplan.HostLightCollective
		ctx  context.Context
		d    cudakv.Descriptor
		want error // nil: any error
	}{
		{name: "nonzero flags", reg: reg, ctx: context.Background(), d: cudakv.Descriptor{SeqID: 1, Generation: 7, Flags: 1}, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "zero identity", reg: reg, ctx: context.Background(), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "nil context", reg: reg, d: cudakv.Descriptor{SeqID: 1, Generation: 7}},
		{name: "zero registry", reg: &zero, ctx: context.Background(), d: cudakv.Descriptor{SeqID: 1, Generation: 7}},
		{name: "nil registry", ctx: context.Background(), d: cudakv.Descriptor{SeqID: 1, Generation: 7}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.reg.Run(tt.ctx, tt.d, words)
			if err == nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("Run: %v, want %v", err, tt.want)
			}
			if len(log.events) != 0 || tr.calls != 0 {
				t.Fatalf("refused command reached %q, %d dispatches", log.events, tr.calls)
			}
		})
	}
	err = reg.Run(context.Background(), cudakv.Descriptor{SeqID: 1, Generation: 7}, words)
	if !errors.Is(err, errPrepare) {
		t.Fatalf("Run with failing prepare: %v, want %v", err, errPrepare)
	}
	if want := []string{"prepare sequence 1"}; !reflect.DeepEqual(log.events, want) || tr.calls != 0 {
		t.Fatalf("failed prepare: events %q, %d dispatches; want %q, none", log.events, tr.calls, want)
	}
	if err := comm.CheckAlive(); err != nil {
		t.Fatalf("communicator after refused prepare: %v", err)
	}
	if err := reg.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestNewHostLightCollectiveRefuses checks that a malformed binding creates
// no registry, so no command can reach prepare or dispatch. Every refusal but
// the buffer ones precedes plan creation; those join the step error with the
// cleanup of plans already created.
func TestNewHostLightCollectiveRefuses(t *testing.T) {
	tests := []struct {
		name   string
		nilCom bool
		change func([]cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding
		want   error // nil: any error
	}{
		{name: "nil communicator", nilCom: true},
		{name: "no bindings", change: func([]cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding { return nil }},
		{name: "too many bindings", change: func([]cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding {
			bs := make([]cudaplan.HostLightCollectiveBinding, 1025)
			for i := range bs {
				bs[i] = hlcBinding(0, uint64(i+1), 7, hlcSum4, 0)
			}
			return bs
		}},
		{name: "zero sequence", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.SeqID = 0 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "zero generation", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Generation = 0 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "duplicate identity", change: func(bs []cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding {
			bs[1].Generation = bs[0].Generation
			return bs
		}, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "wrong rank", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Rank = 1 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "wrong world", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.World = 3 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "wrong communicator generation", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.CommunicatorGeneration = 7 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "all-gather", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) {
			b.Op.Kind, b.Op.Reduction = cudacollective.KindAllGather, 0
		}), want: cudaplan.ErrHostLightCollectiveUnsupported},
		{name: "float32", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Op.DataType = cudair.Float32 }), want: cudaplan.ErrHostLightCollectiveUnsupported},
		{name: "pipeline slot", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Op.PipelineSlot = 1 }), want: cudaplan.ErrHostLightCollectiveUnsupported},
		{name: "zero count", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Op.Count = 0 }), want: cudacollective.ErrCountMismatch},
		{name: "count beyond bound", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Op.Count = 1<<20 + 1 }), want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "unsupported reduction", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Op.Reduction = 0 })},
		{name: "nil buffer", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Recv = 0 }), want: cudacollective.ErrNilBuffer},
		{name: "overlapping buffers", change: hlcChange(func(b *cudaplan.HostLightCollectiveBinding) { b.Recv = b.Send + 4 }), want: cudacollective.ErrBufferOverlap},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := new(hlcLog)
			tr := &hlcCPUTransport{world: 2, log: log}
			comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
			config := cudaplan.HostLightCollectiveConfig{Communicator: comm, Bindings: hlcBindings(0)}
			if tt.nilCom {
				config.Communicator = nil
			}
			if tt.change != nil {
				config.Bindings = tt.change(config.Bindings)
			}
			reg, err := cudaplan.NewHostLightCollective(config)
			if reg != nil || err == nil || tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("NewHostLightCollective = %v, %v; want nil, %v", reg, err, tt.want)
			}
			if tr.calls != 0 || tr.closed {
				t.Fatalf("refused registry: %d dispatches, transport closed %v", tr.calls, tr.closed)
			}
			if err := comm.CheckAlive(); err != nil {
				t.Fatalf("communicator after refused registry: %v", err)
			}
		})
	}
}

// hlcChange returns a change applying f to the last binding, so that, for a
// step refusal, every earlier binding's plan exists first.
func hlcChange(f func(*cudaplan.HostLightCollectiveBinding)) func([]cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding {
	return func(bs []cudaplan.HostLightCollectiveBinding) []cudaplan.HostLightCollectiveBinding {
		f(&bs[len(bs)-1])
		return bs
	}
}

// TestHostLightCollectiveCancel cancels the context from prepare, after the
// command was removed and its upload ran. The command is refused without
// dispatch and stays consumed; the next command stays queued until a live
// context consumes it, and the cancelled one is not resubmitted.
func TestHostLightCollectiveCancel(t *testing.T) {
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log}
	comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cancelPrepare := cancel
	reg := hlcRegistry(t, comm, log, &cancelPrepare)
	ring, c := hlcConsumer(t, 7, cudakv.HostCommandInput{SeqID: 1, Words: []uint32{1, 2, 3, 4}})
	hlcPublish(t, ring, 7, 1, 1)

	batch, err := c.ConsumeBatch(ctx, 2, reg.Run)
	if !errors.Is(err, context.Canceled) || batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 4}) || ring.Len() != 1 {
		t.Fatalf("cancelled batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	prepare := "prepare rank 0 sequence 1 generation 7 values [1 2 3 4]"
	if want := []string{prepare}; !reflect.DeepEqual(log.events, want) || tr.calls != 0 {
		t.Fatalf("after cancellation: events %q, %d dispatches; want %q, none", log.events, tr.calls, want)
	}
	batch, err = c.ConsumeBatch(ctx, 2, reg.Run)
	if !errors.Is(err, context.Canceled) || batch != (cudakv.HostCommandBatch{}) || ring.Len() != 1 {
		t.Fatalf("batch on cancelled context = %+v, %v; ring %d", batch, err, ring.Len())
	}

	cancelPrepare = nil
	batch, err = c.ConsumeBatch(context.Background(), 2, reg.Run)
	if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) || ring.Len() != 0 {
		t.Fatalf("live batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	want := []string{prepare, prepare, hlcDispatch(0, hlcBindings(0)[0])}
	if !reflect.DeepEqual(log.events, want) || tr.calls != 1 {
		t.Fatalf("events %q, %d dispatches; want %q, one", log.events, tr.calls, want)
	}
	if err := reg.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestHostLightCollectiveInFlight calls Close, Run and ConsumeBatch from
// inside the transport, while the dispatched operation has not returned.
// Each is refused without waiting, the in-flight registry keeps its plans,
// and Close succeeds once the operation returns. No goroutine is involved.
func TestHostLightCollectiveInFlight(t *testing.T) {
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log}
	comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
	reg := hlcRegistry(t, comm, log, nil)
	ring, c := hlcConsumer(t, 7, cudakv.HostCommandInput{SeqID: 1, Words: []uint32{1, 2, 3, 4}})
	hlcPublish(t, ring, 7, 1, 1)
	var closeErr, runErr, batchErr error
	tr.during = func() {
		closeErr = reg.Close()
		runErr = reg.Run(context.Background(), cudakv.Descriptor{SeqID: 1, Generation: 7}, []uint32{1, 2, 3, 4})
		_, batchErr = c.ConsumeBatch(context.Background(), 1, reg.Run)
	}
	batch, err := c.ConsumeBatch(context.Background(), 1, reg.Run)
	if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) || ring.Len() != 1 {
		t.Fatalf("batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	if !errors.Is(closeErr, cudaplan.ErrHostLightCollectiveBusy) || !errors.Is(runErr, cudaplan.ErrHostLightCollectiveBusy) || !errors.Is(batchErr, cudakv.ErrHostCommandBusy) {
		t.Fatalf("during dispatch: Close %v, Run %v, ConsumeBatch %v; want busy refusals", closeErr, runErr, batchErr)
	}
	if tr.calls != 1 {
		t.Fatalf("%d dispatches, want one", tr.calls)
	}
	// The refused Close did not close the registry: the queued command runs.
	batch, err = c.ConsumeBatch(context.Background(), 1, reg.Run)
	if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) || tr.calls != 2 {
		t.Fatalf("after refused Close: batch = %+v, %v; %d dispatches", batch, err, tr.calls)
	}
	for i := range 2 {
		if err := reg.Close(); err != nil {
			t.Fatalf("Close %d: %v", i, err)
		}
	}
	if err := reg.Run(context.Background(), cudakv.Descriptor{SeqID: 1, Generation: 7}, []uint32{1, 2, 3, 4}); !errors.Is(err, cudaplan.ErrHostLightCollectiveClosed) {
		t.Fatalf("Run after Close: %v", err)
	}
	if err := comm.CheckAlive(); err != nil || tr.closed || tr.calls != 2 {
		t.Fatalf("after Close: communicator %v, transport closed %v, %d dispatches", err, tr.closed, tr.calls)
	}
}

// TestHostLightCollectiveTransportFails checks that a transport failure
// reaches the consumer's caller with its identity, is not retried, and leaves
// the plan's communicator semantics in place: the generation is failed and
// the next command is refused before prepare.
func TestHostLightCollectiveTransportFails(t *testing.T) {
	errTransport := errors.New("synthetic transport failure")
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log, err: errTransport}
	comm := hlcCommunicator(t, hostLightClique(t, 2), tr, 0)
	reg := hlcRegistry(t, comm, log, nil)
	ring, c := hlcConsumer(t, 7, cudakv.HostCommandInput{SeqID: 1, Words: []uint32{1, 2, 3, 4}})
	hlcPublish(t, ring, 7, 1, 1)

	batch, err := c.ConsumeBatch(context.Background(), 2, reg.Run)
	if !errors.Is(err, errTransport) || batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 4}) || ring.Len() != 1 {
		t.Fatalf("batch = %+v, %v; ring %d", batch, err, ring.Len())
	}
	if err := comm.CheckAlive(); !errors.Is(err, cudacollective.ErrCommunicatorGeneration) || !errors.Is(err, errTransport) {
		t.Fatalf("communicator after transport failure: %v", err)
	}
	batch, err = c.ConsumeBatch(context.Background(), 2, reg.Run)
	if !errors.Is(err, cudacollective.ErrCommunicatorGeneration) || batch != (cudakv.HostCommandBatch{Consumed: 1, Words: 4}) || ring.Len() != 0 {
		t.Fatalf("batch after failure = %+v, %v; ring %d", batch, err, ring.Len())
	}
	want := []string{"prepare rank 0 sequence 1 generation 7 values [1 2 3 4]", hlcDispatch(0, hlcBindings(0)[0])}
	if !reflect.DeepEqual(log.events, want) || tr.calls != 1 {
		t.Fatalf("events %q, %d dispatches; want %q, one", log.events, tr.calls, want)
	}
	if err := reg.Close(); err != nil {
		t.Fatal(err)
	}
}
