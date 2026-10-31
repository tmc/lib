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

// The tests in this file drive the finite CPU program of
// host_light_ring_integration_test.go into the production host-light
// registry: public ring and consumer, a callback that runs the program and
// passes its output words to HostLightCollective.Run, the registry's private
// Plan, and the labelled control-plane transport. The transport moves no
// data. Nothing here is a collective, numerical, native or network result.
// Pending-completion ownership is covered by TestHostLightPendingCompletion
// in package cudacollective and is not repeated here.

var (
	regChainSum4 = cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: hostLightElements, Reduction: cudacollective.ReductionSum}
	regChainMax4 = cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: hostLightElements, Reduction: cudacollective.ReductionMax}
	regChainSum2 = cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 2, Reduction: cudacollective.ReductionSum}
)

// regChainInputs are the consumer's registered input words. Sequence 9 is
// never bound to a collective.
var regChainInputs = map[uint64][]uint32{
	1: {1, 4, 0x7fffffff, 0xffffffff},
	2: {100, 200, 300, 400},
	3: {7, 8, 9, 10},
	9: {5, 6, 7, 8},
}

// regChainBindings are the registrations every rank makes. Sequences 1 and 2
// at ring generation 7 run different operations into different buffers;
// sequence 3 is bound to fewer elements than the program produces; sequence 1
// at generation 8 is a distinct registration from sequence 1 at generation 7.
func regChainBindings(rank int) []cudaplan.HostLightCollectiveBinding {
	return []cudaplan.HostLightCollectiveBinding{
		hlcBinding(rank, 1, hostLightGeneration, regChainSum4, 0),
		hlcBinding(rank, 2, hostLightGeneration, regChainMax4, 1),
		hlcBinding(rank, 3, hostLightGeneration, regChainSum2, 2),
		hlcBinding(rank, 1, hostLightGeneration+1, regChainMax4, 3),
	}
}

func regChainBinding(t *testing.T, rank int, d cudakv.Descriptor) cudaplan.HostLightCollectiveBinding {
	t.Helper()
	for _, b := range regChainBindings(rank) {
		if b.SeqID == d.SeqID && b.Generation == d.Generation {
			return b
		}
	}
	t.Fatalf("no binding for sequence %d generation %d", d.SeqID, d.Generation)
	return cudaplan.HostLightCollectiveBinding{}
}

// regChainTransport is the labelled control-plane hlcCPUTransport, recording
// also the communicator identity each dispatch arrives with.
type regChainTransport struct {
	*hlcCPUTransport
}

func (t regChainTransport) Execute(ctx context.Context, comm *cudacollective.Communicator, op cudacollective.OperationDescriptor, send, recv driver.DevicePtr) error {
	t.log.add("communicator rank %d of %d generation %d", comm.Rank(), comm.Membership().WorldSize(), comm.Generation())
	return t.hlcCPUTransport.Execute(ctx, comm, op, send, recv)
}

// regChainRegistry returns the production registry for comm with
// regChainBindings. Its prepare function records each call.
func regChainRegistry(t *testing.T, comm *cudacollective.Communicator, log *hlcLog) *cudaplan.HostLightCollective {
	t.Helper()
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings:     regChainBindings(comm.Rank()),
		Prepare: func(_ context.Context, c cudaplan.HostLightCollectiveCall) error {
			log.add("prepare rank %d sequence %d generation %d %+v send %#x recv %#x values %v", c.Rank, c.SeqID, c.Generation, c.Op, c.Send, c.Recv, c.Values)
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func regChainConsumer(t *testing.T, gen uint64) (*cudakv.DeviceCommandRing, *cudakv.HostCommandConsumer) {
	t.Helper()
	var inputs []cudakv.HostCommandInput
	for _, seq := range []uint64{1, 2, 3, 9} {
		inputs = append(inputs, cudakv.HostCommandInput{SeqID: seq, Words: regChainInputs[seq]})
	}
	return hlcConsumer(t, gen, inputs...)
}

// regChainCommand returns the ConsumeBatch function under test. It runs the
// finite program for the command's token on the command's input and passes
// the program's output words to reg.Run with the command's own descriptor.
func regChainCommand(reg *cudaplan.HostLightCollective) func(context.Context, cudakv.Descriptor, []uint32) error {
	return func(ctx context.Context, d cudakv.Descriptor, input []uint32) error {
		tasks, ops := hostLightChain(d.PromptToken)
		out, err := hostLightRun(tasks, ops, input)
		if err != nil {
			return err
		}
		return reg.Run(ctx, d, out["bias"])
	}
}

// regChainValues computes (x+token)*3+5 directly in 32-bit wrapping
// arithmetic, sharing no code with the program, as the Int32 values prepare
// must receive.
func regChainValues(input []uint32, token int32) []int32 {
	v := make([]int32, len(input))
	for i, x := range input {
		v[i] = int32((x+uint32(token))*3 + 5)
	}
	return v
}

// regChainEvents returns the events one admitted command produces: prepare
// with the program's values and binding b's identity, operation and buffers,
// then the transport dispatch on communicator rank of 2 at generation
// hlcCommGeneration with b's operation and buffers.
func regChainEvents(rank int, b cudaplan.HostLightCollectiveBinding, token int32) []string {
	return []string{
		fmt.Sprintf("prepare rank %d sequence %d generation %d %+v send %#x recv %#x values %v", rank, b.SeqID, b.Generation, b.Op, b.Send, b.Recv, regChainValues(regChainInputs[b.SeqID], token)),
		fmt.Sprintf("communicator rank %d of 2 generation %d", rank, hlcCommGeneration),
		hlcDispatch(rank, b),
	}
}

// TestHostLightRegisteredChain drives distinct commands on two ranks through
// the public ring and consumer, the finite program and the production
// registry. Prepare must see exactly the program's values, computed here
// independently, and each dispatch must carry the operation, buffers and
// communicator registered for that command's identity, in queued order. A
// callback that skipped the registry, or ran it under another identity,
// fails the events comparison.
func TestHostLightRegisteredChain(t *testing.T) {
	log := new(hlcLog)
	tr := &hlcCPUTransport{world: 2, log: log}
	q := hostLightClique(t, 2)
	var comms []*cudacollective.Communicator
	var want []string
	for rank := range 2 {
		comm := hlcCommunicator(t, q, regChainTransport{tr}, rank)
		comms = append(comms, comm)
		reg := regChainRegistry(t, comm, log)
		ring, c := regChainConsumer(t, hostLightGeneration)
		for _, d := range []cudakv.Descriptor{{SeqID: 1, PromptToken: 2}, {SeqID: 2, PromptToken: 5}, {SeqID: 1, PromptToken: 7}} {
			d.Generation = hostLightGeneration
			d.PromptToken += int32(rank)
			if err := ring.Publish(d); err != nil {
				t.Fatal(err)
			}
			want = append(want, regChainEvents(rank, regChainBinding(t, rank, d), d.PromptToken)...)
		}
		batch, err := c.ConsumeBatch(context.Background(), 4, regChainCommand(reg))
		if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 3, Completed: 3, Words: 12}) || ring.Len() != 0 {
			t.Fatalf("rank %d batch = %+v, %v; ring %d", rank, batch, err, ring.Len())
		}
		if err := reg.Close(); err != nil {
			t.Fatalf("rank %d Close: %v", rank, err)
		}
		if err := comm.CheckAlive(); err != nil {
			t.Fatalf("rank %d communicator after Close: %v", rank, err)
		}
	}
	if !reflect.DeepEqual(log.events, want) {
		t.Fatalf("events\n%q\nwant\n%q", log.events, want)
	}
	if tr.calls != 6 {
		t.Fatalf("%d dispatches, want 6", tr.calls)
	}
	for _, comm := range comms {
		if err := comm.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

// TestHostLightRegisteredChainRefuses publishes a command the chain must
// refuse ahead of a valid one. The refused command reaches neither prepare
// nor the transport, its error keeps its identity, and the valid command
// stays queued and then runs as registered.
func TestHostLightRegisteredChainRefuses(t *testing.T) {
	tests := []struct {
		name   string
		gen    uint64 // ring generation
		d      cudakv.Descriptor
		cancel bool
		batch  cudakv.HostCommandBatch
		queued int
		want   error
	}{
		{name: "unregistered sequence", gen: 7, d: cudakv.Descriptor{SeqID: 9}, batch: cudakv.HostCommandBatch{Consumed: 1, Words: 4}, queued: 1, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "other application generation", gen: 8, d: cudakv.Descriptor{SeqID: 2}, batch: cudakv.HostCommandBatch{Consumed: 1, Words: 4}, queued: 1, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "registered count differs", gen: 7, d: cudakv.Descriptor{SeqID: 3}, batch: cudakv.HostCommandBatch{Consumed: 1, Words: 4}, queued: 1, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "nonzero flags", gen: 7, d: cudakv.Descriptor{SeqID: 1, Flags: 1}, batch: cudakv.HostCommandBatch{Consumed: 1}, queued: 1, want: cudakv.ErrHostCommandInvalid},
		{name: "canceled context", gen: 7, d: cudakv.Descriptor{SeqID: 2, PromptToken: 3}, cancel: true, queued: 2, want: context.Canceled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := new(hlcLog)
			tr := &hlcCPUTransport{world: 2, log: log}
			comm := hlcCommunicator(t, hostLightClique(t, 2), regChainTransport{tr}, 0)
			reg := regChainRegistry(t, comm, log)
			ring, c := regChainConsumer(t, tt.gen)
			first := tt.d
			first.Generation = tt.gen
			next := cudakv.Descriptor{SeqID: 1, Generation: tt.gen, PromptToken: 4}
			for _, d := range []cudakv.Descriptor{first, next} {
				if err := ring.Publish(d); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tt.cancel {
				cancel()
			}
			batch, err := c.ConsumeBatch(ctx, 2, regChainCommand(reg))
			if !errors.Is(err, tt.want) {
				t.Fatalf("ConsumeBatch: %v, want %v", err, tt.want)
			}
			if batch != tt.batch || ring.Len() != tt.queued {
				t.Fatalf("batch = %+v, ring %d; want %+v, %d queued", batch, ring.Len(), tt.batch, tt.queued)
			}
			if len(log.events) != 0 || tr.calls != 0 {
				t.Fatalf("refused command reached %q, %d dispatches", log.events, tr.calls)
			}

			rest := []cudakv.Descriptor{next}
			if tt.cancel {
				rest = []cudakv.Descriptor{first, next}
			}
			var want []string
			for _, d := range rest {
				want = append(want, regChainEvents(0, regChainBinding(t, 0, d), d.PromptToken)...)
			}
			n := len(rest)
			batch, err = c.ConsumeBatch(context.Background(), 2, regChainCommand(reg))
			if err != nil || batch != (cudakv.HostCommandBatch{Consumed: n, Completed: n, Words: 4 * n}) || ring.Len() != 0 {
				t.Fatalf("later batch = %+v, %v; ring %d", batch, err, ring.Len())
			}
			if !reflect.DeepEqual(log.events, want) || tr.calls != n {
				t.Fatalf("events %q, %d dispatches; want %q, %d", log.events, tr.calls, want, n)
			}
			if err := reg.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestHostLightRegisteredChainRunRefuses calls the chain callback directly
// with descriptors the consumer never delivers: nonzero flags, and a context
// cancelled after the command was removed. The registry refuses each after
// the program runs and before prepare, and the next command runs as
// registered.
func TestHostLightRegisteredChainRunRefuses(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		d    cudakv.Descriptor
		want error
	}{
		{name: "nonzero flags", ctx: context.Background(), d: cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration, Flags: 1}, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "canceled context", ctx: canceled, d: cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration}, want: context.Canceled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := new(hlcLog)
			tr := &hlcCPUTransport{world: 2, log: log}
			comm := hlcCommunicator(t, hostLightClique(t, 2), regChainTransport{tr}, 0)
			reg := regChainRegistry(t, comm, log)
			run := regChainCommand(reg)
			if err := run(tt.ctx, tt.d, regChainInputs[1]); !errors.Is(err, tt.want) {
				t.Fatalf("run: %v, want %v", err, tt.want)
			}
			if len(log.events) != 0 || tr.calls != 0 {
				t.Fatalf("refused command reached %q, %d dispatches", log.events, tr.calls)
			}
			d := cudakv.Descriptor{SeqID: 1, Generation: hostLightGeneration, PromptToken: 2}
			if err := run(context.Background(), d, regChainInputs[1]); err != nil {
				t.Fatalf("next run: %v", err)
			}
			if want := regChainEvents(0, regChainBinding(t, 0, d), 2); !reflect.DeepEqual(log.events, want) || tr.calls != 1 {
				t.Fatalf("events %q, %d dispatches; want %q, one", log.events, tr.calls, want)
			}
			if err := reg.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestHostLightRegisteredChainConstructRefuses checks that a registration
// for another rank, world, communicator generation or data type creates no
// registry and leaves the communicator and transport untouched: a correct
// registry on the same communicator then runs a command as registered.
func TestHostLightRegisteredChainConstructRefuses(t *testing.T) {
	tests := []struct {
		name   string
		change func(*cudaplan.HostLightCollectiveBinding)
		want   error
	}{
		{name: "wrong rank", change: func(b *cudaplan.HostLightCollectiveBinding) { b.Rank = 1 }, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "wrong world", change: func(b *cudaplan.HostLightCollectiveBinding) { b.World = 3 }, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "ring generation as communicator generation", change: func(b *cudaplan.HostLightCollectiveBinding) { b.CommunicatorGeneration = hostLightGeneration }, want: cudaplan.ErrHostLightCollectiveBinding},
		{name: "float32", change: func(b *cudaplan.HostLightCollectiveBinding) { b.Op.DataType = cudair.Float32 }, want: cudaplan.ErrHostLightCollectiveUnsupported},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			log := new(hlcLog)
			tr := &hlcCPUTransport{world: 2, log: log}
			comm := hlcCommunicator(t, hostLightClique(t, 2), regChainTransport{tr}, 0)
			bindings := regChainBindings(0)
			tt.change(&bindings[1])
			reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{Communicator: comm, Bindings: bindings})
			if reg != nil || !errors.Is(err, tt.want) {
				t.Fatalf("NewHostLightCollective = %v, %v; want nil, %v", reg, err, tt.want)
			}
			if tr.calls != 0 || tr.closed {
				t.Fatalf("refused registry: %d dispatches, transport closed %v", tr.calls, tr.closed)
			}
			if err := comm.CheckAlive(); err != nil {
				t.Fatalf("communicator after refused registry: %v", err)
			}

			reg = regChainRegistry(t, comm, log)
			ring, c := regChainConsumer(t, hostLightGeneration)
			d := cudakv.Descriptor{SeqID: 2, Generation: hostLightGeneration, PromptToken: 6}
			if err := ring.Publish(d); err != nil {
				t.Fatal(err)
			}
			batch, err := c.ConsumeBatch(context.Background(), 1, regChainCommand(reg))
			if err != nil || batch != (cudakv.HostCommandBatch{Consumed: 1, Completed: 1, Words: 4}) {
				t.Fatalf("batch = %+v, %v", batch, err)
			}
			if want := regChainEvents(0, regChainBinding(t, 0, d), 6); !reflect.DeepEqual(log.events, want) || tr.calls != 1 {
				t.Fatalf("events %q, %d dispatches; want %q, one", log.events, tr.calls, want)
			}
			if err := reg.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
