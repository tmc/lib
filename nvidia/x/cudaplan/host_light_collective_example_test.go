package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func ExampleNewHostLightCollective() {
	// One rank on the control-plane test transport. It checks metadata and
	// moves no data, so this shows dispatch, not a reduction.
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}})
	if err != nil {
		log.Fatal(err)
	}
	q, err := cudacollective.NewClique(m)
	if err != nil {
		log.Fatal(err)
	}
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Clique: q, Generation: 1, Transport: cudacollective.NewDeterministicTestTransport(1)})
	if err != nil {
		log.Fatal(err)
	}
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings: []cudaplan.HostLightCollectiveBinding{{
			SeqID:                  5,
			Generation:             9,
			World:                  1,
			CommunicatorGeneration: 1,
			Op:                     cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 2, Reduction: cudacollective.ReductionSum},
			Send:                   0x1000,
			Recv:                   0x2000,
		}},
		Prepare: func(_ context.Context, c cudaplan.HostLightCollectiveCall) error {
			fmt.Println("prepare sequence", c.SeqID, "values", c.Values)
			return nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	ring, err := cudakv.NewDeviceCommandRing(2, 9)
	if err != nil {
		log.Fatal(err)
	}
	consumer, err := cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{
		Mode:             "host-finite",
		Generation:       9,
		VocabSize:        1,
		MaxEntries:       1,
		MaxRegistryWords: 2,
		MaxBatch:         2,
		MaxBatchWords:    4,
	}, []cudakv.HostCommandInput{{SeqID: 5, Words: []uint32{7, 0xffffffff}}})
	if err != nil {
		log.Fatal(err)
	}
	if err := ring.Publish(cudakv.Descriptor{SeqID: 5, Generation: 9}); err != nil {
		log.Fatal(err)
	}
	batch, err := consumer.ConsumeBatch(context.Background(), 2, reg.Run)
	fmt.Printf("%+v %v\n", batch, err)
	fmt.Println(reg.Close(), comm.Close())
	// Output:
	// prepare sequence 5 values [7 -1]
	// {Consumed:1 Completed:1 Words:2} <nil>
	// <nil> <nil>
}

// exampleHostLightCollective returns a one-rank registry binding sequence 5
// at generation 9 to a two-element Int32 all-reduce on the control-plane
// test transport.
func exampleHostLightCollective() (*cudacollective.Communicator, *cudaplan.HostLightCollective) {
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}})
	if err != nil {
		log.Fatal(err)
	}
	q, err := cudacollective.NewClique(m)
	if err != nil {
		log.Fatal(err)
	}
	comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Clique: q, Generation: 1, Transport: cudacollective.NewDeterministicTestTransport(1)})
	if err != nil {
		log.Fatal(err)
	}
	reg, err := cudaplan.NewHostLightCollective(cudaplan.HostLightCollectiveConfig{
		Communicator: comm,
		Bindings: []cudaplan.HostLightCollectiveBinding{{
			SeqID:                  5,
			Generation:             9,
			World:                  1,
			CommunicatorGeneration: 1,
			Op:                     cudacollective.OperationDescriptor{Kind: cudacollective.KindAllReduce, DataType: cudair.Int32, Count: 2, Reduction: cudacollective.ReductionSum},
			Send:                   0x1000,
			Recv:                   0x2000,
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	return comm, reg
}

func ExampleHostLightCollective_Run() {
	comm, reg := exampleHostLightCollective()
	ctx := context.Background()
	err := reg.Run(ctx, cudakv.Descriptor{SeqID: 5, Generation: 8}, []uint32{7, 8})
	fmt.Println(errors.Is(err, cudaplan.ErrHostLightCollectiveBinding))
	fmt.Println(err)
	fmt.Println(reg.Run(ctx, cudakv.Descriptor{SeqID: 5, Generation: 9}, []uint32{7}))
	fmt.Println(reg.Run(ctx, cudakv.Descriptor{SeqID: 5, Generation: 9}, []uint32{7, 8}))
	fmt.Println(reg.Close(), comm.Close())
	// Output:
	// true
	// cudaplan: host-light collective binding mismatch: sequence 5 generation 8 not registered
	// cudaplan: host-light collective binding mismatch: sequence 5 payload of 1 words, registered 2 Int32 elements
	// <nil>
	// <nil> <nil>
}

func ExampleHostLightCollective_Close() {
	comm, reg := exampleHostLightCollective()
	fmt.Println(reg.Close())
	fmt.Println(reg.Run(context.Background(), cudakv.Descriptor{SeqID: 5, Generation: 9}, []uint32{7, 8}))
	fmt.Println(reg.Close())
	fmt.Println(comm.CheckAlive(), comm.Close())
	// Output:
	// <nil>
	// cudaplan: host-light collective closed
	// <nil>
	// <nil> <nil>
}
