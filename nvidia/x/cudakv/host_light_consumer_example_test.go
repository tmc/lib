package cudakv_test

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleHostCommandConsumer_ConsumeBatch() {
	ring, err := cudakv.NewDeviceCommandRing(4, 1)
	if err != nil {
		log.Fatal(err)
	}
	consumer, err := cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{
		Mode:             "host-finite",
		Generation:       1,
		VocabSize:        16,
		MaxEntries:       2,
		MaxRegistryWords: 4,
		MaxBatch:         2,
		MaxBatchWords:    4,
	}, []cudakv.HostCommandInput{
		{SeqID: 3, Words: []uint32{8, 9}},
		{SeqID: 4, Words: []uint32{100}},
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, d := range []cudakv.Descriptor{{SeqID: 3, PromptToken: 2}, {SeqID: 4, PromptToken: 5}, {SeqID: 3}} {
		d.Generation = 1
		if err := ring.Publish(d); err != nil {
			log.Fatal(err)
		}
	}
	batch, err := consumer.ConsumeBatch(context.Background(), 2, func(_ context.Context, d cudakv.Descriptor, input []uint32) error {
		fmt.Println("sequence", d.SeqID, "token", d.PromptToken, "input", input)
		return nil
	})
	fmt.Printf("%+v %v, %d queued\n", batch, err, ring.Len())
	// Output:
	// sequence 3 token 2 input [8 9]
	// sequence 4 token 5 input [100]
	// {Consumed:2 Completed:2 Words:3} <nil>, 1 queued
}

func ExampleNewHostCommandConsumer_unsupported() {
	ring, err := cudakv.NewDeviceCommandRing(1, 1)
	if err != nil {
		log.Fatal(err)
	}
	_, err = cudakv.NewHostCommandConsumer(ring, cudakv.HostCommandConfig{Mode: "device-poll"}, nil)
	fmt.Println(err)
	// Output:
	// cudakv: host command capability unsupported: mode "device-poll"
}
