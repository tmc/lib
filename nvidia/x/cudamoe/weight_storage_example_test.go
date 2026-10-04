package cudamoe_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func ExampleMoEWeightSource() {
	_, entry, err := cudamoe.MoEWeightSource(cudamoe.MoEWeightSlabConsumer)
	fmt.Println(entry, err)
	// Output: moe_weight_slab <nil>
}

func ExampleMoEWeightSchema() {
	schema, err := cudamoe.MoEWeightSchema()
	fmt.Println(len(schema.Params), schema.TotalParamBytes, err)
	// Output: 9 56 <nil>
}

func ExampleMoEWeightStorage_Stats() {
	var storage *cudamoe.MoEWeightStorage
	fmt.Println(storage.Stats())
	fmt.Println(storage.Close())
	// Output:
	// {0 0 0}
	// <nil>
}

func ExampleMoEWeightConsumer_Wait() {
	var consumer *cudamoe.MoEWeightConsumer
	fmt.Println(consumer.Wait(context.Background()))
	fmt.Println(consumer.Close())
	// Output:
	// <nil>
	// <nil>
}
