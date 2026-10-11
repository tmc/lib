package nvtelemetrybridge_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
	"github.com/tmc/lib/nvidia/x/nvtelemetrybridge"
)

func Example() {
	s := nvtelemetrybridge.Snapshot{
		Owner: [16]byte{1}, Generation: 1, PublishedUnix: 1,
		EpochID: "example-epoch", ClockDomain: "synthetic_fixture", SchemaVersion: 2,
		Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-example"}},
	}
	data, err := nvtelemetrybridge.Encode(s)
	if err != nil {
		panic(err)
	}
	decoded, err := nvtelemetrybridge.Decode(data)
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded.Generation, decoded.Devices[0].UUID, len(data))
	// Output: 1 GPU-example 35024
}
