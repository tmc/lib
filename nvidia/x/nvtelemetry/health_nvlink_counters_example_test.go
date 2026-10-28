package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleNewNVLinkCounterReader() {
	device, clock, config := counterFixture() // host fake; no native producer ran
	reader, err := nvtelemetry.NewNVLinkCounterReader(device, clock, config)
	if err != nil {
		panic(err)
	}
	snapshot, err := reader.Read(context.Background())
	if err != nil {
		panic(err)
	}
	fmt.Println(snapshot.API, snapshot.ResetCoverage)
	// Output: nvmlDeviceGetNvLinkErrorCounter unknown
}

func ExampleReadNVLinkCounterSnapshot() {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	snapshot, _ := r.Read(context.Background())
	raw, _ := json.Marshal(snapshot)
	decoded, err := nvtelemetry.ReadNVLinkCounterSnapshot(bytes.NewReader(raw), 256<<10)
	if err != nil {
		panic(err)
	}
	fmt.Println(decoded.Version, *decoded.Observations[0].Value)
	// Output: 1 9007199254740993
}

func ExampleNVLinkCounterReader_Read() {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	snapshot, err := r.Read(context.Background())
	if err != nil {
		panic(err)
	}
	fmt.Println(snapshot.Before.DeviceUUID, len(snapshot.Observations))
	// Output: GPU-0 5
}

func ExampleNVLinkCounterSnapshot_Validate() {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	snapshot, _ := r.Read(context.Background())
	fmt.Println(snapshot.Validate())
	// Output: <nil>
}

func ExampleNVLinkCounterReadings() {
	d, c, config := counterFixture()
	r, _ := nvtelemetry.NewNVLinkCounterReader(d, c, config)
	snapshot, _ := r.Read(context.Background())
	raw, _ := json.Marshal(snapshot)
	readings, err := nvtelemetry.NVLinkCounterReadings(raw, "synthetic")
	if err != nil {
		panic(err)
	}
	fmt.Println(readings[0].Name, *readings[0].Number.Uint, readings[0].Unit)
	fmt.Println(readings[0].Observation.CounterEpoch == "")
	// Output:
	// nvlink_dl_replay_errors 9007199254740993 count
	// true
}
