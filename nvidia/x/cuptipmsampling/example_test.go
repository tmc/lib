package cuptipmsampling_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func ExampleTriggerMode_String() {
	fmt.Println(cuptipmsampling.TriggerModeSysclkInterval)
	fmt.Println(cuptipmsampling.TriggerModeTimeInterval)
	// Output:
	// SysclkInterval
	// TimeInterval
}

func ExampleAppendMode_String() {
	fmt.Println(cuptipmsampling.AppendModeKeepOldest)
	fmt.Println(cuptipmsampling.AppendModeKeepLatest)
	// Output:
	// KeepOldest
	// KeepLatest
}

func ExampleDecodeStopReason_String() {
	fmt.Println(cuptipmsampling.DecodeStopReasonOther)
	fmt.Println(cuptipmsampling.DecodeStopReasonCounterDataFull)
	fmt.Println(cuptipmsampling.DecodeStopReasonEndOfRecords)
	// Output:
	// Other
	// CounterDataFull
	// EndOfRecords
}

func ExampleSamplerConfig() {
	cfg := cuptipmsampling.SamplerConfig{
		DeviceIndex: 0,
		MetricNames: []string{"sm__throughput"},
	}
	lowLevel := cfg.Config([]byte("mock_config"))
	fmt.Println(lowLevel.DeviceIndex, len(lowLevel.MetricNames), lowLevel.HardwareBufferSize > 0)
	// Output: 0 1 true
}

func ExampleReader() {
	samples := []cuptipmsampling.Sample{
		{
			Index:          0,
			StartTimestamp: 1000,
			EndTimestamp:   2000,
			DurationNs:     1000,
			Values:         map[string]float64{"sm_busy": 78.5},
		},
	}
	r := cuptipmsampling.NewReader(samples)
	sample, err := r.ReadSample()
	if err == nil {
		fmt.Printf("Sample %d: %dns sm_busy=%.1f\n", sample.Index, sample.DurationNs, sample.Values["sm_busy"])
	}
	// Output: Sample 0: 1000ns sm_busy=78.5
}
