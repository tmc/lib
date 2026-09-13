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
