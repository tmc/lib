package cuptipmsampling

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// TriggerMode specifies the periodic sampling trigger type.
type TriggerMode int32

const (
	// TriggerModeSysclkInterval triggers samples based on GPU system clock cycle intervals.
	TriggerModeSysclkInterval TriggerMode = TriggerMode(cupti.PmSamplingTriggerModeGpuSysclkInterval)

	// TriggerModeTimeInterval triggers samples based on GPU time intervals.
	// Supported on GA10x (Ampere) and later architectures.
	TriggerModeTimeInterval TriggerMode = TriggerMode(cupti.PmSamplingTriggerModeGpuTimeInterval)
)

func (m TriggerMode) String() string {
	switch m {
	case TriggerModeSysclkInterval:
		return "SysclkInterval"
	case TriggerModeTimeInterval:
		return "TimeInterval"
	default:
		return fmt.Sprintf("TriggerMode(%d)", m)
	}
}

// AppendMode specifies the hardware buffer overflow policy.
type AppendMode int32

const (
	// AppendModeKeepOldest retains the oldest records in the hardware buffer
	// and flags overflow when full.
	AppendModeKeepOldest AppendMode = AppendMode(cupti.PmSamplingHardwareBufferAppendModeKeepOldest)

	// AppendModeKeepLatest overwrites the oldest records with newer records
	// when the buffer is full. Supported on Ampere and later.
	AppendModeKeepLatest AppendMode = AppendMode(cupti.PmSamplingHardwareBufferAppendModeKeepLatest)
)

func (m AppendMode) String() string {
	switch m {
	case AppendModeKeepOldest:
		return "KeepOldest"
	case AppendModeKeepLatest:
		return "KeepLatest"
	default:
		return fmt.Sprintf("AppendMode(%d)", m)
	}
}

// DecodeStopReason specifies why decoding stopped.
type DecodeStopReason int32

const (
	// DecodeStopReasonOther indicates decoding stopped for an unspecified reason.
	DecodeStopReasonOther DecodeStopReason = DecodeStopReason(cupti.PmSamplingDecodeStopReasonOther)

	// DecodeStopReasonCounterDataFull indicates the destination counter data image became full.
	DecodeStopReasonCounterDataFull DecodeStopReason = DecodeStopReason(cupti.PmSamplingDecodeStopReasonCounterDataFull)

	// DecodeStopReasonEndOfRecords indicates all records in the hardware buffer were drained.
	DecodeStopReasonEndOfRecords DecodeStopReason = DecodeStopReason(cupti.PmSamplingDecodeStopReasonEndOfRecords)
)

func (r DecodeStopReason) String() string {
	switch r {
	case DecodeStopReasonOther:
		return "Other"
	case DecodeStopReasonCounterDataFull:
		return "CounterDataFull"
	case DecodeStopReasonEndOfRecords:
		return "EndOfRecords"
	default:
		return fmt.Sprintf("DecodeStopReason(%d)", r)
	}
}

// CounterDataInfo provides summary statistics for a decoded counter data image.
type CounterDataInfo struct {
	NumTotalSamples     uint
	NumPopulatedSamples uint
	NumCompletedSamples uint
}

// SampleInfo identifies a single sample's time span within the counter data image.
type SampleInfo struct {
	Index          uint
	StartTimestamp uint64 // Nanoseconds
	EndTimestamp   uint64 // Nanoseconds
}

// DecodeResult reports the outcome of a decode operation.
type DecodeResult struct {
	StopReason DecodeStopReason
	Overflow   bool
	Info       CounterDataInfo
}

// Config specifies the parameters for a PM sampling session.
type Config struct {
	// DeviceIndex is the 0-based CUDA device ordinal.
	DeviceIndex uint32

	// ConfigImage is the scheduling configuration image produced by the host profiler
	// for the desired metrics.
	ConfigImage []byte

	// HardwareBufferSize is the size of the GPU hardware buffer in bytes.
	// If zero, a default of 64MB is used.
	HardwareBufferSize uint

	// SamplingInterval specifies the period between samples (in nanoseconds for
	// TriggerModeTimeInterval, or cycles for TriggerModeCycleInterval).
	SamplingInterval uint64

	// TriggerMode selects time or cycle interval sampling.
	TriggerMode TriggerMode

	// AppendMode selects whether overflow keeps oldest or latest records.
	AppendMode AppendMode

	// MaxSamples is the maximum number of samples the counter data image can store.
	MaxSamples uint32

	// MetricNames is the list of metric names scheduled in ConfigImage.
	MetricNames []string
}
