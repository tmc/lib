package hipmega

import (
	"fmt"
	"math"

	"github.com/tmc/lib/rocm/driver"
)

// Progress is the scheduling guarantee required by a worker.
type Progress uint8

const (
	// NoProgress declares no guarantee and is refused.
	NoProgress Progress = iota
	// CoResident requires occupancy admission and a cooperative grid launch.
	CoResident
)

// Request describes one launch of an already compiled worker. The caller must
// supply the geometry that worker expects and declare its required primitives.
// Function, Stream and KernelParams belong to Device's current context.
type Request struct {
	Device          driver.Device
	Function        driver.Function
	Grid            [3]int64
	Block           [3]int64
	SharedMemBytes  int64 // dynamic shared memory, excluding the compiled static size
	Stream          driver.Stream
	KernelParams    uintptr
	Progress        Progress
	RequiresCluster bool
	RequiresTMEM    bool
}

// Limits contains observations for the current device and the exact function
// passed to Operations.Limits. Injected values are host test inputs, not native
// evidence. MaxThreadsPerBlock is the smaller device/function limit.
type Limits struct {
	Device                   driver.Device
	CooperativeLaunch        bool
	Multiprocessors          int64
	MaxGrid                  [3]int64
	MaxBlock                 [3]int64
	MaxThreadsPerBlock       int64
	MaxSharedMemPerBlock     int64
	StaticSharedMemBytes     int64
	MaxDynamicSharedMemBytes int64
}

// Operations is the small HIP boundary used by Launch. Limits and Occupancy
// must describe the same current device and exact Function. CooperativeLaunch
// must perform a cooperative launch, not an ordinary kernel launch. Calls are
// synchronous host submissions; they do not establish device completion.
type Operations struct {
	Limits            func(driver.Function) (Limits, error)
	Occupancy         func(driver.Function, int32, uint) (int32, error)
	CooperativeLaunch func(Request) error
}

// Admission records the checked geometry and occupancy for one submission.
// It is not a reusable launch permit or proof of kernel correctness/completion.
type Admission struct {
	Device                        driver.Device
	GridBlocks                    uint64
	ThreadsPerBlock               int32
	ActiveBlocksPerMultiprocessor int32
	Multiprocessors               int64
	ResidentBlocks                uint64
	Progress                      Progress
}

// Launch checks one request and submits it through the cooperative operation.
// A refusal never calls CooperativeLaunch. Queries and launch errors retain
// their original errors. An error returns a zero Admission. A launch error does
// not prove that device work has completed; the caller retains resource owners.
func Launch(ops Operations, r Request) (Admission, error) {
	var zero Admission
	if ops.Limits == nil || ops.Occupancy == nil || ops.CooperativeLaunch == nil {
		return zero, fmt.Errorf("hipmega: incomplete operations")
	}
	if r.RequiresCluster || r.RequiresTMEM {
		return zero, fmt.Errorf("hipmega: cluster and tmem primitives are unsupported")
	}
	if r.Progress != CoResident {
		return zero, fmt.Errorf("hipmega: co-resident progress is required")
	}
	if r.Device < 0 || r.Function == 0 || r.SharedMemBytes < 0 || r.SharedMemBytes > math.MaxUint32 || uint64(r.SharedMemBytes) > uint64(^uint(0)) {
		return zero, fmt.Errorf("hipmega: invalid device, function or shared memory size")
	}
	blocks, err := geometry(r.Grid)
	if err != nil {
		return zero, fmt.Errorf("hipmega: grid: %w", err)
	}
	threads, err := geometry(r.Block)
	if err != nil {
		return zero, fmt.Errorf("hipmega: block: %w", err)
	}
	if threads > math.MaxInt32 {
		return zero, fmt.Errorf("hipmega: block thread count exceeds occupancy argument width")
	}
	l, err := ops.Limits(r.Function)
	if err != nil {
		return zero, fmt.Errorf("hipmega: limits: %w", err)
	}
	if l.Device != r.Device || !l.CooperativeLaunch || l.Multiprocessors <= 0 || l.MaxThreadsPerBlock <= 0 || l.MaxSharedMemPerBlock <= 0 || l.StaticSharedMemBytes < 0 || l.MaxDynamicSharedMemBytes < 0 {
		return zero, fmt.Errorf("hipmega: missing or incompatible device/function limits")
	}
	for i := range r.Grid {
		if l.MaxGrid[i] <= 0 || l.MaxBlock[i] <= 0 || r.Grid[i] > l.MaxGrid[i] || r.Block[i] > l.MaxBlock[i] {
			return zero, fmt.Errorf("hipmega: geometry exceeds device limits")
		}
	}
	if threads > uint64(l.MaxThreadsPerBlock) || l.StaticSharedMemBytes > l.MaxSharedMemPerBlock || r.SharedMemBytes > l.MaxSharedMemPerBlock-l.StaticSharedMemBytes || r.SharedMemBytes > l.MaxDynamicSharedMemBytes {
		return zero, fmt.Errorf("hipmega: block resources exceed device/function limits")
	}
	active, err := ops.Occupancy(r.Function, int32(threads), uint(r.SharedMemBytes))
	if err != nil {
		return zero, fmt.Errorf("hipmega: occupancy: %w", err)
	}
	if active <= 0 || uint64(l.Multiprocessors) > math.MaxUint64/uint64(active) {
		return zero, fmt.Errorf("hipmega: invalid or overflowing residency")
	}
	resident := uint64(l.Multiprocessors) * uint64(active)
	if blocks > resident {
		return zero, fmt.Errorf("hipmega: grid exceeds co-resident capacity")
	}
	if err := ops.CooperativeLaunch(r); err != nil {
		return zero, fmt.Errorf("hipmega: cooperative launch: %w", err)
	}
	return Admission{
		Device: r.Device, GridBlocks: blocks, ThreadsPerBlock: int32(threads),
		ActiveBlocksPerMultiprocessor: active, Multiprocessors: l.Multiprocessors,
		ResidentBlocks: resident, Progress: CoResident,
	}, nil
}

func geometry(d [3]int64) (uint64, error) {
	n := uint64(1)
	for _, x := range d {
		if x <= 0 || x > math.MaxUint32 || n > math.MaxUint64/uint64(x) {
			return 0, fmt.Errorf("invalid or overflowing dimensions")
		}
		n *= uint64(x)
	}
	return n, nil
}

// HIPOperations returns an adapter to the existing HIP driver bindings. This
// function makes no driver calls. Calling its operations requires the caller's
// initialized HIP runtime and current context on the calling OS thread.
// The existing driver package itself attempts library loading at import init;
// CODEGEN_SKIP_BINDINGS=1 disables that loading for host tests.
func HIPOperations() Operations {
	return Operations{
		Limits: hipLimits,
		Occupancy: func(f driver.Function, block int32, shared uint) (int32, error) {
			var active int32
			err := f.OccupancyMaxActiveBlocksPerMultiprocessor(&active, block, shared)
			return active, err
		},
		CooperativeLaunch: func(r Request) error {
			return r.Function.LaunchCooperativeKernel(
				uint32(r.Grid[0]), uint32(r.Grid[1]), uint32(r.Grid[2]),
				uint32(r.Block[0]), uint32(r.Block[1]), uint32(r.Block[2]),
				uint32(r.SharedMemBytes), r.Stream, r.KernelParams)
		},
	}
}

func hipLimits(f driver.Function) (Limits, error) {
	var l Limits
	if err := l.Device.Get(); err != nil {
		return Limits{}, fmt.Errorf("current device: %w", err)
	}
	queries := []struct {
		attr driver.DeviceAttribute
		dst  *int64
	}{
		{driver.DeviceAttributeMultiprocessorCount, &l.Multiprocessors},
		{driver.DeviceAttributeMaxThreadsPerBlock, &l.MaxThreadsPerBlock},
		{driver.DeviceAttributeMaxSharedMemoryPerBlock, &l.MaxSharedMemPerBlock},
		{driver.DeviceAttributeMaxGridDimX, &l.MaxGrid[0]},
		{driver.DeviceAttributeMaxGridDimY, &l.MaxGrid[1]},
		{driver.DeviceAttributeMaxGridDimZ, &l.MaxGrid[2]},
		{driver.DeviceAttributeMaxBlockDimX, &l.MaxBlock[0]},
		{driver.DeviceAttributeMaxBlockDimY, &l.MaxBlock[1]},
		{driver.DeviceAttributeMaxBlockDimZ, &l.MaxBlock[2]},
	}
	for _, q := range queries {
		n, err := driver.DeviceGetAttribute(q.attr, l.Device)
		if err != nil {
			return Limits{}, fmt.Errorf("device attribute %s: %w", q.attr, err)
		}
		*q.dst = int64(n)
	}
	cooperative, err := driver.DeviceGetAttribute(driver.DeviceAttributeCooperativeLaunch, l.Device)
	if err != nil {
		return Limits{}, fmt.Errorf("cooperative launch capability: %w", err)
	}
	l.CooperativeLaunch = cooperative == 1
	var maxThreads, static, dynamic int32
	for _, q := range []struct {
		attr driver.FunctionAttribute
		dst  *int32
	}{
		{driver.FuncAttributeMaxThreadsPerBlock, &maxThreads},
		{driver.FuncAttributeSharedSizeBytes, &static},
		{driver.FuncAttributeMaxDynamicSharedSizeBytes, &dynamic},
	} {
		if err := f.GetAttribute(q.dst, q.attr); err != nil {
			return Limits{}, fmt.Errorf("function attribute %s: %w", q.attr, err)
		}
	}
	if int64(maxThreads) < l.MaxThreadsPerBlock {
		l.MaxThreadsPerBlock = int64(maxThreads)
	}
	l.StaticSharedMemBytes = int64(static)
	l.MaxDynamicSharedMemBytes = int64(dynamic)
	return l, nil
}
