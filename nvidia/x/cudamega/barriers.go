package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// BarrierPool manages a set of atomic barrier counters resident in GPU global memory.
// Each barrier tracks in-flight producers via an increment-on-launch, decrement-on-finish protocol.
type BarrierPool struct {
	dev     *cuda.Device
	size    int
	counter cudamem.Slice[int32]
}

// NewBarrierPool allocates a pool of numBarriers atomic counter integers in GPU global memory.
func NewBarrierPool(dev *cuda.Device, numBarriers int) (*BarrierPool, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudamega: nil device")
	}
	if numBarriers <= 0 {
		numBarriers = 64
	}

	counters, err := cudamem.Alloc[int32](dev, numBarriers)
	if err != nil {
		return nil, fmt.Errorf("cudamega: alloc barriers: %w", err)
	}

	zeros := make([]int32, numBarriers)
	if err := counters.CopyFromHost(zeros); err != nil {
		_ = counters.Free()
		return nil, fmt.Errorf("cudamega: zero barriers: %w", err)
	}

	return &BarrierPool{
		dev:     dev,
		size:    numBarriers,
		counter: counters,
	}, nil
}

// Reset clears all barrier counters to 0 on the device.
func (p *BarrierPool) Reset() error {
	zeros := make([]int32, p.size)
	return p.counter.CopyFromHost(zeros)
}

// DevicePtr returns the GPU virtual address of the barrier counter array.
func (p *BarrierPool) DevicePtr() uint64 {
	return uint64(p.counter.Ptr())
}

// Size returns the total number of barriers allocated in the pool.
func (p *BarrierPool) Size() int {
	return p.size
}

// Close deallocates the barrier pool GPU memory.
func (p *BarrierPool) Close() error {
	return p.counter.Free()
}
