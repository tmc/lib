package cudamoe

import "github.com/tmc/lib/nvidia/x/cuda"

// MoEWeightBorrowed records the actual owner metadata accompanying a borrowed
// ready range. Shape.ActiveLayers is copied. Slab bytes are admitted allocation
// extents, not driver allocation granularity or peak memory. Metadata copies
// do not extend lifetime; the release callback returned by Borrow owns the pin.
type MoEWeightBorrowed struct {
	Descriptor              MoEWeightDescriptor
	Shape                   MoEWeightShape
	WeightBytes, ScaleBytes uint64
}

// Borrow retains this view's ready generation for a consumer on dev. The
// returned descriptor is copied, and release is safe to call more than once.
// Closing the public view does not release this independent retention.
// Call release only after the consumer's actual device work has completed.
func (v *MoEWeightView) Borrow(dev *cuda.Device) (MoEWeightBorrowed, func(), error) {
	d, release, err := v.retain(dev)
	if err != nil {
		return MoEWeightBorrowed{}, nil, err
	}
	// owner is immutable after View construction. Its separate retained pin
	// prevents Close from freeing either allocation while metadata is copied.
	s := v.owner
	s.mu.Lock()
	shape := s.layout.shape
	shape.ActiveLayers = append([]int(nil), shape.ActiveLayers...)
	b := MoEWeightBorrowed{Descriptor: d, Shape: shape, WeightBytes: uint64(s.layout.weightBytes), ScaleBytes: uint64(s.layout.scaleRows * 4)}
	s.mu.Unlock()
	return b, release, nil
}
