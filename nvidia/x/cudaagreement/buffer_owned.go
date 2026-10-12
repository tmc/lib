package cudaagreement

import (
	"errors"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type bufferOwner struct{ slice cudamem.Slice[uint32] }

// BufferFromSlice binds the exact active uint32 range and its actual owner.
// MegaExecutor borrows this owner before submitting persistent work. Callers
// must exclude mutation until Drain succeeds; changing Ptr or Bytes invalidates
// the binding. A zero or freed slice is refused when used. Ordinary executors
// retain their existing caller-managed raw-pointer ownership boundary.
func BufferFromSlice(s cudamem.Slice[uint32]) Buffer {
	n := s.SizeBytes()
	bytes := -1
	if uint64(n) <= uint64(math.MaxInt) {
		bytes = int(n)
	}
	return Buffer{Ptr: s.Ptr(), Bytes: bytes, owner: &bufferOwner{s}}
}

func (b Buffer) borrow(dev *cuda.Device) (func(), error) {
	if b.owner == nil {
		return nil, errors.New("cudaagreement: persistent buffer requires BufferFromSlice")
	}
	s := b.owner.slice
	if b.Ptr == 0 || b.Bytes <= 0 || b.Bytes%4 != 0 || uint64(b.Ptr) > math.MaxUint64-uint64(b.Bytes) ||
		b.Ptr != s.Ptr() || uint64(b.Bytes) != uint64(s.SizeBytes()) || s.Device() != dev || dev == nil {
		return nil, errors.New("cudaagreement: persistent buffer range or owner differs")
	}
	// Borrow checks the owner again under the allocation lock. A concurrent
	// Free between the checks above and Borrow cannot admit an invalid pointer.
	release, err := s.Borrow(dev)
	if err != nil {
		return nil, err
	}
	if b.Ptr != s.Ptr() || s.Device() != dev {
		release()
		return nil, errors.New("cudaagreement: persistent buffer owner changed")
	}
	return release, nil
}
