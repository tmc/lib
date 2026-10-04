package cudamoe

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// MoEWeightShape describes a global-layer-indexed E4M3FN up-projection slab.
// Each expert has Intermediate gate rows followed by Intermediate up rows in
// its logical input. Storage interleaves eight gate rows with eight up rows;
// the same permutation applies to its FP32 per-row scales. ActiveLayers lists
// readable global layer indices. Other layers reserve space but are unreadable.
type MoEWeightShape struct {
	Layers, Experts, Intermediate, Hidden int
	ActiveLayers                          []int
}

type moeWeightLayout struct {
	shape                  MoEWeightShape
	active                 map[int]bool
	weightBytes, scaleRows int
}

func newMoEWeightLayout(shape MoEWeightShape) (moeWeightLayout, error) {
	if shape.Layers <= 0 || shape.Experts <= 0 || shape.Intermediate <= 0 || shape.Hidden <= 0 || shape.Intermediate%8 != 0 {
		return moeWeightLayout{}, errors.New("cudamoe: invalid weight slab dimensions")
	}
	// This owner admits finite local slabs, not arbitrarily large model loaders.
	// Check each factor before multiplying, including the FP32 scale bytes.
	const maxBytes = 64 << 20
	rows := 1
	for _, n := range []int{shape.Layers, shape.Experts, 2, shape.Intermediate} {
		if n > maxBytes/4/rows {
			return moeWeightLayout{}, errors.New("cudamoe: weight scale slab exceeds byte limit")
		}
		rows *= n
	}
	if shape.Hidden > maxBytes/rows {
		return moeWeightLayout{}, errors.New("cudamoe: weight slab exceeds byte limit")
	}
	if len(shape.ActiveLayers) == 0 || len(shape.ActiveLayers) > shape.Layers {
		return moeWeightLayout{}, errors.New("cudamoe: missing or excessive active layers")
	}
	l := moeWeightLayout{shape: shape, active: make(map[int]bool), weightBytes: rows * shape.Hidden, scaleRows: rows}
	l.shape.ActiveLayers = append([]int(nil), shape.ActiveLayers...)
	for _, layer := range shape.ActiveLayers {
		if layer < 0 || layer >= shape.Layers || l.active[layer] {
			return moeWeightLayout{}, errors.New("cudamoe: invalid or duplicate global layer")
		}
		l.active[layer] = true
	}
	return l, nil
}

func (l moeWeightLayout) offsets(layer, expert int) (weights, scales int, err error) {
	if !l.active[layer] || expert < 0 || expert >= l.shape.Experts {
		return 0, 0, errors.New("cudamoe: unreadable weight layer or expert")
	}
	scales = (layer*l.shape.Experts + expert) * 2 * l.shape.Intermediate
	return scales * l.shape.Hidden, scales, nil
}

func moeWeightPhysicalRow(row, intermediate int) int {
	half, index := row/intermediate, row%intermediate
	return (index/8)*16 + half*8 + index%8
}

func moeWeightLogicalRow(row, intermediate int) int {
	return ((row%16)/8)*intermediate + (row/16)*8 + row%8
}

// Both payloads and scales use this single row permutation. Columns remain in
// logical order. The private helper operates only on host loading buffers.
func interleaveMoEWeightRows[T any](src []T, intermediate, columns int) ([]T, error) {
	if intermediate <= 0 || intermediate%8 != 0 || columns <= 0 || intermediate > math.MaxInt/2/columns || len(src) != 2*intermediate*columns {
		return nil, errors.New("cudamoe: invalid gate/up loading extent")
	}
	dst := make([]T, len(src))
	for row := 0; row < 2*intermediate; row++ {
		physical := moeWeightPhysicalRow(row, intermediate)
		copy(dst[physical*columns:(physical+1)*columns], src[row*columns:(row+1)*columns])
	}
	return dst, nil
}

// NVIDIA E4M3FN has exponent bias seven, signed zero, seven nonzero subnormal
// mantissas and no infinity. Only magnitudes 0x7f are NaN, including sign 0xff.
func decodeMoEWeight(v byte) (float32, error) {
	magnitude := v & 0x7f
	if magnitude == 0x7f {
		return 0, errors.New("cudamoe: nonfinite E4M3FN weight")
	}
	exponent, mantissa := int(magnitude>>3), int(magnitude&7)
	var value float32
	if exponent == 0 {
		value = float32(math.Ldexp(float64(mantissa), -9))
	} else {
		value = float32(math.Ldexp(float64(8+mantissa), exponent-10))
	}
	if v&0x80 != 0 {
		value = -value
	}
	return value, nil
}

func validateMoEWeightPayload(weights []byte, scales []float32) error {
	for i, v := range weights {
		if _, err := decodeMoEWeight(v); err != nil {
			return fmt.Errorf("cudamoe: weight %d: %w", i, err)
		}
	}
	for i, v := range scales {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("cudamoe: nonfinite weight scale %d", i)
		}
	}
	return nil
}

type moeWeightBuffer struct {
	ptr    func() driver.DevicePtr
	write  func(int, []byte) error
	borrow func() (func(), error)
	free   func() error
}

type moeWeightRange struct {
	generation uint64
	ready      bool
	readers    int
}

// MoEWeightStorage owns exactly two device allocations: one E4M3FN weight
// slab and one FP32 scale slab. It must not be copied. Its zero value is closed.
// The device must stay live until Close succeeds. Views retain both allocations;
// Close refuses while any view or asynchronous consumer retains them.
type MoEWeightStorage struct {
	mu                   sync.Mutex
	device               *cuda.Device
	layout               moeWeightLayout
	weights, scales      *moeWeightBuffer
	ranges               map[int]*moeWeightRange
	closed               bool
	requested, allocated uint64
	allocations          int
}

func allocMoEWeightBuffer(dev *cuda.Device, bytes int) (*moeWeightBuffer, error) {
	s, err := cudamem.Alloc[byte](dev, bytes)
	if s.Ptr() == 0 {
		return nil, err
	}
	return &moeWeightBuffer{
		ptr: s.Ptr,
		write: func(offset int, b []byte) error {
			if offset < 0 || offset > s.Len() || len(b) > s.Len()-offset {
				return errors.New("cudamoe: weight load outside slab")
			}
			_, err := cudamem.NewWriter(s.Slice(offset, offset+len(b))).Write(b)
			return err
		},
		borrow: func() (func(), error) { return s.Borrow(dev) },
		free:   s.Free,
	}, err
}

// NewMoEWeightStorage allocates the two slabs without loading any readable
// ranges. Even an error may return an owner retaining an allocation; callers
// must Close every nonnil result. Shape metadata is copied. No BF16 placeholder,
// per-layer device allocation or per-session copy is made.
func NewMoEWeightStorage(dev *cuda.Device, shape MoEWeightShape) (*MoEWeightStorage, error) {
	if dev == nil {
		return nil, errors.New("cudamoe: nil weight storage device")
	}
	if err := dev.WithContext(func() error { return nil }); err != nil {
		return nil, fmt.Errorf("cudamoe: weight storage device: %w", err)
	}
	return newMoEWeightStorage(dev, shape, func(bytes int) (*moeWeightBuffer, error) {
		return allocMoEWeightBuffer(dev, bytes)
	})
}

func newMoEWeightStorage(dev *cuda.Device, shape MoEWeightShape, alloc func(int) (*moeWeightBuffer, error)) (*MoEWeightStorage, error) {
	l, err := newMoEWeightLayout(shape)
	if err != nil {
		return nil, err
	}
	s := &MoEWeightStorage{device: dev, layout: l, ranges: make(map[int]*moeWeightRange)}
	for _, request := range []struct {
		dst   **moeWeightBuffer
		bytes int
	}{{&s.weights, l.weightBytes}, {&s.scales, l.scaleRows * 4}} {
		s.requested += uint64(request.bytes)
		b, err := alloc(request.bytes)
		*request.dst = b
		if b != nil && b.ptr() != 0 {
			s.allocated += uint64(request.bytes)
			s.allocations++
		}
		if err != nil {
			return s, fmt.Errorf("cudamoe: allocate weight slab: %w", err)
		}
		if b == nil || b.ptr() == 0 {
			return s, errors.New("cudamoe: weight allocation returned no storage")
		}
	}
	return s, nil
}

// MoEWeightStorageStats records this owner's base allocation requests and
// admitted bytes. It excludes host loading buffers, CUDA allocation granularity,
// consumer outputs and allocator peak RSS. Creating views changes none of it.
type MoEWeightStorageStats struct {
	RequestedBytes, AllocatedBytes uint64
	BaseAllocations                int
}

// Stats returns the cumulative base allocation ledger, including partial
// construction. AllocatedBytes counts requests with a nonnil allocation owner,
// not a queried driver allocation granularity or the current live-byte count.
func (s *MoEWeightStorage) Stats() MoEWeightStorageStats {
	if s == nil {
		return MoEWeightStorageStats{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return MoEWeightStorageStats{s.requested, s.allocated, s.allocations}
}

// Load fills one active global layer/expert from logical half-split weights and
// little-endian FP32 scales. Each reader supplies exactly the declared range;
// subsequent bytes are not consumed. Both payloads are staged and validated
// before copying. Only completion of both synchronous copies publishes readiness.
// Any failed load invalidates its generation. Retained ranges cannot be rewritten.
func (s *MoEWeightStorage) Load(layer, expert int, weights, scales io.Reader) error {
	if s == nil {
		return errors.New("cudamoe: nil weight owner")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.weights == nil || s.scales == nil || s.weights.ptr() == 0 || s.scales.ptr() == 0 {
		return errors.New("cudamoe: closed or incomplete weight storage")
	}
	wOffset, sOffset, err := s.layout.offsets(layer, expert)
	if err != nil {
		return err
	}
	key := layer*s.layout.shape.Experts + expert
	r := s.ranges[key]
	if r == nil {
		r = &moeWeightRange{}
		s.ranges[key] = r
	}
	if r.readers != 0 {
		return cudamem.ErrInUse
	}
	if r.generation == math.MaxUint64 {
		return errors.New("cudamoe: weight load generation overflow")
	}
	r.generation++
	r.ready = false
	if weights == nil || scales == nil {
		return errors.New("cudamoe: nil weight loading reader")
	}
	rows := 2 * s.layout.shape.Intermediate
	w := make([]byte, rows*s.layout.shape.Hidden)
	scaleBytes := make([]byte, rows*4)
	if _, err := io.ReadFull(weights, w); err != nil {
		return fmt.Errorf("cudamoe: read weight payload: %w", err)
	}
	if _, err := io.ReadFull(scales, scaleBytes); err != nil {
		return fmt.Errorf("cudamoe: read weight scales: %w", err)
	}
	sv := make([]float32, rows)
	for i := range sv {
		sv[i] = math.Float32frombits(binary.LittleEndian.Uint32(scaleBytes[i*4:]))
	}
	if err := validateMoEWeightPayload(w, sv); err != nil {
		return err
	}
	w, err = interleaveMoEWeightRows(w, s.layout.shape.Intermediate, s.layout.shape.Hidden)
	if err != nil {
		return err
	}
	sv, err = interleaveMoEWeightRows(sv, s.layout.shape.Intermediate, 1)
	if err != nil {
		return err
	}
	for i, v := range sv {
		binary.LittleEndian.PutUint32(scaleBytes[i*4:], math.Float32bits(v))
	}
	if err := s.weights.write(wOffset, w); err != nil {
		return fmt.Errorf("cudamoe: copy weight payload: %w", err)
	}
	if err := s.scales.write(sOffset*4, scaleBytes); err != nil {
		return fmt.Errorf("cudamoe: copy weight scales: %w", err)
	}
	r.ready = true
	return nil
}

// MoEWeightDescriptor describes a retained device range. It is metadata, not an
// ownership transfer: keep its originating view live through every actual read.
// Format is e4m3fn-gate-up-interleave8-v1 with FP32 scales and global indices.
type MoEWeightDescriptor struct {
	Device                              *cuda.Device
	WeightBase, ScaleBase               driver.DevicePtr
	WeightOffsetBytes, ScaleOffsetBytes uint64
	Layer, Expert, Intermediate, Hidden int
	Generation                          uint64
	Format                              string
}

// MoEWeightView retains an immutable, ready layer/expert range. Its zero value
// is closed. Copies of the pointer share the same lifetime. Close releases only
// this retention; it never frees either base allocation.
type MoEWeightView struct {
	mu      sync.Mutex
	owner   *MoEWeightStorage
	value   MoEWeightDescriptor
	release func()
}

// View acquires a ready range without copying or allocating any base weights.
// Padded layers and ranges whose loading generation failed are refused.
func (s *MoEWeightStorage) View(layer, expert int) (*MoEWeightView, error) {
	if s == nil {
		return nil, errors.New("cudamoe: nil weight storage")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	d, release, err := s.retain(layer, expert, 0)
	if err != nil {
		return nil, err
	}
	return &MoEWeightView{owner: s, value: d, release: release}, nil
}

// retain requires s.mu. Generation zero requests the current ready generation.
func (s *MoEWeightStorage) retain(layer, expert int, generation uint64) (MoEWeightDescriptor, func(), error) {
	if s.closed || s.weights == nil || s.scales == nil || s.weights.ptr() == 0 || s.scales.ptr() == 0 {
		return MoEWeightDescriptor{}, nil, errors.New("cudamoe: closed or incomplete weight storage")
	}
	w, scale, err := s.layout.offsets(layer, expert)
	if err != nil {
		return MoEWeightDescriptor{}, nil, err
	}
	r := s.ranges[layer*s.layout.shape.Experts+expert]
	if r == nil || !r.ready || generation != 0 && generation != r.generation {
		return MoEWeightDescriptor{}, nil, errors.New("cudamoe: weight range is not ready for this generation")
	}
	a, err := s.weights.borrow()
	if err != nil {
		return MoEWeightDescriptor{}, nil, err
	}
	b, err := s.scales.borrow()
	if err != nil {
		a()
		return MoEWeightDescriptor{}, nil, err
	}
	r.readers++
	var once sync.Once
	release := func() {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			a()
			b()
			r.readers--
		})
	}
	d := MoEWeightDescriptor{
		Device: s.device, WeightBase: s.weights.ptr(), ScaleBase: s.scales.ptr(),
		WeightOffsetBytes: uint64(w), ScaleOffsetBytes: uint64(scale * 4),
		Layer: layer, Expert: expert, Intermediate: s.layout.shape.Intermediate,
		Hidden: s.layout.shape.Hidden, Generation: r.generation,
		Format: "e4m3fn-gate-up-interleave8-v1",
	}
	return d, release, nil
}

// Retain an additional lease for a consumer without extending the public view's
// lifetime. Closing the public view must not release this consumer's read pins.
func (v *MoEWeightView) retain(dev *cuda.Device) (MoEWeightDescriptor, func(), error) {
	if v == nil {
		return MoEWeightDescriptor{}, nil, errors.New("cudamoe: nil weight view")
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.release == nil || v.owner == nil || dev == nil || dev != v.owner.device {
		return MoEWeightDescriptor{}, nil, errors.New("cudamoe: closed view or different weight device")
	}
	v.owner.mu.Lock()
	defer v.owner.mu.Unlock()
	return v.owner.retain(v.value.Layer, v.value.Expert, v.value.Generation)
}

// Descriptor returns copied metadata while this view retains its range.
func (v *MoEWeightView) Descriptor() (MoEWeightDescriptor, error) {
	if v == nil {
		return MoEWeightDescriptor{}, errors.New("cudamoe: nil weight view")
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.release == nil {
		return MoEWeightDescriptor{}, errors.New("cudamoe: closed weight view")
	}
	return v.value, nil
}

// Close releases this view's retention once. Consumer retentions are independent
// and must remain held until their real device completion, even after this call.
func (v *MoEWeightView) Close() error {
	if v == nil {
		return nil
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.release != nil {
		v.release()
		v.release = nil
	}
	return nil
}

// Close frees each base allocation through cudamem after all retained readers
// are released. Failure preserves unfinished allocations for a later retry.
func (s *MoEWeightStorage) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, r := range s.ranges {
		if r.readers != 0 {
			return cudamem.ErrInUse
		}
	}
	s.closed = true
	var errs []error
	for _, b := range []*moeWeightBuffer{s.weights, s.scales} {
		if b != nil {
			errs = append(errs, b.free())
		}
	}
	return errors.Join(errs...)
}
