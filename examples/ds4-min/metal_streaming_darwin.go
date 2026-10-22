//go:build darwin && arm64

package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

// metalStreamingWeights belongs to the model backend, not request State.
// Original mapped tensor bindings stay alive for mixed-class/overflow reads.
type metalStreamingWeights struct {
	source           *dsv4expertcache.Source
	triplets         map[int]dsv4expertcache.Triplet
	cache            *dsv4expertcache.Cache
	upload           *dsv4expertcache.MetalUpload
	strides          [3]uint64
	capacity         int
	gate, down       apple.MTLComputePipelineState
	library          apple.MTLLibrary
	tables           [3]metalmem.Slice[uint64]
	selection        metalStreamingSelection
	consumer         apple.MTLCommandBuffer
	mu               *sync.Mutex
	locked, borrowed bool
}

type metalStreamingSelection interface {
	Ready() bool
	Discard() error
	Release(dsv4expertcache.Completion) error
	Slots() ([]int, error)
	Views(int) ([3]metalmem.Slice[byte], error)
}

// initMetalStreaming installs an owner before returning any initialization
// error. Caller attaches the returned owner before its error check/cleanup.
func initMetalStreaming(dev *libmetal.Device, source *dsv4expertcache.Source, cache *dsv4expertcache.Cache, strides [3]uint64, slots int, budget uint64, experts int) (*metalStreamingWeights, error) {
	w := &metalStreamingWeights{source: source, cache: cache, strides: strides, capacity: slots, mu: new(sync.Mutex)}
	if dev == nil || source == nil || cache == nil || experts <= 0 || experts > 384 {
		return w, errors.New("invalid Metal streaming owner")
	}
	var sizes [3]int
	for i, n := range strides {
		if n == 0 || n > uint64(^uint(0)>>1) {
			return w, errors.New("Metal streaming stride overflow")
		}
		sizes[i] = int(n)
	}
	var err error
	w.upload, err = dsv4expertcache.NewMetalUpload(dev, slots, sizes, budget)
	if err != nil {
		return w, err
	}
	for i := range w.tables {
		w.tables[i], err = metalmem.Alloc[uint64](dev, experts)
		if err != nil {
			return w, err
		}
	}
	sourceText, err := metalStreamingMSL()
	if err != nil {
		return w, err
	}
	w.library, err = dev.CompileLibrary(sourceText)
	if err != nil {
		return w, err
	}
	w.gate, err = dev.NewComputePipeline(w.library, "matvec_iq2_xxs_pair_selected_addresses")
	if err != nil {
		return w, err
	}
	w.down, err = dev.NewComputePipeline(w.library, "matvec_q2_k_selected_addresses")
	if err != nil {
		return w, err
	}
	return w, nil
}

func metalStreamingMSL() (string, error) {
	s := metalPinnedMSL
	changes := [][2]string{
		{"kernel void matvec_iq2_xxs_pair_selected_f32(", "kernel void matvec_iq2_xxs_pair_selected_addresses("},
		{"kernel void matvec_q2_k_selected_f32(", "kernel void matvec_q2_k_selected_addresses("},
		{"const device uchar *selected_gate=gate+uint64_t(expert)*expert_stride;", "const device uchar *selected_gate=reinterpret_cast<device const uchar *>(gate[expert]);"},
		{"const device uchar *selected_up=up+uint64_t(expert)*expert_stride;", "const device uchar *selected_up=reinterpret_cast<device const uchar *>(up[expert]);"},
		{"device const uchar *selected=weights+uint64_t(expert)*expert_stride;", "device const uchar *selected=reinterpret_cast<device const uchar *>(weights[expert]);"},
	}
	for _, p := range changes {
		if strings.Count(s, p[0]) != 1 {
			return "", fmt.Errorf("Metal streaming source site %q", p[0])
		}
		s = strings.Replace(s, p[0], p[1], 1)
	}
	// Restrict buffer type changes to each addressed entry, preserving unrelated
	// full-tensor kernels in the same library.
	for _, p := range [][3]string{
		{"matvec_iq2_xxs_pair_selected_addresses(", "device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],", "device const ulong *gate [[buffer(0)]], device const ulong *up [[buffer(1)]],"},
		{"matvec_q2_k_selected_addresses(", "device const uchar *weights [[buffer(0)]],", "device const ulong *weights [[buffer(0)]],"},
	} {
		i := strings.Index(s, p[0])
		if i < 0 {
			return "", errors.New("missing Metal address entry")
		}
		j := strings.Index(s[i:], p[1])
		if j < 0 {
			return "", errors.New("missing Metal address parameter")
		}
		j += i
		s = s[:j] + strings.Replace(s[j:], p[1], p[2], 1)
	}
	return s, nil
}

// prepare fills whole-layer address tables. Cached experts use protected slab
// views; mixed-size triplets and overflow use the original whole-tensor mapping.
func (w *metalStreamingWeights) prepare(layer int, tensors [3]boundMetalTensor, ids []int32) (err error) {
	if w == nil || w.mu == nil {
		return errors.New("Metal streaming owner unavailable")
	}
	w.mu.Lock()
	w.locked = true
	defer func() {
		if err != nil {
			err = errors.Join(err, w.finish())
		}
	}()
	if w.upload == nil || w.selection != nil || w.consumer != nil || len(ids) == 0 {
		return errors.New("Metal streaming selection active or unavailable")
	}
	experts := len(w.tables[0].HostSlice())
	for _, t := range tensors {
		if len(t.dimensions) != 3 || int(t.dimensions[2]) != experts {
			return errors.New("Metal streaming tensor profile")
		}
	}
	mapped, err := metalStreamingMappedOnly(tensors[0].typeID, tensors[1].typeID, tensors[2].typeID)
	if err != nil {
		return err
	}
	if mapped {
		// The existing Q4 selected consumers retain and read their full mapped
		// tensors. They do not interpret IQ2/Q2 address tables or cache bytes.
		return w.upload.NoteMappedRoutes(w.source, layer, ids)
	}
	unique, _, err := dsv4expertcache.CompactMetalRoutes(ids, uint32(experts))
	if err != nil {
		return err
	}
	for _, t := range w.tables {
		clear(t.HostSlice())
	}
	uniform := true
	for i, t := range tensors {
		if uint64(t.data.Len())/uint64(experts) != w.strides[i] {
			uniform = false
		}
	}
	var cached []int
	if !uniform {
		if err := w.upload.NoteMappedRoutes(w.source, layer, ids); err != nil {
			return err
		}
	}
	if uniform {
		routes := make([]int, len(ids))
		for i, id := range ids {
			routes[i] = int(id)
		}
		// Publish before checking errors: a failed stage can retain ownership.
		w.selection, cached, err = w.upload.SelectRoutes(w.source, w.cache, layer, routes)
		if err != nil {
			var pin *dsv4expertcache.MetalPinFailure
			if !errors.As(err, &pin) {
				return err
			}
			if err := w.selection.Discard(); err != nil {
				return err
			}
			w.selection = nil
			cap, backoff := w.upload.PinBackoff(w.cache, pin)
			if backoff != nil {
				return backoff
			}
			if cap != 0 {
				w.capacity = cap
			}
			// The failed reservation is not consumed. This whole request uses
			// mapped weights; a later request may use the reduced pinned cache.
			cached = nil
		}
	}
	for _, id := range unique {
		var views [3]metalmem.Slice[byte]
		route := -1
		for i, n := range cached {
			if n == int(id) {
				route = i
				break
			}
		}
		if route >= 0 {
			views, err = w.selection.Views(route)
			if err != nil {
				return err
			}
		} else {
			for component, t := range tensors {
				one, e := t.Expert(id)
				if e != nil {
					return e
				}
				views[component] = one.data
			}
		}
		for component, v := range views {
			address := uint64(v.Buffer().GpuAddress())
			if address == 0 || uint64(v.Offset()) > ^uint64(0)-address {
				return errors.New("Metal expert GPU address unavailable")
			}
			w.tables[component].HostSlice()[id] = address + uint64(v.Offset())
		}
	}
	return nil
}

func metalStreamingMappedOnly(gate, up, down uint32) (bool, error) {
	format, err := ExpertFormatForTypes(gate, up, down)
	if err != nil {
		return false, err
	}
	switch format {
	case ExpertsQ4K:
		return true, nil
	case ExpertsIQ2Q2K:
		return false, nil
	default:
		return false, errors.New("Metal streaming quantization route unavailable")
	}
}

// noteToken shares the cache lock with route selection: aging rewrites the
// same counters used to select victims, including counters for other States.
func (w *metalStreamingWeights) noteToken(layer int) error {
	if w == nil || w.mu == nil {
		return errors.New("Metal streaming owner unavailable")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.upload == nil {
		return errors.New("Metal streaming upload unavailable")
	}
	return w.upload.NoteToken(layer)
}

func (w *metalStreamingWeights) resources(enc apple.MTLComputeCommandEncoder, tensors [3]boundMetalTensor, ids []int32) error {
	for _, t := range tensors {
		enc.UseResourceUsage(t.data.Buffer(), apple.MTLResourceUsageRead)
	}
	if w.selection != nil {
		slots, err := w.selection.Slots()
		if err != nil {
			return err
		}
		for i := range slots {
			views, err := w.selection.Views(i)
			if err != nil {
				return err
			}
			for _, v := range views {
				enc.UseResourceUsage(v.Buffer(), apple.MTLResourceUsageRead)
			}
		}
	}
	return nil
}

func (w *metalStreamingWeights) encodeGate(b *metalBatch, tensors [3]boundMetalTensor, ids metalmem.Slice[int32], in, gate, up metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if ids.Len() != 6 || in.Len() != int(tensors[0].dimensions[0]) || gate.Len() != 6*int(tensors[0].dimensions[1]) || up.Len() != gate.Len() {
		return errors.New("Metal streaming gate shape")
	}
	enc := b.encoder()
	if err := w.resources(enc, tensors, ids.HostSlice()); err != nil {
		return err
	}
	enc.SetComputePipelineState(w.gate)
	for i := range 2 {
		enc.SetBufferWithOffsetAtIndex(w.tables[i].Buffer(), w.tables[i].Offset(), uint(i))
	}
	for i, v := range []metalmem.Slice[float32]{in, gate, up} {
		enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i+3))
	}
	enc.SetBufferWithOffsetAtIndex(ids.Buffer(), ids.Offset(), 2)
	for i, n := range []uint32{uint32(in.Len()), uint32(tensors[0].dimensions[1]), uint32(w.tables[0].Len()), uint32(ids.Len())} {
		libmetal.SetScalar(enc, n, i+6)
	}
	enc.DispatchThreadgroupsThreadsPerThreadgroup(apple.MTLSize{Width: uint(tensors[0].dimensions[1]), Height: 6, Depth: 1}, libmetal.Threads(32))
	b.encoded++
	return nil
}

func (w *metalStreamingWeights) encodeDown(b *metalBatch, t boundMetalTensor, ids metalmem.Slice[int32], in, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if ids.Len() != 6 || in.Len() != 6*int(t.dimensions[0]) || out.Len() != 6*int(t.dimensions[1]) {
		return errors.New("Metal streaming down shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(w.down)
	enc.SetBufferWithOffsetAtIndex(w.tables[2].Buffer(), w.tables[2].Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(ids.Buffer(), ids.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	for i, n := range []uint32{uint32(t.dimensions[0]), uint32(t.dimensions[1]), uint32(w.tables[2].Len()), 6} {
		libmetal.SetScalar(enc, n, i+4)
	}
	enc.DispatchThreadsThreadsPerThreadgroup(apple.MTLSize{Width: uint(t.dimensions[1]), Height: 6, Depth: 1}, libmetal.Threads(min(int(t.dimensions[1]), 256)))
	b.encoded++
	return nil
}

func (w *metalStreamingWeights) finish() error {
	if w == nil {
		return nil
	}
	if w.selection != nil {
		if !w.selection.Ready() {
			if err := w.selection.Discard(); err != nil {
				return err
			}
		} else {
			var completion dsv4expertcache.Completion = metalNoConsumer{}
			if w.consumer != nil {
				completion = dsv4expertcache.MetalCommandCompletion{Context: context.Background(), Buffer: w.consumer}
			}
			if err := w.selection.Release(completion); err != nil {
				return err
			}
		}
		w.selection = nil
	} else if w.consumer != nil {
		if err := libmetal.WaitCommandBuffer(context.Background(), w.consumer); err != nil {
			return err
		}
	}
	w.consumer = nil
	if w.locked {
		w.mu.Unlock()
		w.locked = false
	}
	return nil
}

// No command reading experts was submitted before the consumer is published.
type metalNoConsumer struct{}

func (metalNoConsumer) Wait() error { return nil }

func (w *metalStreamingWeights) close() error {
	if w == nil {
		return nil
	}
	if err := w.finish(); err != nil {
		return err
	}
	if w.borrowed {
		return nil
	}
	if w.mu != nil {
		if !w.mu.TryLock() {
			return errors.New("Metal streaming request active")
		}
		defer w.mu.Unlock()
	}
	if w.upload != nil {
		if err := w.upload.Close(); err != nil {
			return err
		}
		w.upload = nil
	}
	if w.cache != nil {
		if err := w.cache.Close(); err != nil {
			return err
		}
		w.cache = nil
	}
	for i, t := range w.tables {
		t.Free()
		w.tables[i] = metalmem.Slice[uint64]{}
	}
	for _, p := range []apple.MTLComputePipelineState{w.gate, w.down} {
		if p != nil {
			objectivec.ObjectFromID(p.GetID()).Release()
		}
	}
	w.gate, w.down = nil, nil
	if w.library != nil {
		objectivec.ObjectFromID(w.library.GetID()).Release()
		w.library = nil
	}
	return nil
}

func (w *metalStreamingWeights) request() *metalStreamingWeights {
	if w == nil {
		return nil
	}
	r := *w
	r.selection, r.consumer = nil, nil
	r.locked, r.borrowed = false, true
	return &r
}
