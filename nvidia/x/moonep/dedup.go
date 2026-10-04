//go:build !windows

package moonep

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// DedupSpec describes checked host fixture metadata for one local FP32 shard.
// Groups are {primary slot, offset into DupLoffs, duplicate count}; only Counts
// prefixes are valid. RouteSlots and ExpertIDs are token-major top-k entries.
// Padding contains half-open slot ranges covering exactly the unused slots.
// SourceID identifies the routing source, not mutable payload values. RankMap
// must be {0}; remote transport and external CuTe cubins are unsupported.
// BufferGeneration must be nonzero; communicator generation must be zero.
// Keep this value and its slices unchanged while the admitted operation lives.
type DedupSpec struct {
	Tokens, Width, K, Slots                  int
	Groups                                   [][3]int32
	DupLoffs                                 []int32
	Counts                                   [2]int32
	RouteSlots, ExpertIDs                    []int32
	Padding                                  [][2]int32
	SourceID                                 string
	RankMap                                  []int32
	BufferGeneration, CommunicatorGeneration uint64
}

type dedupMetadata struct{ primary, token, expert, dst []int32 }

func dedupAdmit(s DedupSpec) (dedupMetadata, error) {
	bad := func(why string) (dedupMetadata, error) { return dedupMetadata{}, fmt.Errorf("moonep: dedup %s", why) }
	if len(s.RankMap) != 1 || s.RankMap[0] != 0 || s.CommunicatorGeneration != 0 {
		return bad("unsupported remote or communicator mode")
	}
	if s.Tokens <= 0 || s.Width <= 0 || s.K <= 0 || s.K > 32 || s.Slots <= 0 || s.SourceID == "" || s.BufferGeneration == 0 {
		return bad("invalid shape or identity")
	}
	if s.Tokens > int32Max/s.K || s.Slots > (int32Max-2)/s.Width || s.Tokens > (int32Max-2)/s.Width {
		return bad("extent overflow")
	}
	n := s.Tokens * s.K
	if n > s.Slots || len(s.RouteSlots) != n || len(s.ExpertIDs) != n || len(s.Groups) > s.Slots || len(s.DupLoffs) > s.Slots {
		return bad("invalid capacity")
	}
	ng, nd := int(s.Counts[0]), int(s.Counts[1])
	if ng < 0 || nd < 0 || ng > len(s.Groups) || nd > len(s.DupLoffs) {
		return bad("invalid compact count")
	}
	m := dedupMetadata{primary: make([]int32, s.Slots), token: make([]int32, s.Slots), expert: make([]int32, s.Slots), dst: make([]int32, n)}
	for i := range m.primary {
		m.primary[i] = -1
		m.token[i] = -1
	}
	for i, slot := range s.RouteSlots {
		if slot < 0 || int(slot) >= s.Slots || m.primary[slot] != -1 || s.ExpertIDs[i] < 0 || s.ExpertIDs[i] >= int32Max {
			return bad("invalid or repeated route slot")
		}
		m.primary[slot] = slot
		m.token[slot] = int32(i / s.K)
		m.expert[slot] = s.ExpertIDs[i]
	}
	used := make([]bool, s.Slots)
	loffs := make([]bool, nd)
	for _, g := range s.Groups[:ng] {
		p, o, c := int(g[0]), int(g[1]), int(g[2])
		if p < 0 || p >= s.Slots || m.primary[p] < 0 || used[p] || o < 0 || c <= 0 || o > nd || c > nd-o {
			return bad("invalid group")
		}
		used[p] = true
		for j := o; j < o+c; j++ {
			d := int(s.DupLoffs[j])
			if loffs[j] || d < 0 || d >= s.Slots || d == p || used[d] || m.primary[d] < 0 || m.token[d] != m.token[p] {
				return bad("overlapping reduction group or destination")
			}
			loffs[j] = true
			used[d] = true
			m.primary[d] = int32(p)
		}
	}
	for _, v := range loffs {
		if !v {
			return bad("unconsumed duplicate prefix")
		}
	}
	pad := make([]bool, s.Slots)
	for _, r := range s.Padding {
		a, b := int(r[0]), int(r[1])
		if a < 0 || b <= a || b > s.Slots {
			return bad("invalid padding")
		}
		for i := a; i < b; i++ {
			if pad[i] || m.primary[i] >= 0 {
				return bad("overlapping padding")
			}
			pad[i] = true
		}
	}
	for i, p := range m.primary {
		if (p < 0) != pad[i] {
			return bad("padding does not cover unused slots")
		}
	}
	for i, slot := range s.RouteSlots {
		p := m.primary[slot]
		m.dst[i] = slot
		if p != slot {
			m.dst[i] = -p - 1
		}
	}
	return m, nil
}

func dedupDigest(s DedupSpec) [32]byte { b, _ := json.Marshal(s); return sha256.Sum256(b) }

type dedupPlanSnapshot struct {
	ptrs  [8]driver.DevicePtr
	shape [6]int
}

func dedupSnapshot(p *CommPlan) dedupPlanSnapshot {
	return dedupPlanSnapshot{[8]driver.DevicePtr{p.Dst, p.CuSeqlens, p.ExpertsToCopy, p.ZeroFillRanges, p.RemoteStats, p.DupGroups, p.DupLoffs, p.DupCounts}, [6]int{p.N, p.R, p.E, p.B, p.NvS, p.K}}
}

func dedupAcquire(p *CommPlan, want dedupPlanSnapshot) (func(), error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || p.active != 0 || dedupSnapshot(p) != want {
		return nil, errors.New("moonep: dedup plan closed, changed or incomplete")
	}
	p.active++
	var once sync.Once
	return func() { once.Do(func() { p.mu.Lock(); p.active--; p.mu.Unlock() }) }, nil
}

// Dedup binds local fixture metadata to an existing CommPlan and device context.
// It permits one incomplete invocation per plan. It owns no transport buffers.
// The zero value is unusable; construct with NewDedup.
type Dedup struct {
	mu           sync.Mutex
	dev          *cuda.Device
	plan         *CommPlan
	spec         *DedupSpec
	digest       [32]byte
	snapshot     dedupPlanSnapshot
	metadata     dedupMetadata
	planIDs      [8]uint64
	inputIDs     [2]uint64
	inputPtrs    [2]driver.DevicePtr
	bound        bool
	planContents [4][]int32
}

func dedupPointerID(dev *cuda.Device, p driver.DevicePtr) (uint64, error) {
	var id uint64
	err := dev.WithContext(func() error {
		current, e := driver.CtxGetCurrent()
		if e != nil {
			return e
		}
		var owner driver.Context
		if e = p.PointerGetAttribute(unsafe.Pointer(&owner), driver.PointerAttributeContext); e != nil {
			return e
		}
		if current != owner {
			return errors.New("moonep: dedup pointer context mismatch")
		}
		return p.PointerGetAttribute(unsafe.Pointer(&id), driver.PointerAttributeBufferId)
	})
	return id, err
}

func dedupPointerExtent(dev *cuda.Device, p driver.DevicePtr, n int) error {
	if n <= 0 || n > int32Max/4 {
		return errors.New("moonep: dedup invalid plan byte extent")
	}
	return dev.WithContext(func() error {
		var start driver.DevicePtr
		var size uint64
		if e := p.PointerGetAttribute(unsafe.Pointer(&start), driver.PointerAttributeRangeStartAddr); e != nil {
			return e
		}
		if e := p.PointerGetAttribute(unsafe.Pointer(&size), driver.PointerAttributeRangeSize); e != nil {
			return e
		}
		if p < start || uint64(p-start) > size || uint64(n)*4 > size-uint64(p-start) {
			return errors.New("moonep: dedup plan allocation too short")
		}
		return nil
	})
}

func (d *Dedup) checkPlan() error {
	for i, p := range d.snapshot.ptrs {
		id, e := dedupPointerID(d.dev, p)
		if e != nil {
			return e
		}
		if id != d.planIDs[i] {
			return errors.New("moonep: dedup plan allocation generation changed")
		}
	}
	read := func(p driver.DevicePtr, n int) ([]int32, error) {
		xs := make([]int32, n)
		if n == 0 {
			return xs, nil
		}
		e := d.dev.WithContext(func() error { return p.CopyToHostPtr(unsafe.Pointer(&xs[0]), uint(n*4)) })
		return xs, e
	}
	counts, e := read(d.plan.DupCounts, 2)
	if e != nil {
		return e
	}
	if counts[0] != d.spec.Counts[0] || counts[1] != d.spec.Counts[1] {
		return errors.New("moonep: dedup device counts changed")
	}
	groups, e := read(d.plan.DupGroups, int(counts[0])*3)
	if e != nil {
		return e
	}
	for i, v := range groups {
		if v != d.spec.Groups[i/3][i%3] {
			return errors.New("moonep: dedup device groups changed")
		}
	}
	loffs, e := read(d.plan.DupLoffs, int(counts[1]))
	if e != nil {
		return e
	}
	if len(loffs) != int(counts[1]) {
		return errors.New("moonep: dedup duplicate offsets changed")
	}
	for i, v := range loffs {
		if v != d.spec.DupLoffs[i] {
			return errors.New("moonep: dedup duplicate offsets changed")
		}
	}
	dst, e := read(d.plan.Dst, d.plan.N)
	if e != nil {
		return e
	}
	if !reflect.DeepEqual(dst, d.metadata.dst) {
		return errors.New("moonep: dedup source routes changed")
	}
	extra := []struct {
		ptr driver.DevicePtr
		n   int
	}{{d.plan.CuSeqlens, d.plan.E + d.plan.B}, {d.plan.ExpertsToCopy, d.plan.R * d.plan.B}, {d.plan.ZeroFillRanges, (d.plan.E + d.plan.B) * 2}, {d.plan.RemoteStats, 2}}
	for i, b := range extra {
		xs, e := read(b.ptr, b.n)
		if e != nil {
			return e
		}
		if i == 2 {
			if len(d.spec.Padding) > b.n/2 {
				return errors.New("moonep: dedup padding exceeds plan capacity")
			}
			for j := 0; j < b.n/2; j++ {
				var want [2]int32
				if j < len(d.spec.Padding) {
					want = d.spec.Padding[j]
				}
				if xs[j*2] != want[0] || xs[j*2+1] != want[1] {
					return errors.New("moonep: dedup device padding changed")
				}
			}
		}
		if d.planContents[i] == nil {
			d.planContents[i] = xs
		} else if !reflect.DeepEqual(xs, d.planContents[i]) {
			return errors.New("moonep: dedup planning metadata changed")
		}
	}
	return nil
}

// NewDedup admits a single-device local fixture using existing plan tensors.
// It checks the actual compact prefixes, routes, allocation generations and
// context. Host metadata cannot stand in for a populated device plan. Callers
// must not mutate plan tensors or exported fields concurrently with this API.
func NewDedup(dev *cuda.Device, p *CommPlan, s *DedupSpec) (*Dedup, error) {
	if dev == nil || p == nil || s == nil {
		return nil, errors.New("moonep: nil dedup device, plan or metadata")
	}
	m, e := dedupAdmit(*s)
	if e != nil {
		return nil, e
	}
	p.mu.Lock()
	snap := dedupSnapshot(p)
	p.mu.Unlock()
	release, e := dedupAcquire(p, snap)
	if e != nil {
		return nil, e
	}
	defer release()
	if snap.shape[0] != s.Tokens*s.K || snap.shape[1] != 1 || snap.shape[4] != s.Slots || snap.shape[5] != s.K {
		return nil, errors.New("moonep: dedup plan shape mismatch or unsupported remote mode")
	}
	if p.E <= 0 || p.B <= 0 || p.B > int32Max/8 || p.E > int32Max/8-p.B || s.Slots > int32Max/12 {
		return nil, errors.New("moonep: dedup invalid plan extents")
	}
	for _, expert := range s.ExpertIDs {
		if int(expert) >= p.E {
			return nil, errors.New("moonep: dedup expert outside plan")
		}
	}
	d := &Dedup{dev: dev, plan: p, spec: s, digest: dedupDigest(*s), snapshot: snap, metadata: m}
	lengths := [8]int{p.N, p.E + p.B, p.R * p.B, (p.E + p.B) * 2, 2, p.NvS * 3, p.NvS, 2}
	for i, ptr := range snap.ptrs {
		if e = dedupPointerExtent(dev, ptr, lengths[i]); e != nil {
			return nil, e
		}
		d.planIDs[i], e = dedupPointerID(dev, ptr)
		if e != nil {
			return nil, e
		}
	}
	if e = d.checkPlan(); e != nil {
		return nil, e
	}
	return d, nil
}

// DedupReceipt preserves the exact source and compiled PTX associated with a
// loaded stage. CompilerKey identifies the complete native compiler request.
type DedupReceipt struct{ Name, Source, PTX, CompilerKey, Target string }

type dedupBuffer struct {
	ptr  driver.DevicePtr
	read func() ([]float32, error)
	free func() error
}
type dedupLoaded struct {
	launch  func([]driver.DevicePtr, int) error
	close   func() error
	receipt DedupReceipt
}
type dedupOps struct {
	alloc   func(int) (*dedupBuffer, error)
	compile func(DedupKernel) (*dedupLoaded, error)
	wait    func(context.Context) error
}

// DedupInvocation retains plan and input borrows, scratch and modules through
// actual completion. A failed wait retains owners; cancellation is not
// completion. Close retries an uncancelled drain and may return unresolved work.
type DedupInvocation struct {
	mu                          sync.Mutex
	ops                         dedupOps
	buffers                     []*dedupBuffer
	modules                     []*dedupLoaded
	pending                     []io.Closer
	releases                    []func()
	receipts                    []DedupReceipt
	attempted, complete, closed bool
}

// DedupResult contains every stage, including token-major combined output.
type DedupResult struct{ Expanded, Expert, Reduced, Combined []float32 }

func (d *DedupInvocation) wait(ctx context.Context) error {
	if d.complete {
		return nil
	}
	if d.attempted {
		if e := d.ops.wait(ctx); e != nil {
			return e
		}
	}
	d.complete = true
	for _, r := range d.releases {
		r()
	}
	d.releases = nil
	return nil
}

// Wait observes completion before releasing plan and input borrows.
func (d *DedupInvocation) Wait(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return errors.New("moonep: dedup invocation closed")
	}
	return d.wait(ctx)
}

// Readback waits for completion and verifies guards, publication and finiteness
// in all four stage outputs. It does not perform a mathematical reference check.
func (d *DedupInvocation) Readback(ctx context.Context) (DedupResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var out DedupResult
	if d.closed {
		return out, errors.New("moonep: dedup invocation closed")
	}
	if e := d.wait(ctx); e != nil {
		return out, e
	}
	if len(d.buffers) != 4 {
		return out, errors.New("moonep: incomplete dedup outputs")
	}
	dsts := []*[]float32{&out.Expanded, &out.Expert, &out.Reduced, &out.Combined}
	for i, b := range d.buffers {
		xs, e := b.read()
		if e != nil {
			return out, e
		}
		*dsts[i] = xs
	}
	return out, nil
}

// Receipts returns copies of the exact compiled stage records.
func (d *DedupInvocation) Receipts() []DedupReceipt {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]DedupReceipt(nil), d.receipts...)
}

// Close drains incomplete work before releasing modules or allocations.
func (d *DedupInvocation) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if e := d.wait(nil); e != nil {
		return fmt.Errorf("moonep: dedup unresolved drain: %w", e)
	}
	var errs []error
	for _, b := range d.buffers {
		if b != nil && b.free != nil {
			if e := b.free(); e != nil {
				errs = append(errs, e)
			} else {
				b.free = nil
			}
		}
	}
	for _, m := range d.modules {
		if m != nil && m.close != nil {
			if e := m.close(); e != nil {
				errs = append(errs, e)
			} else {
				m.close = nil
			}
		}
	}
	for i, c := range d.pending {
		if c != nil {
			if e := c.Close(); e != nil {
				errs = append(errs, e)
			} else {
				d.pending[i] = nil
			}
		}
	}
	if len(errs) == 0 {
		d.closed = true
	}
	return errors.Join(errs...)
}

func dedupRun(ctx context.Context, s DedupSpec, input, weight driver.DevicePtr, ops dedupOps, releases []func()) (*DedupInvocation, error) {
	d := &DedupInvocation{ops: ops, releases: releases}
	fail := func(e error) (*DedupInvocation, error) { return d, errors.Join(e, d.Close()) }
	sources, e := DedupSources(s)
	if e != nil {
		return fail(e)
	}
	for i := 0; i < 4; i++ {
		n := s.Slots * s.Width
		if i == 3 {
			n = s.Tokens * s.Width
		}
		b, e := ops.alloc(n)
		if b != nil {
			d.buffers = append(d.buffers, b)
		}
		if e != nil {
			return fail(e)
		}
	}
	for i, k := range sources {
		if ctx != nil && ctx.Err() != nil {
			return fail(ctx.Err())
		}
		m, e := ops.compile(k)
		if m != nil {
			d.modules = append(d.modules, m)
		}
		if e != nil {
			var c io.Closer
			if errors.As(e, &c) {
				d.pending = append(d.pending, c)
			}
			return fail(e)
		}
		d.receipts = append(d.receipts, m.receipt)
		args := []driver.DevicePtr{input, d.buffers[i].ptr}
		if i == 1 {
			args = []driver.DevicePtr{d.buffers[0].ptr, weight, d.buffers[1].ptr}
		}
		if i > 1 {
			args = []driver.DevicePtr{d.buffers[i-1].ptr, d.buffers[i].ptr}
		}
		n := s.Slots * s.Width
		if i == 3 {
			n = s.Tokens * s.Width
		}
		d.attempted = true
		if e = m.launch(args, n); e != nil {
			return fail(e)
		}
	}
	if e = d.Wait(ctx); e != nil {
		return d, e
	}
	return d, nil
}

func dedupNativeOps(dev *cuda.Device, stream *cuda.Stream) dedupOps {
	return dedupOps{
		wait: stream.Synchronize,
		alloc: func(n int) (*dedupBuffer, error) {
			storage, e := cudamem.Alloc[float32](dev, n+2)
			if storage.Ptr() == 0 {
				return nil, e
			}
			b := &dedupBuffer{ptr: storage.Slice(1, n+1).Ptr(), free: storage.Free, read: func() ([]float32, error) {
				xs := make([]float32, n+2)
				if e := storage.CopyToHost(xs); e != nil {
					return nil, e
				}
				if math.Float32bits(xs[0]) != 0x7fc07901 || math.Float32bits(xs[n+1]) != 0x7fc07901 {
					return nil, errors.New("moonep: dedup output guard changed")
				}
				for _, x := range xs[1 : n+1] {
					if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
						return nil, errors.New("moonep: dedup output unwritten or nonfinite")
					}
				}
				return xs[1 : n+1], nil
			}}
			if e != nil {
				return b, e
			}
			poison := make([]float32, n+2)
			for i := range poison {
				poison[i] = math.Float32frombits(0x7fc07901)
			}
			return b, storage.CopyFromHost(poison)
		},
		compile: func(k DedupKernel) (*dedupLoaded, error) {
			major, minor, e := dev.ComputeCapability()
			if e != nil {
				return nil, e
			}
			cm, e := nvrtcx.Compile(k.Source, nvrtcx.WithArch(major, minor), nvrtcx.WithFlag("--fmad=false"))
			if e != nil {
				return nil, e
			}
			module, _, e := cm.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
			if module == nil {
				return nil, e
			}
			m := &dedupLoaded{close: module.Close, receipt: DedupReceipt{k.Name, k.Source, cm.PTX(), cm.Key(), fmt.Sprintf("compute_%d%d", major, minor)}}
			if e != nil {
				return m, e
			}
			fn, e := module.Function(k.Name)
			if e != nil {
				return m, e
			}
			m.launch = func(ptrs []driver.DevicePtr, n int) error {
				var p Params
				for _, ptr := range ptrs {
					p.AddPtr(ptr)
				}
				p.AddInt32(int32(n))
				return stream.WithHandle(func(raw driver.Stream) error {
					return fn.Launch([3]uint32{uint32((n + 255) / 256), 1, 1}, [3]uint32{256, 1, 1}, 0, raw, p.Ptrs())
				})
			}
			return m, nil
		},
	}
}

// Execute runs expand -> independently weighted affine experts -> reduce ->
// token combine. Payload has Slots*Width FP32 elements, with primary rows ready;
// weights has one independent FP32 value per local slot. Padding weights and
// primary payloads must be finite. Inputs must be ready and unchanged while
// borrowed. Reuse admits changed payload values in the same allocations only;
// it rejects changed metadata, context, generations or buffers before enqueue.
func (d *Dedup) Execute(ctx context.Context, stream *cuda.Stream, payload, weights cudamem.Slice[float32], generation uint64) (*DedupInvocation, error) {
	if d == nil || d.plan == nil {
		return nil, errors.New("moonep: unusable dedup operation")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if stream == nil || stream.Device() != d.dev || generation != d.spec.BufferGeneration || dedupDigest(*d.spec) != d.digest {
		return nil, errors.New("moonep: dedup changed shape, identity, generation or context")
	}
	if ctx != nil && ctx.Err() != nil {
		return nil, ctx.Err()
	}
	release, e := dedupAcquire(d.plan, d.snapshot)
	if e != nil {
		return nil, e
	}
	releases := []func(){release}
	rollback := func() {
		for _, r := range releases {
			r()
		}
	}
	if e = d.checkPlan(); e != nil {
		rollback()
		return nil, e
	}
	s := *d.spec
	if payload.Len() != s.Slots*s.Width || weights.Len() != s.Slots {
		rollback()
		return nil, errors.New("moonep: dedup wrong buffer shape")
	}
	ptrs := [2]driver.DevicePtr{payload.Ptr(), weights.Ptr()}
	sizes := [2]uint64{uint64(payload.SizeBytes()), uint64(weights.SizeBytes())}
	for i, p := range ptrs {
		if p == 0 || uint64(p)%4 != 0 || sizes[i] > math.MaxUint64-uint64(p) {
			rollback()
			return nil, errors.New("moonep: dedup invalid pointer extent")
		}
	}
	if uint64(ptrs[0]) < uint64(ptrs[1])+sizes[1] && uint64(ptrs[1]) < uint64(ptrs[0])+sizes[0] {
		rollback()
		return nil, errors.New("moonep: dedup aliased payload and weight")
	}
	var ids [2]uint64
	for i, b := range []cudamem.Slice[float32]{payload, weights} {
		r, e := b.Borrow(d.dev)
		if e != nil {
			rollback()
			return nil, e
		}
		releases = append(releases, r)
		ids[i], e = dedupPointerID(d.dev, b.Ptr())
		if e != nil {
			rollback()
			return nil, e
		}
	}
	if d.bound && (ptrs != d.inputPtrs || ids != d.inputIDs) {
		rollback()
		return nil, errors.New("moonep: dedup input allocation generation changed")
	}
	xs := make([]float32, payload.Len())
	if e = payload.CopyToHost(xs); e != nil {
		rollback()
		return nil, e
	}
	ws := make([]float32, weights.Len())
	if e = weights.CopyToHost(ws); e != nil {
		rollback()
		return nil, e
	}
	for row, p := range d.metadata.primary {
		if p == int32(row) {
			for _, x := range xs[row*s.Width : (row+1)*s.Width] {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					rollback()
					return nil, errors.New("moonep: dedup nonfinite primary payload")
				}
			}
		}
		if math.IsNaN(float64(ws[row])) || math.IsInf(float64(ws[row]), 0) {
			rollback()
			return nil, errors.New("moonep: dedup nonfinite route weight")
		}
	}
	d.inputPtrs, d.inputIDs, d.bound = ptrs, ids, true
	return dedupRun(ctx, s, ptrs[0], ptrs[1], dedupNativeOps(d.dev, stream), releases)
}
