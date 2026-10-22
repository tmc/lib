//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv41"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_v41_compressed.cu
var cudaV41CompressedSource string

// Preprojected compressor KV/score and indexer query/weight operands are the
// explicit seam. Projection producers outside this seam are not implemented.
// Every input is finite; query/weight projection outputs are already BF16.
type cudaV41CompressedInput struct{ PoolKV, PoolScore, Query, Weight cudamem.Slice[float32] }
type cudaV41CompressedWeights struct {
	Norm, IndexNorm cudamem.Slice[float32]
	Index           cudamem.Slice[uint16]
}

// cudaV41Compressed owns four persistent compressed/index caches, three pair
// carry states, and the selection/mask reused by downstream layers. Steps run
// in order layers2..39 for each token. It is not a model/family backend.
// Foundation and weights are immutable borrows through successful Close.
type cudaV41Compressed struct {
	mu                                             sync.Mutex
	r                                              *cudaFoundation
	w                                              [4]cudaV41CompressedWeights
	module                                         driver.Module
	copy, idcopy, norm, index, score, pick, gather driver.Function
	ops                                            *cudaV41Ops
	pool                                           [3]*cudaV41Pool
	cache, keys                                    [4]cudamem.Slice[float32]
	latent, key, query, frequency, scores, blocks  cudamem.Slice[float32]
	ids                                            cudamem.Slice[int32]
	mask, used                                     cudamem.Slice[uint32]
	pending                                        []cudamem.Slice[float32]
	pendingIDs                                     cudamem.Slice[int32]
	context, position, layer                       uint32
	valid, closing                                 bool
}

func compileCUDAV41Compressed(r *cudaFoundation, w [4]cudaV41CompressedWeights, context uint32, source string) (*cudaV41Compressed, error) {
	if r == nil || r.device == nil || context == 0 || context > 1048576 {
		return nil, errors.New("V4.1 compressed: invalid foundation/context")
	}
	for _, v := range w {
		if v.Norm.Len() != 512 || v.IndexNorm.Len() != 128 || v.Index.Len() != 512*128 || v.Norm.Ptr() == 0 || v.IndexNorm.Ptr() == 0 || v.Index.Ptr() == 0 {
			return nil, errors.New("V4.1 compressed: invalid weight shape")
		}
	}
	s := &cudaV41Compressed{r: r, w: w, context: context, layer: 2}
	p, err := compileCUDAV41Device(source, nvrtcx.WithName("cuda_v41_compressed.cu"), nvrtcx.WithDeviceArch(r.device))
	if err != nil {
		return nil, err
	}
	s.module, _, err = p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err == nil {
		err = r.device.WithContext(func() error {
			for _, v := range []struct {
				name string
				dst  *driver.Function
			}{{"copy", &s.copy}, {"ids", &s.idcopy}, {"norm", &s.norm}, {"index", &s.index}, {"scores", &s.score}, {"pick", &s.pick}, {"gather", &s.gather}} {
				f, e := s.module.Function("ds4_v41_compressed_" + v.name)
				if e != nil {
					return e
				}
				*v.dst = f
			}
			return nil
		})
	}
	if err == nil {
		s.ops, err = newCUDAV41Ops(r)
	}
	for i := range s.pool {
		if err == nil {
			s.pool[i], err = newCUDAV41Pool(r, 512)
		}
	}
	alloc := func(dst *cudamem.Slice[float32], n int) {
		if err == nil {
			*dst, err = cudamem.Alloc[float32](r.device, n)
		}
	}
	for i := range s.cache {
		n := context
		if i < 3 {
			n = max(1, context/2)
		}
		alloc(&s.cache[i], int(n)*512)
		alloc(&s.keys[i], int(n)*128)
	}
	alloc(&s.latent, 512)
	alloc(&s.key, 128)
	alloc(&s.query, 4096)
	alloc(&s.frequency, 32)
	alloc(&s.scores, int(context))
	alloc(&s.blocks, int((context+7)/8))
	if err == nil {
		s.ids, err = cudamem.Alloc[int32](r.device, 512)
	}
	if err == nil {
		s.mask, err = cudamem.Alloc[uint32](r.device, int((context+7)/8))
	}
	if err == nil {
		s.used, err = cudamem.Alloc[uint32](r.device, int(context))
	}
	if err == nil {
		err = s.frequency.CopyFromHost(cudaV41Frequencies(true))
	}
	if err == nil {
		err = s.reset()
	}
	if err != nil {
		if e := s.Close(); e != nil {
			return s, errors.Join(err, e)
		}
		return nil, err
	}
	return s, nil
}

func (s *cudaV41Compressed) drain() error {
	return s.r.device.WithContext(func() error { return syncStream(s.r.stream()) })
}
func (s *cudaV41Compressed) reset() error {
	s.valid = false
	if err := s.drain(); err != nil {
		return err
	}
	for _, p := range s.pool {
		if p != nil {
			if err := p.Reset(); err != nil {
				return err
			}
		}
	}
	for _, a := range s.cache {
		if err := s.copyValues(a, a, true); err != nil {
			return err
		}
	}
	for _, a := range s.keys {
		if err := s.copyValues(a, a, true); err != nil {
			return err
		}
	}
	if err := s.drain(); err != nil {
		return err
	}
	s.pending = nil
	s.pendingIDs = cudamem.Slice[int32]{}
	s.position = 0
	s.layer = 2
	s.valid = true
	return nil
}
func (s *cudaV41Compressed) Reset() error {
	if s == nil {
		return errors.New("V4.1 compressed: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.module == 0 {
		return errors.New("V4.1 compressed: closed")
	}
	return s.reset()
}
func (s *cudaV41Compressed) copyValues(dst, src cudamem.Slice[float32], clear bool) error {
	if dst.Len() == 0 {
		return nil
	}
	var z uint32
	if clear {
		z = 1
	}
	return s.r.device.WithContext(func() error {
		return cudalaunch.Kernel(s.copy).Grid(uint32((dst.Len()+255)/256), 1, 1).Block(256, 1, 1).On(s.r.stream()).Launch(dst, src, uint32(dst.Len()), z)
	})
}

// Step emits selected persisted KV rows in descending-score/lower-ID order.
// Non-refresh layers reuse the preceding IDs. A runtime failure invalidates
// this owner even if a child carry/cache changed; Reset is mandatory. Caller
// borrows must survive until successful drain/Reset/Close, not just Go reachability.
func (s *cudaV41Compressed) Step(out cudamem.Slice[float32], selected cudamem.Slice[int32], in cudaV41CompressedInput, layer, position uint32) error {
	if s == nil {
		return errors.New("V4.1 compressed: closed")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closing || s.module == 0 {
		return errors.New("V4.1 compressed: closed")
	}
	if !s.valid {
		return errors.New("V4.1 compressed: reset required")
	}
	l, e := dsv41.Schedule(int(layer))
	if e != nil || layer < 2 || layer != s.layer || position != s.position || position >= s.context {
		return errors.New("V4.1 compressed: invalid frontier")
	}
	n := (position + 1) / uint32(l.Ratio)
	top := min(n, 512)
	if out.Len() != int(top)*512 || selected.Len() != int(top) {
		return errors.New("V4.1 compressed: invalid output shape")
	}
	wantPool, wantScore, wantQ, wantW := 0, 0, 0, 0
	if l.PublishKV {
		wantPool = 512
		if l.Ratio == 2 {
			wantScore = 512
		}
	}
	if l.RefreshIndex && n > 0 {
		wantQ = 4096
		wantW = 32
	}
	if in.PoolKV.Len() != wantPool || in.PoolScore.Len() != wantScore || in.Query.Len() != wantQ || in.Weight.Len() != wantW {
		return errors.New("V4.1 compressed: invalid operand shape")
	}
	// Public inputs/outputs and all private mutable storage are disjoint.
	type extent struct{ p, n uint64 }
	var ranges []extent
	add := func(p uint64, n int, size uint64) {
		if n != 0 {
			ranges = append(ranges, extent{p, uint64(n) * size})
		}
	}
	for _, a := range []cudamem.Slice[float32]{out, in.PoolKV, in.PoolScore, in.Query, in.Weight, s.latent, s.key, s.query, s.frequency, s.scores, s.blocks} {
		add(uint64(a.Ptr()), a.Len(), 4)
	}
	add(uint64(selected.Ptr()), selected.Len(), 4)
	add(uint64(s.ids.Ptr()), s.ids.Len(), 4)
	add(uint64(s.mask.Ptr()), s.mask.Len(), 4)
	add(uint64(s.used.Ptr()), s.used.Len(), 4)
	for i := range s.cache {
		for _, a := range []cudamem.Slice[float32]{s.cache[i], s.keys[i], s.w[i].Norm, s.w[i].IndexNorm} {
			add(uint64(a.Ptr()), a.Len(), 4)
		}
		add(uint64(s.w[i].Index.Ptr()), s.w[i].Index.Len(), 2)
	}
	for _, p := range s.pool {
		for _, a := range []cudamem.Slice[float32]{p.kv, p.scores} {
			add(uint64(a.Ptr()), a.Len(), 4)
		}
	}
	for i, a := range ranges {
		if a.p == 0 || a.p > math.MaxUint64-a.n {
			return errors.New("V4.1 compressed: invalid device range")
		}
		for _, b := range ranges[:i] {
			if a.p < b.p+b.n && b.p < a.p+a.n {
				return errors.New("V4.1 compressed: overlapping operands")
			}
		}
	}
	s.valid = false
	s.pending = []cudamem.Slice[float32]{out, in.PoolKV, in.PoolScore, in.Query, in.Weight}
	s.pendingIDs = selected
	run := func(fn driver.Function, grid, block uint32, args ...any) {
		if e == nil {
			e = s.r.device.WithContext(func() error {
				return cudalaunch.Kernel(fn).Grid(grid, 1, 1).Block(block, 1, 1).On(s.r.stream()).Launch(args...)
			})
		}
	}
	owner := l.Owner
	if l.PublishKV {
		if l.Ratio == 2 {
			var dst cudamem.Slice[float32]
			if n*2 == position+1 {
				dst = s.latent
			}
			e = s.pool[owner].Append(dst, in.PoolKV, in.PoolScore, position)
		} else {
			e = s.copyValues(s.latent, in.PoolKV, false)
			if e == nil {
				e = s.ops.Quantize(s.latent, 512, 1, dsv41.BFloat16)
			}
		}
		if e == nil && (position+1)%uint32(l.Ratio) == 0 {
			run(s.norm, 1, 1, s.latent, s.w[owner].Norm, uint32(512))
			run(s.index, 4, 32, s.key, s.latent, s.w[owner].Index)
			run(s.norm, 1, 1, s.key, s.w[owner].IndexNorm, uint32(128))
			ropePos := position + 1 - uint32(l.Ratio)
			if e == nil {
				e = s.ops.Rope(s.key, s.frequency, 128, 1, 1, ropePos, 1, false)
			}
			if e == nil {
				e = s.ops.Quantize(s.key, 128, 1, dsv41.FP4E8M0)
			}
			if e == nil {
				e = s.ops.Rope(s.latent, s.frequency, 512, 1, 1, ropePos, 1, false)
			}
			if e == nil {
				e = s.ops.Quantize(s.latent, 512, 1, dsv41.FP4E4M3)
			}
			if e == nil {
				e = s.copyValues(s.cache[owner].Slice(int(n-1)*512, int(n)*512), s.latent, false)
			}
			if e == nil {
				e = s.copyValues(s.keys[owner].Slice(int(n-1)*128, int(n)*128), s.key, false)
			}
		}
	}
	if e == nil && n > 0 && l.RefreshIndex {
		e = s.copyValues(s.query, in.Query, false)
		if e == nil {
			e = s.ops.Rope(s.query, s.frequency, 128, 32, 1, position, 1, false)
		}
		if e == nil {
			e = s.ops.Quantize(s.query, 128, 32, dsv41.FP4E8M0)
		}
		run(s.score, (n+7)/8, 256, s.scores, s.query, in.Weight, s.keys[owner], n, position, uint32(l.Ratio))
		run(s.pick, 1, 1, s.ids, s.scores, s.mask, s.blocks, s.used, n, layer)
	}
	if e == nil && top > 0 {
		run(s.gather, top, 256, out, s.cache[owner], s.ids, n, top)
		run(s.idcopy, (top+255)/256, 256, selected, s.ids, top)
	}
	d := s.drain()
	if d == nil {
		s.pending = nil
		s.pendingIDs = cudamem.Slice[int32]{}
	}
	if e != nil || d != nil {
		return fmt.Errorf("V4.1 compressed step: %w", errors.Join(e, d))
	}
	s.layer++
	if s.layer == 40 {
		s.layer = 2
		s.position++
	}
	s.valid = true
	return nil
}

func (s *cudaV41Compressed) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closing = true
	s.valid = false
	if s.module == 0 && s.ops == nil && s.pool[0] == nil && s.pool[1] == nil && s.pool[2] == nil {
		return nil
	}
	if e := s.drain(); e != nil {
		return e
	}
	s.pending = nil
	s.pendingIDs = cudamem.Slice[int32]{}
	for i, p := range s.pool {
		if p != nil {
			if e := p.Close(); e != nil {
				return e
			}
			s.pool[i] = nil
		}
	}
	if s.ops != nil {
		if e := s.ops.Close(); e != nil {
			return e
		}
		s.ops = nil
	}
	var failures []error
	for _, a := range []*cudamem.Slice[float32]{&s.cache[0], &s.cache[1], &s.cache[2], &s.cache[3], &s.keys[0], &s.keys[1], &s.keys[2], &s.keys[3], &s.latent, &s.key, &s.query, &s.frequency, &s.scores, &s.blocks} {
		if e := a.Free(); e != nil {
			failures = append(failures, e)
		} else {
			*a = cudamem.Slice[float32]{}
		}
	}
	if e := s.ids.Free(); e != nil {
		failures = append(failures, e)
	} else {
		s.ids = cudamem.Slice[int32]{}
	}
	for _, a := range []*cudamem.Slice[uint32]{&s.mask, &s.used} {
		if e := a.Free(); e != nil {
			failures = append(failures, e)
		} else {
			*a = cudamem.Slice[uint32]{}
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	if s.module != 0 {
		if e := s.r.device.WithContext(s.module.Close); e != nil {
			return e
		}
		s.module = 0
	}
	return nil
}
