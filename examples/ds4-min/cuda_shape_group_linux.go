//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_shape_group.cu
var cudaShapeGroupSource string

// cudaShapeGrouping owns its five output buffers and compiled module, never the
// foundation or ids. It is a standalone gated primitive, not Pro admission.
type cudaShapeGrouping struct {
	r                                      *cudaFoundation
	plan                                   shapeGroupingPlan
	module                                 driver.Module
	fn                                     driver.Function
	offsets, pairs, active, count, invalid cudamem.Slice[int32]
	closed                                 bool
	ops                                    shapeGroupingOperations
}

type shapeGroupingOperations struct {
	alloc         func(int) (cudamem.Slice[int32], error)
	load          func() (driver.Module, driver.Function, error)
	run, closeRun func(func() error) error
	drain         func() error
	free          func(*cudamem.Slice[int32]) error
	unload        func(driver.Module) error
}

func shapeGroupingDefaults(r *cudaFoundation) shapeGroupingOperations {
	return shapeGroupingOperations{
		alloc: func(n int) (cudamem.Slice[int32], error) { return cudamem.Alloc[int32](r.device, n) },
		load: func() (driver.Module, driver.Function, error) {
			program, err := nvrtcx.Compile(cudaShapeGroupSource, nvrtcx.WithName("cuda_shape_group.cu"), nvrtcx.WithDeviceArch(r.device))
			if err != nil {
				return 0, 0, err
			}
			module, _, err := program.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
			if err != nil {
				return module, 0, err
			}
			var fn driver.Function
			err = r.withContext(func() error { var err error; fn, err = module.Function("ds4_shape_group_csr"); return err })
			return module, fn, err
		},
		run: r.withContext, closeRun: r.device.WithContext,
		drain:  func() error { return syncStream(r.stream()) },
		free:   func(v *cudamem.Slice[int32]) error { return v.Free() },
		unload: func(m driver.Module) error { return m.Close() },
	}
}

func newCUDAShapeGrouping(r *cudaFoundation, c dsv4shape.Config, tokens int) (*cudaShapeGrouping, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape grouping: missing foundation")
	}
	return newCUDAShapeGroupingWithOperations(r, c, tokens, shapeGroupingDefaults(r))
}

func newCUDAShapeGroupingWithOperations(r *cudaFoundation, c dsv4shape.Config, tokens int, ops shapeGroupingOperations) (*cudaShapeGrouping, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	p, err := planShapeGrouping(c, tokens)
	if err != nil {
		return nil, err
	}
	if r == nil || r.device == nil {
		return nil, errors.New("cuda shape grouping: missing foundation")
	}
	s := &cudaShapeGrouping{r: r, plan: p, ops: ops}
	fail := func(err error) (*cudaShapeGrouping, error) {
		return nil, finishBackendCleanup(err, s)
	}
	for _, a := range []struct {
		dst *cudamem.Slice[int32]
		n   int
	}{{&s.offsets, p.offsets}, {&s.pairs, p.pairs}, {&s.active, p.active}, {&s.count, 1}, {&s.invalid, 1}} {
		v, err := ops.alloc(a.n)
		*a.dst = v
		if err != nil {
			return fail(fmt.Errorf("cuda shape grouping: allocate: %w", err))
		}
		if v.Ptr() == 0 || v.Len() != a.n {
			return fail(errors.New("cuda shape grouping: invalid allocation"))
		}
	}
	s.module, s.fn, err = ops.load()
	if err != nil {
		return fail(err)
	}
	if s.module == 0 || s.fn == 0 {
		return fail(errors.New("cuda shape grouping: missing module or function"))
	}
	return s, nil
}

// EncodeAndValidate groups the ids and drains the owning stream to reject
// invalid ids before any expert consumer. The original ids remain caller-owned.
func (s *cudaShapeGrouping) EncodeAndValidate(ids cudamem.Slice[int32]) error {
	if s == nil || s.closed || s.module == 0 || s.fn == 0 || ids.Len() != s.plan.pairs || ids.Ptr() == 0 {
		return errors.New("cuda shape grouping: invalid or closed operands")
	}
	if err := s.r.checkOpen(); err != nil {
		return err
	}
	ranges := []mxfp4DeviceRange{{uint64(ids.Ptr()), uint64(ids.Len()) * 4}}
	for _, v := range []cudamem.Slice[int32]{s.offsets, s.pairs, s.active, s.count, s.invalid} {
		ranges = append(ranges, mxfp4DeviceRange{uint64(v.Ptr()), uint64(v.Len()) * 4})
	}
	if err := checkMXFP4DeviceRanges(ranges...); err != nil {
		return err
	}
	err := s.ops.run(func() error {
		return cudalaunch.Kernel(s.fn).Grid(1, 1, 1).Block(1, 1, 1).On(s.r.stream()).Launch(ids, s.offsets, s.pairs, s.active, s.count, s.invalid, uint32(s.plan.pairs), uint32(s.plan.experts))
	})
	if err != nil {
		s.closed = true
		return err
	}
	if err := s.ops.run(s.ops.drain); err != nil {
		s.closed = true
		return err
	}
	invalid := make([]int32, 1)
	err = s.invalid.CopyToHost(invalid)
	if err != nil {
		s.closed = true
		return err
	}
	if len(invalid) != 1 || invalid[0] != 0 {
		return fmt.Errorf("cuda shape grouping: invalid ids %v", invalid)
	}
	return nil
}

func (s *cudaShapeGrouping) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.module == 0 && s.offsets.Ptr() == 0 && s.pairs.Ptr() == 0 && s.active.Ptr() == 0 && s.count.Ptr() == 0 && s.invalid.Ptr() == 0 {
		return nil
	}
	if s.r == nil || s.r.device == nil {
		return errors.New("cuda shape grouping: missing owner")
	}
	if err := s.ops.closeRun(s.ops.drain); err != nil {
		return err
	}
	var errs []error
	for _, p := range []*cudamem.Slice[int32]{&s.offsets, &s.pairs, &s.active, &s.count, &s.invalid} {
		if p.Ptr() == 0 {
			continue
		}
		if err := s.ops.free(p); err != nil {
			errs = append(errs, err)
		} else {
			*p = cudamem.Slice[int32]{}
		}
	}
	if s.module != 0 {
		if err := mxfp4UnloadInContext(s.ops.closeRun, func() error { return s.ops.unload(s.module) }, func() { s.module = 0; s.fn = 0 }); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
