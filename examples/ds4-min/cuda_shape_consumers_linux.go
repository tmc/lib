//go:build linux

package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// cudaShapeConsumers owns only its modules and extra scratch. The foundation,
// forward/cache/HC state and every uploaded model weight remain borrowed.
// Constructor cleanup failures retain this owner through finishBackendCleanup.
type cudaShapeConsumers struct {
	r       *cudaFoundation
	config  dsv4shape.Config
	context uint64
	buffers map[string]cudamem.Slice[byte]
	modules map[string]driver.Module
	entries map[string]driver.Function
	ops     shapeConsumerOperations
	closed  bool
}

type shapeConsumerOperations struct {
	available func() (uint64, error)
	alloc     func(int) (cudamem.Slice[byte], error)
	load      func(string, string) (driver.Module, driver.Function, error)
	drain     func() error
	free      func(*cudamem.Slice[byte]) error
	closeRun  func(func() error) error
	unload    func(driver.Module) error
}

func newCUDAShapeConsumers(r *cudaFoundation, c dsv4shape.Config, context, reserve uint64) (*cudaShapeConsumers, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	if r.device == nil {
		return nil, errors.New("cuda shape consumers: missing device")
	}
	ops := shapeConsumerOperations{
		available: func() (uint64, error) {
			var free uint
			err := r.withContext(func() error { var err error; free, _, err = driver.MemGetInfo(); return err })
			return uint64(free), err
		},
		alloc: func(n int) (cudamem.Slice[byte], error) { return cudamem.Alloc[byte](r.device, n) },
		load: func(source, entry string) (driver.Module, driver.Function, error) {
			p, err := nvrtcx.Compile(source, nvrtcx.WithName(entry+".cu"), nvrtcx.WithDeviceArch(r.device))
			if err != nil {
				return 0, 0, err
			}
			m, _, err := p.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
			if err != nil {
				return m, 0, err
			}
			var f driver.Function
			err = r.withContext(func() error { var err error; f, err = m.Function(entry); return err })
			return m, f, err
		},
		drain: func() error { return r.device.WithContext(func() error { return syncStream(r.stream()) }) },
		free:  func(v *cudamem.Slice[byte]) error { return v.Free() }, closeRun: r.device.WithContext, unload: func(m driver.Module) error { return m.Close() },
	}
	return newCUDAShapeConsumersWithOperations(r, c, context, reserve, ops)
}

func newCUDAShapeConsumersWithOperations(r *cudaFoundation, c dsv4shape.Config, context, reserve uint64, ops shapeConsumerOperations) (*cudaShapeConsumers, error) {
	if err := r.checkOpen(); err != nil {
		return nil, err
	}
	plan, bytes, err := planShapeConsumerBuffers(c, context)
	if err != nil {
		return nil, err
	}
	available, err := ops.available()
	if err != nil {
		return nil, err
	}
	if reserve > available || bytes > available-reserve {
		return nil, errors.New("cuda shape consumers: scratch exceeds available memory")
	}
	s := &cudaShapeConsumers{r: r, config: c, context: context, ops: ops, buffers: make(map[string]cudamem.Slice[byte]), modules: make(map[string]driver.Module), entries: make(map[string]driver.Function)}
	fail := func(err error) (*cudaShapeConsumers, error) { return nil, finishBackendCleanup(err, s) }
	for _, b := range plan {
		v, err := ops.alloc(b.bytes)
		s.buffers[b.name] = v
		if err != nil {
			return fail(fmt.Errorf("allocate %s: %w", b.name, err))
		}
		if v.Ptr() == 0 || v.Len() != b.bytes {
			return fail(fmt.Errorf("cuda shape consumers: invalid allocation %s", b.name))
		}
	}
	for _, v := range []struct{ source, entry string }{{cudaShapeAttentionExternalSource, "ds4_shape_attention_decode_external"}, {cudaShapeRouterSource, "ds4_shape_router_select"}, {cudaShapeRopeSource, "ds4_shape_rope_tail"}, {cudaShapeOutputSource, "ds4_shape_output_hc_weights"}, {cudaShapeRawCommitSource, "ds4_shape_raw_commit"}, {cudaShapeIndexerSource, "ds4_shape_indexer_select"}} {
		m, f, err := ops.load(v.source, v.entry)
		s.modules[v.entry], s.entries[v.entry] = m, f
		if err != nil {
			return fail(fmt.Errorf("load %s: %w", v.entry, err))
		}
		if m == 0 || f == 0 {
			return fail(fmt.Errorf("cuda shape consumers: missing entry %s", v.entry))
		}
	}
	return s, nil
}

func (s *cudaShapeConsumers) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	owned := false
	for _, v := range s.buffers {
		owned = owned || v.Ptr() != 0
	}
	for _, m := range s.modules {
		owned = owned || m != 0
	}
	if !owned {
		return nil
	}
	if err := s.ops.drain(); err != nil {
		return err
	}
	var errs []error
	for name, v := range s.buffers {
		if v.Ptr() == 0 {
			continue
		}
		if err := s.ops.free(&v); err != nil {
			s.buffers[name] = v
			errs = append(errs, err)
		} else {
			s.buffers[name] = cudamem.Slice[byte]{}
		}
	}
	for name, m := range s.modules {
		if m == 0 {
			continue
		}
		errs = append(errs, mxfp4UnloadInContext(s.ops.closeRun, func() error { return s.ops.unload(m) }, func() { s.modules[name] = 0; s.entries[name] = 0 }))
	}
	return errors.Join(errs...)
}
