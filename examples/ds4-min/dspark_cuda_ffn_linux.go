//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_dspark_ffn.cu
var cudaDSparkFFNSource string

// cudaDSparkFFN is request-private, single-stream scratch. Its owner must
// synchronize before close and must keep resident support weights alive.
// Partial output after an enqueue failure is never a published draft.
type cudaDSparkFFN struct {
	device                                               *cuda.Device
	floats                                               cudamem.Slice[float32]
	bytes                                                cudamem.Slice[byte]
	ids                                                  cudamem.Slice[int32]
	pre, logits, scores, weights                         cudamem.Slice[float32]
	gate, up, down, sharedGate, sharedUp, sharedOut, sum cudamem.Slice[float32]
	qInput, qExperts, qShared                            cudamem.Slice[byte]
	norms, sharedMid                                     cudamem.Slice[float32]
	module                                               driver.Module
	mid                                                  cudamem.Slice[float32]
	iqFn, q2Fn, midFn, quantD2Fn                         driver.Function
	sumFn                                                driver.Function
	copyFn, mmqFn                                        driver.Function
	closed                                               bool
	failed                                               bool
	synchronize                                          func() error
	releases                                             []func() error
}

func newCUDADSparkFFN(device *cuda.Device) (_ *cudaDSparkFFN, err error) {
	if device == nil {
		return nil, fmt.Errorf("cuda dspark FFN: device required")
	}
	s := &cudaDSparkFFN{device: device, synchronize: func() error { return device.WithContext(driver.CtxSynchronize) }}
	defer func() {
		if err != nil {
			err = s.finishConstruction(err)
		}
	}()
	const floatCount = 8192 + 5*256 + 256 + 30 + 3*30*2048 + 30*4096 + 2*5*2048 + 5*4096 + 4096 + 5*4096 + 5*2048
	err = s.acquire(func() (func() error, error) {
		var e error
		s.floats, e = cudamem.Alloc[float32](device, floatCount)
		return func() error { return s.floats.Free() }, e
	})
	if err != nil {
		return nil, err
	}
	at := 0
	take := func(n int) cudamem.Slice[float32] { v := s.floats.Slice(at, at+n); at += n; return v }
	s.pre, s.logits, s.scores, s.weights = take(8192), take(5*256), take(256), take(30)
	s.gate, s.up, s.down, s.mid = take(30*2048), take(30*2048), take(30*4096), take(30*2048)
	s.sharedGate, s.sharedUp, s.sharedOut, s.sum = take(5*2048), take(5*2048), take(5*4096), take(4096)
	s.norms, s.sharedMid = take(5*4096), take(5*2048)
	const inputBytes = 5 * 4096 / 128 * 144
	const expertBytes = 30 * 2048 / 128 * 144
	const sharedBytes = 4096 / 128 * 5 * 144
	err = s.acquire(func() (func() error, error) {
		var e error
		s.bytes, e = cudamem.Alloc[byte](device, inputBytes+expertBytes+sharedBytes)
		return func() error { return s.bytes.Free() }, e
	})
	if err != nil {
		return nil, err
	}
	s.qInput = s.bytes.Slice(0, inputBytes)
	s.qExperts = s.bytes.Slice(inputBytes, inputBytes+expertBytes)
	s.qShared = s.bytes.Slice(inputBytes+expertBytes, s.bytes.Len())
	err = s.acquire(func() (func() error, error) {
		var e error
		s.ids, e = cudamem.Alloc[int32](device, 30)
		return func() error { return s.ids.Free() }, e
	})
	if err != nil {
		return nil, err
	}
	opts := append(cudaMathHeaderOptions(), nvrtcx.WithName("cuda_dspark_ffn.cu"), nvrtcx.WithDeviceArch(device))
	program, err := nvrtcx.Compile(cudaMathCompilationSource()+cudaDSparkFFNSource, opts...)
	if err != nil {
		return nil, err
	}
	s.module, _, err = program.LoadOutputWithLog(device, nvrtcx.OutputCUBIN)
	if s.module != 0 {
		s.releases = append(s.releases, func() error {
			return s.releaseModule(s.device.WithContext, func(module driver.Module) error { return module.Close() })
		})
	}
	if err != nil {
		return nil, err
	}
	err = device.WithContext(func() error {
		for name, dst := range map[string]*driver.Function{"ds4_dspark_ffn_sum": &s.sumFn, "ds4_dspark_ffn_copy": &s.copyFn, "ds4_dspark_ffn_q8_mmq": &s.mmqFn, "ds4_dspark_ffn_iq2_mmq": &s.iqFn, "ds4_dspark_ffn_q2_mmq": &s.q2Fn, "ds4_dspark_ffn_weighted_mid": &s.midFn, "ds4_dspark_ffn_quant_d2s6": &s.quantD2Fn} {
			fn, e := s.module.Function(name)
			if e != nil {
				return e
			}
			*dst = fn
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}

// Close preserves failed release callbacks for retry. Constructor failures
// expose this owner through their returned io.Closer error boundary.
func (s *cudaDSparkFFN) Close() error { return s.close() }

func (s *cudaDSparkFFN) finishConstruction(cause error) error {
	return finishBackendCleanup(cause, s)
}

// A successful unload consumes ownership even if restoring the caller context
// then fails. Preserve that error, but never unload the consumed handle again.
func (s *cudaDSparkFFN) releaseModule(withContext func(func() error) error, unload func(driver.Module) error) error {
	if s.module == 0 {
		return nil
	}
	return withContext(func() error {
		if err := unload(s.module); err != nil {
			return err
		}
		s.module = 0
		return nil
	})
}

func (s *cudaDSparkFFN) close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	if s.synchronize == nil {
		return fmt.Errorf("cuda dspark FFN: missing synchronization owner")
	}
	if err := s.synchronize(); err != nil {
		return err
	}
	var errs []error
	for i := len(s.releases) - 1; i >= 0; i-- {
		if s.releases[i] == nil {
			continue
		}
		if err := s.releases[i](); err != nil {
			errs = append(errs, err)
		} else {
			s.releases[i] = nil
		}
	}
	return errors.Join(errs...)
}

// acquire retains a returned release callback even on partial acquisition
// failure. Failed releases stay owned; a retry never repeats a successful free.
func (s *cudaDSparkFFN) acquire(op func() (func() error, error)) error {
	if s.closed {
		return fmt.Errorf("cuda dspark FFN: owner closed")
	}
	release, err := op()
	if release != nil {
		s.releases = append(s.releases, release)
	}
	if err != nil {
		return err
	}
	if release == nil {
		return fmt.Errorf("cuda dspark FFN: missing release")
	}
	return nil
}

// enqueue wraps an actual pending row program. Any operation error poisons the
// request and prevents a second program from observing a partial output.
func (s *cudaDSparkFFN) enqueue(op func() error) error {
	if s.closed || s.failed {
		return fmt.Errorf("cuda dspark FFN: request unavailable")
	}
	err := op()
	if err != nil {
		s.failed = true
	}
	return err
}

// encode consumes actual admitted support tensors. Rows are evaluated on one
// request stream; no device buffer is shared with target State or another draft.
func (s *cudaDSparkFFN) encode(w *cudaDSparkWeights, r *cudaRuntime, stage int, input, output cudamem.Slice[float32]) (err error) {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if s == nil || s.closed || s.failed || w == nil || r == nil || r.cudaFoundation == nil || s.device != r.device || s.device != w.device || w.model == nil || w.model.block != 5 || stage < 0 || stage >= w.model.stages || input.Len() != w.model.block*16384 || output.Len() != input.Len() || input.Ptr() == 0 || output.Ptr() == 0 || dsparkDeviceOverlap(input.Ptr(), output.Ptr(), uint64(input.Len())*4) {
		return fmt.Errorf("cuda dspark FFN: owner, stage or shape mismatch")
	}
	for _, span := range []struct {
		ptr   driver.DevicePtr
		bytes uint64
	}{{s.floats.Ptr(), uint64(s.floats.Len()) * 4}, {s.bytes.Ptr(), uint64(s.bytes.Len())}, {s.ids.Ptr(), uint64(s.ids.Len()) * 4}} {
		if dsparkFFNRangeOverlap(input.Ptr(), uint64(input.Len())*4, span.ptr, span.bytes) || dsparkFFNRangeOverlap(output.Ptr(), uint64(output.Len())*4, span.ptr, span.bytes) {
			return fmt.Errorf("cuda dspark FFN: input or output aliases request scratch")
		}
	}
	names := []string{"hc_ffn_fn.weight", "hc_ffn_scale.weight", "hc_ffn_base.weight", "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}
	t := make([]cudaModelTensor, len(names))
	for i, name := range names {
		var err error
		t[i], err = w.tensor(fmt.Sprintf("mtp.%d.%s", stage, name))
		if err != nil {
			return err
		}
		if err := dsparkFFNTensorProfile(t[i], i); err != nil {
			return err
		}
	}
	// This first native implementation admits only the verified support formats.
	if t[4].typeID != 8 || t[6].typeID != 16 || t[7].typeID != 16 || t[8].typeID != 10 || t[9].typeID != 8 || t[10].typeID != 8 || t[11].typeID != 8 {
		return fmt.Errorf("cuda dspark FFN: unsupported support quantization")
	}
	return s.enqueue(func() error { return s.encodeRows(w, r, stage, input, output, t) })
}

func (s *cudaDSparkFFN) encodeRows(w *cudaDSparkWeights, r *cudaRuntime, stage int, input, output cudamem.Slice[float32], t []cudaModelTensor) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	for row := 0; row < 5; row++ {
		if err := r.encodeHCPre(input.Slice(row*16384, (row+1)*16384), t[0], t[1], t[2], t[3], s.pre); err != nil {
			return err
		}
		if err := r.withContext(func() error {
			return cudalaunch.Kernel(s.copyFn).Grid(16, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(s.pre.Slice(4096, 8192), s.norms.Slice(row*4096, (row+1)*4096), uint32(4096))
		}); err != nil {
			return err
		}
	}
	if err := r.encodeQuantMMQ(s.norms, s.qShared, 4096, 4096, 5); err != nil {
		return err
	}
	project := func(t cudaModelTensor, q cudamem.Slice[byte], out cudamem.Slice[float32]) error {
		return r.withContext(func() error {
			return cudalaunch.Kernel(s.mmqFn).Grid(uint32(t.dimensions[1]*5), 1, 1).Block(32, 1, 1).On(r.stream()).Launch(t.device, q, out, uint32(t.dimensions[0]), uint32(t.dimensions[1]), uint32(5))
		})
	}
	if err := project(t[4], s.qShared, s.logits); err != nil {
		return err
	}
	if err := r.encodeQuantMMQ(s.norms, s.qInput, 4096, 4096, 5); err != nil {
		return err
	}
	for row := 0; row < 5; row++ {
		if err := r.encodeRouterBiased(s.logits.Slice(row*256, (row+1)*256), t[5], s.scores, s.ids.Slice(row*6, (row+1)*6), s.weights.Slice(row*6, (row+1)*6)); err != nil {
			return err
		}
	}
	if err := r.withContext(func() error {
		return cudalaunch.Kernel(s.iqFn).Grid(30*2048, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(t[6].device, t[7].device, s.ids, s.qInput, s.gate, s.up, uint32(4096), uint32(2048), uint32(5), uint32(6), uint32(256))
	}); err != nil {
		return err
	}
	if err := r.withContext(func() error {
		return cudalaunch.Kernel(s.midFn).Grid(30*2048/256, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(s.gate, s.up, s.weights, s.mid, uint32(2048), uint32(30))
	}); err != nil {
		return err
	}
	if err := r.withContext(func() error {
		return cudalaunch.Kernel(s.quantD2Fn).Grid(30*2048/128, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(s.mid, s.qExperts, uint32(2048), uint32(30))
	}); err != nil {
		return err
	}
	if err := r.withContext(func() error {
		return cudalaunch.Kernel(s.q2Fn).Grid(30*4096, 1, 1).Block(32, 1, 1).On(r.stream()).Launch(t[8].device, s.ids, s.qExperts, s.down, uint32(2048), uint32(4096), uint32(30), uint32(256))
	}); err != nil {
		return err
	}
	if err := project(t[9], s.qShared, s.sharedGate); err != nil {
		return err
	}
	if err := project(t[10], s.qShared, s.sharedUp); err != nil {
		return err
	}
	if err := r.encodeSwiGLU(s.sharedGate, s.sharedUp, s.sharedMid, 1); err != nil {
		return err
	}
	if err := r.encodeQuantMMQ(s.sharedMid, s.qShared, 2048, 2048, 5); err != nil {
		return err
	}
	if err := project(t[11], s.qShared, s.sharedOut); err != nil {
		return err
	}
	for row := 0; row < w.model.block; row++ {
		in, out := input.Slice(row*16384, (row+1)*16384), output.Slice(row*16384, (row+1)*16384)
		if err := r.withContext(func() error {
			return cudalaunch.Kernel(s.sumFn).Grid(16, 1, 1).Block(256, 1, 1).On(r.stream()).Launch(s.down.Slice(row*6*4096, (row+1)*6*4096), s.sharedOut.Slice(row*4096, (row+1)*4096), s.sum, uint32(4096), uint32(6))
		}); err != nil {
			return err
		}
		if err := r.encodeHCPost(in, s.sum, t[0], t[1], t[2], out); err != nil {
			return err
		}
	}
	return nil
}

func dsparkDeviceOverlap(a, b driver.DevicePtr, size uint64) bool {
	return dsparkFFNRangeOverlap(a, size, b, size)
}

func dsparkFFNRangeOverlap(a driver.DevicePtr, aSize uint64, b driver.DevicePtr, bSize uint64) bool {
	if aSize == 0 || bSize == 0 {
		return false
	}
	if a > b {
		a, b = b, a
		aSize, bSize = bSize, aSize
	}
	return uint64(b-a) < aSize
}

// Every dimension and exact raw byte extent is checked before enqueue. The
// support owner can admit other expert formats, but this kernel path cannot.
func dsparkFFNTensorProfile(t cudaModelTensor, index int) error {
	dims := [][]uint64{{16384, 24}, {3}, {24}, {4096}, {4096, 256}, {256}, {4096, 2048, 256}, {4096, 2048, 256}, {2048, 4096, 256}, {4096, 2048}, {4096, 2048}, {2048, 4096}}
	if index < 0 || index >= len(dims) || t.device == 0 || len(t.dimensions) != len(dims[index]) {
		return fmt.Errorf("cuda dspark FFN: invalid tensor %q", t.name)
	}
	var count uint64 = 1
	for i, n := range dims[index] {
		if t.dimensions[i] != n {
			return fmt.Errorf("cuda dspark FFN: invalid dimensions for %q", t.name)
		}
		count *= n
	}
	var bytes uint64
	switch t.typeID {
	case 0:
		bytes = count * 4
	case 1:
		bytes = count * 2
	case 8:
		bytes = count / 32 * 34
	case 16:
		bytes = count / 256 * 66
	case 10:
		bytes = count / 256 * 84
	default:
		return fmt.Errorf("cuda dspark FFN: unsupported tensor type %d", t.typeID)
	}
	if t.bytes != bytes {
		return fmt.Errorf("cuda dspark FFN: truncated or oversized tensor %q", t.name)
	}
	if (index == 0 && t.typeID != 1) || ((index == 1 || index == 2 || index == 3 || index == 5) && t.typeID != 0) || (index == 4 && t.typeID != 0 && t.typeID != 1 && t.typeID != 8) {
		return fmt.Errorf("cuda dspark FFN: invalid tensor type for %q", t.name)
	}
	return nil
}
