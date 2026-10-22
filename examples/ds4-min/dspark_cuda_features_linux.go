//go:build linux

package main

import (
	_ "embed"
	"errors"
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_dspark_features.cu
var cudaDSparkFeatureSource string

// cudaDSparkFeatures belongs to one target request. Its pending feature set
// is valid only after every selected layer was captured and the successful
// target Eval has advanced position. State reset/restore must invalidate or
// restore this object; integrating those hooks is a separate ownership gate.
// No terminal-HC alias can replace the selected per-layer captures.
type cudaDSparkFeatures struct {
	r          *cudaRuntime
	request    *cudaRequestState
	generation uint64
	modelSHA   string
	block      int
	targets    []int
	captured   []bool
	features   cudamem.Slice[float32]
	status     cudamem.Slice[uint32]
	module     driver.Module
	function   driver.Function
	position   int
	token      int
	published  bool
	closed     bool
	history    *cudaDSparkHistory
	grouped    *cudaDSparkGroupedCapture
}

func newCUDADSparkFeatures(request *cudaRequestState, model *dsparkModel) (_ *cudaDSparkFeatures, err error) {
	if request == nil || request.closed || request.runtime == nil || request.dspark != nil {
		return nil, fmt.Errorf("cuda dspark features: fresh request binding required")
	}
	r := request.runtime
	if err := dsparkRequestOpen(r); err != nil {
		return nil, err
	}
	if r.cudaFoundation == nil || r.device == nil || model == nil || len(model.targets) == 0 || len(model.sha256) != 64 {
		return nil, fmt.Errorf("cuda dspark features: request and verified support required")
	}
	f := &cudaDSparkFeatures{r: r, request: request, modelSHA: model.sha256, block: model.block, targets: append([]int(nil), model.targets...), captured: make([]bool, len(model.targets)), position: -1, token: -1}
	defer func() {
		if err != nil {
			err = finishBackendCleanup(err, f)
		}
	}()
	f.features, err = cudamem.Alloc[float32](r.device, len(f.targets)*Flash.Embed)
	if err != nil {
		return nil, err
	}
	f.status, err = cudamem.Alloc[uint32](r.device, 1)
	if err != nil {
		return nil, err
	}
	compiled, err := nvrtcx.Compile(cudaDSparkFeatureSource, nvrtcx.WithName("cuda_dspark_features.cu"), nvrtcx.WithDeviceArch(r.device))
	if err != nil {
		return nil, err
	}
	// Physical CUBIN only: a load error cannot silently choose a PTX JIT.
	f.module, _, err = compiled.LoadOutputWithLog(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		return nil, err
	}
	err = r.device.WithContext(func() error {
		var err error
		f.function, err = f.module.Function("ds4_dspark_capture_feature")
		return err
	})
	if err != nil {
		return nil, err
	}
	request.dspark = f
	f.history, err = newCUDADSparkHistory(f, model.block)
	if err != nil {
		request.dspark = nil
		return nil, err
	}
	return f, nil
}

func (s *cudaRequestState) invalidateDSpark() {
	if s.generation == ^uint64(0) {
		s.runtime.failed = errors.Join(dsv4.ErrDesynchronized, fmt.Errorf("cuda dspark features: history generation exhausted"))
	} else {
		s.generation++
	}
	if s.dspark != nil {
		s.dspark.invalidate()
		if err := s.dspark.history.reset(); err != nil {
			s.runtime.failed = errors.Join(dsv4.ErrDesynchronized, err)
		}
	}
}

// Target Eval has already advanced when this is called. A derived-feature
// failure cannot be retryable: poison the request rather than report an
// ordinary error against a silently advanced target frontier.
func (s *cudaRequestState) finishDSparkEval() error {
	if err := s.advanceSpeculativeEpoch(); err != nil {
		return err
	}
	if s.generation == ^uint64(0) {
		s.runtime.failed = errors.Join(dsv4.ErrDesynchronized, fmt.Errorf("cuda state: history generation exhausted"))
		return s.runtime.failed
	}
	s.generation++
	if s.dspark != nil {
		if err := s.dspark.seal(); err != nil {
			s.runtime.failed = errors.Join(dsv4.ErrDesynchronized, err)
			return s.runtime.failed
		}
	}
	return nil
}

func (f *cudaDSparkFeatures) bound() bool {
	return f != nil && !f.closed && f.request != nil && !f.request.closed && f.request.runtime == f.r && f.request.dspark == f && f.r != nil
}

func (f *cudaDSparkFeatures) begin(position, token int) error {
	if !f.bound() || position < 0 || position >= f.r.ctx || position != f.r.pos || token < 0 || token >= Flash.Vocab || f.r.failed != nil {
		return fmt.Errorf("cuda dspark features: invalid request position")
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return err
	}
	f.position, f.published = position, false
	f.token = token
	f.generation = f.request.generation
	clear(f.captured)
	if err := f.status.CopyFromHost([]uint32{0}); err != nil {
		f.invalidate()
		return err
	}
	return nil
}

// capture runs immediately after a selected layer's post-FFN HC expansion,
// on the same request stream. Metadata remains pending until seal succeeds.
func (f *cudaDSparkFeatures) capture(layer, position int, hc cudamem.Slice[float32]) (err error) {
	defer func() {
		if err != nil && f != nil {
			f.invalidate()
		}
	}()
	slot, err := f.captureSlot(layer, position, hc.Len())
	if err != nil || slot < 0 {
		return err
	}
	if err := dsparkRequestOpen(f.r); err != nil {
		return err
	}
	err = f.r.withContext(func() error {
		return cudalaunch.Kernel(f.function).Grid(16, 1, 1).Block(256, 1, 1).On(f.r.stream()).Launch(hc, f.features, f.status, uint32(slot))
	})
	if err != nil {
		return err
	}
	f.captured[slot] = true
	return nil
}

// captureSlot is the exact pure preflight used before any device launch.
// It permits CPU guard mutations with the correct HC shape, so a different
// invalid operand cannot conceal a missing identity or generation check.
func (f *cudaDSparkFeatures) captureSlot(layer, position, width int) (int, error) {
	if !f.bound() {
		return -1, fmt.Errorf("cuda dspark features: request binding mismatch")
	}
	if f.r.dspark != f {
		return -1, fmt.Errorf("cuda dspark features: inactive request capture")
	}
	if f.request.generation != f.generation {
		return -1, fmt.Errorf("cuda dspark features: history generation mismatch")
	}
	if position != f.position || f.published {
		return -1, fmt.Errorf("cuda dspark features: no pending capture")
	}
	for slot, selected := range f.targets {
		if layer != selected {
			continue
		}
		if slot >= len(f.captured) || f.captured[slot] {
			return -1, fmt.Errorf("cuda dspark features: duplicate selected layer")
		}
		if width != Flash.HC*Flash.Embed {
			return -1, fmt.Errorf("cuda dspark features: wrong HC shape")
		}
		return slot, nil
	}
	return -1, nil
}

func (f *cudaDSparkFeatures) seal() (err error) {
	defer func() {
		if err != nil && f != nil {
			f.invalidate()
		}
	}()
	if !f.bound() || f.r.dspark != f || f.request.generation != f.generation+1 || f.position < 0 || f.r.failed != nil || f.r.pos != f.position+1 {
		return fmt.Errorf("cuda dspark features: target Eval did not complete")
	}
	for _, present := range f.captured {
		if !present {
			return fmt.Errorf("cuda dspark features: selected layer missing")
		}
	}
	if err := syncStream(f.r.stream()); err != nil {
		return err
	}
	var status [1]uint32
	if err := f.status.CopyToHost(status[:]); err != nil {
		return err
	}
	if status[0] != 0 {
		return fmt.Errorf("cuda dspark features: nonfinite selected HC mean")
	}
	f.published = true
	if f.history != nil {
		if err := f.history.record(f); err != nil {
			return err
		}
	}
	return nil
}

func (f *cudaDSparkFeatures) invalidate() {
	if f != nil {
		f.position, f.published, f.token = -1, false, -1
		f.grouped = nil
		clear(f.captured)
	}
}

func (f *cudaDSparkFeatures) close() error {
	if f == nil || f.closed && f.module == 0 && f.features.Ptr() == 0 && f.status.Ptr() == 0 && f.history == nil {
		return nil
	}
	f.closed = true
	if err := syncStream(f.r.stream()); err != nil {
		return err
	}
	var errs []error
	if f.history != nil {
		if err := f.history.close(); err != nil {
			return err
		}
		f.history = nil
	}
	if f.module != 0 {
		if err := f.r.device.WithContext(func() error {
			if err := f.module.Close(); err != nil {
				return err
			}
			f.module = 0
			return nil
		}); err != nil {
			return err
		}
	}
	if err := f.features.Free(); err != nil {
		errs = append(errs, err)
	} else {
		f.features = cudamem.Slice[float32]{}
	}
	if err := f.status.Free(); err != nil {
		errs = append(errs, err)
	} else {
		f.status = cudamem.Slice[uint32]{}
	}
	f.invalidate()
	return errors.Join(errs...)
}

func (f *cudaDSparkFeatures) Close() error {
	if err := f.close(); err != nil {
		return err
	}
	if f != nil && f.request != nil && f.request.dspark == f {
		f.request.dspark = nil
	}
	return nil
}
