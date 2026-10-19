package cudasearch

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

type sequenceStage struct {
	raw    driver.Module
	module *cuda.Module
	kernel *cuda.Kernel
	args   []any
}
type sequenceLaunch struct{ grid, block [3]uint32 }
type sequenceOutput struct {
	buffer, name string
	bytes        bool
}

// load and addBuffer preserve partial owners on every error. Callers return
// the closed sequence to their caller for retryable cleanup until setup ends.
func (x *sequenceExecutable) load(artifacts []*cudakernel.Artifact) error {
	return measurePreparation(&x.preparation, "load_modules_stream_events", 0, 0, func() error { return x.loadResources(artifacts) })
}

func (x *sequenceExecutable) loadResources(artifacts []*cudakernel.Artifact) error {
	if len(artifacts) == 0 || len(artifacts) != len(x.launches) {
		return fmt.Errorf("cudasearch: sequence artifact/launch count mismatch")
	}
	for _, a := range artifacts {
		if a == nil {
			return fmt.Errorf("cudasearch: nil sequence artifact")
		}
	}
	for i, a := range artifacts {
		x.stages = append(x.stages, sequenceStage{})
		s := &x.stages[i]
		x.identity += fmt.Sprintf("/%s/%s/%v/%v", a.Digest, a.EntryPoint, x.launches[i].grid, x.launches[i].block)
		var err error
		s.raw, err = a.Load(x.dev)
		if err != nil {
			return err
		}
		s.module, err = cuda.OwnModule(x.dev, s.raw)
		if err != nil {
			return err
		}
		s.raw = 0
		s.kernel, err = s.module.Function(a.EntryPoint)
		if err != nil {
			return err
		}
	}
	var err error
	x.stream, err = x.dev.NewStream()
	if err != nil {
		return err
	}
	return x.dev.WithContext(func() error {
		for i := range x.events {
			var err error
			x.events[i], err = driver.NewEvent(0)
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func (x *sequenceExecutable) addBuffer(name string, layout cudalayout.Tile) error {
	size, err := layout.FootprintBytes()
	if err != nil {
		return err
	}
	return measurePreparation(&x.preparation, "allocate_"+name, 0, int64(size), func() error { return x.allocateBuffer(name, layout) })
}

func (x *sequenceExecutable) allocateBuffer(name string, layout cudalayout.Tile) error {
	return x.allocateBufferWith(name, layout, func(size int) (cudamem.Slice[byte], error) { return cudamem.Alloc[byte](x.dev, size) })
}

func (x *sequenceExecutable) allocateBufferWith(name string, layout cudalayout.Tile, allocate func(int) (cudamem.Slice[byte], error)) error {
	if _, ok := x.buffers[name]; ok {
		return fmt.Errorf("cudasearch: duplicate sequence buffer %q", name)
	}
	size, err := layout.FootprintBytes()
	if err != nil {
		return err
	}
	r, err := x.memory.reserve(int64(size))
	if err != nil {
		return err
	}
	if x.memoryReservations == nil {
		x.memoryReservations = make(map[string]*memoryReservation)
	}
	x.memoryReservations[name] = r
	b, err := allocate(size)
	x.buffers[name] = b
	x.bufferOrder = append(x.bufferOrder, name)
	if err != nil {
		return err
	}
	if b.Ptr()%driver.DevicePtr(layout.AlignmentBytes()) != 0 {
		return fmt.Errorf("cudasearch: sequence buffer %q misaligned", name)
	}
	return nil
}

type sequenceExecutable struct {
	outputMemory       *outputMemoryOwner
	memory             *memoryBudget
	memoryReservations map[string]*memoryReservation
	preparation        []PreparationPhase
	mu                 sync.Mutex
	dev                *cuda.Device
	checkLease         func() bool
	closed             bool
	identity           string
	launches           []sequenceLaunch
	outputs            []sequenceOutput
	poison             []string
	bufferOrder        []string
	stages             []sequenceStage
	stream             *cuda.Stream
	events             [2]driver.Event
	buffers            map[string]cudamem.Slice[byte]
}

func (x *sequenceExecutable) Identity() string                { return x.identity }
func (x *sequenceExecutable) Preparation() []PreparationPhase { return slices.Clone(x.preparation) }

func (x *sequenceExecutable) upload(name string, data []byte) error {
	return measurePreparation(&x.preparation, "upload_"+name, int64(len(data)), int64(len(data)), func() error {
		buffer, ok := x.buffers[name]
		if !ok {
			return fmt.Errorf("cudasearch: missing upload buffer %q", name)
		}
		return buffer.CopyFromHost(data)
	})
}

// Outputs retains its returned readbacks until Close. outputsOwned transfers
// each call's lifetime independently to the evaluator, including on a baseline.
func (x *sequenceExecutable) Outputs(ctx context.Context) (map[string][]float32, error) {
	return x.outputsLegacyWith(ctx, x.outputOps())
}

func (x *sequenceExecutable) outputsLegacyWith(ctx context.Context, ops outputReadOps) (map[string][]float32, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	values, owner, err := x.outputsWith(ctx, ops)
	retainBuiltInOutputs(&x.outputMemory, owner)
	return values, err
}

func (x *sequenceExecutable) outputsOwned(ctx context.Context) (map[string][]float32, io.Closer, error) {
	return x.outputsOwnedWith(ctx, x.outputOps())
}

func (x *sequenceExecutable) outputsOwnedWith(ctx context.Context, ops outputReadOps) (map[string][]float32, io.Closer, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.outputsWith(ctx, ops)
}

func (x *sequenceExecutable) outputOps() outputReadOps {
	return outputReadOps{
		run: func(ctx context.Context) error { _, err := x.run(ctx, false); return err },
		length: func(i int) (int, error) {
			b, ok := x.buffers[x.outputs[i].buffer]
			if !ok {
				return 0, ErrMemoryAccountingUnavailable
			}
			return b.Len(), nil
		},
		readBytes: func(i int, data []byte) error { return x.buffers[x.outputs[i].buffer].CopyToHost(data) },
		decode:    func(i int, data []byte) ([]float32, error) { return x.outputs[i].decode(data) },
	}
}

func (x *sequenceExecutable) outputsWith(ctx context.Context, ops outputReadOps) (map[string][]float32, io.Closer, error) {
	read, decode := ops.readBytes, ops.decode
	ops.readBytes = func(i int, data []byte) error {
		err := read(i, data)
		if err != nil {
			x.closed = true
		}
		return err
	}
	ops.decode = func(i int, data []byte) ([]float32, error) {
		values, err := decode(i, data)
		if err != nil {
			x.closed = true
		}
		return values, err
	}
	return readBuiltInOutputs(ctx, len(x.outputs), func(i int) (outputDescription, error) {
		output := x.outputs[i]
		n, err := ops.length(i)
		format := outputFloatBytes
		if output.bytes {
			format = outputBytes
		}
		return outputDescription{name: output.name, length: n, format: format}, err
	}, ops)
}

func (output sequenceOutput) decode(data []byte) ([]float32, error) {
	if output.bytes {
		return byteValues(data), nil
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("cudasearch: output %q has incomplete float32 data", output.name)
	}
	v := make([]float32, len(data)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	return v, nil
}
func (x *sequenceExecutable) Run(ctx context.Context) (cudatrial.Receipt, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.run(ctx, true)
}
func (x *sequenceExecutable) run(ctx context.Context, measure bool) (r cudatrial.Receipt, err error) {
	return x.runWith(ctx, measure, sequenceRunOps{
		poison: func() error {
			for _, name := range x.poison {
				b, ok := x.buffers[name]
				if !ok {
					continue
				}
				reservation, err := reserveSearchMemory(ctx, int64(b.Len()))
				if err != nil {
					return err
				}
				defer reservation.release()
				data := make([]byte, b.Len())
				for i := range data {
					data[i] = 0xa5
				}
				if err := b.CopyFromHost(data); err != nil {
					return err
				}
			}
			return nil
		},
		record: func(i int) error {
			return x.stream.WithHandle(func(s driver.Stream) error { return x.dev.WithContext(func() error { return x.events[i].Record(s) }) })
		},
		launch: func(i int) error {
			s, l := x.stages[i], x.launches[i]
			return cudalaunch.OwnedKernel(s.kernel).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]).On(x.stream).Launch(s.args...)
		},
		wait: x.stream.Synchronize,
		elapsed: func() (float32, error) {
			var ms float32
			err := x.dev.WithContext(func() error { var err error; ms, err = x.events[0].ElapsedTime(x.events[1]); return err })
			return ms, err
		},
	})
}

type sequenceRunOps struct {
	poison         func() error
	record, launch func(int) error
	wait           func(context.Context) error
	elapsed        func() (float32, error)
}

func (x *sequenceExecutable) runWith(ctx context.Context, measure bool, ops sequenceRunOps) (r cudatrial.Receipt, err error) {
	if x.closed {
		return r, fmt.Errorf("cudasearch: sequence executable closed")
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	if !x.checkLease() {
		return r, cudatrial.ErrHardwareLockRequired
	}
	defer func() {
		if err != nil {
			x.closed = true
		}
	}()
	if err = ops.poison(); err != nil {
		return r, err
	}
	if measure {
		if err = ops.record(0); err != nil {
			return r, err
		}
	}
	for i := range x.stages {
		if err = ops.launch(i); err != nil {
			return r, err
		}
	}
	if measure {
		if err = ops.record(1); err != nil {
			return r, err
		}
	}
	if err = ops.wait(ctx); err != nil {
		return r, err
	}
	if !measure {
		return cudatrial.Receipt{Complete: true}, nil
	}
	var ms float32
	ms, err = ops.elapsed()
	if err != nil {
		return r, err
	}
	if math.IsNaN(float64(ms)) || math.IsInf(float64(ms), 0) || ms <= 0 || float64(ms) > float64(math.MaxInt64)/float64(time.Millisecond) {
		return r, cudatrial.ErrTimingUnavailable
	}
	r = cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingGPUBusy, Provenance: map[string]string{"timer": "cuda_event", "scope": "complete kernel sequence; conversion and transfer excluded", "executable_identity": x.identity}}
	err = r.RecordTiming(cudatrial.TimingGPUBusy, time.Duration(float64(ms)*float64(time.Millisecond)))
	return r, err
}

func (x *sequenceExecutable) Close() error {
	return x.closeWith(x.closeResources)
}

func (x *sequenceExecutable) closeWith(closeResources func() error) error {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.closed = true
	return closeBuiltInOutputs(&x.outputMemory, closeResources)
}

func (x *sequenceExecutable) closeResources() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if x.stream != nil {
		if err := x.stream.Synchronize(ctx); err != nil {
			return err
		}
	}
	for i := range x.stages {
		s := &x.stages[i]
		if s.module != nil {
			if err := s.module.Close(); err != nil {
				return err
			}
			s.module = nil
		}
		if s.raw != 0 {
			if err := x.dev.WithContext(func() error {
				if err := s.raw.Close(); err != nil {
					return err
				}
				s.raw = 0
				return nil
			}); err != nil {
				return err
			}
		}
	}
	for i := range x.events {
		if x.events[i] != 0 {
			if err := x.dev.WithContext(func() error {
				if err := x.events[i].Close(); err != nil {
					return err
				}
				x.events[i] = 0
				return nil
			}); err != nil {
				return err
			}
		}
	}
	if x.stream != nil {
		if err := x.stream.Close(); err != nil {
			return err
		}
		x.stream = nil
	}
	for i := len(x.bufferOrder) - 1; i >= 0; i-- {
		name := x.bufferOrder[i]
		if b, ok := x.buffers[name]; ok {
			if err := x.freeBuffer(name, b.Free); err != nil {
				return err
			}
		}
	}
	return nil
}

func (x *sequenceExecutable) freeBuffer(name string, free func() error) error {
	if err := free(); err != nil {
		return err
	}
	delete(x.buffers, name)
	x.memoryReservations[name].release()
	delete(x.memoryReservations, name)
	return nil
}
