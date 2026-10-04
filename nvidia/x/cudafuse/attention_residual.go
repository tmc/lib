package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// AttentionResidualInputs supplies resident candidates [C,R,D], actual
// multipliers g[D] and pseudo-query w[D]. Inputs and output [R,D] must have
// disjoint active FP32 byte ranges. Callers exclude mutation and raw frees
// through successful Wait or Close. Resident values are validated on device.
type AttentionResidualInputs struct{ Values, Multiplier, Query cudamem.Slice[float32] }

// AttentionResidual owns immutable loaded kernels for one shape/math identity.
// Its zero value is closed. Separate executions may use independent streams;
// Close refuses while an execution still owns its modules. Keep the device and
// stream live, and do not concurrently close caller buffers during execution.
type AttentionResidual struct {
	mu      sync.Mutex
	device  *cuda.Device
	config  AttentionResidualConfig
	path    AttentionResidualPath
	modules []*cuda.Module
	raw     []driver.Module
	kernels []*cuda.Kernel
	active  int
	closed  bool
}

// LoadAttentionResidual loads trusted compiled artifacts of the exact sources,
// ABI, full owner header, actual FP32 fragment and numerical flags. Descriptor
// slices are copied. Receipt checks bind inputs; they do not authenticate the
// compiler or prove arbitrary image semantics. Close every nonnil error result.
func LoadAttentionResidual(dev *cuda.Device, c AttentionResidualConfig, path AttentionResidualPath, artifacts []*cudakernel.Artifact) (*AttentionResidual, error) {
	sources, err := AttentionResidualSource(c, path)
	if err != nil {
		return nil, err
	}
	return loadAttentionResidualSources(dev, c, path, artifacts, sources)
}

func loadAttentionResidualSources(dev *cuda.Device, c AttentionResidualConfig, path AttentionResidualPath, artifacts []*cudakernel.Artifact, sources []FusedKernel) (*AttentionResidual, error) {
	if dev == nil {
		return nil, errors.New("cudafuse: nil attention residual device")
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if len(artifacts) != len(sources) || len(sources) != 1 && len(sources) != 4 {
		return nil, errors.New("cudafuse: attention residual artifact count differs")
	}
	var owned []*cudakernel.Artifact
	for i, a := range artifacts {
		v, err := attentionResidualArtifact(a, sources[i])
		if err != nil {
			return nil, err
		}
		owned = append(owned, v)
	}
	x := &AttentionResidual{device: dev, config: c, path: path}
	for _, a := range owned {
		raw, err := a.Load(dev)
		if raw != 0 {
			x.raw = append(x.raw, raw)
		}
		if err != nil {
			return x, err
		}
		m, err := cuda.OwnModule(dev, raw)
		if err != nil {
			return x, err
		}
		x.raw = x.raw[:len(x.raw)-1]
		x.modules = append(x.modules, m)
		k, err := m.Function(a.EntryPoint)
		if err != nil {
			return x, err
		}
		x.kernels = append(x.kernels, k)
	}
	return x, nil
}

type attentionResidualRange struct {
	ptr, bytes uint64
	length     int
}

func attentionResidualFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func attentionResidualRanges(c AttentionResidualConfig, rs []attentionResidualRange) error {
	if err := c.validate(); err != nil {
		return err
	}
	lengths := []int{c.Candidates * c.Rows * c.Width, c.Width, c.Width, c.Rows * c.Width}
	if len(rs) != 4 {
		return errors.New("cudafuse: attention residual binding count differs")
	}
	for i, r := range rs {
		if r.length != lengths[i] || r.bytes != uint64(lengths[i])*4 || r.ptr == 0 || r.ptr%4 != 0 || r.ptr > math.MaxUint64-r.bytes {
			return fmt.Errorf("cudafuse: invalid attention residual range %d", i)
		}
		for _, s := range rs[:i] {
			if r.ptr < s.ptr+s.bytes && s.ptr < r.ptr+r.bytes {
				return errors.New("cudafuse: aliased attention residual inputs or output")
			}
		}
	}
	return nil
}

// AttentionResidualExecution owns guarded scratch and input leases until a
// real stream drain, and the output lease through publication or Close.
// Output is published only after every resident status and
// scratch write is checked. Cancellation retains unpublished work for Wait;
// Close drains and discards unpublished work. Close is required on error too.
type AttentionResidualExecution struct {
	mu               sync.Mutex
	wait             func(context.Context) error
	inspect          func() error
	publish          func() error
	cleanup          []func() error
	finishRelease    func()
	finished, closed bool
	abandoned        bool
	failure          error
	launches         int
	scratchBytes     int
}

// Wait observes real completion, validates statuses and scratch, and publishes
// output once. Nonfinite resident input/status1 or FP32 overflow/status2 refuses
// publication. Missing or unknown statuses and unwritten scratch also fail.
func (e *AttentionResidualExecution) Wait(ctx context.Context) error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed || e.abandoned {
		return errors.New("cudafuse: closed attention residual execution")
	}
	if e.wait == nil {
		return errors.New("cudafuse: uninitialized attention residual execution")
	}
	if e.finished {
		return e.failure
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if err := e.wait(ctx); err != nil {
		return err
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	if e.failure == nil {
		e.failure = e.inspect()
	}
	if e.failure == nil {
		e.failure = e.publish()
	}
	e.finished = true
	if e.finishRelease != nil {
		e.finishRelease()
		e.finishRelease = nil
	}
	return e.failure
}

// Close waits without cancellation before releasing scratch and remaining
// caller leases. Failed drains retain everything for retry. Publication is
// deliberately skipped when closing an abandoned execution.
func (e *AttentionResidualExecution) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.wait == nil {
		e.closed = true
		return nil
	}
	if err := e.wait(context.Background()); err != nil {
		return err
	}
	e.abandoned = true
	var errs []error
	for i, f := range e.cleanup {
		if f != nil {
			if err := f(); err != nil {
				errs = append(errs, err)
			} else {
				e.cleanup[i] = nil
			}
		}
	}
	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	e.closed = true
	return nil
}

// attentionResidualCleanup keeps failed allocation cleanup reachable even when
// no device work was submitted or an inner execution has already closed.
func attentionResidualCleanup(e *AttentionResidualExecution, cause error, free func() error) (*AttentionResidualExecution, error) {
	if err := free(); err != nil {
		if e == nil {
			e = &AttentionResidualExecution{}
		}
		e.mu.Lock()
		if e.wait == nil || e.closed {
			e.wait = func(context.Context) error { return nil }
		}
		e.closed = false
		e.abandoned = true
		e.cleanup = append(e.cleanup, free)
		e.mu.Unlock()
		return e, errors.Join(cause, err)
	}
	return e, cause
}

// Stats reports actual submitted kernels and requested scratch bytes. It does
// not report kernel registers, allocator peaks, device traffic or GPU timing.
func (e *AttentionResidualExecution) Stats() (launches, scratchBytes int) {
	if e == nil {
		return 0, 0
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.launches, e.scratchBytes
}

const attentionResidualGuard = float32(9876.5)

var attentionResidualPoison = math.Float32frombits(0x7fc001ab)

func attentionResidualStatus(status []int32) error {
	if len(status) == 0 {
		return errors.New("cudafuse: missing attention residual status")
	}
	for _, v := range status {
		switch v {
		case 0:
		case 1:
			return errors.New("cudafuse: nonfinite resident attention residual input")
		case 2:
			return errors.New("cudafuse: attention residual FP32 intermediate overflow")
		default:
			return errors.New("cudafuse: missing or unknown attention residual device status")
		}
	}
	return nil
}

// Execute dispatches four composed stages or one fused stage. All validation
// happens before output publication; no host readback of inputs certifies
// resident data. The nonnil execution owns scratch and leases on every result.
func (x *AttentionResidual) Execute(ctx context.Context, stream *cuda.Stream, in AttentionResidualInputs, out cudamem.Slice[float32]) (*AttentionResidualExecution, error) {
	return x.execute(ctx, stream, in, out, nil, nil)
}

func (x *AttentionResidual) execute(ctx context.Context, stream *cuda.Stream, in AttentionResidualInputs, out cudamem.Slice[float32], extra []func(), setup func() error) (*AttentionResidualExecution, error) {
	releaseExtra := func() {
		for _, r := range extra {
			r()
		}
	}
	if x == nil {
		releaseExtra()
		return nil, errors.New("cudafuse: nil attention residual consumer")
	}
	x.mu.Lock()
	if x.closed || len(x.kernels) == 0 || stream == nil || stream.Device() != x.device {
		x.mu.Unlock()
		releaseExtra()
		return nil, errors.New("cudafuse: closed consumer or different attention residual stream")
	}
	x.active++
	x.mu.Unlock()
	var releases []func()
	var outputRelease func()
	releaseOutput := func() {
		if outputRelease != nil {
			outputRelease()
			outputRelease = nil
		}
	}
	releases = append(releases, extra...)
	releases = append(releases, func() { x.mu.Lock(); x.active--; x.mu.Unlock() })
	releaseAll := func() {
		for _, r := range releases {
			r()
		}
		releases = nil
		releaseOutput()
	}
	if ctx != nil && ctx.Err() != nil {
		releaseAll()
		return nil, ctx.Err()
	}
	inputs := []cudamem.Slice[float32]{in.Values, in.Multiplier, in.Query, out}
	var rs []attentionResidualRange
	for i, s := range inputs {
		r, err := s.Borrow(x.device)
		if err != nil {
			releaseAll()
			return nil, err
		}
		if i == len(inputs)-1 {
			outputRelease = r
		} else {
			releases = append(releases, r)
		}
		rs = append(rs, attentionResidualRange{uint64(s.Ptr()), uint64(s.SizeBytes()), s.Len()})
	}
	if err := attentionResidualRanges(x.config, rs); err != nil {
		releaseAll()
		return nil, err
	}
	e := &AttentionResidualExecution{}
	e.finishRelease = releaseOutput
	e.cleanup = append(e.cleanup, func() error { releaseOutput(); return nil })
	failPre := func(err error) (*AttentionResidualExecution, error) {
		releaseAll()
		e.failure = err
		e.finished = true
		return e, errors.Join(err, e.Close())
	}
	// Until completion adopts callbacks, cleanup has no queued device work.
	e.wait = func(context.Context) error { return nil }
	var buffers []cudamem.Slice[float32]
	for _, n := range []int{x.config.Candidates * x.config.Rows, x.config.Candidates * x.config.Rows, x.config.Rows * x.config.Width} {
		b, err := cudamem.Alloc[float32](x.device, n+2)
		if b.Ptr() != 0 {
			buffers = append(buffers, b)
			e.cleanup = append(e.cleanup, b.Free)
			e.scratchBytes += (n + 2) * 4
		}
		if err != nil {
			return failPre(err)
		}
		h := make([]float32, n+2)
		for i := range h {
			h[i] = attentionResidualPoison
		}
		h[0], h[n+1] = attentionResidualGuard, attentionResidualGuard
		if err := b.CopyFromHost(h); err != nil {
			return failPre(err)
		}
	}
	status, err := cudamem.Alloc[int32](x.device, x.config.Rows+2)
	if status.Ptr() != 0 {
		e.cleanup = append(e.cleanup, status.Free)
		e.scratchBytes += (x.config.Rows + 2) * 4
	}
	if err != nil {
		return failPre(err)
	}
	sv := make([]int32, x.config.Rows+2)
	for i := range sv {
		sv[i] = -1
	}
	sv[0], sv[len(sv)-1] = 0x13579, 0x13579
	if err := status.CopyFromHost(sv); err != nil {
		return failPre(err)
	}
	completion, err := cudaplan.NewStreamCompletion(stream, releases...)
	if err != nil {
		return failPre(err)
	}
	releases = nil
	e.wait = completion.Wait
	scores := buffers[0].Slice(1, buffers[0].Len()-1)
	prob := buffers[1].Slice(1, buffers[1].Len()-1)
	tmp := buffers[2].Slice(1, buffers[2].Len()-1)
	stat := status.Slice(1, status.Len()-1)
	e.inspect = func() error {
		if err := status.CopyToHost(sv); err != nil {
			return err
		}
		if sv[0] != 0x13579 || sv[len(sv)-1] != 0x13579 {
			return errors.New("cudafuse: attention residual status guard changed")
		}
		if err := attentionResidualStatus(sv[1 : len(sv)-1]); err != nil {
			return err
		}
		for i, b := range buffers {
			h := make([]float32, b.Len())
			if err := b.CopyToHost(h); err != nil {
				return err
			}
			if h[0] != attentionResidualGuard || h[len(h)-1] != attentionResidualGuard {
				return errors.New("cudafuse: attention residual scratch guard changed")
			}
			if x.path == AttentionResidualFused && i < 2 {
				continue
			}
			for _, v := range h[1 : len(h)-1] {
				if !attentionResidualFinite(v) {
					return errors.New("cudafuse: nonfinite or unwritten attention residual scratch")
				}
			}
		}
		return nil
	}
	e.publish = func() error { return out.CopyDeviceToDevice(tmp) }
	ptrs := []driver.DevicePtr{in.Values.Ptr(), in.Multiplier.Ptr(), in.Query.Ptr(), scores.Ptr(), prob.Ptr(), tmp.Ptr(), stat.Ptr()}
	args := attentionResidualArguments(ptrs)
	if setup != nil {
		if err := setup(); err != nil {
			e.failure = err
			return e, err
		}
	}
	for _, k := range x.kernels {
		if ctx != nil && ctx.Err() != nil {
			return e, ctx.Err()
		}
		err := stream.WithHandle(func(raw driver.Stream) error {
			return k.Launch([3]uint32{uint32(x.config.Rows), 1, 1}, [3]uint32{32, 1, 1}, 0, raw, args)
		})
		if err != nil {
			e.failure = err
			return e, err
		}
		e.launches++
	}
	return e, e.Wait(ctx)
}

func attentionResidualArguments(ptrs []driver.DevicePtr) []unsafe.Pointer {
	args := make([]unsafe.Pointer, len(ptrs))
	for i := range ptrs {
		args[i] = unsafe.Pointer(&ptrs[i])
	}
	return args
}

// Close releases modules only when no execution retains them. Failed cleanup
// leaves the remaining module owned for retry. It never closes the device.
func (x *AttentionResidual) Close() error {
	if x == nil {
		return nil
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.active != 0 {
		return cudamem.ErrInUse
	}
	x.closed = true
	for i, m := range x.modules {
		if m != nil {
			if err := m.Close(); err != nil {
				return err
			}
			x.modules[i] = nil
		}
	}
	for i, m := range x.raw {
		if m != 0 {
			if err := x.device.WithContext(func() error { return m.Close() }); err != nil {
				return err
			}
			x.raw[i] = 0
		}
	}
	x.kernels = nil
	return nil
}

// AttentionResidualSnapshots retains immutable completed blocks [R,D] across
// calls. Execute appends the caller's current partial block last, packs actual
// device values and keeps original snapshots/partial alive through completion.
// Close refuses during a pending call; it releases views, not caller buffers.
type AttentionResidualSnapshots struct {
	mu          sync.Mutex
	device      *cuda.Device
	rows, width int
	completed   []cudamem.Slice[float32]
	releases    []func()
	active      int
	closed      bool
}

// NewAttentionResidualSnapshots retains completed snapshots. Their order is
// oldest first; at most seven completed snapshots plus one partial are admitted.
func NewAttentionResidualSnapshots(dev *cuda.Device, rows, width int, completed []cudamem.Slice[float32]) (*AttentionResidualSnapshots, error) {
	c := AttentionResidualConfig{len(completed) + 1, rows, width, 1e-5}
	if err := c.validate(); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, errors.New("cudafuse: nil snapshot device")
	}
	s := &AttentionResidualSnapshots{device: dev, rows: rows, width: width, completed: append([]cudamem.Slice[float32](nil), completed...)}
	for _, v := range s.completed {
		if v.Len() != rows*width {
			_ = s.Close()
			return nil, errors.New("cudafuse: completed snapshot extent differs")
		}
		r, err := v.Borrow(dev)
		if err != nil {
			_ = s.Close()
			return nil, err
		}
		s.releases = append(s.releases, r)
	}
	return s, nil
}

// Execute mixes retained snapshots plus partial, using the consumer's shape
// and math identity. Calls are serial until their actual drain; replacing the
// partial after a completed call never rewrites an old completed snapshot.
func (s *AttentionResidualSnapshots) Execute(ctx context.Context, x *AttentionResidual, stream *cuda.Stream, partial, g, w, out cudamem.Slice[float32]) (*AttentionResidualExecution, error) {
	if s == nil || x == nil {
		return nil, errors.New("cudafuse: nil snapshot caller or consumer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.active != 0 {
		return nil, cudamem.ErrInUse
	}
	if x.device != s.device || x.config.Candidates != len(s.completed)+1 || x.config.Rows != s.rows || x.config.Width != s.width || partial.Len() != s.rows*s.width {
		return nil, errors.New("cudafuse: snapshot consumer shape differs")
	}
	all := append(append([]cudamem.Slice[float32](nil), s.completed...), partial, g, w, out)
	for i, a := range all {
		for _, b := range all[:i] {
			ap, bp := uint64(a.Ptr()), uint64(b.Ptr())
			ab, bb := uint64(a.SizeBytes()), uint64(b.SizeBytes())
			if ap > math.MaxUint64-ab || bp > math.MaxUint64-bb || ap < bp+bb && bp < ap+ab {
				return nil, errors.New("cudafuse: aliased snapshot inputs or output")
			}
		}
	}
	r, err := partial.Borrow(s.device)
	if err != nil {
		return nil, err
	}
	packed, err := cudamem.Alloc[float32](s.device, x.config.Candidates*s.rows*s.width)
	if err != nil {
		r()
		if packed.Ptr() != 0 {
			return attentionResidualCleanup(nil, err, packed.Free)
		}
		return nil, err
	}
	s.active++
	extra := []func(){r, func() { s.mu.Lock(); s.active--; s.mu.Unlock() }}
	// Drop the caller mutex before execution: completion releases take it.
	s.mu.Unlock()
	e, err := x.execute(ctx, stream, AttentionResidualInputs{packed, g, w}, out, extra, func() error {
		for i, v := range append(append([]cudamem.Slice[float32](nil), s.completed...), partial) {
			if err := packed.Slice(i*s.rows*s.width, (i+1)*s.rows*s.width).CopyDeviceToDeviceAsync(v, stream); err != nil {
				return err
			}
		}
		return nil
	})
	s.mu.Lock()
	if e == nil {
		return attentionResidualCleanup(nil, err, packed.Free)
	}
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return attentionResidualCleanup(e, err, packed.Free)
	}
	e.cleanup = append(e.cleanup, packed.Free)
	e.mu.Unlock()
	return e, err
}

// Close releases completed snapshot leases after all queued reads drain.
func (s *AttentionResidualSnapshots) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active != 0 {
		return cudamem.ErrInUse
	}
	if !s.closed {
		for _, r := range s.releases {
			r()
		}
		s.releases = nil
		s.closed = true
	}
	return nil
}
