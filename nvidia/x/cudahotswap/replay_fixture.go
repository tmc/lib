package cudahotswap

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// These private declarations are intentionally explicit. PTX parameter widths
// cannot establish pointer intent, allocation extent, or read-only effects.
type replayBuffer struct {
	Arg, Bytes int
	Stable     bool
}

type replayOutput struct {
	Arg  int
	Elem string
}

type replaySpec struct {
	Signature cudalaunch.Signature
	Args      []any // Declared pointer positions must be nil.
	Buffers   []replayBuffer
	Outputs   []replayOutput
	Config    cudalaunch.Config
}

type replayPolicy struct {
	Rounds      int
	MaxAbs, RMS float64
	MaxULP      uint32
	Exact       bool
	NoiseMaxAbs float64
	NoiseRMS    float64
}

// Release releases a retained module pin, not an unpinned raw function. The
// fixture takes ownership after the entrypoint's presence/availability checks,
// including on subsequent failure. It runs Release only after confirmed
// completion, under its request lease; it must not acquire a second
// exclusive lease. Failed releases remain reachable for Close to retry.
type replayArm struct {
	Kernel  Kernel
	Release func() error
}

// A backend retains unresolved work and owners on errors. End must drain before
// releasing its lease. Close must be retryable and retain failed owners.
type replayBackend interface {
	Begin(context.Context) error
	Prepare(context.Context, replaySpec) error
	Drain(context.Context) error
	End(context.Context) error
	Read(int) ([]byte, error)
	Write(int, []byte) error
	Run(context.Context, replayArm, replaySpec, bool) (time.Duration, error)
	Close(context.Context) error
	Timing() (cudatrial.TimingKind, bool)
}

type replayComparison struct {
	Arg       int
	Noise     ReplayMetrics
	Candidate ReplayMetrics
	WriteOnly bool
	Pass      bool
}

type replayResult struct {
	Outputs                 []replayComparison
	Trials                  []cudatrial.Receipt
	Baseline, Candidate     time.Duration
	CandidateWins, BaseWins int
	SignP                   float64
	Timing                  cudatrial.TimingKind
	Native                  bool
	Pass                    bool
}

type replayFixture struct {
	mu             sync.Mutex
	spec           replaySpec
	policy         replayPolicy
	backend        replayBackend
	baseline       replayArm
	pins           []func() error
	initial        map[int][]byte
	want           map[int][]float32
	noise          map[int]ReplayMetrics
	writeOnly      map[int]bool
	ready, closed  bool
	stableRecovery bool
	cleanupStarted bool
}

const replayMaxBytes = 64 << 20

func validateReplay(spec replaySpec, p replayPolicy) error {
	if p.Rounds < 1 || p.Rounds > 64 {
		return fmt.Errorf("replay rounds must be in [1,64]")
	}
	for _, v := range []float64{p.MaxAbs, p.RMS, p.NoiseMaxAbs, p.NoiseRMS} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("replay tolerances must be finite and nonnegative")
		}
	}
	if len(spec.Args) == 0 || len(spec.Args) > 64 || len(spec.Args) != len(spec.Signature.Params) || len(spec.Buffers) == 0 || len(spec.Outputs) == 0 {
		return fmt.Errorf("replay requires a bounded complete argument and output schema")
	}
	g, b := spec.Config.Grid, spec.Config.Block
	if g.X == 0 || g.X > 1<<31-1 || g.Y == 0 || g.Y > 65535 || g.Z == 0 || g.Z > 65535 || b.X == 0 || b.Y == 0 || b.Z == 0 || uint64(b.X)*uint64(b.Y) > 1024 || uint64(b.X)*uint64(b.Y)*uint64(b.Z) > 1024 {
		return fmt.Errorf("replay launch geometry is unsupported")
	}
	declared := make(map[int]replayBuffer)
	total := 0
	args := append([]any(nil), spec.Args...)
	sig := spec.Signature
	sig.Params = append([]cudalaunch.Param(nil), sig.Params...)
	for _, buf := range spec.Buffers {
		if buf.Arg < 0 || buf.Arg >= len(args) || buf.Bytes <= 0 || buf.Bytes > replayMaxBytes-total || args[buf.Arg] != nil {
			return fmt.Errorf("replay buffer has an invalid argument or extent")
		}
		if _, ok := declared[buf.Arg]; ok {
			return fmt.Errorf("replay duplicate buffer argument %d", buf.Arg)
		}
		if sig.Params[buf.Arg].Size != 8 {
			return fmt.Errorf("replay pointer argument %d is not 64-bit", buf.Arg)
		}
		declared[buf.Arg] = buf
		total += buf.Bytes
		sig.Params[buf.Arg].IsPointer = true
		args[buf.Arg] = driver.DevicePtr(1)
	}
	for i, arg := range args {
		if _, ok := declared[i]; ok {
			continue
		}
		if sig.Params[i].IsPointer {
			return fmt.Errorf("replay undeclared pointer argument %d", i)
		}
		switch arg.(type) {
		case int8, uint8, int16, uint16, int32, uint32, int64, uint64, float32, float64:
		default:
			return fmt.Errorf("replay scalar argument %d has an unsupported type", i)
		}
	}
	if err := cudalaunch.Validate(sig, args...); err != nil {
		return fmt.Errorf("replay arguments: %w", err)
	}
	outputs := make(map[int]bool)
	for _, out := range spec.Outputs {
		buf, ok := declared[out.Arg]
		if !ok || buf.Stable || out.Elem != "f32" || buf.Bytes%4 != 0 || outputs[out.Arg] {
			return fmt.Errorf("replay output %d has an unsupported extent, element, alias, or effect", out.Arg)
		}
		outputs[out.Arg] = true
	}
	return nil
}

// recordReplay returns a retained owner for every error after adopting the
// baseline pin. Missing context/backend/kernel/pin returns nil before adoption.
// The caller must Close a nonnil result with a valid context before releasing
// its device or source allocations.
func recordReplay(ctx context.Context, backend replayBackend, baseline replayArm, spec replaySpec, p replayPolicy) (*replayFixture, error) {
	if ctx == nil || backend == nil || baseline.Kernel == nil || baseline.Release == nil {
		return nil, fmt.Errorf("record replay: nil context, backend, kernel, or pin")
	}
	f := &replayFixture{backend: backend, baseline: baseline, policy: p, initial: make(map[int][]byte), want: make(map[int][]float32), noise: make(map[int]ReplayMetrics), writeOnly: make(map[int]bool)}
	if baseline.Release != nil {
		f.pins = append(f.pins, baseline.Release)
	}
	if err := validateReplay(spec, p); err != nil {
		return f, err
	}
	f.spec = spec
	f.spec.Args = append([]any(nil), spec.Args...)
	f.spec.Signature.Params = append([]cudalaunch.Param(nil), spec.Signature.Params...)
	f.spec.Buffers = append([]replayBuffer(nil), spec.Buffers...)
	f.spec.Outputs = append([]replayOutput(nil), spec.Outputs...)
	if err := backend.Begin(ctx); err != nil {
		return f, err
	}
	err := backend.Prepare(ctx, f.spec)
	if err == nil {
		err = f.record(ctx)
	}
	err = errors.Join(err, f.finish(ctx))
	f.ready = err == nil
	return f, err
}

func (f *replayFixture) record(ctx context.Context) error {
	for _, buf := range f.spec.Buffers {
		data, err := f.backend.Read(buf.Arg)
		if err != nil {
			return err
		}
		if len(data) != buf.Bytes {
			return fmt.Errorf("replay snapshot extent mismatch for argument %d", buf.Arg)
		}
		f.initial[buf.Arg] = append([]byte(nil), data...)
	}
	first, err := f.arm(ctx, f.baseline, false, nil)
	if err != nil {
		return err
	}
	second, err := f.arm(ctx, f.baseline, false, nil)
	if err != nil {
		return err
	}
	for _, out := range f.spec.Outputs {
		if bytes.Equal(first[out.Arg], f.initial[out.Arg]) {
			return fmt.Errorf("replay baseline output %d is unchanged", out.Arg)
		}
		want, err := replayFloats(first[out.Arg])
		if err != nil {
			return err
		}
		got, err := replayFloats(second[out.Arg])
		if err != nil {
			return err
		}
		noise, err := MeasureReplay(got, want)
		if err != nil {
			return err
		}
		if noise.NonFinite != 0 || *noise.MaxAbs > f.policy.NoiseMaxAbs || *noise.RMS > f.policy.NoiseRMS {
			return fmt.Errorf("replay baseline output %d exceeds noise tolerance", out.Arg)
		}
		f.want[out.Arg], f.noise[out.Arg] = want, noise
		poison := make([]byte, len(first[out.Arg]))
		for i := 0; i < len(poison); i += 4 {
			binary.LittleEndian.PutUint32(poison[i:], 0x7fc12345)
		}
		readback, err := f.arm(ctx, f.baseline, false, map[int][]byte{out.Arg: poison})
		if err != nil {
			return err
		}
		values, err := replayFloats(readback[out.Arg])
		if err != nil {
			return err
		}
		m, err := MeasureReplay(values, want)
		if err != nil {
			return err
		}
		// In-place read/modify/write outputs may propagate the poison. That is
		// recorded as an effect, not mistaken for a write-only output.
		f.writeOnly[out.Arg] = replayPass(m, f.policy)
	}
	return f.restore()
}

func (f *replayFixture) restore() error {
	for _, buf := range f.spec.Buffers {
		if buf.Stable {
			continue
		}
		if err := f.backend.Write(buf.Arg, f.initial[buf.Arg]); err != nil {
			return err
		}
	}
	return nil
}

// Recovery also runs after launch failures, but only after real completion.
// A canceled drain leaves the request lease and every owner retained.
func (f *replayFixture) recover() error {
	for _, buf := range f.spec.Buffers {
		initial, ok := f.initial[buf.Arg]
		if !ok {
			continue
		}
		if buf.Stable {
			if !f.stableRecovery {
				continue
			}
			got, err := f.backend.Read(buf.Arg)
			if err != nil {
				return err
			}
			if bytes.Equal(got, initial) {
				continue
			}
		}
		if err := f.backend.Write(buf.Arg, initial); err != nil {
			return err
		}
	}
	f.stableRecovery = false
	return nil
}

func (f *replayFixture) finish(ctx context.Context) error {
	if err := f.backend.Drain(ctx); err != nil {
		return err
	}
	if err := f.recover(); err != nil {
		return err
	}
	return f.backend.End(ctx)
}

func (f *replayFixture) arm(ctx context.Context, arm replayArm, timed bool, poison map[int][]byte) (map[int][]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := f.restore(); err != nil {
		return nil, err
	}
	for arg, data := range poison {
		if err := f.backend.Write(arg, data); err != nil {
			return nil, err
		}
	}
	f.stableRecovery = true
	_, err := f.backend.Run(ctx, arm, f.spec, timed)
	if err != nil {
		return nil, err
	}
	return f.readArm()
}

func (f *replayFixture) readArm() (map[int][]byte, error) {
	out := make(map[int][]byte)
	for _, buf := range f.spec.Buffers {
		if !buf.Stable {
			continue
		}
		got, err := f.backend.Read(buf.Arg)
		if err != nil {
			return nil, err
		}
		if !bytes.Equal(got, f.initial[buf.Arg]) {
			// Stable pointers remain in-place. Recovery is explicit and checked,
			// rather than leaving a candidate's weight mutation live.
			return nil, errors.Join(fmt.Errorf("replay stable argument %d was modified", buf.Arg), f.backend.Write(buf.Arg, f.initial[buf.Arg]))
		}
	}
	f.stableRecovery = false
	for _, output := range f.spec.Outputs {
		data, err := f.backend.Read(output.Arg)
		if err != nil {
			return nil, err
		}
		if len(data) != len(f.initial[output.Arg]) {
			return nil, fmt.Errorf("replay output extent changed")
		}
		out[output.Arg] = data
	}
	return out, nil
}

func (f *replayFixture) checkStable() error {
	for _, buf := range f.spec.Buffers {
		if !buf.Stable {
			continue
		}
		got, err := f.backend.Read(buf.Arg)
		if err != nil {
			return err
		}
		if !bytes.Equal(got, f.initial[buf.Arg]) {
			return fmt.Errorf("replay stable argument %d changed outside this fixture", buf.Arg)
		}
	}
	return nil
}

func replayFloats(b []byte) ([]float32, error) {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil, fmt.Errorf("replay output is not a nonempty FP32 array")
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, nil
}

func replayPass(m ReplayMetrics, p replayPolicy) bool {
	if m.NonFinite != 0 || m.MaxAbs == nil || m.RMS == nil || m.MaxULP == nil {
		return false
	}
	return *m.MaxAbs <= p.MaxAbs && *m.RMS <= p.RMS && ((p.Exact && m.Exact) || (!p.Exact && *m.MaxULP <= p.MaxULP))
}

func (f *replayFixture) Check(ctx context.Context, candidate replayArm) (result replayResult, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if ctx == nil || !f.ready || f.closed || candidate.Kernel == nil || candidate.Release == nil {
		return result, fmt.Errorf("check replay: nil context or pin, or unavailable fixture")
	}
	f.pins = append(f.pins, candidate.Release)
	pinIndex := len(f.pins) - 1
	if err := f.backend.Begin(ctx); err != nil {
		f.ready = false
		return result, err
	}
	defer func() {
		cleanup := f.backend.Drain(ctx)
		if cleanup == nil {
			cleanup = f.recover()
		}
		if cleanup == nil {
			cleanup = f.pins[pinIndex]()
			if cleanup == nil {
				f.pins = f.pins[:pinIndex]
			}
		}
		if cleanup == nil {
			cleanup = f.backend.End(ctx)
		}
		err = errors.Join(err, cleanup)
		if err != nil {
			f.ready = false
		}
	}()
	result.Timing, result.Native = f.backend.Timing()
	if err := f.checkStable(); err != nil {
		return result, err
	}
	result.Pass = true
	metrics := make(map[int]ReplayMetrics)
	makeArm := func(name string, arm replayArm) cudatrial.Arm {
		return cudatrial.Arm{Name: name, Run: func(ctx context.Context) (cudatrial.Receipt, error) {
			var receipt cudatrial.Receipt
			if err := f.restore(); err != nil {
				return receipt, err
			}
			f.stableRecovery = true
			duration, err := f.backend.Run(ctx, arm, f.spec, true)
			if err != nil {
				return receipt, err
			}
			if duration <= 0 {
				return receipt, fmt.Errorf("replay event interval is unresolved")
			}
			values, err := f.readArm()
			if err != nil {
				return receipt, err
			}
			for _, out := range f.spec.Outputs {
				v, err := replayFloats(values[out.Arg])
				if err != nil {
					return receipt, err
				}
				m, err := MeasureReplay(v, f.want[out.Arg])
				if err != nil {
					return receipt, err
				}
				if name == "baseline" {
					if m.NonFinite != 0 || *m.MaxAbs > f.policy.NoiseMaxAbs || *m.RMS > f.policy.NoiseRMS {
						return receipt, fmt.Errorf("replay baseline drift")
					}
				} else {
					result.Pass = result.Pass && replayPass(m, f.policy)
					old, ok := metrics[out.Arg]
					if !ok {
						metrics[out.Arg] = m
					} else {
						metrics[out.Arg] = replayWorst(old, m)
					}
				}
			}
			hash := sha256.New()
			for _, out := range f.spec.Outputs {
				hash.Write(values[out.Arg])
			}
			receipt.Digest = fmt.Sprintf("%x", hash.Sum(nil))
			receipt.Complete = true
			receipt.DeviceNs, receipt.QueueNs, receipt.GpuIntervalUnionNs = cudatrial.Unmeasured, cudatrial.Unmeasured, cudatrial.Unmeasured
			if err := receipt.RecordTiming(result.Timing, duration); err != nil {
				return receipt, err
			}
			return receipt, nil
		}}
	}
	result.Trials, err = cudatrial.Interleave(ctx, []cudatrial.Arm{makeArm("baseline", f.baseline), makeArm("candidate", candidate)}, f.policy.Rounds)
	if err != nil {
		return result, err
	}
	var base, cand []time.Duration
	paired := make(map[int]map[string]time.Duration)
	for _, trial := range result.Trials {
		d, err := trial.DurationFor(result.Timing)
		if err != nil {
			return result, err
		}
		if trial.Name == "baseline" {
			base = append(base, d)
		} else {
			cand = append(cand, d)
		}
		if paired[trial.Round] == nil {
			paired[trial.Round] = make(map[string]time.Duration)
		}
		paired[trial.Round][trial.Name] = d
	}
	for _, pair := range paired {
		if pair["candidate"] < pair["baseline"] {
			result.CandidateWins++
		}
		if pair["candidate"] > pair["baseline"] {
			result.BaseWins++
		}
	}
	result.Baseline, result.Candidate = replayMedian(base), replayMedian(cand)
	result.SignP = replaySignP(result.CandidateWins, result.BaseWins)
	for _, out := range f.spec.Outputs {
		result.Outputs = append(result.Outputs, replayComparison{Arg: out.Arg, Noise: f.noise[out.Arg], Candidate: metrics[out.Arg], WriteOnly: f.writeOnly[out.Arg], Pass: replayPass(metrics[out.Arg], f.policy)})
	}
	err = f.restore()
	return result, err
}

func replayWorst(a, b ReplayMetrics) ReplayMetrics {
	a.Exact = a.Exact && b.Exact
	a.NonFinite = max(a.NonFinite, b.NonFinite)
	if a.NonFinite != 0 {
		a.MaxAbs, a.RMS, a.MaxULP = nil, nil, nil
		return a
	}
	abs, rms, ulp := max(*a.MaxAbs, *b.MaxAbs), max(*a.RMS, *b.RMS), max(*a.MaxULP, *b.MaxULP)
	a.MaxAbs, a.RMS, a.MaxULP = &abs, &rms, &ulp
	return a
}

func replayMedian(v []time.Duration) time.Duration {
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	if len(v)%2 == 1 {
		return v[len(v)/2]
	}
	a, b := v[len(v)/2-1], v[len(v)/2]
	return a + (b-a)/2
}

func replaySignP(wins, losses int) float64 {
	n := wins + losses
	if n == 0 {
		return 1
	}
	term := math.Ldexp(1, -n)
	sum := term
	for k := 1; k <= min(wins, losses); k++ {
		term *= float64(n-k+1) / float64(k)
		sum += term
	}
	return min(1, 2*sum)
}

func (f *replayFixture) Close(ctx context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if ctx == nil {
		return fmt.Errorf("close replay: nil context")
	}
	if f.closed {
		return nil
	}
	f.ready = false
	if f.backend == nil {
		return fmt.Errorf("close replay: missing backend")
	}
	if err := f.backend.Begin(ctx); err != nil {
		return err
	}
	// Drain first; no release may precede confirmed device completion.
	if err := f.backend.Drain(ctx); err != nil {
		return err
	}
	if !f.cleanupStarted {
		if err := f.recover(); err != nil {
			return err
		}
		for i, release := range f.pins {
			if release == nil {
				continue
			}
			if err := release(); err != nil {
				return err
			}
			f.pins[i] = nil
		}
		// Later retry may find scratch already freed. Do not accidentally
		// recover through the original caller binding after that point.
		f.cleanupStarted = true
	}
	if err := f.backend.Close(ctx); err != nil {
		return err
	}
	f.closed = true
	return nil
}

// Native sources are borrowed until Close. Every nonstable argument is replayed
// from an owned scratch allocation; stable arguments retain their original
// address and are checked against the bounded host snapshot after every arm.
type replayDeviceBinding struct {
	Arg   int
	Slice cudamem.Slice[byte]
}
type cudaReplayBackend struct {
	dev          *cuda.Device
	bindings     map[int]cudamem.Slice[byte]
	scratch      map[int]cudamem.Slice[byte]
	borrows      []func()
	stream       *cuda.Stream
	timer        *cuda.EventTimer
	eviction     cudamem.Slice[byte]
	lease        io.Closer
	pending      bool
	initialized  bool
	declarations []replayBuffer
	acquire      func(context.Context) (io.Closer, error)
}

func newCUDAReplayBackend(dev *cuda.Device, spec replaySpec, p replayPolicy, bindings []replayDeviceBinding) (*cudaReplayBackend, error) {
	b := &cudaReplayBackend{dev: dev, bindings: make(map[int]cudamem.Slice[byte]), scratch: make(map[int]cudamem.Slice[byte])}
	if dev == nil {
		return b, fmt.Errorf("replay device is nil")
	}
	if err := validateReplay(spec, p); err != nil {
		return b, err
	}
	b.declarations = append([]replayBuffer(nil), spec.Buffers...)
	if len(bindings) != len(spec.Buffers) {
		return b, fmt.Errorf("replay binding count mismatch")
	}
	for _, binding := range bindings {
		if _, ok := b.bindings[binding.Arg]; ok {
			return b, fmt.Errorf("replay duplicate binding")
		}
		var decl *replayBuffer
		for i := range spec.Buffers {
			if spec.Buffers[i].Arg == binding.Arg {
				decl = &spec.Buffers[i]
			}
		}
		ptr := uint64(binding.Slice.Ptr())
		if decl == nil || binding.Slice.Device() != dev || binding.Slice.SizeBytes() != uintptr(decl.Bytes) || ptr == 0 || ptr+uint64(decl.Bytes) < ptr {
			return b, fmt.Errorf("replay binding owner or extent mismatch")
		}
		for _, out := range spec.Outputs {
			if out.Arg == binding.Arg && ptr%4 != 0 {
				return b, fmt.Errorf("replay FP32 output is not aligned")
			}
		}
		for _, old := range b.bindings {
			lo, hi := uint64(old.Ptr()), uint64(old.Ptr())+uint64(old.SizeBytes())
			if ptr < hi && lo < ptr+uint64(decl.Bytes) {
				return b, fmt.Errorf("replay overlapping bindings are unsupported")
			}
		}
		b.bindings[binding.Arg] = binding.Slice
	}
	b.acquire = func(ctx context.Context) (io.Closer, error) {
		h, err := cudalock.Acquire(ctx, cudalock.WithLabel("cudahotswap:replay"))
		if h == nil {
			return nil, err
		}
		return requestLease{h}, err
	}
	return b, nil
}

func (b *cudaReplayBackend) Begin(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("replay request has nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.lease != nil {
		return nil
	}
	if b.acquire == nil {
		return fmt.Errorf("replay backend is not initialized")
	}
	lease, err := b.acquire(ctx)
	b.lease = lease // Adopt a nonnil partial lease before inspecting its error.
	if err != nil {
		return err
	}
	if lease == nil {
		return fmt.Errorf("replay acquire returned no lease")
	}
	return nil
}

func (b *cudaReplayBackend) Prepare(ctx context.Context, spec replaySpec) error {
	if len(spec.Buffers) != len(b.declarations) {
		return fmt.Errorf("replay binding declarations changed")
	}
	for i, buf := range spec.Buffers {
		if buf != b.declarations[i] {
			return fmt.Errorf("replay binding declarations changed")
		}
	}
	if b.initialized {
		return nil
	}
	if b.lease == nil {
		return fmt.Errorf("replay preparation without lease")
	}
	if spec.Config.Stream == nil || spec.Config.Stream.Device() != b.dev {
		return fmt.Errorf("replay requires the owning input stream")
	}
	if err := spec.Config.Stream.Synchronize(ctx); err != nil {
		return err
	}
	if b.stream == nil {
		stream, err := b.dev.NewStream()
		b.stream = stream
		if err != nil {
			return err
		}
	}
	for _, buf := range spec.Buffers {
		if _, ok := b.scratch[buf.Arg]; ok {
			continue
		}
		release, err := b.bindings[buf.Arg].Borrow(b.dev)
		if release != nil {
			b.borrows = append(b.borrows, release)
		}
		if err != nil {
			return err
		}
		if buf.Stable {
			continue
		}
		owned, err := cudamem.Alloc[byte](b.dev, buf.Bytes)
		b.scratch[buf.Arg] = owned
		if err != nil {
			return err
		}
		data := make([]byte, buf.Bytes)
		if err := b.bindings[buf.Arg].CopyToHost(data); err != nil {
			return err
		}
		if err := owned.CopyFromHost(data); err != nil {
			return err
		}
	}
	if b.timer == nil {
		timer, err := cuda.NewEventTimer(b.dev, []*cuda.Stream{b.stream})
		b.timer = timer
		if err != nil {
			return err
		}
		if !timer.Checked() {
			return fmt.Errorf("replay event timer is unchecked")
		}
	}
	if b.eviction.Ptr() == 0 {
		l2, err := b.dev.Attribute(driver.DevAttrL2CacheSize)
		if err != nil {
			return err
		}
		if l2 <= 0 || l2 > 256<<20 {
			return fmt.Errorf("replay L2 extent is unsupported")
		}
		owned, err := cudamem.Alloc[byte](b.dev, 2*l2)
		b.eviction = owned
		if err != nil {
			return err
		}
	}
	b.initialized = true
	return nil
}

func (b *cudaReplayBackend) Read(arg int) ([]byte, error) {
	if b.lease == nil {
		return nil, fmt.Errorf("replay read without lease")
	}
	s, ok := b.scratch[arg]
	if !ok {
		s, ok = b.bindings[arg]
	}
	if !ok {
		return nil, fmt.Errorf("replay undeclared read")
	}
	v := make([]byte, s.Len())
	return v, s.CopyToHost(v)
}

func (b *cudaReplayBackend) Write(arg int, data []byte) error {
	if b.lease == nil {
		return fmt.Errorf("replay write without lease")
	}
	s, ok := b.scratch[arg]
	if !ok {
		s, ok = b.bindings[arg]
	}
	if !ok || len(data) != s.Len() {
		return fmt.Errorf("replay write extent mismatch")
	}
	return s.CopyFromHost(data)
}

func (b *cudaReplayBackend) Run(ctx context.Context, arm replayArm, spec replaySpec, timed bool) (time.Duration, error) {
	if b.lease == nil {
		return 0, fmt.Errorf("replay launch without lease")
	}
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if !b.initialized {
		return 0, fmt.Errorf("replay backend is not prepared")
	}
	kernel, ok := arm.Kernel.(*nativeKernel)
	if !ok || kernel == nil || kernel.dev != b.dev {
		return 0, fmt.Errorf("replay kernel owner does not match device")
	}
	args := append([]any(nil), spec.Args...)
	for _, buf := range spec.Buffers {
		s := b.bindings[buf.Arg]
		if !buf.Stable {
			s = b.scratch[buf.Arg]
		}
		args[buf.Arg] = s.Ptr()
	}
	cfg := spec.Config
	cfg.Stream = b.stream
	if timed {
		b.pending = true
		err := b.dev.WithContext(func() error {
			return b.stream.WithHandle(func(s driver.Stream) error { return b.eviction.Ptr().MemsetD8Async(0, uint(b.eviction.Len()), s) })
		})
		if err != nil {
			return 0, err
		}
		if err := b.stream.Synchronize(ctx); err != nil {
			return 0, err
		}
		b.pending = false
	}
	b.pending = true // An error can occur after submission; never infer completion.
	var duration time.Duration
	var err error
	if timed {
		duration, err = b.timer.Measure(func() error { return arm.Kernel.Launch(cfg, args...) })
	} else {
		err = arm.Kernel.Launch(cfg, args...)
	}
	if err != nil {
		return 0, err
	}
	if err := arm.Kernel.Wait(ctx, b.stream); err != nil {
		return 0, err
	}
	b.pending = false
	return duration, nil
}

func (b *cudaReplayBackend) End(ctx context.Context) error {
	if b.lease == nil {
		return nil
	}
	if err := b.Drain(ctx); err != nil {
		return err
	}
	if err := b.lease.Close(); err != nil {
		return err
	}
	b.lease = nil
	return nil
}

func (b *cudaReplayBackend) Drain(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("replay drain has nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if b.pending {
		if err := b.stream.Synchronize(ctx); err != nil {
			return err
		}
		b.pending = false
	}
	return nil
}

func (b *cudaReplayBackend) Close(ctx context.Context) error {
	if b.pending {
		if err := b.stream.Synchronize(ctx); err != nil {
			return err
		}
		b.pending = false
	}
	if b.timer != nil {
		if err := b.timer.Close(); err != nil {
			return err
		}
		b.timer = nil
	}
	for arg, owner := range b.scratch {
		if err := owner.Free(); err != nil {
			return err
		}
		delete(b.scratch, arg)
	}
	if err := b.eviction.Free(); err != nil {
		return err
	}
	for i, release := range b.borrows {
		if release != nil {
			release()
			b.borrows[i] = nil
		}
	}
	if b.stream != nil {
		if err := b.stream.Close(); err != nil {
			return err
		}
		b.stream = nil
	}
	return b.End(ctx)
}

func (*cudaReplayBackend) Timing() (cudatrial.TimingKind, bool) { return "cuda_event_interval", true }
