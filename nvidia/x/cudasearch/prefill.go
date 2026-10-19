package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// PrefillCompiler binds two W256 alternatives to private, already grouped
// one-expert FP16 matrices. It does not reorder tokens or prove independence;
// the frontend must establish that grouping before construction. Conversion
// and prepacking precede this adapter and are not included in kernel timing.
type PrefillCompiler struct {
	dev         *cuda.Device
	target      string
	k, n        int
	a, b        []uint16
	bindings    WorkloadBindings
	checkLease  func() bool
	preparation []PreparationPhase
}

// NewPrefillCompiler copies row-major A[256,K] and B[K,N]. Values must be finite
// IEEE binary16, dimensions must satisfy cudamoe.PrefillW256Layouts, and target
// must be sm_100a or sm_121a. checkLease must verify a live hardware lease.
// Construction performs no CUDA calls. Keep dev open until all executables close.
func NewPrefillCompiler(dev *cuda.Device, target string, k, n int, a, b []uint16, checkLease func() bool) (*PrefillCompiler, error) {
	if dev == nil || checkLease == nil {
		return nil, fmt.Errorf("cudasearch: prefill requires device and lease checker")
	}
	if target != "sm_100a" && target != "sm_121a" {
		return nil, fmt.Errorf("cudasearch: unsupported prefill target %q", target)
	}
	if _, err := cudamoe.PrefillW256Layouts(k, n); err != nil {
		return nil, err
	}
	if n > math.MaxInt32/2 || len(a) != 256*k || len(b) != k*n {
		return nil, fmt.Errorf("cudasearch: prefill extent or grid limit mismatch")
	}
	for _, data := range [][]uint16{a, b} {
		for _, v := range data {
			if v&0x7c00 == 0x7c00 {
				return nil, fmt.Errorf("cudasearch: prefill input contains nonfinite half")
			}
		}
	}
	shape, err := cudashape.New(256, k)
	if err != nil {
		return nil, err
	}
	c := &PrefillCompiler{dev: dev, target: target, k: k, n: n, a: slices.Clone(a), b: slices.Clone(b), checkLease: checkLease}
	c.preparation = []PreparationPhase{{Name: "external_conversion", WallNs: cudatrial.Unmeasured}, {Name: "external_prepack", WallNs: cudatrial.Unmeasured}, {Name: "external_expert_grouping", WallNs: cudatrial.Unmeasured}}
	c.bindings = WorkloadBindings{Name: fmt.Sprintf("prefill-w256-k%d-n%d-%s", k, n, target), Shape: shape, DType: cudair.Float16, Layout: "row_major", InputNames: []string{"a", "b"}, InputIdentities: map[string]string{"a": halfDigest(a)}, WeightIdentities: map[string]string{"b": halfDigest(b)}, NumericalPolicy: NumericalPolicyFP16, Objective: ObjectiveMinLatency}
	return c, nil
}

func halfDigest(data []uint16) string {
	h := sha256.New()
	var word [2]byte
	for _, v := range data {
		binary.LittleEndian.PutUint16(word[:], v)
		h.Write(word[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// Bindings returns detached workload identities for Compile and the evaluator.
func (c *PrefillCompiler) Bindings() WorkloadBindings {
	b := c.bindings
	b.InputNames = slices.Clone(b.InputNames)
	b.InputIdentities = maps.Clone(b.InputIdentities)
	b.WeightIdentities = maps.Clone(b.WeightIdentities)
	return b
}

// Compile accepts only leaf expressions prefill_scalar and prefill_wmma with
// nil Data. Both represent exactly this bound workload, not arbitrary GEMMs.
// The returned executable owns resources even on error and must be closed.
// This method compiles, loads and uploads inputs; it requires hardware authority.
func (c *PrefillCompiler) Compile(ctx context.Context, expr *egraph.Expr, b WorkloadBindings) (Executable, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.admit(expr, b); err != nil {
		return nil, err
	}
	if !c.checkLease() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	major, minor, err := c.dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	if fmt.Sprintf("sm_%d%da", major, minor) != c.target {
		return nil, fmt.Errorf("cudasearch: prefill device target mismatch")
	}
	compile := cudamoe.PrefillW256Artifact
	grid := [3]uint32{uint32(c.n / 16), 4, 1}
	block := [3]uint32{128, 1, 1}
	if expr.Op == "prefill_scalar" {
		compile = cudamoe.PrefillW256ScalarArtifact
		grid = [3]uint32{uint32(2 * c.n), 1, 1}
	}
	artifact, err := compile(c.target)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !c.checkLease() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	x := &prefillExecutable{memory: searchMemoryBudget(ctx), dev: c.dev, checkLease: c.checkLease, closed: true, k: c.k, n: c.n, grid: grid, block: block}
	x.preparation = c.Preparation()
	x.identity = fmt.Sprintf("%s/%s/%v/%v/%s", artifact.Digest, artifact.EntryPoint, grid, block, c.bindings.Key())
	x.raw, err = artifact.Load(c.dev)
	if err != nil {
		return x, err
	}
	x.module, err = cuda.OwnModule(c.dev, x.raw)
	if err != nil {
		return x, err
	}
	x.raw = 0
	x.kernel, err = x.module.Function(artifact.EntryPoint)
	if err != nil {
		return x, err
	}
	x.stream, err = c.dev.NewStream()
	if err != nil {
		return x, err
	}
	err = c.dev.WithContext(func() error {
		for i := range x.events {
			var err error
			x.events[i], err = driver.NewEvent(0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return x, err
	}
	release, err := ReserveSearchMemory(ctx, int64(len(c.a))*2+int64(len(c.b))*2+int64(256*c.n)*4)
	if err != nil {
		return x, err
	}
	x.releaseMemory = release
	x.deviceBytes = int64(len(c.a))*2 + int64(len(c.b))*2 + int64(256*c.n)*4
	x.a, err = cudamem.Alloc[uint16](c.dev, len(c.a))
	if err != nil {
		return x, err
	}
	x.b, err = cudamem.Alloc[uint16](c.dev, len(c.b))
	if err != nil {
		return x, err
	}
	x.out, err = cudamem.Alloc[float32](c.dev, 256*c.n)
	if err != nil {
		return x, err
	}
	if x.a.Ptr()%32 != 0 || x.b.Ptr()%32 != 0 || x.out.Ptr()%32 != 0 {
		return x, fmt.Errorf("cudasearch: prefill buffers misaligned")
	}
	if err := x.a.CopyFromHost(c.a); err != nil {
		return x, err
	}
	if err := x.b.CopyFromHost(c.b); err != nil {
		return x, err
	}
	x.args = []any{x.a, x.b, x.out, uint32(c.k), uint32(c.n)}
	if _, err := artifact.Schema.Pack(x.args...); err != nil {
		return x, err
	}
	x.closed = false
	return x, nil
}

func (c *PrefillCompiler) admit(expr *egraph.Expr, b WorkloadBindings) error {
	if expr == nil || (expr.Op != "prefill_scalar" && expr.Op != "prefill_wmma") || expr.Data != nil || len(expr.Children) != 0 {
		return fmt.Errorf("cudasearch: unsupported prefill expression")
	}
	if b.Key() != c.bindings.Key() {
		return fmt.Errorf("%w: prefill workload identity differs", ErrMissingBinding)
	}
	return nil
}

type prefillExecutable struct {
	outputMemory  *outputMemoryOwner
	memory        *memoryBudget
	releaseMemory func()
	deviceBytes   int64
	preparation   []PreparationPhase
	mu            sync.Mutex
	dev           *cuda.Device
	checkLease    func() bool
	closed        bool
	identity      string
	k, n          int
	grid, block   [3]uint32
	raw           driver.Module
	module        *cuda.Module
	kernel        *cuda.Kernel
	stream        *cuda.Stream
	events        [2]driver.Event
	a, b          cudamem.Slice[uint16]
	out           cudamem.Slice[float32]
	args          []any
}

func (x *prefillExecutable) Identity() string { return x.identity }

// Outputs retains its returned readbacks until Close. The evaluator uses
// outputsOwned instead so a borrowed baseline does not retain every call.
func (x *prefillExecutable) Outputs(ctx context.Context) (map[string][]float32, error) {
	return x.outputsLegacyWith(ctx, x.outputOps())
}

func (x *prefillExecutable) outputsLegacyWith(ctx context.Context, ops outputReadOps) (map[string][]float32, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	values, owner, err := x.outputsWith(ctx, ops)
	retainBuiltInOutputs(&x.outputMemory, owner)
	return values, err
}

func (x *prefillExecutable) outputsOwned(ctx context.Context) (map[string][]float32, io.Closer, error) {
	return x.outputsOwnedWith(ctx, x.outputOps())
}

func (x *prefillExecutable) outputsOwnedWith(ctx context.Context, ops outputReadOps) (map[string][]float32, io.Closer, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.outputsWith(ctx, ops)
}

func (x *prefillExecutable) outputOps() outputReadOps {
	return outputReadOps{
		run:        func(ctx context.Context) error { _, err := x.run(ctx, false); return err },
		length:     func(int) (int, error) { return x.out.Len(), nil },
		readFloats: func(_ int, data []float32) error { return x.out.CopyToHost(data) },
	}
}

// outputsWith is called under mu; only its per-call device operations are
// replaced by host controls. Output metadata and allocation stay on this path.
func (x *prefillExecutable) outputsWith(ctx context.Context, ops outputReadOps) (map[string][]float32, io.Closer, error) {
	read := ops.readFloats
	ops.readFloats = func(i int, data []float32) error {
		err := read(i, data)
		if err != nil {
			x.closed = true
		}
		return err
	}
	return readBuiltInOutputs(ctx, 1, func(i int) (outputDescription, error) {
		n, err := ops.length(i)
		return outputDescription{name: "out", length: n, format: outputFloats}, err
	}, ops)
}
func (x *prefillExecutable) Run(ctx context.Context) (cudatrial.Receipt, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.run(ctx, true)
}
func (x *prefillExecutable) run(ctx context.Context, measure bool) (r cudatrial.Receipt, err error) {
	return x.runWith(ctx, measure, prefillRunOps{
		poison: func() error {
			reservation, err := reserveSearchMemory(ctx, int64(x.out.Len())*4)
			if err != nil {
				return err
			}
			defer reservation.release()
			data := make([]float32, x.out.Len())
			for i := range data {
				data[i] = float32(math.NaN())
			}
			return x.out.CopyFromHost(data)
		},
		record: func(i int) error {
			return x.stream.WithHandle(func(s driver.Stream) error { return x.dev.WithContext(func() error { return x.events[i].Record(s) }) })
		},
		launch: func() error {
			return cudalaunch.OwnedKernel(x.kernel).Grid(x.grid[0], x.grid[1], x.grid[2]).Block(x.block[0], x.block[1], x.block[2]).On(x.stream).Launch(x.args...)
		},
		wait: x.stream.Synchronize,
		elapsed: func() (float32, error) {
			var ms float32
			err := x.dev.WithContext(func() error { var err error; ms, err = x.events[0].ElapsedTime(x.events[1]); return err })
			return ms, err
		},
	})
}

type prefillRunOps struct {
	poison  func() error
	record  func(int) error
	launch  func() error
	wait    func(context.Context) error
	elapsed func() (float32, error)
}

func (x *prefillExecutable) runWith(ctx context.Context, measure bool, ops prefillRunOps) (r cudatrial.Receipt, err error) {
	if x.closed {
		return r, fmt.Errorf("cudasearch: prefill executable closed")
	}
	if err := ctx.Err(); err != nil {
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
	if err = ops.launch(); err != nil {
		return r, err
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
	r = cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingSummedKernel, Provenance: map[string]string{"timer": "cuda_event", "executable_identity": x.identity, "scope": "one kernel; input conversion and transfer excluded"}}
	err = r.RecordTiming(cudatrial.TimingSummedKernel, time.Duration(float64(ms)*float64(time.Millisecond)))
	return r, err
}

func (x *prefillExecutable) Close() error {
	return x.closeWith(x.closeResources)
}

func (x *prefillExecutable) closeWith(closeResources func() error) error {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.closed = true
	return closeBuiltInOutputs(&x.outputMemory, closeResources)
}

func (x *prefillExecutable) closeResources() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if x.stream != nil {
		if err := x.stream.Synchronize(ctx); err != nil {
			return err
		}
	}
	if x.module != nil {
		if err := x.module.Close(); err != nil {
			return err
		}
		x.module = nil
	}
	if x.raw != 0 {
		if err := x.dev.WithContext(func() error {
			if err := x.raw.Close(); err != nil {
				return err
			}
			x.raw = 0
			return nil
		}); err != nil {
			return err
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
	for _, free := range []func() error{x.a.Free, x.b.Free, x.out.Free} {
		if err := free(); err != nil {
			return err
		}
	}
	if x.releaseMemory != nil {
		x.releaseMemory()
		x.releaseMemory = nil
		x.deviceBytes = 0
	}
	return nil
}
