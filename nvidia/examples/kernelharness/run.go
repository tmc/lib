package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Config sets how hard the harness looks. The defaults are deliberately small
// enough to run in a refinement loop and large enough that a median means
// something.
type Config struct {
	// Trials is the number of correctness trials. Each regenerates its inputs
	// from a distinct seed.
	Trials int
	// Seed is the first trial's seed; trial i uses Seed+i.
	Seed int64
	// Warmup launches run before timing and are discarded.
	Warmup int
	// Samples is the number of event-timed samples per arm.
	Samples int
	// LaunchesPerSample is how many launches one sample spans. One launch of a
	// short kernel can fall below the device timer's resolution; batching keeps
	// the measured interval well above it.
	LaunchesPerSample int
	// Timeout bounds one correctness launch. Exceeding it yields outcome
	// timeout. Nothing here kills the running kernel: the context is poisoned
	// and the process must be abandoned.
	Timeout time.Duration
	// Baseline enables the A/B arm against the task's own baseline kernel.
	Baseline bool
}

// DefaultConfig returns the configuration the CLI uses.
func DefaultConfig() Config {
	return Config{
		Trials: 5, Seed: 1, Warmup: 32, Samples: 9, LaunchesPerSample: 32,
		Timeout: 10 * time.Second, Baseline: true,
	}
}

func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// fail builds a terminal report for err, classified.
func (r *Report) fail(err error) *Report {
	outcome, recognized := classify(err)
	if outcome == OutcomeOK {
		// classify only returns ok for a nil error; reaching here with a
		// non-nil one would be a harness bug reported as a pass.
		outcome, recognized = OutcomeHarnessError, false
	}
	r.Outcome = outcome
	r.Recognized = recognized
	r.ContextPoisoned = outcome.poisonsContext()
	r.Reason = err.Error()
	return r
}

// Run compiles, checks and measures one candidate against one task.
//
// It returns a Report for every outcome including failure; the error result is
// reserved for a harness fault that leaves the report meaningless.
func Run(dev *cuda.Device, task *Task, name, source string, cfg Config) (rep *Report, err error) {
	rep = &Report{
		Task: task.Name, TaskDef: task.Summary, Candidate: name,
		Entry: task.Entry, Outcome: OutcomeHarnessError, Recognized: true,
		Unmeasured: unmeasured(),
	}

	var closers []func() error
	defer func() {
		// Reverse dependency order, as elsewhere in these examples. A cleanup
		// failure is reported, not swallowed.
		for i := len(closers) - 1; i >= 0; i-- {
			if cerr := closers[i](); cerr != nil {
				err = errors.Join(err, cerr)
				return
			}
		}
	}()
	cleanup := func(fn func() error) { closers = append(closers, fn) }

	rep.Device = describeDevice(dev)

	// Compile. A compile failure is the one outcome that carries the compiler
	// log as its whole evidence, so it is reported verbatim.
	start := time.Now()
	compiled, cerr := nvrtcx.Compile(source, nvrtcx.WithName(name), nvrtcx.WithDeviceArch(dev), nvrtcx.WithLineInfo())
	wall := time.Since(start).Nanoseconds()
	rep.Compile = &CompileReport{
		SourceSHA256: digest([]byte(source)), SourceBytes: len(source),
		Arch: rep.Device.Compute, WallNanos: wall,
	}
	if cerr != nil {
		rep.Outcome, rep.Recognized = OutcomeCompileError, true
		rep.Reason = cerr.Error()
		return rep, nil
	}
	ptx := compiled.PTX()
	rep.Compile.PTXSHA256 = digest([]byte(ptx))
	rep.Compile.PTXBytes = len(ptx)
	rep.Compile.Log = compiled.Log()

	module, merr := compiled.Load(dev)
	if merr != nil {
		return rep.fail(merr), nil
	}
	cleanup(func() error { return dev.WithContext(module.Close) })

	var fn driver.Function
	if ferr := dev.WithContext(func() error {
		var e error
		fn, e = module.Function(task.Entry)
		return e
	}); ferr != nil {
		return rep.fail(fmt.Errorf("entry %q: %w", task.Entry, ferr)), nil
	}

	stream, serr := dev.NewStreamWithFlags(driver.StreamNonBlocking)
	if serr != nil {
		return rep.fail(serr), nil
	}
	cleanup(stream.Close)

	bufs, berr := allocBuffers(dev, task.N)
	if berr != nil {
		return rep.fail(berr), nil
	}
	cleanup(bufs.free)

	launch := func(f driver.Function) error {
		return cudalaunch.Kernel(f).Grid(task.Grid, 1, 1).Block(task.Block, 1, 1).On(stream).
			Launch(bufs.out, bufs.a, bufs.b, task.Alpha, int32(task.N))
	}

	// Correctness, before any timing. A kernel that faults or hangs must not be
	// handed to the timer, whose completion wait has no deadline.
	if done := rep.check(task, cfg, bufs, stream, launch, fn); !done {
		return rep, nil
	}

	// Timing. The candidate is known to terminate and to be correct by here.
	timer, terr := cuda.NewEventTimer(dev, []*cuda.Stream{stream})
	if timer != nil {
		cleanup(timer.Close)
	}
	if terr != nil {
		return rep.fail(terr), nil
	}

	arms := []struct {
		name string
		fn   driver.Function
		out  **TimingReport
	}{{name: "candidate", fn: fn, out: &rep.Timing}}

	if cfg.Baseline && task.Baseline != "" {
		baseCompiled, berr := nvrtcx.Compile(task.Baseline,
			nvrtcx.WithName(task.Name+"_baseline.cu"), nvrtcx.WithDeviceArch(dev))
		if berr != nil {
			return nil, fmt.Errorf("task %s: its own baseline kernel does not compile: %w", task.Name, berr)
		}
		baseModule, lerr := baseCompiled.Load(dev)
		if lerr != nil {
			return nil, fmt.Errorf("task %s: loading its own baseline kernel: %w", task.Name, lerr)
		}
		cleanup(func() error { return dev.WithContext(baseModule.Close) })
		var baseFn driver.Function
		if gerr := dev.WithContext(func() error {
			var e error
			baseFn, e = baseModule.Function(task.Entry)
			return e
		}); gerr != nil {
			return nil, fmt.Errorf("task %s: its baseline kernel lacks entry %q: %w", task.Name, task.Entry, gerr)
		}
		arms = append(arms, struct {
			name string
			fn   driver.Function
			out  **TimingReport
		}{name: "baseline", fn: baseFn, out: &rep.Baseline})
	}

	if terr := measureArms(timer, stream, cfg, launch, arms); terr != nil {
		return rep.fail(terr), nil
	}
	for _, arm := range arms {
		(*arm.out).summarize()
	}

	if rep.Baseline != nil && rep.Timing != nil && rep.Timing.Resolved && rep.Baseline.Resolved &&
		rep.Timing.MedianNanosPerLaunch > 0 {
		rep.Speedup = &SpeedupReport{
			Method: fmt.Sprintf("ABBA interleaved, %d samples per arm, median of samples", cfg.Samples),
			Ratio:  rep.Baseline.MedianNanosPerLaunch / rep.Timing.MedianNanosPerLaunch,
			Note:   "baseline median / candidate median; greater than 1 means the candidate is faster. Both arms ran on the same uploaded inputs under the same launch geometry.",
		}
	} else {
		rep.Unmeasured = append(rep.Unmeasured,
			"speedup: no baseline arm resolved, so no ratio is reported")
	}

	rep.Outcome = OutcomeOK
	return rep, nil
}

// buffers holds the device memory for one task instance.
type buffers struct {
	out, a, b cudamem.Slice[float32]
}

func allocBuffers(dev *cuda.Device, n int) (buffers, error) {
	var bufs buffers
	var err error
	if bufs.out, err = cudamem.Alloc[float32](dev, n); err != nil {
		return bufs, err
	}
	if bufs.a, err = cudamem.Alloc[float32](dev, n); err != nil {
		return bufs, errors.Join(err, bufs.out.Free())
	}
	if bufs.b, err = cudamem.Alloc[float32](dev, n); err != nil {
		return bufs, errors.Join(err, bufs.a.Free(), bufs.out.Free())
	}
	return bufs, nil
}

func (b buffers) free() error {
	return errors.Join(b.b.Free(), b.a.Free(), b.out.Free())
}

// check runs the correctness trials. It reports whether the candidate passed;
// on any failure it has already filled in the report's outcome.
func (r *Report) check(task *Task, cfg Config, bufs buffers, stream *cuda.Stream,
	launch func(driver.Function) error, fn driver.Function) bool {

	cr := &CorrectnessReport{
		Reference: task.ReferenceDoc, Tolerance: task.Tolerance, Elements: task.N,
		OutputPrefill: "NaN before every launch, so an element the kernel never writes is counted as non-finite rather than read as a correct zero",
	}
	r.Correctness = cr

	prefill := make([]float32, task.N)
	nan := float32(math.NaN())
	for i := range prefill {
		prefill[i] = nan
	}
	got := make([]float32, task.N)

	for i := 0; i < cfg.Trials; i++ {
		seed := cfg.Seed + int64(i)
		a, b := genInputs(seed, task.N)
		want := task.Reference(a, b, task.Alpha)

		if err := errors.Join(
			bufs.a.CopyFromHost(a), bufs.b.CopyFromHost(b), bufs.out.CopyFromHost(prefill),
		); err != nil {
			r.fail(err)
			return false
		}
		if err := launch(fn); err != nil {
			r.fail(err)
			return false
		}
		ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
		serr := stream.Synchronize(ctx)
		cancel()
		if serr != nil {
			r.fail(serr)
			return false
		}
		if err := bufs.out.CopyToHost(got); err != nil {
			r.fail(err)
			return false
		}
		cmp, cerr := Compare(got, want, task.Tolerance)
		if cerr != nil {
			r.fail(cerr)
			return false
		}
		cr.Trials = append(cr.Trials, TrialReport{Seed: seed, Comparison: cmp})
		if !cmp.OK() {
			r.Outcome, r.Recognized = OutcomeWrongAnswer, true
			r.Reason = fmt.Sprintf("trial seed %d: %d of %d elements outside tolerance, %d non-finite",
				seed, cmp.Mismatches, cmp.Elements, cmp.NonFinite)
			return false
		}
	}
	cr.Passed = true
	return true
}

// measureArms runs the arms interleaved. With two arms the order is ABBA, so a
// drift over the measurement window falls on both arms alike instead of on
// whichever ran first.
func measureArms(timer *cuda.EventTimer, stream *cuda.Stream, cfg Config,
	launch func(driver.Function) error,
	arms []struct {
		name string
		fn   driver.Function
		out  **TimingReport
	}) error {

	reports := make([]*TimingReport, len(arms))
	for i, arm := range arms {
		reports[i] = &TimingReport{
			Source: "cuda_event_device_span", Warmup: cfg.Warmup,
			LaunchesPerSample: cfg.LaunchesPerSample,
			InputPolicy:       "one host input set uploaded once and reused by every sample of every arm; device cache state is uncontrolled and not measured",
		}
		*arm.out = reports[i]
		for j := 0; j < cfg.Warmup; j++ {
			if err := launch(arm.fn); err != nil {
				return err
			}
		}
	}
	if err := stream.Synchronize(nil); err != nil {
		return err
	}

	wallStart := time.Now()
	for s := 0; s < cfg.Samples; s++ {
		// Reverse the arm order on odd samples: ABBA over each pair.
		order := make([]int, len(arms))
		for i := range order {
			if s%2 == 0 {
				order[i] = i
			} else {
				order[i] = len(arms) - 1 - i
			}
		}
		for _, i := range order {
			fn := arms[i].fn
			d, err := timer.Measure(func() error {
				for k := 0; k < cfg.LaunchesPerSample; k++ {
					if lerr := launch(fn); lerr != nil {
						return lerr
					}
				}
				return nil
			})
			if err != nil {
				return err
			}
			reports[i].SampleNanos = append(reports[i].SampleNanos, d.Nanoseconds())
		}
	}
	wall := time.Since(wallStart).Nanoseconds()
	for _, rep := range reports {
		rep.WallNanos = wall
	}
	return nil
}

func describeDevice(dev *cuda.Device) *DeviceReport {
	d := &DeviceReport{Ordinal: dev.Ordinal(), Compute: "unknown"}
	if name, err := dev.Name(); err == nil {
		d.Name = name
	}
	if major, minor, err := dev.ComputeCapability(); err == nil {
		d.Compute = fmt.Sprintf("sm_%d%d", major, minor)
	}
	if n, err := dev.MultiprocessorCount(); err == nil {
		d.SMCount = n
	}
	return d
}
