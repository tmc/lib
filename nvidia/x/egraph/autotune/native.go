package autotune

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// CountSpills returns the local-memory store (STL) and load (LDL)
// instructions in function fn of dis.
func CountSpills(dis *cudasass.Disassembly, fn string) (stl, ldl int, err error) {
	f := dis.Function(fn)
	if f == nil {
		return 0, 0, fmt.Errorf("count spills: no function %q in disassembly", fn)
	}
	for _, in := range f.Instructions {
		switch cudasass.BaseOpcode(in.Opcode) {
		case "STL":
			stl++
		case "LDL":
			ldl++
		}
	}
	return stl, ldl, nil
}

// NVRTCBuilder compiles candidates with NVRTC to a cubin for Target and
// counts spills in the entry function's SASS with nvdisasm. The source is
// [KnobPreamble] for the candidate followed by Source(c), which must define
// an extern "C" kernel named Entry.
type NVRTCBuilder struct {
	Twin         HardwareTwin
	Target       string // physical target such as "sm_121a"; default Twin.Target
	Entry        string
	Source       func(c ScheduleCandidate) (string, error)
	Disassembler cudasass.Disassembler
}

// Build compiles c and disassembles the result.
func (b NVRTCBuilder) Build(ctx context.Context, c ScheduleCandidate) (Variant, error) {
	if b.Source == nil || b.Entry == "" {
		return Variant{}, errors.New("nvrtc build: Source and Entry are required")
	}
	pre, err := KnobPreamble(b.Twin, c)
	if err != nil {
		return Variant{}, err
	}
	body, err := b.Source(c)
	if err != nil {
		return Variant{}, fmt.Errorf("nvrtc build %s: source: %w", c.Name, err)
	}
	target := b.Target
	if target == "" {
		target = b.Twin.Target
	}
	src := pre + body
	cm, err := nvrtcx.CompileContext(ctx, src,
		nvrtcx.WithName(c.Name+".cu"),
		nvrtcx.WithPhysicalArchTarget(target),
		nvrtcx.WithNoCache())
	if err != nil {
		return Variant{}, fmt.Errorf("nvrtc build %s: %w", c.Name, err)
	}
	cubin := cm.CUBIN()
	if len(cubin) == 0 {
		return Variant{}, fmt.Errorf("nvrtc build %s: no cubin for %s", c.Name, target)
	}
	out, err := b.Disassembler.Disassemble(ctx, cubin, cudasass.Options{})
	if err != nil {
		return Variant{}, fmt.Errorf("nvrtc build %s: disassemble: %w", c.Name, err)
	}
	dis, err := cudasass.Parse(bytes.NewReader(out.Stdout))
	if err != nil {
		return Variant{}, fmt.Errorf("nvrtc build %s: parse sass: %w", c.Name, err)
	}
	stl, ldl, err := CountSpills(dis, b.Entry)
	if err != nil {
		return Variant{}, fmt.Errorf("nvrtc build %s: %w", c.Name, err)
	}
	return Variant{
		Candidate:   c,
		Source:      cudahotswap.Source{Code: src, Filename: c.Name + ".cu", Entry: b.Entry},
		PTX:         cm.PTX(),
		Cubin:       cubin,
		SpillStores: stl,
		SpillLoads:  ldl,
	}, nil
}

// CUPTICollector loads a variant's cubin on Device, times Iters launches
// with CUDA events, and samples them again under CUPTI PC sampling. Run
// launches the kernel once on the device's primary context, which is
// current when it is called.
//
// Collection follows [cuptipcsampling]: Linux only, an admitted CUPTI
// version, exclusive use of PC sampling, and profiling permitted for the
// user. Callers must call [driver.ProfilerStop] before the process exits.
type CUPTICollector struct {
	Device *cuda.Device
	Run    func(fn driver.Function, c ScheduleCandidate) error
	Period uint32 // PC sampling period exponent, 5 to 31; default 12
}

// Execute loads v and launches it once, synchronously.
func (col CUPTICollector) Execute(ctx context.Context, v Variant) error {
	return col.withFunction(v, func(fn driver.Function) error {
		if err := col.Run(fn, v.Candidate); err != nil {
			return err
		}
		return driver.CtxSynchronize()
	})
}

func (col CUPTICollector) withFunction(v Variant, f func(driver.Function) error) error {
	if col.Device == nil || col.Run == nil {
		return errors.New("cupti collector: Device and Run are required")
	}
	return col.Device.WithContext(func() (err error) {
		mod, err := driver.ModuleLoadData(v.Cubin)
		if err != nil {
			return fmt.Errorf("load %s: %w", v.Candidate.Name, err)
		}
		defer func() { err = errors.Join(err, mod.Close()) }()
		fn, err := mod.Function(v.Source.Entry)
		if err != nil {
			return fmt.Errorf("function %s: %w", v.Source.Entry, err)
		}
		return f(fn)
	})
}

// SampleKernel implements [LiveCollector].
func (col CUPTICollector) SampleKernel(ctx context.Context, v Variant, iters int) (ProfileObservation, StallHistogram, error) {
	var obs ProfileObservation
	var hist StallHistogram
	err := col.withFunction(v, func(fn driver.Function) error {
		regs, err := fn.Attribute(driver.FuncAttributeNumRegs)
		if err != nil {
			return err
		}
		obs.Metrics.RegisterCount = int(regs)
		// Warm up once, then time each launch.
		if err := col.Run(fn, v.Candidate); err != nil {
			return err
		}
		if obs.MedianUS, err = col.time(fn, v.Candidate, iters); err != nil {
			return err
		}
		hist, err = col.sample(ctx, fn, v, iters)
		return err
	})
	if err != nil {
		return ProfileObservation{}, StallHistogram{}, fmt.Errorf("cupti collector %s: %w", v.Candidate.Name, err)
	}
	return obs, hist, nil
}

func (col CUPTICollector) time(fn driver.Function, c ScheduleCandidate, iters int) (us float64, err error) {
	start, err := driver.NewEvent(driver.EventDefault)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, start.Close()) }()
	end, err := driver.NewEvent(driver.EventDefault)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, end.Close()) }()
	ts := make([]float64, 0, iters)
	for range max(iters, 1) {
		if err := start.Record(0); err != nil {
			return 0, err
		}
		if err := col.Run(fn, c); err != nil {
			return 0, err
		}
		if err := end.Record(0); err != nil {
			return 0, err
		}
		if err := end.Synchronize(); err != nil {
			return 0, err
		}
		ms, err := start.ElapsedTime(end)
		if err != nil {
			return 0, err
		}
		ts = append(ts, float64(ms)*1000)
	}
	slices.Sort(ts)
	return ts[len(ts)/2], nil
}

// sample runs iters launches under PC sampling and sums the entry
// function's per-PC histograms.
func (col CUPTICollector) sample(ctx context.Context, fn driver.Function, v Variant, iters int) (h StallHistogram, err error) {
	cuCtx, err := driver.CtxGetCurrent()
	if err != nil {
		return h, err
	}
	disable, err := cuptipcsampling.Enable(cuCtx)
	if err != nil {
		if disable != nil {
			err = errors.Join(err, disable())
		}
		return h, fmt.Errorf("enable pc sampling: %w", err)
	}
	inv, err := cuptipcsampling.StallReasons(cuCtx, 1<<10)
	if err = errors.Join(err, disable()); err != nil {
		return h, fmt.Errorf("stall reasons: %w", err)
	}
	period := col.Period
	if period == 0 {
		period = 12
	}
	cfg := cuptipcsampling.Config{
		Period: period, WorkerSleepMS: 10,
		MaxReasons:   len(inv.Reasons),
		ScratchBytes: 8 << 20, HardwareBytes: 8 << 20, MaxNativeBytes: 64 << 20,
		PCsPerBuffer: 512, MaxOutputBytes: 8 << 20, MaxReads: 256,
		MaxNames: 2*512 + 64, MaxNameBytes: 256, MaxTotalNameBytes: (2*512 + 64) * 256,
	}
	for _, r := range inv.Reasons {
		cfg.ReasonIDs = append(cfg.ReasonIDs, r.Index)
	}
	session, err := cuptipcsampling.Start(cuCtx, cfg)
	if err != nil {
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		return h, fmt.Errorf("start pc sampling: %w", err)
	}
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, session.Close())
		}
	}()
	for range max(iters, 1) {
		if err := ctx.Err(); err != nil {
			return h, err
		}
		if err := col.Run(fn, v.Candidate); err != nil {
			return h, err
		}
	}
	if err := driver.CtxSynchronize(); err != nil {
		return h, err
	}
	if err := session.Stop(); err != nil {
		return h, fmt.Errorf("stop pc sampling: %w", err)
	}
	add := func(b cuptipcsampling.Batch) {
		for _, x := range HistogramsFromBatch(b, inv, v.Source.Entry) {
			h = h.Add(x)
		}
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		b, err := session.Read()
		if err != nil {
			return h, fmt.Errorf("read pc samples: %w", err)
		}
		add(b)
		if b.RemainingPCs == 0 || time.Now().After(deadline) {
			break
		}
	}
	// The final buffer is available only after Close disables sampling.
	closed = true
	if err := session.Close(); err != nil {
		return h, fmt.Errorf("close pc sampling: %w", err)
	}
	if b, ok := session.Final(); ok {
		add(b)
	}
	return h, nil
}
