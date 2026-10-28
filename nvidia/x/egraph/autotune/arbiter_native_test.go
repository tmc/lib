package autotune_test

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// arbiterSteps selects which parts of the CUPTI collector's per-candidate
// sequence a sampling arbiter run performs. The zero value is the bare
// sequence of TestNativeSessionsShareContext: inventory enable, then a
// started and closed session with no module and no launches.
type arbiterSteps struct {
	module        bool // load a cubin and launch its kernel under sampling
	moduleOnce    bool // load one module before the first session, unload after the last
	keepModules   bool // unload every session's module only after the last session
	sameCubin     bool // every session uses the first candidate's cubin
	compileAhead  bool // build every cubin before the first session
	warmup        bool // launch once before sampling, as SampleKernel does
	timing        bool // time Iters launches with CUDA events before sampling
	inventoryOnce bool // query the reason inventory once, not per session
	noLaunch      bool // load the module but launch nothing while sampling
	keepSessions  bool // keep every Session reachable until the last session ends
	sync          bool // synchronize the context after each disable and Close
}

// arbiterVariants starts from the collector's sequence and removes one
// step at a time. "collector" calls CUPTICollector.SampleKernel itself;
// "replica" is the same sequence written out step by step.
var arbiterVariants = []struct {
	name  string
	steps arbiterSteps
	real  bool
}{
	{name: "bare", steps: arbiterSteps{}},
	{name: "replica", steps: arbiterSteps{module: true, warmup: true, timing: true}},
	{name: "same_cubin", steps: arbiterSteps{module: true, warmup: true, timing: true, sameCubin: true}},
	{name: "module_once", steps: arbiterSteps{module: true, warmup: true, timing: true, sameCubin: true, moduleOnce: true}},
	{name: "keep_modules", steps: arbiterSteps{module: true, warmup: true, timing: true, keepModules: true}},
	{name: "compile_ahead", steps: arbiterSteps{module: true, warmup: true, timing: true, compileAhead: true}},
	{name: "no_warmup", steps: arbiterSteps{module: true, timing: true}},
	{name: "no_timing", steps: arbiterSteps{module: true, warmup: true}},
	{name: "inventory_once", steps: arbiterSteps{module: true, warmup: true, timing: true, inventoryOnce: true}},
	{name: "no_sampled_launch", steps: arbiterSteps{module: true, warmup: true, timing: true, noLaunch: true}},
	{name: "replica_keep_sessions", steps: arbiterSteps{module: true, warmup: true, timing: true, keepSessions: true}},
	{name: "replica_sync", steps: arbiterSteps{module: true, warmup: true, timing: true, sync: true}},
	{name: "collector", real: true},
}

// arbiterCandidates are the first non-spilling candidates the daemon tries.
var arbiterCandidates = []string{"t32_k8_s2", "t32_k16_s3", "t16_k16_s3", "t32_k16_s2"}

// TestAutotuneSamplingArbiter runs four PC-sampling sessions on one primary
// context for each variant in arbiterVariants and reports the first session
// whose step fails. Each variant opens its own device, so its primary
// context is released before the next one starts unless something else
// retains it; run variants in separate processes for full isolation.
// It is gated like TestAutotuneNativeQualification.
func TestAutotuneSamplingArbiter(t *testing.T) {
	if os.Getenv(autotuneNativeEnv) != "1" {
		t.Skip("native arbiter pending: CUDA_EGRAPH_AUTOTUNE_NATIVE=1 required")
	}
	if err := checkAutotuneNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	nvdisasm, err := cudatool.LookTool("nvdisasm", "")
	if err != nil {
		t.Fatalf("nvdisasm: %v", err)
	}
	for _, v := range arbiterVariants {
		t.Run(v.name, func(t *testing.T) {
			dev, err := cuda.NewDevice(0)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := dev.WithContext(driver.ProfilerStop); err != nil {
					t.Errorf("ProfilerStop: %v", err)
				}
				if err := dev.Close(); err != nil {
					t.Error(err)
				}
			}()
			major, minor, err := dev.ComputeCapability()
			if err != nil {
				t.Fatal(err)
			}
			twin, err := autotune.AutoDiscoverTarget(major, minor, false, false)
			if err != nil {
				t.Fatal(err)
			}
			a := arbiter{
				t:   t,
				dev: dev,
				builder: autotune.NVRTCBuilder{
					Twin:         twin,
					Target:       twin.Target,
					Entry:        gemmEntry,
					Source:       gemmSourceFor,
					Disassembler: cudasass.Disassembler{Path: nvdisasm},
				},
			}
			if err := a.run(v.steps, v.real); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type arbiter struct {
	t       *testing.T
	dev     *cuda.Device
	builder autotune.NVRTCBuilder
	a, b, c cudamem.Slice[float32]
}

const arbiterDim, arbiterIters = 256, 20

func (ar *arbiter) run(steps arbiterSteps, real bool) (err error) {
	t := ar.t
	cands := make([]autotune.ScheduleCandidate, 0, len(arbiterCandidates))
	for _, name := range arbiterCandidates {
		for _, c := range gemmCandidates() {
			if c.Name == name {
				cands = append(cands, c)
			}
		}
	}
	const n = arbiterDim
	for _, p := range []*cudamem.Slice[float32]{&ar.a, &ar.b, &ar.c} {
		buf, err := cudamem.Alloc[float32](ar.dev, n*n)
		if err != nil {
			return err
		}
		defer buf.Free()
		if err := buf.CopyFromHost(make([]float32, n*n)); err != nil {
			return err
		}
		*p = buf
	}
	launch := func(fn driver.Function, c autotune.ScheduleCandidate) error {
		tm := uint32(c.TileM)
		return cudalaunch.Kernel(fn).Grid(n/tm, n/tm, 1).Block(16, 16, 1).
			Launch(ar.a, ar.b, ar.c, int32(n), int32(n), int32(n))
	}

	if real {
		col := autotune.CUPTICollector{Device: ar.dev, Run: launch}
		for i, c := range cands {
			v, err := ar.builder.Build(t.Context(), c)
			if err != nil {
				return fmt.Errorf("session %d: build %s: %w", i, c.Name, err)
			}
			obs, h, err := col.SampleKernel(t.Context(), v, arbiterIters)
			if err != nil {
				return fmt.Errorf("session %d (%s): %w", i, c.Name, err)
			}
			t.Logf("session %d %s: %.2fus regs=%d samples=%d", i, c.Name, obs.MedianUS, obs.Metrics.RegisterCount, h.TotalSamples)
		}
		return nil
	}

	cubin := func(i int) ([]byte, error) {
		c := cands[i]
		if steps.sameCubin {
			c = cands[0]
		}
		v, err := ar.builder.Build(t.Context(), c)
		return v.Cubin, err
	}
	var ahead [][]byte
	if steps.compileAhead {
		for i := range cands {
			b, err := cubin(i)
			if err != nil {
				return fmt.Errorf("build %d: %w", i, err)
			}
			ahead = append(ahead, b)
		}
	}
	var kept []driver.Module
	defer func() {
		err = errors.Join(err, ar.dev.WithContext(func() error {
			var errs []error
			for _, m := range kept {
				errs = append(errs, m.Close())
			}
			return errors.Join(errs...)
		}))
	}()
	var inv *cuptipcsampling.ReasonInventory
	var sessions []*cuptipcsampling.Session
	defer func() { runtime.KeepAlive(sessions) }()
	for i := range cands {
		var bin []byte
		if steps.module && !(steps.moduleOnce && i > 0) {
			if steps.compileAhead {
				bin = ahead[i]
			} else {
				b, err := cubin(i)
				if err != nil {
					return fmt.Errorf("session %d: build: %w", i, err)
				}
				bin = b
			}
		}
		err := ar.dev.WithContext(func() (err error) {
			var fn driver.Function
			if steps.module {
				var mod driver.Module
				switch {
				case steps.moduleOnce && i > 0:
					mod = kept[0]
				default:
					m, loadErr := driver.ModuleLoadData(bin)
					if loadErr != nil {
						return fmt.Errorf("load module: %w", loadErr)
					}
					mod = m
					if steps.moduleOnce || steps.keepModules {
						kept = append(kept, m)
					} else {
						defer func() { err = errors.Join(err, m.Close()) }()
					}
				}
				f, err := mod.Function(gemmEntry)
				if err != nil {
					return err
				}
				fn = f
				if _, err := fn.Attribute(driver.FuncAttributeNumRegs); err != nil {
					return err
				}
				if steps.warmup {
					if err := launch(fn, cands[i]); err != nil {
						return fmt.Errorf("warmup: %w", err)
					}
				}
				if steps.timing {
					if err := timeLaunches(func() error { return launch(fn, cands[i]) }, arbiterIters); err != nil {
						return fmt.Errorf("timing: %w", err)
					}
				}
			}
			cuCtx, err := driver.CtxGetCurrent()
			if err != nil {
				return err
			}
			if inv == nil || !steps.inventoryOnce {
				disable, err := cuptipcsampling.Enable(cuCtx)
				if err != nil {
					if disable != nil {
						err = errors.Join(err, disable())
					}
					return fmt.Errorf("enable for inventory: %w", err)
				}
				got, err := cuptipcsampling.StallReasons(cuCtx, 1<<10)
				if err = errors.Join(err, disable()); err != nil {
					return fmt.Errorf("inventory: %w", err)
				}
				inv = &got
				if steps.sync {
					if err := driver.CtxSynchronize(); err != nil {
						return fmt.Errorf("synchronize after inventory disable: %w", err)
					}
				}
			}
			s, err := sampleSession(cuCtx, *inv, func() error {
				if !steps.module || steps.noLaunch {
					return nil
				}
				for range arbiterIters {
					if err := launch(fn, cands[i]); err != nil {
						return err
					}
				}
				return nil
			})
			if steps.keepSessions {
				sessions = append(sessions, s)
			}
			if err == nil && steps.sync {
				if err := driver.CtxSynchronize(); err != nil {
					return fmt.Errorf("synchronize after close: %w", err)
				}
			}
			return err
		})
		if err != nil {
			return fmt.Errorf("session %d (%s): %w", i, cands[i].Name, err)
		}
		t.Logf("session %d %s: ok", i, cands[i].Name)
	}
	return nil
}

// sampleSession is CUPTICollector's sampling sequence with its
// configuration: Start, the workload, synchronize, Stop, Read until no PCs
// remain, Close, Final. It returns the session, closed, for callers that
// keep it reachable.
func sampleSession(cuCtx driver.Context, inv cuptipcsampling.ReasonInventory, work func() error) (s *cuptipcsampling.Session, err error) {
	cfg := cuptipcsampling.Config{
		Period: 12, WorkerSleepMS: 10,
		MaxReasons:   len(inv.Reasons),
		ScratchBytes: 8 << 20, HardwareBytes: 8 << 20, MaxNativeBytes: 64 << 20,
		PCsPerBuffer: 512, MaxOutputBytes: 8 << 20, MaxReads: 256,
		MaxNames: 2*512 + 64, MaxNameBytes: 256, MaxTotalNameBytes: (2*512 + 64) * 256,
	}
	for _, r := range inv.Reasons {
		cfg.ReasonIDs = append(cfg.ReasonIDs, r.Index)
	}
	s, err = cuptipcsampling.Start(cuCtx, cfg)
	if err != nil {
		if s != nil {
			err = errors.Join(err, s.Close())
		}
		return s, fmt.Errorf("start: %w", err)
	}
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, s.Close())
		}
	}()
	if err := work(); err != nil {
		return s, fmt.Errorf("sampled launches: %w", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		return s, err
	}
	if err := s.Stop(); err != nil {
		return s, fmt.Errorf("stop: %w", err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		b, err := s.Read()
		if err != nil {
			return s, fmt.Errorf("read: %w", err)
		}
		if b.RemainingPCs == 0 || time.Now().After(deadline) {
			break
		}
	}
	closed = true
	if err := s.Close(); err != nil {
		return s, fmt.Errorf("close: %w", err)
	}
	s.Final()
	return s, nil
}

// timeLaunches records a CUDA event pair around each of iters launches, as
// CUPTICollector does before sampling.
func timeLaunches(launch func() error, iters int) (err error) {
	start, err := driver.NewEvent(driver.EventDefault)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, start.Close()) }()
	end, err := driver.NewEvent(driver.EventDefault)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, end.Close()) }()
	for range iters {
		if err := start.Record(0); err != nil {
			return err
		}
		if err := launch(); err != nil {
			return err
		}
		if err := end.Record(0); err != nil {
			return err
		}
		if err := end.Synchronize(); err != nil {
			return err
		}
		if _, err := start.ElapsedTime(end); err != nil {
			return err
		}
	}
	return nil
}
