package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// contextProbeBallast is the size of contextProbeSource's device global,
// large enough that loading and unloading the module moves cuMemGetInfo's
// free memory by far more than its usual drift.
const contextProbeBallast = 256 << 20

// contextProbeSource defines one kernel and a contextProbeBallast-byte
// device global, with C linkage so both are found by their source names.
const contextProbeSource = `extern "C" {
__device__ char contextProbeBallast[256 << 20];
__global__ void contextProbe(char *out) { out[0] = contextProbeBallast[threadIdx.x]; }
}`

// A contextProbe is what one arm of TestModuleContextExperiment observed
// for one module: the context current at each step, the results of a
// function lookup and an unload run under the arm's context, and free
// device memory before the load, after it, after the arm's unload and
// after any cleanup unload.
type contextProbe struct {
	loadErr          error
	before           driver.Context // current before the load
	afterLoad        driver.Context // current after nvrtcx Load returned
	primary          driver.Context // current inside dev.WithContext
	atCall           driver.Context // current when the arm's calls ran
	globalBytes      uint
	globalErr        error
	lookupErr        error
	sameHandle       bool // lookup equals one made under the primary context
	refErr           error
	closeErr         error
	cleanupErr       error
	cleaned          bool // the arm's unload failed and cleanup ran
	freeBefore       uint
	freeLoaded       uint
	freeClosed       uint
	freeCleaned      uint
	memErr           error
	currentAfterCall driver.Context // current after the arm's last call
}

// freeMemory returns cuMemGetInfo's free bytes, read under dev's primary
// context.
func freeMemory(dev *cuda.Device) (uint, error) {
	var free uint
	err := dev.WithContext(func() error {
		var err error
		free, _, err = driver.MemGetInfo()
		return err
	})
	return free, err
}

// current returns the context current on the calling thread, or a sentinel
// all-ones handle if the query fails, so that a failed query never reads
// as "no context".
func current() driver.Context {
	c, err := driver.CtxGetCurrent()
	if err != nil {
		return ^driver.Context(0)
	}
	return c
}

// probeModule compiles contextProbeSource, loads it onto dev, and looks up
// its kernel and unloads it with run, which runs each call under the arm's
// context. It must be called on a goroutine locked to its OS thread.
func probeModule(dev *cuda.Device, run func(func() error) error) (p contextProbe) {
	cm, err := nvrtcx.Compile(contextProbeSource,
		nvrtcx.WithName("context_probe.cu"),
		nvrtcx.WithDeviceArch(dev),
	)
	if err != nil {
		p.loadErr = fmt.Errorf("compile context probe: %w", releaseErrorOwner(err))
		return p
	}
	p.before = current()
	keep := func(free uint, err error) uint {
		if err != nil && p.memErr == nil {
			p.memErr = err
		}
		return free
	}
	p.freeBefore = keep(freeMemory(dev))
	mod, err := cm.Load(dev)
	if err != nil {
		p.loadErr = releaseErrorOwner(err)
		return p
	}
	p.afterLoad = current()
	// Under lazy loading the global may not be allocated until it is
	// used, so look it up first, under the primary context, so that
	// freeLoaded counts it. This touches data, not the kernel the arm
	// looks up.
	p.globalErr = dev.WithContext(func() error {
		p.primary = current()
		var err error
		_, p.globalBytes, err = mod.Global("contextProbeBallast")
		return err
	})
	p.freeLoaded = keep(freeMemory(dev))

	var fn driver.Function
	p.lookupErr = run(func() error {
		p.atCall = current()
		var err error
		fn, err = mod.Function("contextProbe")
		return err
	})
	// The reference lookup comes after the arm's, so that the arm's is
	// the first lookup of the kernel in the module.
	if p.lookupErr == nil {
		p.refErr = dev.WithContext(func() error {
			ref, err := mod.Function("contextProbe")
			p.sameHandle = err == nil && ref == fn
			return err
		})
	}
	p.closeErr = run(func() error {
		return mod.Close()
	})
	p.currentAfterCall = current()
	p.freeClosed = keep(freeMemory(dev))
	if mod != 0 {
		p.cleaned = true
		p.cleanupErr = dev.WithContext(func() error {
			return mod.Close()
		})
		p.freeCleaned = keep(freeMemory(dev))
	}
	return p
}

// onLockedThread runs fn on a new goroutine locked to its OS thread and
// waits for it. The goroutine exits still locked, so the runtime retires
// the thread with whatever context it holds instead of reusing it.
func onLockedThread(fn func()) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		runtime.LockOSThread()
		fn()
	}()
	<-done
}

// errField formats err for a report line: ok, or the quoted message.
func errField(err error) string {
	if err == nil {
		return "ok"
	}
	return strconv.Quote(err.Error())
}

// report formats p as the fields of an "F6 arm=" line.
func (p contextProbe) report() string {
	return fmt.Sprintf("load=%s current_before=%#x current_after_load=%#x primary=%#x current_at_call=%#x global=%s global_bytes=%d lookup=%s same_handle=%t ref_lookup=%s close=%s current_after_close=%#x cleanup_ran=%t cleanup=%s",
		errField(p.loadErr), p.before, p.afterLoad, p.primary, p.atCall,
		errField(p.globalErr), p.globalBytes, errField(p.lookupErr), p.sameHandle, errField(p.refErr),
		errField(p.closeErr), p.currentAfterCall, p.cleaned, errField(p.cleanupErr))
}

// memoryReport formats p's free-memory readings as the fields of an
// "F6 arm=D" line. Deltas are signed: load_drop is the free memory the
// load took and close_return what the arm's unload gave back.
func (p contextProbe) memoryReport() string {
	loadDrop := int64(p.freeBefore) - int64(p.freeLoaded)
	closeReturn := int64(p.freeClosed) - int64(p.freeLoaded)
	s := fmt.Sprintf("free_before=%d free_loaded=%d free_closed=%d load_drop=%d close_return=%d loaded=%t released=%t mem=%s",
		p.freeBefore, p.freeLoaded, p.freeClosed, loadDrop, closeReturn,
		loadDrop >= contextProbeBallast/2, closeReturn >= contextProbeBallast/2, errField(p.memErr))
	if p.cleaned {
		s += fmt.Sprintf(" free_cleaned=%d cleanup_return=%d", p.freeCleaned, int64(p.freeCleaned)-int64(p.freeClosed))
	}
	return s
}

// TestModuleContextExperiment is the experiment for review finding F6: the
// parity fixtures look up kernels in and unload modules loaded by nvrtcx,
// whose Load restores the context that was current before it, from
// goroutines that hold no context. It does not assert what the driver
// does then. It reports, in lines beginning "F6 arm=", what a function
// lookup and a module unload return and whether the unload frees the
// module's memory, under no context (arm A), the device's primary context
// (arm B), and another context on the same device (arm C), with free
// device memory for each (arm D). It fails only if arm A does not start
// with no context current, if arm B's calls fail, or if an arm cannot be
// set up or cleaned up.
func TestModuleContextExperiment(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
	}()
	version, verr := driver.DriverGetVersion()
	mode, merr := driver.ModuleGetLoadingMode()
	name, nerr := dev.Name()
	t.Logf("F6 arm=env driver_version=%d driver_version_err=%s loading_mode=%v loading_mode_err=%s CUDA_MODULE_LOADING=%q device=%q device_err=%s ballast=%d",
		version, errField(verr), mode, errField(merr), os.Getenv("CUDA_MODULE_LOADING"), name, errField(nerr), contextProbeBallast)

	// Arm A: no context current at any step but the load, which nvrtcx
	// runs under the primary context.
	var a contextProbe
	var aStart driver.Context
	var aStartErr error
	onLockedThread(func() {
		aStart, aStartErr = driver.CtxGetCurrent()
		if aStartErr != nil || aStart != 0 {
			return
		}
		a = probeModule(dev, func(f func() error) error { return f() })
	})
	if aStartErr != nil || aStart != 0 {
		t.Fatalf("F6 arm=A precondition: current context %#x, %v; want none", aStart, aStartErr)
	}
	t.Logf("F6 arm=A %s", a.report())

	// Arm B, the positive control: the same calls under the primary
	// context, as dev.WithContext binds it.
	var b contextProbe
	onLockedThread(func() {
		b = probeModule(dev, dev.WithContext)
	})
	t.Logf("F6 arm=B %s", b.report())

	// Arm C: another context on the same device current for the calls,
	// to show whether the driver checks the context at all.
	var c contextProbe
	var foreignID driver.Context // foreign's handle; Close zeroes foreign
	var setupErr, restoreErr, destroyErr error
	onLockedThread(func() {
		foreign, err := driver.NewContext(dev.Raw(), 0)
		if err != nil {
			setupErr = err
			return
		}
		foreignID = foreign
		// NewContext made foreign current; pop it so the load starts
		// from no context, as in arm A.
		if setupErr = driver.Context(0).SetCurrent(); setupErr != nil {
			destroyErr = foreign.Close()
			return
		}
		c = probeModule(dev, func(f func() error) error {
			if err := foreign.SetCurrent(); err != nil {
				return fmt.Errorf("set foreign context current: %w", err)
			}
			err := f()
			if rerr := driver.Context(0).SetCurrent(); rerr != nil && restoreErr == nil {
				restoreErr = rerr
			}
			return err
		})
		destroyErr = foreign.Close()
	})
	if setupErr != nil {
		t.Errorf("F6 arm=C setup: %v", setupErr)
	} else {
		t.Logf("F6 arm=C foreign=%#x %s", foreignID, c.report())
	}
	if restoreErr != nil {
		t.Errorf("F6 arm=C restore no context: %v", restoreErr)
	}
	if destroyErr != nil {
		t.Errorf("F6 arm=C destroy foreign context: %v", destroyErr)
	}

	// Arm D: free memory around each arm's load and unload. Arm B's
	// load_drop and close_return calibrate the reading; arm A's
	// close_return says whether an unload with no context frees memory.
	for _, r := range []struct {
		arm string
		p   contextProbe
		ran bool
	}{
		{"A", a, true},
		{"B", b, true},
		{"C", c, setupErr == nil},
	} {
		if r.ran && r.p.loadErr == nil {
			t.Logf("F6 arm=D measured=%s %s", r.arm, r.p.memoryReport())
		}
	}

	for _, r := range []struct {
		arm string
		p   contextProbe
	}{
		{"A", a},
		{"B", b},
		{"C", c},
	} {
		if r.p.loadErr != nil {
			t.Errorf("F6 arm=%s load: %v", r.arm, r.p.loadErr)
		}
		if r.p.cleanupErr != nil {
			t.Errorf("F6 arm=%s cleanup unload under the primary context: %v", r.arm, r.p.cleanupErr)
		}
	}
	// Positive control: under the primary context every call succeeds.
	// Whether two lookups return the same handle is reported, not
	// asserted: the driver does not promise it.
	if b.globalErr != nil || b.lookupErr != nil || b.refErr != nil || b.closeErr != nil {
		t.Errorf("F6 arm=B positive control failed: global %v, lookup %v, reference lookup %v, close %v",
			b.globalErr, b.lookupErr, b.refErr, b.closeErr)
	}
	if b.atCall != b.primary || b.primary == 0 {
		t.Errorf("F6 arm=B context at call %#x, want the primary context %#x", b.atCall, b.primary)
	}
}
