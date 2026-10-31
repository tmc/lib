package cudapdl_test

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"
	"unsafe"
	"weak"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudapdl"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Bias is real consumer parameter storage. The padding excludes tiny-object
// allocation, but does not make weak.Pointer clearing deterministic.
type pdlParameterOwner struct {
	Bias float32
	pad  [124]byte
}

func TestPDLPackedPointerContract(t *testing.T) {
	driverLoaded, compilerLoaded := driver.Loaded(), nvrtc.Loaded()
	owner := &pdlParameterOwner{Bias: nativePDLBias}
	if unsafe.Sizeof(*owner) != 128 {
		t.Fatal("parameter owner must occupy 128 bytes")
	}
	launches, err := newPDLReferenceFixture("sm_90").launches(cudalaunch.Kernel(11), cudalaunch.Kernel(22), pdlFixturePointer(1), pdlFixturePointer(2), pdlFixturePointer(3))
	if err != nil {
		t.Fatal(err)
	}
	launches[1].args[2] = &owner.Bias
	packed, err := cudalaunch.PackArgs(launches[1].args...)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != 5 || packed[2] != unsafe.Pointer(&owner.Bias) {
		t.Fatal("consumer bias must use the caller's scalar storage")
	}
	owner = nil
	launches = [2]pdlReferenceLaunch{}
	runtime.GC()
	if got := *(*float32)(packed[2]); got != nativePDLBias {
		t.Fatalf("packed bias=%g, want %g", got, nativePDLBias)
	}
	runtime.KeepAlive(packed)
	if driver.Loaded() != driverLoaded || nvrtc.Loaded() != compilerLoaded {
		t.Fatal("packing loaded a vendor library")
	}
}

const pdlOwnerDelaySource = `
extern "C" __global__ void pdl_owner_delay(unsigned long long ticks) {
    unsigned long long start = clock64();
    while (clock64() - start < ticks) {}
}
`

func drainPDLOwnerStream(stream *cuda.Stream) error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return stream.Synchronize(ctx)
}

func loadPDLOwnerKernel(t *testing.T, dev *cuda.Device, custody *pdlFixtureCustody, source, name string) *cuda.Kernel {
	t.Helper()
	compiled, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	custody.ownError(name+" compiler", err)
	if err != nil {
		t.Fatalf("compile %s: %v", name, err)
	}
	module, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if module != nil {
		custody.own(name+" module", module.Close)
	}
	custody.ownError(name+" module load", err)
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	kernel, err := module.Function(name)
	if err != nil {
		t.Fatal(err)
	}
	return kernel
}

// Only the executable and weak pointer escape this frame. In particular, a
// bound graph.Destroy callback, Node, or launch argument slice must not escape:
// each would keep the original packed owner alive independently of Exec.
//
//go:noinline
func instantiatePDLPackedOwner(t *testing.T, dev *cuda.Device, custody *pdlFixtureCustody, fixture pdlReferenceFixture, arm pdlReferenceArm, producer, consumer *cuda.Kernel, input, intermediate, output cudamem.Slice[float32]) (*cudagraph.Exec, weak.Pointer[pdlParameterOwner]) {
	t.Helper()
	owner := &pdlParameterOwner{Bias: nativePDLBias}
	weakOwner := weak.Make(owner)
	launches, err := fixture.launches(cudalaunch.OwnedKernel(producer), cudalaunch.OwnedKernel(consumer), input, intermediate, output)
	if err != nil {
		t.Fatal(err)
	}
	launches[1].args[2] = &owner.Bias
	g, err := cudagraph.New(dev)
	if g != nil {
		// Capture the nullable holder, never a bound method value. On success
		// this callback remains in custody but retains no source graph.
		custody.own("source graph", func() error {
			if g == nil {
				return nil
			}
			if err := g.Destroy(); err != nil {
				return err
			}
			g = nil
			return nil
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	prodNode, err := g.AddLaunch(launches[0].desc, launches[0].args)
	if err != nil {
		t.Fatal(err)
	}
	consNode, err := g.AddLaunch(launches[1].desc, launches[1].args)
	if err != nil {
		t.Fatal(err)
	}
	composition, err := cudapdl.Compose(g, prodNode, consNode, fixture.eligibility(arm.programmatic))
	if err != nil {
		t.Fatal(err)
	}
	if composition.Policy != arm.wantPolicy {
		t.Fatalf("policy=%s, want %s: %s", composition.Policy, arm.wantPolicy, composition.FallbackReason)
	}
	exec, err := g.Instantiate()
	if exec != nil {
		custody.own("graph executable", exec.Destroy)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := g.Destroy(); err != nil {
		custody.unresolved = fmt.Errorf("source graph destruction unresolved: %w", err)
		t.Fatal(custody.unresolved)
	}
	g = nil
	// The helper frame, its Nodes, owner and launches cease to be roots on
	// return. Do not call weakOwner.Value here to acquire another strong alias.
	return exec, weakOwner
}

// This is a physical lifetime observation, not a deterministic omission kill.
// CUDA may copy parameter values, and weak.Value need not eventually be nil.
// A ready/error witness refuses qualification without retrying either arm.
func TestNativePDLPackedOwnerWhilePending(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("SKIPPED, NOT VERIFIED: CUDA_COMPILER_TEST=1 and a separately authorized native gate required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: inherited GPU lock required: %v", err)
	}
	if unresolvedPDLFixtures.unresolved() {
		t.Fatal("previous fixture ownership unresolved; refusing submission")
	}
	admission, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dev, deviceOwner := cudatest.RequireOwned(admission, t)
	custody := new(pdlFixtureCustody)
	var stream *cuda.Stream
	deviceOwner.Defer(func() error {
		err := custody.release(0, func() error { return drainPDLOwnerStream(stream) })
		if err != nil {
			unresolvedPDLFixtures.keep(custody)
		}
		return err // DeviceOwner retains its device and lease on failure.
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if major < 9 || version < 12030 || !driver.HasGraphAddDependenciesV2() {
		t.Skip("SKIPPED, NOT VERIFIED: SM90+ and CUDA12.3+ typed graph edges required")
	}
	if err := nvrtc.Load(""); err != nil {
		t.Fatal(err)
	}
	compilerMajor, compilerMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	narch, err := nvrtc.GetNumSupportedArchs()
	if err != nil || narch <= 0 {
		t.Fatalf("NVRTC architecture count=%d: %v", narch, err)
	}
	archs := make([]int32, narch)
	if err := nvrtc.GetSupportedArchs(&archs[0]); err != nil {
		t.Fatal(err)
	}
	if reason := pdlReferenceUnsupported(major, minor, version, true, compilerMajor, compilerMinor, archs); reason != "" {
		t.Skipf("SKIPPED, NOT VERIFIED: %s", reason)
	}
	fixture := newPDLReferenceFixture(fmt.Sprintf("sm_%d%d", major, minor))
	t.Logf("target=%s driver=%d NVRTC=%d.%d driverPath=%q compilerPath=%q", fixture.target, version, compilerMajor, compilerMinor, driver.Path(), nvrtc.Path())
	stream, err = dev.NewStream()
	if stream != nil {
		custody.own("stream", stream.Close)
	}
	if err != nil {
		if stream == nil {
			custody.unresolved = fmt.Errorf("stream constructor custody unresolved: %w", err)
		}
		t.Fatal(err)
	}
	producer := loadPDLOwnerKernel(t, dev, custody, fixture.producer.Source, fixture.producer.Name)
	consumer := loadPDLOwnerKernel(t, dev, custody, fixture.consumer.Source, fixture.consumer.Name)
	delay := loadPDLOwnerKernel(t, dev, custody, pdlOwnerDelaySource, "pdl_owner_delay")
	var buffers [3]cudamem.Slice[float32]
	for i := range buffers {
		buffers[i], err = cudamem.Alloc[float32](dev, nativePDLN)
		custody.own("buffer", buffers[i].Free)
		if err != nil {
			t.Fatal(err)
		}
	}
	input, want := pdlReferenceInputs()
	poison := make([]float32, nativePDLN)
	for i := range poison {
		poison[i] = -12345
	}
	for _, arm := range fixture.arms {
		if !t.Run(arm.name, func(t *testing.T) {
			mark := len(custody.owners)
			defer func() {
				if err := custody.release(mark, func() error { return drainPDLOwnerStream(stream) }); err != nil {
					t.Error(err)
				}
			}()
			for i, data := range [][]float32{input, poison, poison} {
				if err := buffers[i].CopyFromHost(data); err != nil {
					t.Fatal(err)
				}
			}
			exec, weakOwner := instantiatePDLPackedOwner(t, dev, custody, fixture, arm, producer, consumer, buffers[0], buffers[1], buffers[2])
			witness, err := dev.NewEventWithFlags(driver.EventDisableTiming)
			if witness != nil {
				custody.own("pending witness", witness.Close)
			}
			if err != nil {
				if witness == nil {
					custody.unresolved = fmt.Errorf("event constructor custody unresolved: %w", err)
				}
				t.Fatal(err)
			}
			// Exactly one finite one-thread delay per arm. No polling for a
			// desired liveness outcome and no update while work is pending.
			if err := custody.launch(func() error {
				return cudalaunch.OwnedKernel(delay).Grid(1, 1, 1).Block(1, 1, 1).On(stream).Launch(uint64(1 << 27))
			}); err != nil {
				t.Fatal(err)
			}
			if err := witness.Record(stream); err != nil {
				t.Fatal(err)
			}
			if err := custody.launch(func() error { return exec.Launch(stream) }); err != nil {
				t.Fatal(err)
			}
			runtime.GC()
			alive := weakOwner.Value() != nil
			ready, queryErr := witness.Query()
			runtime.KeepAlive(exec)
			t.Logf("owner_alive=%v earlier_witness_ready=%v query_error=%v; overlap unmeasured", alive, ready, queryErr)
			if queryErr != nil || ready {
				t.Fatalf("NOT VERIFIED: no pending witness after owner observation: ready=%v err=%v", ready, queryErr)
			}
			if !alive {
				t.Fatal("packed owner released while graph completion was proved pending")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if err := custody.observe(func() error { return stream.Synchronize(ctx) }); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, nativePDLN)
			if err := buffers[2].CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("out[%d]=%g, want %g", i, got[i], want[i])
				}
			}
		}) {
			return // No retry or second arm after uncertainty/failure.
		}
	}
}
