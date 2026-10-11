//go:build linux

package cuptirange

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/internal/cuptiowner"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var retainedDisableDiagnostic []any

// TestPhysicalDisableRetry deliberately supplies one invalid StructSize. It is
// not spontaneous backend-failure qualification or a test of Plan.pending.
func TestPhysicalDisableRetry(t *testing.T) {
	if os.Getenv("CUPTIBENCH_PHYSICAL_TEST") != "1" {
		t.Skip("unqualified: requires exclusive device and CUPTIBENCH_PHYSICAL_TEST=1")
	}
	var keep []any
	var closers []func() error
	retain := false
	t.Cleanup(func() {
		if !retain {
			for i := len(closers) - 1; i >= 0; i-- {
				if err := closers[i](); err != nil {
					t.Error(err)
					retain = true
					break
				}
			}
		}
		if retain {
			retainedDisableDiagnostic = append(retainedDisableDiagnostic, keep...)
			retainedDisableDiagnostic = append(retainedDisableDiagnostic, closers)
			t.Log("retained native resources until process exit; no cleanup loop")
		}
	})
	check := func(err error) {
		t.Helper()
		if errors.Is(err, errors.ErrUnsupported) {
			t.Skipf("installed CUPTI has no qualified layout: %v", err)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	own := func(v any, close func() error) { keep = append(keep, v); closers = append(closers, close) }
	dev, err := cuda.NewDevice(0)
	check(err)
	own(dev, dev.Close)
	stream, err := dev.NewStreamWithFlags(driver.StreamNonBlocking)
	check(err)
	own(stream, stream.Close)
	out, err := cudamem.Alloc[uint32](dev, 32)
	check(err)
	own(out, out.Free)
	const source = `extern "C" __global__ void disable_control(unsigned *out) { out[threadIdx.x]=threadIdx.x+17; }`
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	check(err)
	module, err := cm.Load(dev)
	check(err)
	own(module, func() error { return dev.WithContext(module.Close) })
	var fn driver.Function
	check(dev.WithContext(func() error { var err error; fn, err = module.Function("disable_control"); return err }))
	var syncFailure error
	synchronize := func() error {
		if syncFailure != nil {
			return syncFailure
		}
		syncFailure = stream.Synchronize(nil)
		if syncFailure != nil {
			retain = true
		}
		return syncFailure
	}
	poison := make([]uint32, 32)
	for i := range poison {
		poison[i] = ^uint32(0)
	}
	output := func() error {
		values := make([]uint32, 32)
		if err := out.CopyToHost(values); err != nil {
			return err
		}
		for i, v := range values {
			if v != uint32(i+17) {
				return fmt.Errorf("output[%d]=%d want=%d", i, v, i+17)
			}
		}
		return nil
	}
	replay := Replay{Prepare: func(uint) error { return out.CopyFromHost(poison) }, Run: func(uint, *Scope) error { return cudalaunch.Kernel(fn).Block(32, 1, 1).On(stream).Launch(out) }, Synchronize: synchronize}
	availability, err := CounterAvailability(dev, 16<<20)
	check(err)
	var chip string
	check(dev.WithContext(func() error { var err error; chip, err = cuptisass.DeviceChipName(uint32(dev.Ordinal())); return err }))
	plan, err := NewPlan(chip, availability, []string{"sm__ctas_launched.sum"}, 16<<20)
	if plan != nil {
		own(plan, plan.Close)
	}
	check(err)
	// Drain before any teardown; synchronization failure is never retried.
	closers = append(closers, synchronize)
	limits := Limits{MaxRanges: 8, MaxRangeTreeNodes: 8, MaxCounterBytes: 32 << 20, MaxPasses: 64}
	options := CaptureOptions{Mode: Whole, ReplayMode: UserReplay, MaxDepth: 1, CachePolicy: CachePreserve}
	version, err := cupti.GetVersion()
	check(err)
	uuid, err := dev.Raw().Uuid()
	check(err)
	driverVersion, err := driver.DriverGetVersion()
	check(err)
	t.Logf("device_uuid=%x driver_api=%d metric=sm__ctas_launched.sum replay=user mode=whole", uuid, driverVersion)
	t.Logf("deliberate invalid-argument injection chip=%s cupti=%d availability_sha256=%x source_sha256=%x disable_size=%d limits=%+v", chip, version, sha256.Sum256(availability), sha256.Sum256([]byte(source)), unsafe.Sizeof(cupti.RangeProfilerDisableParams{}), limits)
	public := func(label string) {
		t.Helper()
		c, err := plan.CaptureReplay(dev, limits, options, replay)
		if c != nil {
			keep = append(keep, c)
		}
		if err != nil {
			retain = true
			t.Fatalf("%s public capture: %v", label, err)
		}
		check(output())
		if c == nil || c.Ranges != 1 || !c.Quality.Decoded || !c.Quality.AllPassesSubmitted || c.Quality.DroppedRanges != 0 {
			t.Fatalf("%s capture=%+v", label, c)
		}
		values, err := plan.Evaluate(c.Image, 0)
		check(err)
		if len(values) != 1 || values[0] != 1 {
			t.Fatalf("%s CTAs=%v want=[1]", label, values)
		}
		check(c.Close())
		t.Logf("%s quality=%+v CTAs=%v", label, c.Quality, values)
	}
	public("baseline")
	ctx, err := dev.PrimaryContext()
	check(err)
	c := &Capture{context: ctx}
	keep = append(keep, c)
	// From this point, failed assertions retain dependencies unless correct retry
	// proves native teardown succeeded. No deferred Close can retry a bad pointer.
	retain = true
	c.lease, err = cuptiowner.AcquireDevice(uint32(dev.Ordinal()))
	check(err)
	injection := &disableInjection{call: cupti.RangeProfilerDisable}
	var originalObject uintptr
	err = ctx.Run(func() error {
		store, err := newCounterStore(driver.Context(ctx.Raw()), []string{"sm__ctas_launched.sum"}, limits.MaxRanges, limits.MaxRangeTreeNodes, limits.MaxCounterBytes, counterAPI{cupti.ProfilerInitialize, cupti.RangeProfilerEnable, injection.disable, cupti.RangeProfilerGetCounterDataSize, cupti.RangeProfilerCounterDataImageInitialize})
		c.store = store
		if err != nil {
			return err
		}
		originalObject = store.host
		count, info, err := store.captureReplay(plan.config, limits, options, replay, true)
		c.RangeInfo = info
		c.retainCollection(store, count, err)
		return err
	})
	check(err)
	check(output())
	store := c.store
	if c.Ranges != 1 || !c.Quality.Decoded || !c.Quality.AllPassesSubmitted || c.Quality.DroppedRanges != 0 {
		t.Fatalf("injection precondition capture=%+v", c)
	}
	// Validate only before the failure demotes this image.
	values, err := plan.Evaluate(c.Image, 0)
	check(err)
	if len(values) != 1 || values[0] != 1 {
		t.Fatalf("pre-injection CTAs=%v", values)
	}
	failure := c.finishCollection(nil)
	t.Logf("first disable attempts=%+v error=%v quality=%+v diagnostic_bytes=%d diagnostic_sha256=%x", injection.attempts, failure, c.Quality, len(c.DiagnosticImage), sha256.Sum256(c.DiagnosticImage))
	if failure == nil {
		// Native success really destroyed the object; finishCollection already cleared
		// its handle. Never issue another Disable against originalObject.
		retain = false
		t.Fatal("malformed native Disable unexpectedly succeeded; no retry")
	}
	if !errors.Is(failure, cupti.ErrInvalidParameter) {
		t.Fatalf("unexpected native rejection: %v", failure)
	}
	if len(injection.attempts) != 1 || injection.attempts[0].Size != 0 || injection.attempts[0].Object != originalObject || originalObject == 0 {
		t.Fatal("invalid injection receipt")
	}
	if c.store != store || store.host != originalObject || store.names == nil || len(store.image) == 0 || c.context != ctx || c.lease == nil {
		t.Fatal("failed Disable released native dependencies")
	}
	if len(c.Image) != 0 || c.Ranges != 0 || len(c.RangeInfo) != 0 || len(c.DiagnosticImage) == 0 || !bytes.Equal(c.DiagnosticImage, store.image) || &c.DiagnosticImage[0] == &store.image[0] {
		t.Fatal("failed cleanup image not independently demoted")
	}
	diagnostic := append([]byte(nil), c.DiagnosticImage...)
	lease, err := cuptiowner.AcquireDevice(uint32(dev.Ordinal()))
	if err == nil {
		lease.Release()
		t.Fatal("failed Disable released device lease")
	}
	// Exactly one correct retry on the same object.
	check(c.Close())
	if len(injection.attempts) != 2 || injection.attempts[1].Object != originalObject || injection.attempts[1].Size != uint(unsafe.Sizeof(cupti.RangeProfilerDisableParams{})) || injection.attempts[1].Err != nil {
		t.Fatalf("retry attempts=%+v", injection.attempts)
	}
	if store.host != 0 || store.names != nil || c.context != nil || c.lease != nil || c.store != nil {
		t.Fatal("successful retry retained native owner")
	}
	retain = false
	check(c.Close())
	if len(injection.attempts) != 2 {
		t.Fatal("idempotent Close issued another Disable")
	}
	if len(c.Image) != 0 || c.Ranges != 0 || !bytes.Equal(c.DiagnosticImage, diagnostic) {
		t.Fatal("retry promoted rejected measurement")
	}
	lease, err = cuptiowner.AcquireDevice(uint32(dev.Ordinal()))
	check(err)
	lease.Release()
	t.Logf("correct same-object retry attempts=%+v; image remains diagnostic-only", injection.attempts)
	public("after-retry")
}
