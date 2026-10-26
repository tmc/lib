package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// A ready witness cannot establish pending work. Errors likewise cannot be
// interpreted as evidence of contention or incompletion.
func requirePendingWitness(ready bool, err error) error {
	if err != nil {
		return fmt.Errorf("pending witness query: %w", err)
	}
	if ready {
		return fmt.Errorf("pending native observation unproven: witness already ready")
	}
	return nil
}

const pendingDelayTicks uint64 = 1 << 27

func pendingDelayArtifact(target string) (*cudakernel.Artifact, error) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "ticks", Type: "u64"})
	if err != nil {
		return nil, err
	}
	// One finite CTA/thread. Unsigned clock subtraction tolerates wraparound;
	// no host-controlled gate or unbounded progress dependency is introduced.
	ptx := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry delay(.param .u64 ticks) {
 .reg .b64 start, now, elapsed, limit;
 .reg .pred again;
 ld.param.u64 limit, [ticks];
 mov.u64 start, %%clock64;
loop:
 mov.u64 now, %%clock64;
 sub.u64 elapsed, now, start;
 setp.lt.u64 again, elapsed, limit;
 @again bra loop;
 ret;
}
`, target)
	return cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "delay", schema, target)
}

func TestNativeCanceledPlanStillPending(t *testing.T) {
	dev, target := selectedDevice(t)
	var x *selectedExecutable
	var witness *cuda.Event
	var raw driver.Module
	var module *cuda.Module
	t.Cleanup(func() {
		if x != nil {
			if err := x.Close(); err != nil {
				t.Errorf("retain pending owners/device/borrowed lease: %v", err)
				return
			}
		}
		if witness != nil {
			if err := witness.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if module != nil {
			if err := module.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if raw != 0 {
			if err := closeFixtureRawModule(&raw, dev.WithContext, func(m driver.Module) error { return m.Close() }); err != nil {
				t.Error(err)
				return
			}
		}
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatal("target/device mismatch")
	}
	data := replayInputs(1)
	want := make([]float32, replayElements)
	for i := range want {
		want[i] = float32(float32(data["x"][i]*data["a"][i]) + data["b"][i])
	}
	x, err = newSelectedExecutable(dev, affineSelection(), target, data, want)
	if err != nil {
		t.Fatal(err)
	}
	art, err := pendingDelayArtifact(target)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = art.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	module, err = cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	raw = 0
	kernel, err := module.Function(art.EntryPoint())
	if err != nil {
		t.Fatal(err)
	}
	witness, err = dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	if err = x.poison(); err != nil {
		t.Fatal(err)
	}
	if err = cudalaunch.OwnedKernel(kernel).Grid(1, 1, 1).Block(1, 1, 1).On(x.stream).Launch(pendingDelayTicks); err != nil {
		t.Fatal(err)
	}
	// Stream order: finite delay -> witness -> entire submitted plan. This
	// witness is queried only AFTER the canceled wait has returned.
	if err = witness.Record(x.stream); err != nil {
		t.Fatal(err)
	}
	if err = x.submit(); err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err = x.completion.Wait(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled wait = %v", err)
	}
	ready, queryErr := witness.Query()
	if err = requirePendingWitness(ready, queryErr); err != nil {
		t.Fatal(err)
	} // no retry if the delay has already completed
	if err = x.output.Free(); !errors.Is(err, cudamem.ErrInUse) {
		t.Fatalf("pending output free = %v", err)
	}
	for name, b := range x.inputs {
		if err = b.Free(); !errors.Is(err, cudamem.ErrInUse) {
			t.Fatalf("pending input %s free = %v", name, err)
		}
	}
	ctx, cancelDrain := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelDrain()
	if err = x.drain(ctx); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, replayElements)
	if err = x.output.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !sameReplayOutput(got, want) {
		t.Fatal("pending plan output differs from independent affine arithmetic")
	}
	if err = x.output.Free(); err != nil {
		t.Fatalf("output not released after actual drain: %v", err)
	}
	for name, b := range x.inputs {
		if err = b.Free(); err != nil {
			t.Fatalf("input %s not released after actual drain: %v", name, err)
		}
	}
	t.Logf("pending witness observed after cancellation; finite delay=%d ticks; actual drain/readback succeeded", pendingDelayTicks)
}

// This fixture-only helper records the successful inner release before an
// outer context-restore error can be returned. A dead raw handle is never
// retried; the caller retains the device and outer error for explicit cleanup.
func closeFixtureRawModule(raw *driver.Module, withContext func(func() error) error, unload func(driver.Module) error) error {
	if *raw == 0 {
		return nil
	}
	return withContext(func() error {
		if err := unload(*raw); err != nil {
			return err
		}
		*raw = 0
		return nil
	})
}
