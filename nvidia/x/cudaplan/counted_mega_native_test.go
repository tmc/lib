package cudaplan

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestNativeCountedMegaReference(t *testing.T) {
	if os.Getenv("CUDAPLAN_COUNTED_NATIVE") != "1" {
		t.Skip("requires CUDAPLAN_COUNTED_NATIVE=1 and an admitted device lease")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native counted worker requires verified GPU lease")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAPLAN_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("set explicit CUDAPLAN_NATIVE_DEVICE")
	}
	wantUUID := os.Getenv("CUDAPLAN_COUNTED_UUID")
	if len(wantUUID) != 32 {
		t.Fatal("set CUDAPLAN_COUNTED_UUID to the 32 hexadecimal UUID digits")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	if err := cuda.Init(); err != nil {
		_ = lease.Release()
		t.Fatal(err)
	}
	raw, err := driver.DeviceGet(ordinal)
	if err != nil {
		_ = lease.Release()
		t.Fatal(err)
	}
	uuid, err := raw.Uuid()
	if err != nil {
		_ = lease.Release()
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", uuid.Bytes) != wantUUID {
		_ = lease.Release()
		t.Fatalf("device identity differs: got %x want %s", uuid.Bytes, wantUUID)
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		_ = lease.Release()
		t.Fatal(err)
	}
	var stream *cuda.Stream
	var plan *ExecutablePlan
	var completion *Completion
	var buffers []cudamem.Slice[float32]
	t.Cleanup(func() {
		if completion != nil {
			if err := completion.Wait(nil); err != nil && !completion.Done() {
				t.Errorf("retain native dependencies after drain failure: %v", err)
				return
			}
		}
		if plan != nil {
			if err := plan.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, buffer := range buffers {
			if err := buffer.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := dev.Close(); err != nil {
			t.Error(err)
			return
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if want := os.Getenv("CUDAPLAN_COUNTED_TARGET"); want != "sm_"+strconv.Itoa(major)+strconv.Itoa(minor) {
		t.Fatalf("set exact CUDAPLAN_COUNTED_TARGET=sm_%d%d", major, minor)
	}
	snap := countedFixture(t)
	plan, err = Compile(snap, WithDevice(dev), WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	alloc := func(n int) cudamem.Slice[float32] {
		buffer, err := cudamem.Alloc[float32](dev, n)
		buffers = append(buffers, buffer)
		if err != nil {
			t.Fatal(err)
		}
		return buffer
	}
	input, output := alloc(1026), alloc(2)
	for round := range 3 {
		data := make([]float32, 1026)
		data[0] = 16777216
		data[1] = 1
		data[2] = -16777216
		data[513] = -16777216
		data[514] = -1
		data[515] = 16777216
		if round == 1 {
			for i := range data {
				data[i] = -data[i]
			}
		}
		if err := input.CopyFromHost(data); err != nil {
			t.Fatal(err)
		}
		if err := output.CopyFromHost([]float32{float32(math.NaN()), float32(math.NaN())}); err != nil {
			t.Fatal(err)
		}
		binding := NewBindings()
		if err := SetInputSlice(binding, "x", input, dev); err != nil {
			t.Fatal(err)
		}
		name := snap.Outputs()[0].Name
		if err := SetOutputSlice(binding, name, output, dev); err != nil {
			t.Fatal(err)
		}
		completion, err = plan.Execute(stream, binding)
		if err != nil {
			t.Fatal(err)
		}
		if err := completion.Wait(ctx); err != nil {
			t.Fatal(err)
		}
		for _, step := range completion.ExecutedSteps() {
			if step.Backend != BackendMega {
				t.Fatalf("unexpected native backend %+v", step)
			}
		}
		got := make([]float32, 2)
		if err := output.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		reference, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"x": data})
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range reference[name] {
			if math.Float32bits(got[i]) != math.Float32bits(want) {
				t.Fatalf("round %d output %d got %g want %g", round, i, got[i], want)
			}
		}
		t.Logf("round=%d uuid=%x target=sm_%d%d CTAs=2 artifact=%s outcome_counts=exact independent_FP32_reference=pass", round, uuid.Bytes, major, minor, plan.megaArtifact.Digest)
	}
}
