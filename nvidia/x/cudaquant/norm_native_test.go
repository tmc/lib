package cudaquant

import (
	"bytes"
	"context"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func normNativeCapability(target string) (major, minor int, ok bool) {
	switch target {
	case "sm_90", "sm_90a":
		return 9, 0, true
	case "sm_100a":
		return 10, 0, true
	case "sm_121a":
		return 12, 1, true
	}
	return 0, 0, false
}

func normNativeDeviceMatches(target string, major, minor int) bool {
	wantMajor, wantMinor, ok := normNativeCapability(target)
	return ok && major == wantMajor && minor == wantMinor
}

func TestNativeNormQuantReference(t *testing.T) {
	if os.Getenv("CUDAQUANT_NATIVE") != "1" {
		t.Skip("CUDAQUANT_NATIVE=1 requires scoped hardware authorization")
	}
	target := os.Getenv("CUDAQUANT_TARGET")
	if _, _, ok := normNativeCapability(target); !ok {
		t.Fatal("CUDAQUANT_TARGET must name sm_90, sm_90a, sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAQUANT_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("CUDAQUANT_DEVICE must name an explicit ordinal")
	}
	lockCtx, cancelLock := context.WithTimeout(context.Background(), time.Minute)
	defer cancelLock()
	lease, err := cudalock.Acquire(lockCtx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	var dev *cuda.Device
	var stream *cuda.Stream
	var raw driver.Module
	var module *cuda.Module
	var inputs []cudamem.Slice[float32]
	var outputs []cudamem.Slice[byte]
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		// A launch error does not prove that work was not submitted. Do not
		// release dependencies or the lease until the stream has drained.
		if stream != nil {
			if err := stream.Synchronize(ctx); err != nil {
				t.Errorf("retain native resources after wait failure: %v", err)
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
			if err := dev.WithContext(func() error {
				if err := raw.Close(); err != nil {
					return err
				}
				raw = 0
				return nil
			}); err != nil {
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
		for _, b := range inputs {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, b := range outputs {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		if dev != nil {
			if err := dev.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	dev, err = cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if !normNativeDeviceMatches(target, major, minor) {
		t.Fatalf("device SM%d%d differs from %s", major, minor, target)
	}
	artifact, err := RMSNormQ8_1Artifact(target)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = artifact.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	module, err = cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	raw = 0
	kernel, err := module.Function(artifact.EntryPoint)
	if err != nil {
		t.Fatal(err)
	}
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	rtcMajor, rtcMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d uuid=%v target=%s driver=%d nvrtc=%d.%d artifact=%s resources=%+v flags=%v lease=%+v", ordinal, uuid, target, version, rtcMajor, rtcMinor, artifact.Digest, artifact.Resources, artifact.CompilerOpts, lease.Owner())
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	for _, f := range normFixtures() {
		want := normBytes(f.want)
		ref, err := RMSNormQ8_1(f.input, f.weight, f.rows, f.cols, f.epsilon, f.weighted)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(normBytes(ref), want) {
			t.Fatalf("%s: host reference differs from analytic fixture", f.name)
		}
		input, err := cudamem.Alloc[float32](dev, len(f.input))
		inputs = append(inputs, input)
		if err != nil {
			t.Fatal(err)
		}
		weight, err := cudamem.Alloc[float32](dev, len(f.weight))
		inputs = append(inputs, weight)
		if err != nil {
			t.Fatal(err)
		}
		storage, err := cudamem.Alloc[byte](dev, len(want)+64)
		outputs = append(outputs, storage)
		if err != nil {
			t.Fatal(err)
		}
		out := storage.Slice(32, 32+len(want))
		if input.Ptr()%4 != 0 || weight.Ptr()%4 != 0 || out.Ptr()%2 != 0 {
			t.Fatal("unaligned fixture storage")
		}
		if err := input.CopyFromHost(f.input); err != nil {
			t.Fatal(err)
		}
		if err := weight.CopyFromHost(f.weight); err != nil {
			t.Fatal(err)
		}
		readback := bytes.Repeat([]byte{0xa5}, storage.Len())
		if err := storage.CopyFromHost(readback); err != nil {
			t.Fatal(err)
		}
		weighted := uint32(0)
		if f.weighted {
			weighted = 1
		}
		args := []any{input, weight, out, uint32(f.rows), uint32(f.cols), f.epsilon, weighted}
		if _, err := artifact.Schema.Pack(args...); err != nil {
			t.Fatal(err)
		}
		// An extra CTA and inactive lanes exercise the generated bounds checks.
		if err := cudalaunch.OwnedKernel(kernel).Grid(uint32(f.rows+1), 1, 1).Block(32, 1, 1).On(stream).Launch(args...); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(ctx); err != nil {
			t.Fatal(err)
		}
		if err := storage.CopyToHost(readback); err != nil {
			t.Fatal(err)
		}
		got := readback[32 : 32+len(want)]
		if !bytes.Equal(got, want) {
			t.Fatalf("%s: packed output=%x want=%x", f.name, got, want)
		}
		for i := range 32 {
			if readback[i] != 0xa5 || readback[len(readback)-1-i] != 0xa5 {
				t.Fatal("output guard overwritten")
			}
		}
		got[len(got)-1] ^= 1
		if bytes.Equal(got, want) {
			t.Fatal("accepted corrupted output")
		}
		t.Logf("fixture=%s rows=%d cols=%d weighted=%v epsilon=%g blocks=%d all_bytes=exact guards=preserved", f.name, f.rows, f.cols, f.weighted, f.epsilon, len(f.want))
	}
}
