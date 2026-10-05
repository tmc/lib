//go:build linux && native

package cudaoutputcoverage_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"syscall"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

var recordedOutputRegistry = flag.String("recorded-output-registry", "", "owned source/PTX registry directory")

var recordedOutputArm = flag.String("recorded-output-arm", "", "owned output worker launch arm")

const recordedOutputSource = `extern "C" __global__ void output_contract(float* out, int n, int corrupt) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned int)n) out[i] = corrupt && i == 29 ? -9.0f : (float)(i + 1);
}`

func recordedHash(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func recordedImage(path string) ([]byte, error) { return recordedFile(path, 4<<20) }

func recordedFile(path string, limit int64) ([]byte, error) {
	if limit <= 0 || limit > 4<<20 {
		return nil, fmt.Errorf("recorded output: invalid file bound")
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > limit {
		return nil, fmt.Errorf("recorded output: file size/type")
	}
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if int64(len(b)) != st.Size() || after.Size() != st.Size() || !after.ModTime().Equal(st.ModTime()) {
		return nil, fmt.Errorf("recorded output: file changed")
	}
	return b, nil
}

type recordedOutputOwner struct {
	dev       *cuda.Device
	module    driver.Module
	buffer    cudamem.Slice[uint32]
	allocated bool
}

var retainedRecordedOutputOwners []*recordedOutputOwner

func (o *recordedOutputOwner) close() error {
	if o.dev != nil {
		if err := o.dev.WithContext(driver.CtxSynchronize); err != nil {
			return err
		}
	}
	if o.allocated {
		if err := o.buffer.Free(); err != nil {
			return err
		}
		o.allocated = false
	}
	if o.module != 0 {
		if err := o.dev.WithContext(o.module.Close); err != nil {
			return err
		}
		o.module = 0
	}
	if o.dev != nil {
		if err := o.dev.Close(); err != nil {
			return err
		}
		o.dev = nil
	}
	return nil
}

// The keeper ACK precedes every CUDA call. This worker loads the actual supplied
// PTX; the explicit arm controls geometry/parameter, never its output verdict.
func TestRecordedOutputWorker(t *testing.T) {
	if *recordedOutputArm == "" {
		return
	}
	grid, corrupt, err := recordedOutputLaunch(*recordedOutputArm)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudaschedule.RunRecordedHelper(context.Background(), []string{"worker-ready"}); err != nil {
		t.Fatal(err)
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	image, err := recordedImage(os.Args[len(os.Args)-1])
	if err != nil {
		t.Fatal(err)
	}
	sourceDigest := recordedHash([]byte(recordedOutputSource))
	if *recordedOutputRegistry != "" {
		if *recordedOutputArm != "healthy" {
			t.Fatal("registry worker requires fixed healthy launch arguments")
		}
		selected, err := recordedRegisteredOutput(*recordedOutputRegistry, image)
		if err != nil {
			t.Fatal(err)
		}
		sourceDigest = selected.SourceDigest
	}
	owner := &recordedOutputOwner{}
	t.Cleanup(func() {
		if err := owner.close(); err != nil {
			retainedRecordedOutputOwners = append(retainedRecordedOutputOwners, owner)
			t.Errorf("retain output worker cleanup owner: %v", err)
		}
	})
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	owner.dev, err = cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := owner.dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Fatalf("output worker target sm_%d%d differs from sm_121", major, minor)
	}
	if err := owner.dev.WithContext(func() error {
		var err error
		owner.module, err = driver.ModuleLoadData(append(append([]byte(nil), image...), 0))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stdout, "CUDASCHEDULE_LOADED_PTX_SHA256:%s\n", recordedHash(image))
	var fn driver.Function
	if err := owner.dev.WithContext(func() error { var err error; fn, err = owner.module.Function("output_contract"); return err }); err != nil {
		t.Fatal(err)
	}
	owner.buffer, err = cudamem.Alloc[uint32](owner.dev, 1008)
	owner.allocated = owner.buffer.Ptr() != 0
	if err != nil {
		t.Fatal(err)
	}
	spec := recordedOutputSpec(grid, corrupt)
	spec.SourceDigest = sourceDigest
	launch := func(_ context.Context, fill uint32) ([]uint32, error) {
		base := make([]uint32, 1008)
		for i := range base {
			base[i] = fill
		}
		base[0] = math.Float32bits(77)
		base[1007] = math.Float32bits(-77)
		return recordedOutputExecute(owner, fn, spec, corrupt, base, fill)
	}
	completed, _, err := cudaoutputcoverage.Observe(context.Background(), spec, launch)
	if err != nil {
		t.Fatal(err)
	}
	ordinary := make([]uint32, 1008)
	ordinary[0] = math.Float32bits(77)
	ordinary[1007] = math.Float32bits(-77)
	copy(ordinary[1:], spec.Reference)
	got, err := recordedOutputExecute(owner, fn, spec, corrupt, ordinary, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Successful explicit retirement must precede the receipt. Failed owners stay
	// retained and no output receipt is produced.
	if err := owner.close(); err != nil {
		t.Fatal(err)
	}
	out := &cudaoutputcoverage.RecordedOutput{PTXDigest: recordedHash(image), TwoFill: completed, Ordinary: got, Policy: recordedOutputPolicy(spec)}
	if err := cudaschedule.WriteOutputReceipt(os.Stdout, out); err != nil {
		t.Fatal(err)
	}
}

func recordedOutputExecute(o *recordedOutputOwner, fn driver.Function, spec cudaoutputcoverage.TwoFill, corrupt int32, base []uint32, padding uint32) ([]uint32, error) {
	if err := o.buffer.CopyFromHost(base); err != nil {
		return nil, err
	}
	ptr := o.buffer.Slice(1, 1007).Ptr()
	n := int32(1003)
	if err := o.dev.WithContext(func() error {
		if err := fn.LaunchKernel(driver.Dims(spec.Geometry.Grid[0], 1, 1), driver.Dims(256, 1, 1), 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr), unsafe.Pointer(&n), unsafe.Pointer(&corrupt)}); err != nil {
			return err
		}
		return driver.CtxSynchronize()
	}); err != nil {
		return nil, err
	}
	if err := o.buffer.CopyToHost(base); err != nil {
		return nil, err
	}
	if base[0] != math.Float32bits(77) || base[1007] != math.Float32bits(-77) {
		return nil, fmt.Errorf("output guard overwritten")
	}
	for i := 1004; i < 1007; i++ {
		if base[i] != padding {
			return nil, fmt.Errorf("excluded padding overwritten at %d", i)
		}
	}
	return append([]uint32(nil), base[1:1007]...), nil
}
