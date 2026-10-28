package cudadecomp

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// nativeTarget returns the target the native tests assemble for.
// CUDA_DECOMP_NATIVE_ARCH names another.
func nativeTarget() string {
	if target := os.Getenv("CUDA_DECOMP_NATIVE_ARCH"); target != "" {
		return target
	}
	return "sm_121a"
}

func requireNative(t *testing.T) {
	t.Helper()
	if os.Getenv("CUDA_DECOMP_NATIVE") != "1" {
		t.Skip("set CUDA_DECOMP_NATIVE=1 to assemble the merge kernel and query the device")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate open without inherited gpu lock: %v", err)
	}
}

// TestNativeAssembleMerge requires ptxas to accept the merge kernel.
func TestNativeAssembleMerge(t *testing.T) {
	requireNative(t)
	ptxas, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Fatalf("ptxas unavailable: %v", err)
	}
	target := nativeTarget()
	ptx, err := EmitBytePlaneMergePTX("merge_planes", target)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "merge.ptx")
	if err := os.WriteFile(src, []byte(ptx), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	out, err := cudatool.Run(ctx, ptxas, dir, nil, "-arch="+target, "-v", "-o", filepath.Join(dir, "merge.cubin"), src)
	if err != nil {
		t.Fatalf("ptxas %s: %v\n%s%s", target, err, out.Stdout, out.Stderr)
	}
	t.Logf("ptxas %s:\n%s", ptxas, out.Stderr)
}

// TestNativeDecompressAttributes reports device 0's hardware decompression
// algorithms and checks a compressed tensor against them. It does not
// submit a batch.
func TestNativeDecompressAttributes(t *testing.T) {
	requireNative(t)
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	mask, err := dev.Attribute(driver.DevAttrMemDecompressAlgorithmMask)
	if err != nil {
		t.Fatalf("decompress algorithm mask: %v", err)
	}
	maxLen, err := dev.Attribute(driver.DevAttrMemDecompressMaximumLength)
	if err != nil {
		t.Fatalf("decompress maximum length: %v", err)
	}
	algos := driver.MemDecompressAlgorithm(mask)
	t.Logf("decompress algorithms %#x, maximum length %d", mask, maxLen)
	if algos == driver.MemDecompressUnsupported {
		t.Skip("device has no hardware decompression")
	}
	for _, codec := range []Codec{CodecLZ4, CodecDeflate} {
		ct, err := CompressBF16("w", weights(DefaultChunkSize, false, 41), codec)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%v: CheckDevice = %v", codec, ct.CheckDevice(algos, int(maxLen)))
	}
}
