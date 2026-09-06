//go:build linux

package cudnn_test

import (
	"bytes"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/cudnn"
)

func TestNativeDirectReturns(t *testing.T) {
	if os.Getenv("CUDNN_DIRECT_RETURN_TEST") != "1" {
		t.Skip("set CUDNN_DIRECT_RETURN_TEST=1 for native cuDNN query validation")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, tt := range []struct {
		name string
		call func() (uint, error)
	}{
		{"version", cudnn.GetVersion}, {"maximum SM version", cudnn.GetMaxDeviceVersion}, {"linked CUDA runtime", cudnn.GetCudartVersion},
	} {
		v, err := tt.call()
		if err != nil {
			t.Fatal(err)
		}
		if v == 0 {
			t.Fatalf("%s returned zero", tt.name)
		}
		t.Logf("%s=%d", tt.name, v)
	}
	// This per-thread diagnostic query needs no handle, context or GPU operation.
	var storage [1026]byte
	for i := range storage {
		storage[i] = 0xa5
	}
	if err := cudnn.GetLastErrorString(&storage[1], 1024); err != nil {
		t.Fatal(err)
	}
	if storage[0] != 0xa5 || storage[1025] != 0xa5 {
		t.Fatal("diagnostic buffer guard overwritten")
	}
	if bytes.IndexByte(storage[1:1025], 0) < 0 {
		t.Fatal("diagnostic buffer not NUL terminated")
	}
}
