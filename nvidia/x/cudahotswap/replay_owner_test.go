package cudahotswap

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestReplayTypedNilNativeOwnerRefused(t *testing.T) {
	for _, timed := range []bool{false, true} {
		lease := new(serviceCloser)
		backend := &cudaReplayBackend{dev: new(cuda.Device), lease: lease, initialized: true}
		var kernel *nativeKernel
		var owner Kernel = kernel
		if owner == nil {
			t.Fatal("test requires a typed-nil interface")
		}
		duration, err := backend.Run(context.Background(), replayArm{Kernel: owner}, replaySpec{}, timed)
		if err == nil || duration != 0 {
			t.Fatalf("typed-nil owner admitted: %v, %v", duration, err)
		}
		if backend.pending || backend.lease != lease || lease.calls != 0 {
			t.Fatal("owner refusal submitted work or discarded retained lease")
		}
	}
}
