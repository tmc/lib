package cudnn_test

import (
	"math"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudnn"
	"github.com/tmc/lib/nvidia/driver"
)

// TestNativeActivationForward validates the legacy FP32 ReLU/scaling API.
// It does not exercise the cuDNN backend graph API.
func TestNativeActivationForward(t *testing.T) {
	if os.Getenv("CUDNN_HARDWARE_TEST") != "1" {
		t.Skip("set CUDNN_HARDWARE_TEST=1 for native activation validation")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	clean := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("cleanup %s: %v", what, err)
		}
	}
	check("CUDA init", driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check("device", err)
	var uuid driver.Uuid
	check("UUID", dev.GetUuid(&uuid))
	ctx, err := dev.PrimaryCtxRetain()
	check("retain context", err)
	defer func() { clean("release context", dev.PrimaryCtxRelease()) }()
	check("select context", ctx.SetCurrent())
	stream, err := driver.NewStream(0)
	check("stream", err)
	defer func() { clean("destroy stream", stream.Destroy()) }()
	h, err := cudnn.Create()
	check("handle", err)
	defer func() { clean("destroy handle", h.Destroy()) }()
	check("set stream", h.SetStream(cudnn.Stream(stream)))
	desc, err := cudnn.CreateTensorDescriptor()
	check("tensor descriptor", err)
	defer func() { clean("destroy tensor descriptor", desc.Destroy()) }()
	check("tensor layout", desc.SetTensor4dDescriptor(cudnn.TensorNchw, cudnn.DataFloat, 1, 1, 2, 4))
	activation, err := cudnn.CreateActivationDescriptor()
	check("activation descriptor", err)
	defer func() { clean("destroy activation descriptor", activation.Destroy()) }()
	check("ReLU", activation.Set(cudnn.ActivationRelu, cudnn.PropagateNan, 0))
	input := [8]float32{-3, -1, 0, 1, 2, 4, -2, 8}
	initial := [8]float32{1, 2, 3, 4, 5, 6, 7, 8}
	const bytes = uint(8 * 4)
	x, err := driver.MemAlloc(bytes)
	check("allocate input", err)
	defer func() { clean("free input", x.MemFree()) }()
	y, err := driver.MemAlloc(bytes)
	check("allocate output", err)
	defer func() { clean("free output", y.MemFree()) }()
	check("upload input", x.MemcpyHtoD(unsafe.Pointer(&input[0]), bytes))
	check("upload initial output", y.MemcpyHtoD(unsafe.Pointer(&initial[0]), bytes))
	alpha, beta := float32(2), float32(0.5)
	var pins runtime.Pinner
	pins.Pin(&alpha)
	pins.Pin(&beta)
	defer pins.Unpin()
	// Registered last so pending work is drained before buffers/scalars and
	// descriptors are released, including when the launch reports an error.
	defer func() { clean("synchronize before release", stream.Synchronize()) }()
	check("activation forward", h.ActivationForward(activation, unsafe.Pointer(&alpha), desc, uintptr(x), unsafe.Pointer(&beta), desc, uintptr(y)))
	check("synchronize", stream.Synchronize())
	var got [8]float32
	check("download", y.MemcpyDtoH(unsafe.Pointer(&got[0]), bytes))
	var want [8]float32
	for i, value := range input {
		relu := value
		if relu < 0 {
			relu = 0
		}
		want[i] = alpha*relu + beta*initial[i]
		if math.IsNaN(float64(got[i])) || got[i] != want[i] {
			t.Errorf("output[%d] = %g, want %g", i, got[i], want[i])
		}
	}
	t.Logf("device=%d UUID=%x legacy FP32 ReLU outputs=%v want=%v", dev, uuid, got, want)
}
