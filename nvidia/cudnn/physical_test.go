package cudnn_test

import (
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/cudnn"
)

// TestNativeTensorDescriptor requires the ops sublibrary, even when the
// top-level cuDNN loader succeeds. This test does not execute GPU work.
func TestNativeTensorDescriptor(t *testing.T) {
	if os.Getenv("CUDNN_LIBRARY_TEST") != "1" {
		t.Skip("set CUDNN_LIBRARY_TEST=1 for native descriptor validation")
	}
	d, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := d.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	if err := d.Set4D(cudnn.TensorNchw, cudnn.DataFloat, 2, 3, 4, 5); err != nil {
		t.Fatal(err)
	}
	c, err := d.Get4D()
	if err != nil {
		t.Fatal(err)
	}
	if c != (cudnn.Tensor4DConfig{DataType: cudnn.DataFloat, N: 2, C: 3, H: 4, W: 5, NStride: 60, CStride: 20, HStride: 5, WStride: 1}) {
		t.Fatalf("Get4D = %+v", c)
	}
	typ, dims, strides, err := d.Get(8)
	if err != nil {
		t.Fatal(err)
	}
	if typ != cudnn.DataFloat || !slices.Equal(dims, []int32{2, 3, 4, 5}) || !slices.Equal(strides, []int32{60, 20, 5, 1}) {
		t.Fatalf("Get = %v, %v, %v", typ, dims, strides)
	}
	t.Logf("type=%v dims=%v strides=%v", typ, dims, strides)
}
