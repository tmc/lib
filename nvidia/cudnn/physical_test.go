package cudnn_test

import (
	"os"
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
	if err := d.SetTensor4dDescriptor(cudnn.TensorNchw, cudnn.DataFloat, 2, 3, 4, 5); err != nil {
		t.Fatal(err)
	}
	var typ cudnn.DataType
	var dims, strides [4]int32
	err = d.GetTensor4dDescriptor(&typ, &dims[0], &dims[1], &dims[2], &dims[3], &strides[0], &strides[1], &strides[2], &strides[3])
	if err != nil {
		t.Fatal(err)
	}
	if typ != cudnn.DataFloat || dims != [4]int32{2, 3, 4, 5} || strides != [4]int32{60, 20, 5, 1} {
		t.Fatalf("type=%v dims=%v strides=%v", typ, dims, strides)
	}
	t.Logf("type=%v dims=%v strides=%v", typ, dims, strides)
}
