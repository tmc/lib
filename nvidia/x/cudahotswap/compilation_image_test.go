package cudahotswap

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCompilationImageRefusesMissingOwner(t *testing.T) {
	for _, c := range []*Compilation{nil, {}, {loader: &NVRTCLoader{}}} {
		if image, err := c.CUBIN(); err == nil || len(image) != 0 {
			t.Fatal("accepted missing native owner")
		}
	}
}

func TestCompilationImageActualCPU(t *testing.T) {
	if os.Getenv("HOTSWAP_IMAGE_PREPARE_ONLY") != "1" {
		t.Skip("HOTSWAP_IMAGE_PREPARE_ONLY=1 enables CPU compiler controls")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cm, err := nvrtcx.CompileContext(ctx, `extern "C" __global__ void image_entry(float* out) { out[0] = 7.0f; }`,
		nvrtcx.WithName("compilation_image.cu"), nvrtcx.WithPhysicalArch(12, 1), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	loader := &NVRTCLoader{}
	c := &Compilation{PTX: cm.PTX(), Key: cm.Key(), Entry: "image_entry", native: cm, entry: "image_entry", loader: loader}
	image, err := c.CUBIN()
	if err != nil || len(image) == 0 {
		t.Fatal(err)
	}
	original := append([]byte(nil), image...)
	image[0] ^= 255
	again, err := c.CUBIN()
	if err != nil || !bytes.Equal(again, original) || !bytes.Equal(cm.CUBIN(), original) {
		t.Fatal("image copy altered native body")
	}
	for _, tt := range []struct {
		name   string
		change func(*Compilation)
	}{
		{"key", func(c *Compilation) { c.Key = "changed" }},
		{"ptx", func(c *Compilation) { c.PTX += "\n//changed" }},
		{"entry", func(c *Compilation) { c.Entry = "other" }},
		{"private entry", func(c *Compilation) { c.entry = "other" }},
		{"native absent", func(c *Compilation) { c.native = nil }},
		{"loader absent", func(c *Compilation) { c.loader = nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			copy := *c
			tt.change(&copy)
			if image, err := copy.CUBIN(); err == nil || len(image) != 0 {
				t.Fatal("changed compilation exposed image")
			}
		})
	}
	if err := (&NVRTCLoader{}).validateCompilation(c); err == nil {
		t.Fatal("foreign loader accepted native owner")
	}
	ptx, err := nvrtcx.CompileContext(ctx, `extern "C" __global__ void image_entry(float* out) { out[0] = 7.0f; }`,
		nvrtcx.WithName("compilation_image_ptx.cu"), nvrtcx.WithArch(12, 1), nvrtcx.WithOutput(nvrtcx.OutputPTX), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	missing := &Compilation{PTX: ptx.PTX(), Key: ptx.Key(), Entry: "image_entry", native: ptx, entry: "image_entry", loader: loader}
	if image, err := missing.CUBIN(); err == nil || len(image) != 0 {
		t.Fatal("PTX-only compilation exposed native image")
	}
}
func ExampleCompilation_CUBIN() {
	var c Compilation
	_, err := c.CUBIN()
	fmt.Println(err)
	// Output: compilation image: missing native loader
}
