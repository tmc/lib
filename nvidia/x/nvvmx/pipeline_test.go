package nvvmx_test

import (
	"crypto/sha256"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/nvvm"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

// TestCompilerPipeline compiles and links only. It neither loads a device
// module nor proves kernel execution. Run under the host coordination protocol.
func TestCompilerPipeline(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 required")
	}
	arch := os.Getenv("CUDA_COMPILER_ARCH")
	libdevicePath := os.Getenv("CUDA_LIBDEVICE")
	if arch == "" || libdevicePath == "" {
		t.Fatal("CUDA_COMPILER_ARCH (for example 100) and CUDA_LIBDEVICE required")
	}
	ir, err := os.ReadFile("../../nvvm/testdata/sin.ll")
	if err != nil {
		t.Fatal(err)
	}
	libdevice, err := os.ReadFile(libdevicePath)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvvm.Version()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("NVVM %d.%d path=%s arch=%s IR SHA256=%x libdevice=%s SHA256=%x", major, minor, nvvm.Path(), arch, sha256.Sum256(ir), libdevicePath, sha256.Sum256(libdevice))
	inputs := []nvvmx.Input{{Name: "sin.ll", Data: ir}, {Name: "libdevice", Data: libdevice, Lazy: true}}
	for _, lto := range []bool{false, true} {
		name := "ptx"
		options := []string{"-arch=compute_" + arch}
		if lto {
			name = "ltoir"
			options = append(options, "-gen-lto")
		}
		t.Run(name, func(t *testing.T) {
			r, err := nvvmx.Compile(inputs, options...)
			if r != nil {
				t.Cleanup(func() {
					if err := r.Close(); err != nil {
						t.Error(err)
					}
				})
				t.Logf("compiler log: %s", r.Log)
			}
			if err != nil {
				t.Fatal(err)
			}
			data, kind := r.PTX, nvjitlink.InputPtx
			linkOptions := []string{"-arch=sm_" + arch}
			if lto {
				data, kind = r.LTOIR, nvjitlink.InputLtoir
				linkOptions = append(linkOptions, "-lto")
			}
			if len(data) == 0 {
				t.Fatal("empty compiler artifact")
			}
			t.Logf("artifact SHA256=%x", sha256.Sum256(data))
			linked, err := nvjitlinkx.Link([]nvjitlinkx.Input{{Name: name, Kind: kind, Data: data}}, linkOptions...)
			if linked != nil {
				t.Cleanup(func() {
					if err := linked.Close(); err != nil {
						t.Error(err)
					}
				})
				t.Logf("link logs: %s\n%s", linked.InfoLog, linked.ErrorLog)
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(linked.CUBIN) < 4 || string(linked.CUBIN[:4]) != "\x7fELF" {
				t.Fatal("no ELF CUBIN")
			}
			t.Logf("CUBIN SHA256=%x", sha256.Sum256(linked.CUBIN))
		})
	}
	t.Run("invalid IR", func(t *testing.T) {
		r, err := nvvmx.Compile([]nvvmx.Input{{Name: "invalid", Data: []byte("not NVVM IR")}})
		if r != nil {
			t.Cleanup(func() {
				if err := r.Close(); err != nil {
					t.Error(err)
				}
			})
		}
		if err == nil || r == nil || r.Log == "" {
			t.Fatalf("invalid IR: result=%+v error=%v", r, err)
		}
		t.Logf("expected failure: %v", err)
	})
}
