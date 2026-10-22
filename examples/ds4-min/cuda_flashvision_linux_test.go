//go:build linux

package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This is a physical gate, not a host skip counted as qualification. The
// reviewed runner supplies the extracted pinned C kernels and GPU lease.
func TestCUDAFlashVisionMatmul(t *testing.T) {
	if os.Getenv("DS4_VISION_DEVICE") != "1" {
		t.Skip("DS4_VISION_DEVICE=1 requires the sealed vision GPU runner")
	}
	if err := validateVisionLease(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := validateVisionLease(); err != nil {
			t.Error(err)
		}
	}()
	if err := nvrtc.Load(os.Getenv("DS4_VISION_NVRTC_LIBRARY")); err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	lib, err := os.ReadFile(nvrtc.Path())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual NVRTC library=%s SHA256=%x version=%d.%d", nvrtc.Path(), sha256.Sum256(lib), major, minor)
	c, err := os.ReadFile(os.Getenv("DS4_VISION_MATMUL_ORACLE"))
	if err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	ccMajor, ccMinor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual GPU=%s compute=%d.%d driver=%d", name, ccMajor, ccMinor, driverVersion)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()
	var blas cublas.Handle
	if err := dev.WithContext(func() error { var err error; blas, err = cublas.Create(); return err }); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(blas.Close); err != nil {
			t.Error(err)
		}
	}()
	// An actual source-site mutation changes every matvec's accumulator. It is
	// compared against the same C/float64 checks as the ordinary baseline.
	source := flashVisionCUDA
	if os.Getenv("DS4_VISION_MATMUL_MUTANT") == "1" {
		const old = "float sum = 0.0f;"
		if strings.Count(source, old) != 1 {
			t.Fatal("mutation site is not unique")
		}
		source = strings.Replace(source, old, "float sum = 1.0f;", 1)
	}
	if os.Getenv("DS4_VISION_MATMUL_MUTANT") == "conversion" {
		const old = "__float2bfloat16(x[i])"
		if strings.Count(source, old) != 1 {
			t.Fatal("conversion mutation site is not unique")
		}
		source = strings.Replace(source, old, "__float2bfloat16(x[i] + 1.0f)", 1)
	}
	load := func(src string) (driver.Module, error) {
		cm, err := compileFlashVisionCUDA(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithFlag("--use_fast_math"), nvrtcx.WithFlag("--include-path="+os.Getenv("DS4_VISION_CUDA_INCLUDE")), nvrtcx.WithNoCache())
		if err != nil {
			return 0, err
		}
		return cm.Load(dev)
	}
	mod, err := load(source)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(mod.Close); err != nil {
			t.Error(err)
		}
	}()
	oracle, err := load(string(c))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(oracle.Close); err != nil {
			t.Error(err)
		}
	}()
	builtins, err := filepath.EvalSymlinks(os.Getenv("DS4_VISION_NVRTC_BUILTINS"))
	if err != nil {
		t.Fatal(err)
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(maps), builtins) {
		t.Fatalf("actual NVRTC builtins is not admitted %s", builtins)
	}
	builtinsBytes, err := os.ReadFile(builtins)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("actual NVRTC builtins=%s SHA256=%x", builtins, sha256.Sum256(builtinsBytes))
	fn := func(m driver.Module, name string) driver.Function {
		var f driver.Function
		if err := dev.WithContext(func() error { var err error; f, err = m.Function(name); return err }); err != nil {
			t.Fatal(err)
		}
		return f
	}
	for _, rows := range []int{1, 8, 9} {
		t.Run(fmt.Sprint(rows), func(t *testing.T) {
			const in, out = 588, 1024
			a := cudamem.NewArena(dev)
			defer func() {
				if err := stream.Synchronize(context.Background()); err != nil {
					t.Error(err)
					return // preserve storage still referenced by failed work
				}
				if err := a.Free(); err != nil {
					t.Error(err)
				}
			}()
			w := make([]uint16, in*out)
			x := make([]float32, in*rows)
			for i := range w {
				w[i] = uint16(math.Float32bits(float32((i%29)-14)*0.03125) >> 16)
			}
			for i := range x {
				x[i] = float32((i%37)-18) * 0.018321
			}
			wd := cudamem.AllocIn[uint16](a, "weights", len(w))
			xd := cudamem.AllocIn[float32](a, "input", len(x))
			gd := cudamem.AllocIn[float32](a, "got", rows*out)
			cd := cudamem.AllocIn[float32](a, "oracle", rows*out)
			scratch := cudamem.AllocIn[uint16](a, "activation", len(x))
			if err := a.Err(); err != nil {
				t.Fatal(err)
			}
			if err := wd.CopyFromHost(w); err != nil {
				t.Fatal(err)
			}
			if err := xd.CopyFromHost(x); err != nil {
				t.Fatal(err)
			}
			gfn, gconv := fn(mod, "ds4_vision_matvec_bf16"), fn(mod, "ds4_vision_f32_to_bf16")
			cfn, cconv := fn(oracle, "glm53_matvec_bf16_f32_kernel"), fn(oracle, "glm53_f32_to_bf16_kernel")
			if err := dev.WithContext(func() error {
				if err := flashVisionMatmul(blas, stream, gfn, gconv, wd, xd, gd, scratch, in, out, rows); err != nil {
					return err
				}
				return flashVisionMatmul(blas, stream, cfn, cconv, wd, xd, cd, scratch, in, out, rows)
			}); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}
			got, cwant := make([]float32, rows*out), make([]float32, rows*out)
			if err := gd.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			if err := cd.CopyToHost(cwant); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < rows; r++ {
				for col := 0; col < out; col++ {
					var ref, abs float64
					for k := 0; k < in; k++ {
						v := x[r*in+k]
						if rows > 8 {
							bits := math.Float32bits(v)
							bits += 0x7fff + (bits >> 16 & 1)
							v = math.Float32frombits(bits & 0xffff0000)
						}
						p := float64(math.Float32frombits(uint32(w[col*in+k])<<16)) * float64(v)
						ref += p
						abs += math.Abs(p)
					}
					i := r*out + col
					bound := (float64((in+31)/32+7)*0x1p-24)*abs + 1e-6
					if rows > 8 {
						// cuBLAS owns its accumulation schedule; use a full dot
						// roundoff bound rather than assume our warp tree.
						bound = float64(in+2)*0x1p-24*abs + 1e-6
					}
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || math.Abs(float64(got[i])-ref) > bound {
						t.Fatalf("float64 sample %d: %g vs %g bound %g", i, got[i], ref, bound)
					}
					if math.Float32bits(got[i]) != math.Float32bits(cwant[i]) {
						t.Fatalf("C sample %d: %08x vs %08x", i, math.Float32bits(got[i]), math.Float32bits(cwant[i]))
					}
				}
			}
		})
	}
}
