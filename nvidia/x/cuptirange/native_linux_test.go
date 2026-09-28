//go:build linux

package cuptirange_test

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestNativeRangeCapture(t *testing.T) {
	if os.Getenv("CUPTI_RANGE_TEST") != "1" {
		t.Skip("CUPTI_RANGE_TEST=1 required")
	}
	chip := os.Getenv("CUPTI_CHIP")
	if chip == "" {
		t.Fatal("CUPTI_CHIP must identify the qualified device chip")
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
	cm, err := nvrtcx.Compile(`extern "C" __global__ void measured(unsigned* out) { out[0] = 7; }`, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	module, err := cm.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	}()
	var fn driver.Function
	err = dev.WithContext(func() error {
		var err error
		fn, err = module.Function("measured")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	output, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := output.Free(); err != nil {
			t.Error(err)
		}
	}()
	plan, err := cuptirange.NewPlan(chip, nil, []string{"sm__cycles_elapsed.avg"}, 16<<20)
	if plan != nil {
		defer func() {
			if err := plan.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	image, passes := plan.Configuration()
	if len(image) == 0 || passes == 0 {
		t.Fatal("empty profiling configuration")
	}
	launches := uint(0)
	capture, err := plan.Capture(dev, cuptirange.Limits{MaxRanges: 4, MaxRangeTreeNodes: 8, MaxCounterBytes: 16 << 20, MaxPasses: 16}, func(pass uint) error {
		if pass != launches {
			t.Errorf("pass=%d expected %d", pass, launches)
		}
		launches++
		return cudalaunch.Kernel(fn).Launch(output)
	})
	if capture != nil {
		defer func() {
			if err := capture.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, 1)
	if err := output.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 || launches == 0 || capture.Ranges != 1 {
		t.Fatalf("output=%d launches=%d ranges=%d", got[0], launches, capture.Ranges)
	}
	values, err := plan.Evaluate(capture.Image, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 1 || values[0] <= 0 {
		t.Fatalf("invalid cycle metric %v", values)
	}
	t.Logf("chip=%s passes=%d launches=%d ranges=%d cycles=%g; no timing comparison", chip, passes, launches, capture.Ranges, values[0])
}

func TestNativeRangeIsolation(t *testing.T) {
	if os.Getenv("CUPTI_RANGE_TEST") != "1" {
		t.Skip("CUPTI_RANGE_TEST=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("native qualification pending: inherited gpu-lock required: %v", err)
	}
	chip := os.Getenv("CUPTI_CHIP")
	if chip == "" {
		t.Fatal("CUPTI_CHIP must identify the qualified device chip")
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

	const src = `
extern "C" __global__ void light_kernel(unsigned int *out) {
    out[0] = 42;
}

extern "C" __global__ void heavy_kernel(unsigned int *out, unsigned int trips) {
    unsigned int v = 1;
    #pragma unroll 1
    for (unsigned int i = 0; i < trips; i++) {
        v = v * 1664525u + 1013904223u;
    }
    out[0] = v;
}
`
	cm, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache(), nvrtcx.WithLineInfo())
	if err != nil {
		t.Fatal(err)
	}
	module, err := cm.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.WithContext(module.Close); err != nil {
			t.Error(err)
		}
	}()

	var fnLight, fnHeavy driver.Function
	err = dev.WithContext(func() error {
		var err error
		fnLight, err = module.Function("light_kernel")
		if err != nil {
			return err
		}
		fnHeavy, err = module.Function("heavy_kernel")
		return err
	})
	if err != nil {
		t.Fatal(err)
	}

	outLight, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := outLight.Free(); err != nil {
			t.Error(err)
		}
	}()

	outHeavy, err := cudamem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := outHeavy.Free(); err != nil {
			t.Error(err)
		}
	}()

	const metricName = "sm__inst_executed.sum"

	plan, err := cuptirange.NewPlan(chip, nil, []string{metricName}, 16<<20)
	if plan != nil {
		defer func() {
			if err := plan.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	static, err := loopBodySASSInstructions(t, dev, cm.PTX(), src)
	if err != nil {
		t.Fatal(err)
	}

	runCapture := func(trips uint32, combined bool) (heavyVal, lightVal float64, err error) {
		options := cuptirange.CaptureOptions{
			Mode:     cuptirange.Regions,
			MaxDepth: 4,
		}
		limits := cuptirange.Limits{
			MaxRanges:         8,
			MaxRangeTreeNodes: 16,
			MaxCounterBytes:   16 << 20,
			MaxPasses:         16,
		}
		replay := cuptirange.Replay{
			Synchronize: driver.CtxSynchronize,
			Run: func(pass uint, s *cuptirange.Scope) error {
				if combined {
					return s.Region("combined", func() error {
						if err := cudalaunch.Kernel(fnHeavy).Launch(outHeavy, trips); err != nil {
							return err
						}
						return cudalaunch.Kernel(fnLight).Launch(outLight)
					})
				}
				if err := s.Region("heavy", func() error {
					return cudalaunch.Kernel(fnHeavy).Launch(outHeavy, trips)
				}); err != nil {
					return err
				}
				return s.Region("light", func() error {
					return cudalaunch.Kernel(fnLight).Launch(outLight)
				})
			},
		}

		capture, err := plan.CaptureReplay(dev, limits, options, replay)
		if capture != nil {
			defer func() {
				if cErr := capture.Close(); cErr != nil {
					t.Error(cErr)
				}
			}()
		}
		if err != nil {
			return 0, 0, err
		}

		var heavyInfo, lightInfo, combinedInfo *cuptirange.RangeInfo
		for i := range capture.RangeInfo {
			switch capture.RangeInfo[i].Name {
			case "heavy":
				heavyInfo = &capture.RangeInfo[i]
			case "light":
				lightInfo = &capture.RangeInfo[i]
			case "combined":
				combinedInfo = &capture.RangeInfo[i]
			}
		}
		if combined {
			if combinedInfo == nil {
				return 0, 0, fmt.Errorf("missing combined range in range info: %+v", capture.RangeInfo)
			}
			value, err := plan.Evaluate(capture.Image, combinedInfo.Index)
			if err != nil {
				return 0, 0, fmt.Errorf("evaluate combined: %w", err)
			}
			return value[0], value[0], nil
		}
		if heavyInfo == nil || lightInfo == nil {
			return 0, 0, fmt.Errorf("missing heavy or light range in range info: %+v", capture.RangeInfo)
		}

		vHeavy, err := plan.Evaluate(capture.Image, heavyInfo.Index)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluate heavy: %w", err)
		}
		vLight, err := plan.Evaluate(capture.Image, lightInfo.Index)
		if err != nil {
			return 0, 0, fmt.Errorf("evaluate light: %w", err)
		}
		return vHeavy[0], vLight[0], nil
	}

	trips1 := uint32(10000)
	h1, l1, err := runCapture(trips1, false)
	if err != nil {
		t.Fatalf("capture 1: %v", err)
	}

	trips2 := uint32(20000)
	h2, l2, err := runCapture(trips2, false)
	if err != nil {
		t.Fatalf("capture 2: %v", err)
	}

	if err := checkIsolation(h1, l1, h2, l2, uint64(static), trips1, trips2); err != nil {
		t.Fatalf("range isolation failed (static loop body=%d SASS instructions): %v; h1=%g l1=%g h2=%g l2=%g", static, err, h1, l1, h2, l2)
	}
	combined, _, err := runCapture(trips1, true)
	if err != nil {
		t.Fatalf("combined range control: %v", err)
	}
	if err := checkIsolation(combined, combined, combined, combined, uint64(static), trips1, trips2); err == nil {
		t.Fatalf("single range containing both kernels passed isolation check: combined=%g static=%d", combined, static)
	}

	t.Logf("range isolation confirmed on %s (%s): loop_body_sass=%d heavy(%d)=%g heavy(%d)=%g light=%g combined_control=%g",
		chip, metricName, static, trips1, h1, trips2, h2, l1, combined)
}

func loopBodySASSInstructions(t *testing.T, dev *cuda.Device, ptx, source string) (int, error) {
	t.Helper()
	findTool := func(name string) (string, error) {
		for _, root := range []string{os.Getenv("CUDA"), os.Getenv("CUDA_HOME")} {
			if root == "" {
				continue
			}
			candidate := filepath.Join(root, "bin", name)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
		return exec.LookPath(name)
	}
	ptxas, err := findTool("ptxas")
	if err != nil {
		t.Skipf("static loop bound requires ptxas: %v", err)
	}
	nvdisasm, err := findTool("nvdisasm")
	if err != nil {
		t.Skipf("static loop bound requires nvdisasm: %v", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return 0, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return 0, err
	}
	tmp := filepath.Join(home, "tmp")
	if err := os.MkdirAll(tmp, 0o700); err != nil {
		return 0, err
	}
	dir, err := os.MkdirTemp(tmp, "cuptirange-sass-")
	if err != nil {
		return 0, err
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	ptxPath := filepath.Join(dir, "range.ptx")
	cubinPath := filepath.Join(dir, "range.cubin")
	if err := os.WriteFile(ptxPath, []byte(ptx), 0o600); err != nil {
		return 0, err
	}
	compile := exec.Command(ptxas, "-arch=sm_"+fmt.Sprintf("%d%d", major, minor), "-lineinfo", "-o", cubinPath, ptxPath)
	if out, err := compile.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("ptxas: %w: %s", err, out)
	}
	disassemble := exec.Command(nvdisasm, "-g", cubinPath)
	out, err := disassemble.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("nvdisasm: %w: %s", err, out)
	}
	parsed, err := cudasass.Parse(bytes.NewReader(out))
	if err != nil {
		return 0, fmt.Errorf("parse loop SASS: %w", err)
	}
	f := parsed.Function("heavy_kernel")
	if f == nil {
		return 0, fmt.Errorf("SASS omitted heavy_kernel")
	}
	marker := "v = v * 1664525u + 1013904223u;"
	line := 0
	for i, text := range strings.Split(source, "\n") {
		if strings.Contains(text, marker) {
			line = i + 1
			break
		}
	}
	if line == 0 {
		return 0, fmt.Errorf("loop body marker not found in source")
	}
	count := 0
	for _, inst := range f.Instructions {
		if inst.SourceLine == line {
			count++
		}
	}
	if count == 0 {
		return 0, fmt.Errorf("SASS has no instructions attributed to loop body source line %d", line)
	}
	t.Logf("static loop bound: ptxas %s -arch=sm_%d%d -lineinfo; nvdisasm -g; source line %d maps to %d SASS instructions", ptxas, major, minor, line, count)
	return count, nil
}
