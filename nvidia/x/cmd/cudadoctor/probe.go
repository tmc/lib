package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func probeMain(name, arch string, out, errout io.Writer) int {
	c := check{Name: name, Status: "pass"}
	var err error
	switch name {
	case "nvrtc", "tileir":
		err = nvrtc.Load("")
		c.Path = nvrtc.Path()
		if err == nil {
			var major, minor int32
			major, minor, err = nvrtc.Version()
			c.Version = fmt.Sprintf("%d.%d", major, minor)
		}
		c.Detail = "NVRTC loaded and version queried; no compilation performed"
		c.Remedy = "Install NVRTC or set NVRTC_LIB_PATH to the intended library before starting the process."
		if err == nil && name == "tileir" {
			if !nvrtc.HasGetTileIR() || !nvrtc.HasGetTileIRSize() {
				err = fmt.Errorf("loaded NVRTC lacks TileIR extraction entry points")
			}
			c.Detail = "TileIR extraction symbols present; Tile C++ headers, compilation, and execution are not validated"
			c.Remedy = "Select a TileIR-capable NVRTC and provide the Tile C++ headers for your compiler workflow."
		}
	case "driver":
		err = driver.Load("")
		c.Path = driver.Path()
		c.Detail = "driver library loaded; no device/context initialization or execution probe"
		c.Remedy = "Install the NVIDIA driver or correct CUDA_LIB_PATH; a toolkit stub is not a usable execution driver."
	case "cupti":
		err = cupti.Load("")
		c.Path = cupti.Path()
		if err == nil {
			var v uint32
			v, err = cupti.GetVersion()
			c.Version = fmt.Sprint(v)
		}
		c.Detail = "CUPTI loaded; profiling permission, device support, and capture completeness are not validated"
		c.Remedy = "Install CUPTI or set CUPTI_LIB_PATH; successful loading alone does not establish profiling permissions."
	case "smoke-compile", "smoke-load", "smoke-run":
		err = smokeTest(strings.TrimPrefix(name, "smoke-"), arch)
		c.Detail = map[string]string{"smoke-compile": "compiled a fresh PTX kernel (cache disabled)", "smoke-load": "compiled and loaded a PTX kernel on device 0", "smoke-run": "compiled, loaded, launched, synchronized and checked device output equals 42"}[name]
		c.Remedy = "Check the compiler target and selected driver; use the failing stage's diagnostic below."
	default:
		fmt.Fprintln(errout, "unknown probe", name)
		return 2
	}
	if err != nil {
		c.Status = "fail"
		c.Detail = err.Error()
	} else {
		c.Remedy = ""
	}
	if c.Path != "" && !strings.Contains(c.Path, "/") {
		c.Detail += "; library path is the loader request, resolved by the system dynamic linker"
	}
	if os.Getenv("CODEGEN_SKIP_BINDINGS") != "" {
		c.Status = "fail"
		c.Detail = "CODEGEN_SKIP_BINDINGS disables bindings; readiness cannot be established"
		c.Remedy = "Unset CODEGEN_SKIP_BINDINGS."
	}
	if err := json.NewEncoder(out).Encode(c); err != nil {
		fmt.Fprintln(errout, err)
		return 1
	}
	return 0
}

func smokeTest(stage, arch string) (err error) {
	if !strings.HasPrefix(arch, "compute_") {
		return fmt.Errorf("smoke compilation needs a virtual compute_* target")
	}
	cm, err := nvrtcx.Compile(`extern "C" __global__ void doctor(int *out) { out[0] = 42; }`, nvrtcx.WithFlag("--gpu-architecture="+arch), nvrtcx.WithOutput(nvrtcx.OutputPTX), nvrtcx.WithNoCache())
	if err != nil {
		return fmt.Errorf("compile: %w", err)
	}
	if len(cm.PTX()) == 0 {
		return fmt.Errorf("compile: empty PTX")
	}
	if stage == "compile" {
		return nil
	}
	if err := cuda.Init(); err != nil {
		return fmt.Errorf("initialize device: %w", err)
	}
	devices, err := cuda.Devices()
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		return fmt.Errorf("no CUDA devices")
	}
	for _, dev := range devices {
		defer func() { err = errors.Join(err, dev.Close()) }()
	}
	dev := devices[0]
	var kernels struct {
		Doctor *cuda.Kernel `cuda:"doctor"`
	}
	mod, err := cm.Bind(dev, &kernels)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	defer func() { err = errors.Join(err, mod.Close()) }()
	if stage == "load" {
		return nil
	}
	buf, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, buf.Free()) }()
	if err = buf.CopyFromHost([]int32{0}); err != nil {
		return err
	}
	stream := dev.DefaultStream()
	if err = cudalaunch.OwnedKernel(kernels.Doctor).Grid(1, 1, 1).Block(1, 1, 1).On(stream).Launch(buf); err != nil {
		return fmt.Errorf("launch: %w", err)
	}
	if err = stream.Synchronize(context.Background()); err != nil {
		return fmt.Errorf("synchronize: %w", err)
	}
	out := make([]int32, 1)
	if err = buf.CopyToHost(out); err != nil {
		return err
	}
	if out[0] != 42 {
		return fmt.Errorf("output = %d, want 42", out[0])
	}
	return nil
}
