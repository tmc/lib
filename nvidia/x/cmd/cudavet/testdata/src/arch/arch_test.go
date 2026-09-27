package arch

import (
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
)

// Tests and examples that only compile never run their output on a GPU.

func ExampleProgram_Compile() {
	p, _ := nvrtc.CreateProgram("", "k.cu")
	p.Compile("--gpu-architecture=compute_70")
}

func TestCompileOnly(t *testing.T) {
	t.Run("sub", func(t *testing.T) {
		p, _ := nvrtc.CreateProgram("", "k.cu")
		p.Compile("-arch=sm_80")
	})
}

func TestCompileWithHostHelper(t *testing.T) {
	p, _ := nvrtc.CreateProgram("", "k.cu")
	p.Compile("-arch=sm_80")
	check(t)
}

func check(t *testing.T) {}

// Once one loads or launches, the fixed architecture must match the GPU.

func TestCompileThenLoad(t *testing.T) {
	p, _ := nvrtc.CreateProgram("", "k.cu")
	p.Compile("-arch=sm_80") // want `Compile fixes the GPU architecture at sm_80`
	driver.ModuleLoadDataEx(nil, 0, nil, nil)
}

func TestCompileThenLaunchInLiteral(t *testing.T) {
	p, _ := nvrtc.CreateProgram("", "k.cu")
	p.Compile("-arch=sm_80") // want `Compile fixes the GPU architecture at sm_80`
	t.Run("run", func(t *testing.T) {
		var f driver.Function
		f.LaunchKernel(driver.Dim3{}, driver.Dim3{}, 0, 0, []unsafe.Pointer{})
	})
}

func TestCompileThenHelperLaunches(t *testing.T) {
	p, _ := nvrtc.CreateProgram("", "k.cu")
	p.Compile("-arch=sm_80") // want `Compile fixes the GPU architecture at sm_80`
	run(nil)
}

func run(image []byte) { launch(image) }

func launch(image []byte) {
	var g driver.GraphExec
	g.Launch(0)
}

// A helper may hand its output to a caller that loads it.
func compileHelper(p nvrtc.Program) error {
	return p.Compile("-arch=sm_80") // want `Compile fixes the GPU architecture at sm_80`
}
