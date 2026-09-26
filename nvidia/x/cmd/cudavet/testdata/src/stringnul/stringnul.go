package stringnul

import "github.com/tmc/lib/nvidia/nvrtc"

const src = "extern \"C\" __global__ void k() {}"

func compile(p nvrtc.Program) error {
	return p.Compile("-default-device\x00") // want `Compile argument contains a NUL byte; Compile rejects it at run time`
}

func create() (nvrtc.Program, error) {
	return nvrtc.CreateProgram(src, "k.cu\x00") // want `CreateProgram argument contains a NUL byte; CreateProgram rejects it at run time`
}

func clean() (nvrtc.Program, error) {
	p, err := nvrtc.CreateProgram(src, "k.cu")
	if err != nil {
		return 0, err
	}
	return p, p.Compile("-default-device")
}
