package arch

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
)

func fixed(p nvrtc.Program) error {
	return p.Compile("--gpu-architecture=sm_90") // want `Compile fixes the GPU architecture at sm_90; derive it from Device.ComputeCapability`
}

func split(p nvrtc.Program) error {
	return p.Compile("-default-device", "-arch", "compute_80") // want `Compile fixes the GPU architecture at compute_80; derive it from Device.ComputeCapability`
}

func derived(p nvrtc.Program, dev driver.Device) error {
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	return p.Compile(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor))
}

func unrelated(p nvrtc.Program) error {
	return p.Compile("-default-device", "sm_90")
}
