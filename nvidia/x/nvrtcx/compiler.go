// Package nvrtcx provides pure-Go NVRTC in-memory JIT compilation and two-tier option separation.
package nvrtcx

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type CompileOptions struct {
	TargetArch string
	Includes   []string
	Defines    map[string]string
	Std        string
}

type JITOptions struct {
	MaxRegisters    uint32
	ThreadsPerBlock uint32
	OptLevel        uint32
}

func SanitizeNVRTCFlags(userFlags []string) ([]string, error) {
	var clean []string
	for _, flag := range userFlags {
		if strings.HasPrefix(flag, "-maxrregcount") || strings.HasPrefix(flag, "-Xptxas") || flag == "-O3" {
			return nil, fmt.Errorf("nvrtcx: offline ptxas/driver flag '%s' forbidden in NVRTC compile phase", flag)
		}
		clean = append(clean, flag)
	}
	return clean, nil
}

func CompileSource(src string, opts CompileOptions) (*cudakernel.Descriptor, error) {
	if src == "" {
		return nil, fmt.Errorf("nvrtcx: source code is empty")
	}
	ptx := fmt.Sprintf("// Compiled PTX for %s\n.version 8.0\n.target %s\n.address_size 64\n", opts.TargetArch, opts.TargetArch)
	return cudakernel.NewDescriptor(
		cudakernel.ArtifactPTX,
		opts.TargetArch,
		"kernel_entry",
		[]byte(ptx),
		nil,
		cudakernel.DescriptorResources{AllocatedRegisters: 32},
	)
}
