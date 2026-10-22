package nvjitlinkx

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Option configures a link. The zero value is invalid.
type Option func(*linkOptions) error

type linkOptions struct{ flags []string }

// LinkWithOptions links inputs using validated nvJitLink options.
// Link remains available for callers that pass native option strings.
func LinkWithOptions(inputs []Input, opts ...Option) (*Result, error) {
	var o linkOptions
	for _, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("nvjitlinkx: nil option")
		}
		if err := opt(&o); err != nil {
			return nil, err
		}
	}
	return Link(inputs, o.flags...)
}

// WithFlag adds a documented nvJitLink option.
func WithFlag(flag string) Option {
	return func(o *linkOptions) error {
		if err := nvjitlink.ValidateOption(flag); err != nil {
			return err
		}
		o.flags = append(o.flags, flag)
		return nil
	}
}

// WithArch selects an SM target such as sm_90.
func WithArch(arch string) Option { return WithFlag(nvjitlink.FlagArch + "=" + arch) }

// WithLTO enables link-time optimization.
func WithLTO() Option { return WithFlag(nvjitlink.FlagLTO) }

// WithPTX requests PTX output. It requires WithLTO.
func WithPTX() Option { return WithFlag(nvjitlink.FlagPTX) }

// WithMaxRegisters sets the maximum register count.
func WithMaxRegisters(count int) Option {
	return WithFlag(nvjitlink.FlagMaxrregcount + "=" + strconv.Itoa(count))
}

// WithVerbose enables linker diagnostic messages.
func WithVerbose() Option { return WithFlag(nvjitlink.FlagVerbose) }

// InputFromArtifact selects the linkable output of an NVRTC compilation.
// It prefers CUBIN for OutputAuto and requires explicit LTO for LTOIR inputs.
func InputFromArtifact(name string, cm *nvrtcx.CompiledModule) (Input, error) {
	if name == "" || strings.ContainsRune(name, 0) || cm == nil {
		return Input{}, fmt.Errorf("nvjitlinkx: invalid compiled artifact")
	}
	var in Input
	in.Name = name
	switch cm.Output() {
	case nvrtcx.OutputLTOIR:
		in.Kind, in.Data = nvjitlink.InputLtoir, cm.LTOIR()
	case nvrtcx.OutputCUBIN:
		in.Kind, in.Data = nvjitlink.InputCubin, cm.CUBIN()
	case nvrtcx.OutputPTX:
		in.Kind, in.Data = nvjitlink.InputPtx, []byte(cm.PTX())
	case nvrtcx.OutputAuto:
		if data := cm.CUBIN(); len(data) > 0 {
			in.Kind, in.Data = nvjitlink.InputCubin, data
		} else {
			in.Kind, in.Data = nvjitlink.InputPtx, []byte(cm.PTX())
		}
	default:
		return Input{}, fmt.Errorf("nvjitlinkx: output %d cannot be linked", cm.Output())
	}
	if len(in.Data) == 0 {
		return Input{}, fmt.Errorf("nvjitlinkx: artifact has no linkable output")
	}
	return in, nil
}

// InputFromKernelArtifact selects the linkable input from a cudakernel.Artifact.
func InputFromKernelArtifact(name string, a *cudakernel.Artifact) (Input, error) {
	if name == "" || strings.ContainsRune(name, 0) || a == nil {
		return Input{}, fmt.Errorf("nvjitlinkx: invalid kernel artifact")
	}
	var in Input
	in.Name = name
	owned := *a
	switch owned.Kind() {
	case cudakernel.KindLTOIR:
		in.Kind, in.Data = nvjitlink.InputLtoir, owned.Bytes()
	case cudakernel.KindCUBIN:
		in.Kind, in.Data = nvjitlink.InputCubin, owned.Bytes()
	case cudakernel.KindPTX:
		in.Kind, in.Data = nvjitlink.InputPtx, owned.Bytes()
	default:
		return Input{}, fmt.Errorf("nvjitlinkx: artifact kind %s cannot be linked", owned.Kind())
	}
	if len(in.Data) == 0 {
		return Input{}, fmt.Errorf("nvjitlinkx: artifact has no data")
	}
	return in, nil
}
