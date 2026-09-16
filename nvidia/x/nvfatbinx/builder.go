package nvfatbinx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Builder collects device images for one fatbinary. Its zero value is ready
// for use. A Builder must not be mutated concurrently with Build.
type Builder struct {
	inputs  []Input
	options []string
}

// NewBuilder creates a builder with native nvFatbin options.
func NewBuilder(options ...string) *Builder {
	return &Builder{options: append([]string(nil), options...)}
}

// AddInput adds one image after checking its architecture and format.
func (b *Builder) AddInput(input Input) error {
	if b == nil {
		return fmt.Errorf("nvfatbinx: nil builder")
	}
	if input.Name == "" || strings.ContainsRune(input.Name, 0) || len(input.Data) == 0 {
		return fmt.Errorf("nvfatbinx: invalid input")
	}
	switch input.Kind {
	case CUBIN, PTX, LTOIR:
		if !slices.Contains(cudatool.KnownGPUArchitectures, "sm_"+input.Arch) {
			return fmt.Errorf("nvfatbinx: unknown architecture %q", input.Arch)
		}
		if input.Kind == CUBIN && input.JITOptions != "" {
			return fmt.Errorf("nvfatbinx: CUBIN does not accept JIT options")
		}
	case TileIR:
		if input.Arch != "" {
			return fmt.Errorf("nvfatbinx: TileIR does not accept an architecture")
		}
	default:
		return fmt.Errorf("nvfatbinx: unsupported input kind %d", input.Kind)
	}
	if strings.ContainsRune(input.JITOptions, 0) {
		return fmt.Errorf("nvfatbinx: JIT options contain NUL")
	}
	input.Data = append([]byte(nil), input.Data...)
	b.inputs = append(b.inputs, input)
	return nil
}

// AddArtifact adds the CUBIN, PTX, or LTOIR of an NVRTC compilation.
// A mixed or TileIR result requires an explicit Input.
func (b *Builder) AddArtifact(name, arch string, cm *nvrtcx.CompiledModule) error {
	if cm == nil {
		return fmt.Errorf("nvfatbinx: nil artifact")
	}
	in := Input{Name: name, Arch: arch}
	switch cm.Output() {
	case nvrtcx.OutputCUBIN:
		in.Kind, in.Data = CUBIN, cm.CUBIN()
	case nvrtcx.OutputPTX:
		in.Kind, in.Data = PTX, []byte(cm.PTX())
	case nvrtcx.OutputLTOIR:
		in.Kind, in.Data = LTOIR, cm.LTOIR()
	case nvrtcx.OutputAuto:
		if data := cm.CUBIN(); len(data) > 0 {
			in.Kind, in.Data = CUBIN, data
		} else {
			in.Kind, in.Data = PTX, []byte(cm.PTX())
		}
	default:
		return fmt.Errorf("nvfatbinx: output %d cannot be added as an architecture image", cm.Output())
	}
	return b.AddInput(in)
}

// Build packages the accumulated inputs. It leaves the builder reusable.
func (b *Builder) Build() (*Result, error) {
	if b == nil {
		return nil, fmt.Errorf("nvfatbinx: nil builder")
	}
	return Build(b.inputs, b.options...)
}
