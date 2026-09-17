package nvrtcx

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/nvrtc"
)

// OutputKind selects an NVRTC artifact format.
type OutputKind uint8

const (
	// OutputAuto preserves the default PTX and optional CUBIN output.
	OutputAuto OutputKind = iota
	OutputPTX
	OutputCUBIN
	OutputLTOIR
	OutputTileIR
	// OutputPTXTileIR preserves both outputs from a mixed SIMT/Tile compilation.
	OutputPTXTileIR
	OutputOptiXIR
)

func outputKindName(k OutputKind) string {
	switch k {
	case OutputAuto:
		return "OutputAuto"
	case OutputPTX:
		return "OutputPTX"
	case OutputCUBIN:
		return "OutputCUBIN"
	case OutputLTOIR:
		return "OutputLTOIR"
	case OutputTileIR:
		return "OutputTileIR"
	case OutputPTXTileIR:
		return "OutputPTXTileIR"
	case OutputOptiXIR:
		return "OutputOptiXIR"
	default:
		return fmt.Sprintf("OutputKind(%d)", uint8(k))
	}
}

// WithOutput selects the requested output format or formats. LTOIR enables link-time optimization;
// TileIR enables tile-only compilation. Other language and target options remain
// caller-owned. Mixed PTX/TileIR enables Tile constructs without suppressing
// SIMT output. Unsupported formats return compiler capability errors.
func WithOutput(kind OutputKind) Option {
	return func(o *options) {
		if kind > OutputOptiXIR {
			o.err = fmt.Errorf("nvrtcx: invalid output kind %d", kind)
			return
		}
		if o.physicalArch && (kind == OutputLTOIR || kind == OutputTileIR || kind == OutputPTXTileIR || kind == OutputOptiXIR) {
			o.err = fmt.Errorf("nvrtcx: physical architecture cannot combine with WithOutput(%s)", outputKindName(kind))
			return
		}
		o.output = kind
		o.outputSet = true
	}
}

// Output returns the selected output format, or OutputAuto for a nil module.
func (cm *CompiledModule) Output() OutputKind {
	if cm == nil {
		return OutputAuto
	}
	return cm.output
}

// LTOIR returns an owned copy of link-time optimization IR, if selected.
// LTOIR must be linked before it can be loaded as a CUDA module.
func (cm *CompiledModule) LTOIR() []byte {
	if cm == nil {
		return nil
	}
	return append([]byte(nil), cm.ltoir...)
}

// TileIR returns an owned copy of CUDA Tile IR, if selected.
// Loading requires a driver supporting the emitted format.
func (cm *CompiledModule) TileIR() []byte {
	if cm == nil {
		return nil
	}
	return append([]byte(nil), cm.tileir...)
}

// OptiXIR returns an owned copy of OptiX IR, if selected.
// OptiX IR must be loaded via optixModuleCreate rather than the CUDA driver loader.
func (cm *CompiledModule) OptiXIR() []byte {
	if cm == nil {
		return nil
	}
	return append([]byte(nil), cm.optixir...)
}

func outputFlags(kind OutputKind, flags []string) ([]string, error) {
	flags = append([]string(nil), flags...)
	for _, flag := range flags {
		name, _, _ := strings.Cut(flag, "=")
		switch name {
		case "-dlto", "--dlink-time-opt":
			if kind != OutputLTOIR {
				return nil, fmt.Errorf("nvrtcx: LTO flag requires OutputLTOIR")
			}
		case "-tile-only", "--tile-only":
			if kind != OutputTileIR {
				return nil, fmt.Errorf("nvrtcx: tile-only flag requires OutputTileIR")
			}
		case "-simt-only", "--simt-only":
			if kind == OutputTileIR || kind == OutputPTXTileIR {
				return nil, fmt.Errorf("nvrtcx: simt-only flag conflicts with TileIR output")
			}
		case "-optix-ir", "--optix-ir":
			if kind != OutputOptiXIR {
				return nil, fmt.Errorf("nvrtcx: OptiX IR flag requires OutputOptiXIR")
			}
		}
	}
	switch kind {
	case OutputPTXTileIR:
		flags = append(flags, nvrtc.FlagEnableTile)
	case OutputLTOIR:
		flags = append(flags, nvrtc.FlagDLTO)
	case OutputTileIR:
		flags = append(flags, nvrtc.FlagTileOnly)
	case OutputOptiXIR:
		flags = append(flags, nvrtc.FlagOptiXIR)
	}
	return flags, nil
}

// ptxText converts the PTX buffer NVRTC returns into text. nvrtcGetPTXSize
// counts the trailing NUL, which is not part of the assembly and makes the
// result compare, hash, and grep as binary.
func ptxText(b []byte) string {
	return string(bytes.TrimSuffix(b, []byte{0}))
}

func getArtifact(prog nvrtc.Program, kind OutputKind) (*CompiledModule, error) {
	cm := &CompiledModule{output: kind}
	var data []byte
	var err error
	switch kind {
	case OutputAuto, OutputPTX:
		data, err = prog.GetPTX()
		cm.ptx = ptxText(data)
		if kind == OutputAuto && err == nil {
			cm.cubin, _ = prog.GetCUBIN()
		}
	case OutputCUBIN:
		data, err = prog.GetCUBIN()
		cm.cubin = data
		if err == nil {
			if ptxData, ptxErr := prog.GetPTX(); ptxErr == nil {
				cm.ptx = ptxText(ptxData)
			}
		}
	case OutputLTOIR:
		data, err = prog.GetLTOIR()
		cm.ltoir = data
	case OutputPTXTileIR:
		data, err = prog.GetPTX()
		if err != nil {
			return nil, fmt.Errorf("nvrtcx: get PTX: %w", err)
		}
		if len(data) == 0 {
			return nil, fmt.Errorf("nvrtcx: compiler returned empty PTX")
		}
		cm.ptx = ptxText(data)
		fallthrough
	case OutputTileIR:
		data, err = prog.GetTileIR()
		cm.tileir = data
	case OutputOptiXIR:
		data, err = prog.GetOptiXIR()
		cm.optixir = data
	default:
		return nil, fmt.Errorf("nvrtcx: invalid output kind %d", kind)
	}
	if err != nil {
		return nil, fmt.Errorf("nvrtcx: get output %d: %w", kind, err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("nvrtcx: compiler returned empty output %d", kind)
	}
	return cm, nil
}

func (cm *CompiledModule) validArtifact() bool {
	if cm == nil {
		return false
	}
	switch cm.output {
	case OutputAuto:
		return (cm.ptx != "" || len(cm.cubin) > 0) && len(cm.ltoir) == 0 && len(cm.tileir) == 0 && len(cm.optixir) == 0
	case OutputPTX:
		return cm.ptx != "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) == 0 && len(cm.optixir) == 0
	case OutputCUBIN:
		return len(cm.cubin) > 0 && len(cm.ltoir) == 0 && len(cm.tileir) == 0 && len(cm.optixir) == 0
	case OutputLTOIR:
		return cm.ptx == "" && len(cm.cubin) == 0 && len(cm.ltoir) > 0 && len(cm.tileir) == 0 && len(cm.optixir) == 0
	case OutputPTXTileIR:
		return cm.ptx != "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) > 0 && len(cm.optixir) == 0
	case OutputTileIR:
		return cm.ptx == "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) > 0 && len(cm.optixir) == 0
	case OutputOptiXIR:
		return cm.ptx == "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) == 0 && len(cm.optixir) > 0
	}
	return false
}
