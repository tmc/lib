package nvrtcx

import (
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
)

// WithOutput selects the requested output format or formats. LTOIR enables link-time optimization;
// TileIR enables tile-only compilation. Other language and target options remain
// caller-owned. Mixed PTX/TileIR enables Tile constructs without suppressing
// SIMT output. Unsupported formats return compiler capability errors.
func WithOutput(kind OutputKind) Option {
	return func(o *options) {
		if kind > OutputPTXTileIR {
			o.err = fmt.Errorf("nvrtcx: invalid output kind %d", kind)
			return
		}
		o.output = kind
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
			return nil, fmt.Errorf("nvrtcx: OptiX IR output is not supported")
		}
	}
	switch kind {
	case OutputPTXTileIR:
		flags = append(flags, "--enable-tile")
	case OutputLTOIR:
		flags = append(flags, "-dlto")
	case OutputTileIR:
		flags = append(flags, "--tile-only")
	}
	return flags, nil
}

func getArtifact(prog nvrtc.Program, kind OutputKind) (*CompiledModule, error) {
	cm := &CompiledModule{output: kind}
	var data []byte
	var err error
	switch kind {
	case OutputAuto, OutputPTX:
		data, err = prog.GetPTX()
		cm.ptx = string(data)
		if kind == OutputAuto && err == nil {
			cm.cubin, _ = prog.GetCUBIN()
		}
	case OutputCUBIN:
		data, err = prog.GetCUBIN()
		cm.cubin = data
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
		cm.ptx = string(data)
		fallthrough
	case OutputTileIR:
		data, err = prog.GetTileIR()
		cm.tileir = data
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
		return (cm.ptx != "" || len(cm.cubin) > 0) && len(cm.ltoir) == 0 && len(cm.tileir) == 0
	case OutputPTX:
		return cm.ptx != "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) == 0
	case OutputCUBIN:
		return cm.ptx == "" && len(cm.cubin) > 0 && len(cm.ltoir) == 0 && len(cm.tileir) == 0
	case OutputLTOIR:
		return cm.ptx == "" && len(cm.cubin) == 0 && len(cm.ltoir) > 0 && len(cm.tileir) == 0
	case OutputPTXTileIR:
		return cm.ptx != "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) > 0
	case OutputTileIR:
		return cm.ptx == "" && len(cm.cubin) == 0 && len(cm.ltoir) == 0 && len(cm.tileir) > 0
	}
	return false
}
