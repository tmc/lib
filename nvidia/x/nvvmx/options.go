package nvvmx

import (
	"crypto/sha256"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/nvvm"
)

// Option configures an NVVM compilation. The zero value is invalid.
type Option func(*compileOptions) error

type compileOptions struct {
	flags          []string
	inferAddrSpace bool
}

// CompileWithOptions compiles inputs using validated NVVM options.
// Compile remains available for callers that pass native option strings.
func CompileWithOptions(inputs []Input, opts ...Option) (*Result, error) {
	return compileWithOptions(inputs, newProgram, opts...)
}

func compileWithOptions(inputs []Input, create func() (program, error), opts ...Option) (*Result, error) {
	var o compileOptions
	for _, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("nvvmx: nil option")
		}
		if err := opt(&o); err != nil {
			return nil, err
		}
	}
	prepared, reports, err := prepareInferenceInputs(inputs, o.inferAddrSpace)
	if err != nil {
		return nil, err
	}
	result, err := compile(prepared, o.flags, create)
	if err != nil || result == nil || !o.inferAddrSpace || len(result.PTX) == 0 {
		return result, err
	}
	provenance := &CompilationProvenance{PTXSHA256: fmt.Sprintf("%x", sha256.Sum256(result.PTX))}
	for i, report := range reports {
		if report == nil {
			continue
		}
		snapshot, e := report.Provenance()
		if e != nil {
			return result, e
		}
		provenance.Inputs = append(provenance.Inputs, InferredInput{Name: prepared[i].Name, Provenance: *snapshot})
	}
	result.inference = provenance
	return result, nil
}

func prepareInputs(inputs []Input, inferAddrSpace bool) ([]Input, error) {
	prepared, _, err := prepareInferenceInputs(inputs, inferAddrSpace)
	return prepared, err
}

func prepareInferenceInputs(inputs []Input, inferAddrSpace bool) ([]Input, []*InferenceReport, error) {
	prepared := append([]Input(nil), inputs...)
	reports := make([]*InferenceReport, len(inputs))
	if !inferAddrSpace {
		return prepared, reports, nil
	}
	for i, input := range inputs {
		if input.Lazy || len(input.Data) == 0 {
			continue
		}
		rewritten, report, err := InferAddressSpaces(input.Data)
		if err != nil {
			return nil, nil, fmt.Errorf("nvvmx: infer address spaces for %q: %w", input.Name, err)
		}
		prepared[i].Data = rewritten
		reports[i] = report
	}
	return prepared, reports, nil
}

// WithAddressSpaceInference enables the address-space inference pass on input IR.
func WithAddressSpaceInference() Option {
	return func(o *compileOptions) error {
		o.inferAddrSpace = true
		return nil
	}
}

// WithFlag adds a documented NVVM compiler option after validating it.
func WithFlag(flag string) Option {
	return func(o *compileOptions) error {
		if err := nvvm.ValidateOption(flag); err != nil {
			return err
		}
		o.flags = append(o.flags, flag)
		return nil
	}
}

// WithArch selects a virtual target architecture such as compute_80 or compute_90a.
func WithArch(arch string) Option {
	if !strings.HasPrefix(arch, "compute_") {
		arch = "compute_" + arch
	}
	return WithFlag(nvvm.FlagArch + "=" + arch)
}

// WithOpt sets optimization level: 0 to disable optimizations, 3 to enable optimizations.
func WithOpt(level int) Option {
	return WithFlag(fmt.Sprintf("%s=%d", nvvm.FlagOpt, level))
}

// WithGenLTO requests generation of LTO IR instead of PTX.
func WithGenLTO() Option {
	return WithFlag(nvvm.FlagGenLTO)
}

// WithDebug enables generation of full debugging information (-g).
// Note: per NVVM specification, debug support is only valid with -opt=0.
func WithDebug() Option {
	return WithFlag(nvvm.FlagG)
}

// WithFTZ controls flushing of denormals to zero in single-precision floating-point operations.
func WithFTZ(flush bool) Option {
	val := "0"
	if flush {
		val = "1"
	}
	return WithFlag(nvvm.FlagFtz + "=" + val)
}

// WithPrecSqrt controls whether IEEE round-to-nearest mode (true, default) or a faster approximation (false)
// is used for single-precision square root.
func WithPrecSqrt(ieee bool) Option {
	val := "0"
	if ieee {
		val = "1"
	}
	return WithFlag(nvvm.FlagPrecSqrt + "=" + val)
}

// WithPrecDiv controls whether IEEE round-to-nearest mode (true, default) or a faster approximation (false)
// is used for single-precision division.
func WithPrecDiv(ieee bool) Option {
	val := "0"
	if ieee {
		val = "1"
	}
	return WithFlag(nvvm.FlagPrecDiv + "=" + val)
}

// WithFMA controls contraction of floating-point multiply-add operations.
func WithFMA(enable bool) Option {
	val := "0"
	if enable {
		val = "1"
	}
	return WithFlag(nvvm.FlagFma + "=" + val)
}

// WithJumpTableDensity specifies the minimal switch case density percentage (0 to 101).
func WithJumpTableDensity(density int) Option {
	return WithFlag(fmt.Sprintf("%s=%d", nvvm.FlagJumpTableDensity, density))
}
