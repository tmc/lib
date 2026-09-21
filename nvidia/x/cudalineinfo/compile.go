package cudalineinfo

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Options configures [Compile].
type Options struct {
	// Arch is the physical architecture to compile for, such as sm_121. It
	// is required: attribution is read from a CUBIN, and a virtual target
	// would leave the final compilation to the driver.
	Arch string

	// Nvdisasm is the path to nvdisasm. Empty searches PATH. The tool must
	// understand Arch; a CUDA 12 nvdisasm rejects sm_121.
	Nvdisasm string

	// NoDirectives compiles the unit's source as written, without #line
	// directives. Every tier then names only the generated file, which is
	// the control that shows the directives carry the attribution.
	NoDirectives bool

	// NVRTC holds further compiler options, such as [nvrtcx.WithFastMath].
	NVRTC []nvrtcx.Option
}

// Compile compiles the unit with line information, disassembles the result,
// and attributes every tier to its sites.
func Compile(ctx context.Context, unit Unit, opts Options) (*Attribution, error) {
	if opts.Arch == "" {
		return nil, errors.New("cudalineinfo: no architecture")
	}
	src := unit.SourceWithLineDirectives()
	if opts.NoDirectives {
		src = unit.Source
	}
	nopts := []nvrtcx.Option{
		nvrtcx.WithContext(ctx),
		nvrtcx.WithName(unit.GeneratedName()),
		nvrtcx.WithLineInfo(),
		nvrtcx.WithPhysicalArchTarget(opts.Arch),
	}
	nopts = append(nopts, opts.NVRTC...)
	mod, err := nvrtcx.Compile(src, nopts...)
	if err != nil {
		return nil, fmt.Errorf("compile %s: %w", unit.Name, err)
	}
	out, err := cudasass.Disassembler{Path: opts.Nvdisasm}.Disassemble(ctx, mod.CUBIN(), cudasass.Options{
		PrintCode:   true,
		SourceLines: true,
	})
	if err != nil {
		return nil, fmt.Errorf("disassemble %s: %w", unit.Name, err)
	}
	dis, err := cudasass.ParseString(string(out.Stdout))
	if err != nil {
		return nil, fmt.Errorf("parse disassembly of %s: %w", unit.Name, err)
	}
	a := Attribute(unit, mod.PTX(), dis)
	a.Module = mod
	return a, nil
}
