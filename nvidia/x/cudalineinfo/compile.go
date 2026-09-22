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

	// Name is the file name the compiler records for the unit's source, and
	// so the name every tier reports its own lines under. Empty uses
	// [Unit.GeneratedName]. Set it when the unit is named for something
	// other than its file, such as a hand-written source holding several
	// kernels.
	Name string

	// NoDirectives compiles the unit's source as written, without #line
	// directives. Every tier then names only the generated file, which is
	// the control that shows the directives carry the attribution.
	NoDirectives bool

	// NVRTC holds further compiler options, such as [nvrtcx.WithFastMath].
	// They are applied after the ones Compile sets, but an option that
	// appends rather than assigns, such as [nvrtcx.WithPhysicalArchTarget],
	// does not replace what is already there. Use Arch and Name instead of
	// repeating those here.
	NVRTC []nvrtcx.Option
}

// name is the file name to compile the unit under.
func (o Options) name(u Unit) string {
	if o.Name != "" {
		return o.Name
	}
	return u.GeneratedName()
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
		nvrtcx.WithName(opts.name(unit)),
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
