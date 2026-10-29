package hopper

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// TileIRStep represents a hopper kernel step (RowKernel, TileGEMM, SkinnyGEMM, or PlanKernel)
// that can be lowered to a Tile IR Module and verified against CUDA host emulation.
type TileIRStep interface {
	stepKernel() PlanKernel
}

func (pk PlanKernel) stepKernel() PlanKernel { return pk }

func (k RowKernel) stepKernel() PlanKernel {
	rk := k
	return PlanKernel{Row: &rk}
}

func (g TileGEMM) stepKernel() PlanKernel {
	tg := g
	return PlanKernel{Tile: &tg}
}

func (g SkinnyGEMM) stepKernel() PlanKernel {
	sg := g
	return PlanKernel{GEMM: &sg}
}

// NewRMSNormKernel returns a RowKernel configured for RMSNorm over row width D:
//
//	out[m, c] = x[m, c] * rsqrt(mean(x[m, :]^2) + eps) * weight[c]
func NewRMSNormKernel(name string, d int, eps float32) RowKernel {
	return fromTileBridgeRow(tilebridge.NewRMSNormKernel(name, d, eps))
}

// NewResidualRMSNormKernel returns a RowKernel configured for fused ResidualAdd + RMSNorm:
//
//	h[m, c]   = x[m, c] + residual[m, c]
//	out[m, c] = h[m, c] * rsqrt(mean(h[m, :]^2) + eps) * weight[c]
func NewResidualRMSNormKernel(name string, d int, eps float32) RowKernel {
	return fromTileBridgeRow(tilebridge.NewResidualRMSNormKernel(name, d, eps))
}

// NewSwiGLUKernel returns a RowKernel configured for elementwise SwiGLU:
//
//	out[m, c] = silu(gate[m, c]) * up[m, c]
func NewSwiGLUKernel(name string, d int) RowKernel {
	return fromTileBridgeRow(tilebridge.NewSwiGLUKernel(name, d))
}

// FormatTileIR formats a Tile IR Module into its textual representation.
func FormatTileIR(mod *tileir.Module) string { return tilebridge.FormatTileIR(mod) }

// ParseTileIR parses textual Tile IR into a Module.
func ParseTileIR(src string) (*tileir.Module, error) { return tilebridge.ParseTileIR(src) }

// EncodeTileIR encodes a Tile IR Module into binary Tile IR bytecode.
func EncodeTileIR(mod *tileir.Module) ([]byte, error) { return tilebridge.EncodeTileIR(mod) }

// EvalTileIRReference executes a lowered Tile IR Module on the given input buffers
// and problem dimensions (M, N, K) using a pure-Go Tile IR SSA interpreter.
func EvalTileIRReference(mod *tileir.Module, inputs map[string][]float32, M, N, K int, eps float32) ([]float32, error) {
	return tilebridge.EvalTileIRReference(mod, inputs, M, N, K, eps)
}

// DualTargetReport summarizes the dual-target verification between the
// hopper CUDA C++ backend (executed via cudahostemu) and the Tile IR
// lowering path (formatted, parsed, bytecode-encoded, and evaluated).
type DualTargetReport = tilebridge.DualTargetReport

// StepToTileIR lowers a hopper step (RowKernel, TileGEMM, SkinnyGEMM, or PlanKernel)
// into a valid, bytecode-encodable *tileir.Module.
func StepToTileIR(step TileIRStep) (*tileir.Module, error) {
	tb, err := toTileBridgeStep(step, 0, false)
	if err != nil {
		return nil, err
	}
	return tilebridge.StepToTileIR(tb)
}

// ToTileIRModules lowers all supported steps in a Plan into Tile IR modules,
// verifying that every module round-trips through Format -> Parse and encodes
// cleanly to Tile IR bytecode.
func (p *Plan) ToTileIRModules() ([]*tileir.Module, error) {
	if p == nil {
		return nil, fmt.Errorf("hopper: nil plan")
	}
	var steps []tilebridge.Step
	for i, pk := range p.Kernels {
		if pk.Row == nil && pk.Tile == nil && pk.GEMM == nil {
			continue
		}
		st, err := toTileBridgeStep(pk, 0, false)
		if err != nil {
			return nil, fmt.Errorf("hopper: step %d: %w", i, err)
		}
		steps = append(steps, st)
	}
	return tilebridge.StepsToTileIRModules(steps)
}

// VerifyDualTargetAgreement lowers step to both CUDA C++ (executed via cudahostemu.Run)
// and a *tileir.Module (verified through Format -> Parse -> EncodeBytecode and evaluated
// via EvalTileIRReference) on identical synthetic inputs, returning an error if
// maxAbsDiff >= 2e-2.
func VerifyDualTargetAgreement(ctx context.Context, step TileIRStep, M, N, K int, eps float32) (*DualTargetReport, error) {
	tb, err := toTileBridgeStep(step, M, true)
	if err != nil {
		return nil, err
	}
	return tilebridge.VerifyDualTargetAgreement(ctx, tb, M, N, K, eps)
}

func fromTileBridgeRow(tb tilebridge.RowKernel) RowKernel {
	rk := RowKernel{Name: tb.Name, D: tb.D}
	for _, in := range tb.Inputs {
		rk.Inputs = append(rk.Inputs, RowTensor{
			Name:   in.Name,
			DType:  DType(in.DType),
			Kind:   RowKind(in.Kind),
			Offset: in.Offset,
			Buffer: in.Buffer,
		})
	}
	for _, v := range tb.Values {
		rk.Values = append(rk.Values, RowValue{
			Name:   v.Name,
			Reduce: v.Reduce,
			Body:   MustLambda(v.Body.String()),
			Args:   v.Args,
		})
	}
	for _, o := range tb.Outputs {
		rk.Outputs = append(rk.Outputs, RowTensor{
			Name:  o.Name,
			DType: DType(o.DType),
			Kind:  RowKind(o.Kind),
			Value: o.Value,
		})
	}
	return rk
}

func toTileBridgeStep(step TileIRStep, m int, withCUDA bool) (tilebridge.Step, error) {
	if step == nil {
		return tilebridge.Step{}, fmt.Errorf("hopper: nil step")
	}
	pk := step.stepKernel()
	switch {
	case pk.Row != nil:
		k := *pk.Row
		if _, err := k.check(); err != nil {
			return tilebridge.Step{}, err
		}
		rk := &tilebridge.RowKernel{Name: k.Name, D: k.D}
		for _, in := range k.Inputs {
			rk.Inputs = append(rk.Inputs, tilebridge.RowTensor{
				Name:   in.Name,
				DType:  tilebridge.DType(in.DType),
				Kind:   tilebridge.RowKind(in.Kind),
				Offset: in.Offset,
				Buffer: in.buffer(),
			})
		}
		for _, v := range k.Values {
			l, err := tilebridge.ParseLambda(v.Body.String())
			if err != nil {
				return tilebridge.Step{}, err
			}
			rk.Values = append(rk.Values, tilebridge.RowValue{
				Name:   v.Name,
				Reduce: v.Reduce,
				Body:   l,
				Args:   v.Args,
			})
		}
		for _, o := range k.Outputs {
			rk.Outputs = append(rk.Outputs, tilebridge.RowTensor{
				Name:  o.Name,
				DType: tilebridge.DType(o.DType),
				Kind:  tilebridge.RowKind(o.Kind),
				Value: o.Value,
			})
		}
		if withCUDA {
			src, err := k.CUDA()
			if err != nil {
				return tilebridge.Step{}, err
			}
			rk.CUDASource = Prelude + src
		}
		return tilebridge.Step{Row: rk}, nil
	case pk.Tile != nil:
		g := *pk.Tile
		gm, err := toTileBridgeGEMM("tile_gemm", g.Name, g.N, g.K, g.GLU, g.Out, g.Epilogue, g.Extras)
		if err != nil {
			return tilebridge.Step{}, err
		}
		if withCUDA {
			src, err := g.CUDA()
			if err != nil {
				return tilebridge.Step{}, err
			}
			gm.CUDASource = Prelude + src
			gm.Grid, gm.Block = g.Grid(m, 1)
			gm.SharedBytes = g.SharedBytes()
		}
		return tilebridge.Step{GEMM: gm}, nil
	case pk.GEMM != nil:
		g := *pk.GEMM
		gm, err := toTileBridgeGEMM("skinny_gemm", g.Name, g.N, g.K, false, g.Out, g.Epilogue, g.Extras)
		if err != nil {
			return tilebridge.Step{}, err
		}
		if withCUDA {
			src, err := g.CUDA()
			if err != nil {
				return tilebridge.Step{}, err
			}
			grid, block, err := g.Grid()
			if err != nil {
				return tilebridge.Step{}, err
			}
			gm.CUDASource = Prelude + src
			gm.Grid, gm.Block = grid, block
			gm.WorkspaceBytes = g.WorkspaceBytes()
		}
		return tilebridge.Step{GEMM: gm}, nil
	default:
		return tilebridge.Step{}, fmt.Errorf("hopper: unsupported PlanKernel step (need Row, Tile, or GEMM)")
	}
}

func toTileBridgeGEMM(kind, name string, n, k int, glu bool, out DType, epilogue Lambda, extras []Operand) (*tilebridge.GEMM, error) {
	ep := tilebridge.Identity
	if epilogue.root != nil {
		var err error
		ep, err = tilebridge.ParseLambda(epilogue.String())
		if err != nil {
			return nil, err
		}
	}
	gm := &tilebridge.GEMM{
		Kind:     kind,
		Name:     name,
		N:        n,
		K:        k,
		GLU:      glu,
		Out:      tilebridge.DType(out),
		Epilogue: ep,
	}
	for _, ex := range extras {
		gm.Extras = append(gm.Extras, tilebridge.Operand{
			Name:      ex.Name,
			DType:     tilebridge.DType(ex.DType),
			Broadcast: tilebridge.Broadcast(ex.Broadcast),
		})
	}
	return gm, nil
}
