package tilebridge

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/tileir"
)

// TileIRStep represents a hopper kernel step ([hopper.RowKernel],
// [hopper.TileGEMM], [hopper.SkinnyGEMM], or [hopper.PlanKernel]) that can be
// lowered to a Tile IR Module and verified against CUDA host emulation.
type TileIRStep any

func toPlanKernel(step TileIRStep) (hopper.PlanKernel, error) {
	switch s := step.(type) {
	case hopper.PlanKernel:
		return s, nil
	case *hopper.PlanKernel:
		if s == nil {
			return hopper.PlanKernel{}, fmt.Errorf("tilebridge: nil *hopper.PlanKernel")
		}
		return *s, nil
	case hopper.RowKernel:
		rk := s
		return hopper.PlanKernel{Row: &rk}, nil
	case *hopper.RowKernel:
		if s == nil {
			return hopper.PlanKernel{}, fmt.Errorf("tilebridge: nil *hopper.RowKernel")
		}
		rk := *s
		return hopper.PlanKernel{Row: &rk}, nil
	case hopper.TileGEMM:
		tg := s
		return hopper.PlanKernel{Tile: &tg}, nil
	case *hopper.TileGEMM:
		if s == nil {
			return hopper.PlanKernel{}, fmt.Errorf("tilebridge: nil *hopper.TileGEMM")
		}
		tg := *s
		return hopper.PlanKernel{Tile: &tg}, nil
	case hopper.SkinnyGEMM:
		sg := s
		return hopper.PlanKernel{GEMM: &sg}, nil
	case *hopper.SkinnyGEMM:
		if s == nil {
			return hopper.PlanKernel{}, fmt.Errorf("tilebridge: nil *hopper.SkinnyGEMM")
		}
		sg := *s
		return hopper.PlanKernel{GEMM: &sg}, nil
	default:
		return hopper.PlanKernel{}, fmt.Errorf("tilebridge: unsupported TileIRStep %T", step)
	}
}

// NewHopperRMSNormKernel returns a [hopper.RowKernel] configured for RMSNorm
// over row width D:
//
//	out[m, c] = x[m, c] * rsqrt(mean(x[m, :]^2) + eps) * weight[c]
func NewHopperRMSNormKernel(name string, d int, eps float32) hopper.RowKernel {
	return FromTileBridgeRow(NewRMSNormKernel(name, d, eps))
}

// NewHopperResidualRMSNormKernel returns a [hopper.RowKernel] configured for
// fused ResidualAdd + RMSNorm:
//
//	h[m, c]   = x[m, c] + residual[m, c]
//	out[m, c] = h[m, c] * rsqrt(mean(h[m, :]^2) + eps) * weight[c]
func NewHopperResidualRMSNormKernel(name string, d int, eps float32) hopper.RowKernel {
	return FromTileBridgeRow(NewResidualRMSNormKernel(name, d, eps))
}

// NewHopperSwiGLUKernel returns a [hopper.RowKernel] configured for
// elementwise SwiGLU:
//
//	out[m, c] = silu(gate[m, c]) * up[m, c]
func NewHopperSwiGLUKernel(name string, d int) hopper.RowKernel {
	return FromTileBridgeRow(NewSwiGLUKernel(name, d))
}

// StepHopperToTileIR lowers a hopper step ([hopper.RowKernel],
// [hopper.TileGEMM], [hopper.SkinnyGEMM], or [hopper.PlanKernel]) into a
// valid, bytecode-encodable [*tileir.Module].
func StepHopperToTileIR(step TileIRStep) (*tileir.Module, error) {
	tb, err := ToTileBridgeStep(step, 0, false)
	if err != nil {
		return nil, err
	}
	return StepToTileIR(tb)
}

// PlanToTileIRModules lowers all supported steps in a [hopper.Plan] into Tile
// IR modules, verifying that every module round-trips through Format -> Parse
// and encodes cleanly to Tile IR bytecode.
func PlanToTileIRModules(p *hopper.Plan) ([]*tileir.Module, error) {
	if p == nil {
		return nil, fmt.Errorf("tilebridge: nil plan")
	}
	var steps []Step
	for i, pk := range p.Kernels {
		if pk.Row == nil && pk.Tile == nil && pk.GEMM == nil {
			continue
		}
		st, err := ToTileBridgeStep(pk, 0, false)
		if err != nil {
			return nil, fmt.Errorf("tilebridge: step %d: %w", i, err)
		}
		steps = append(steps, st)
	}
	return StepsToTileIRModules(steps)
}

// VerifyHopperDualTargetAgreement lowers step to both CUDA C++ (executed via
// [cudahostemu.Run]) and a [*tileir.Module] (verified through Format -> Parse
// -> EncodeBytecode and evaluated via [EvalTileIRReference]) on identical
// synthetic inputs, returning an error if maxAbsDiff >= 2e-2.
func VerifyHopperDualTargetAgreement(ctx context.Context, step TileIRStep, M, N, K int, eps float32) (*DualTargetReport, error) {
	tb, err := ToTileBridgeStep(step, M, true)
	if err != nil {
		return nil, err
	}
	return VerifyDualTargetAgreement(ctx, tb, M, N, K, eps)
}

// FromTileBridgeRow converts a [RowKernel] into a [hopper.RowKernel].
func FromTileBridgeRow(tb RowKernel) hopper.RowKernel {
	rk := hopper.RowKernel{Name: tb.Name, D: tb.D}
	for _, in := range tb.Inputs {
		rk.Inputs = append(rk.Inputs, hopper.RowTensor{
			Name:   in.Name,
			DType:  hopper.DType(in.DType),
			Kind:   hopper.RowKind(in.Kind),
			Offset: in.Offset,
			Buffer: in.Buffer,
		})
	}
	for _, v := range tb.Values {
		rk.Values = append(rk.Values, hopper.RowValue{
			Name:   v.Name,
			Reduce: v.Reduce,
			Body:   hopper.MustLambda(v.Body.String()),
			Args:   v.Args,
		})
	}
	for _, o := range tb.Outputs {
		rk.Outputs = append(rk.Outputs, hopper.RowTensor{
			Name:  o.Name,
			DType: hopper.DType(o.DType),
			Kind:  hopper.RowKind(o.Kind),
			Value: o.Value,
		})
	}
	return rk
}

// ToTileBridgeStep converts a hopper step into a [Step].
func ToTileBridgeStep(step TileIRStep, m int, withCUDA bool) (Step, error) {
	if step == nil {
		return Step{}, fmt.Errorf("tilebridge: nil step")
	}
	pk, err := toPlanKernel(step)
	if err != nil {
		return Step{}, err
	}
	switch {
	case pk.Row != nil:
		k := *pk.Row
		if err := k.Validate(); err != nil {
			return Step{}, err
		}
		rk := &RowKernel{Name: k.Name, D: k.D}
		for _, in := range k.Inputs {
			rk.Inputs = append(rk.Inputs, RowTensor{
				Name:   in.Name,
				DType:  DType(in.DType),
				Kind:   RowKind(in.Kind),
				Offset: in.Offset,
				Buffer: in.BufferName(),
			})
		}
		for _, v := range k.Values {
			l, err := ParseLambda(v.Body.String())
			if err != nil {
				return Step{}, err
			}
			rk.Values = append(rk.Values, RowValue{
				Name:   v.Name,
				Reduce: v.Reduce,
				Body:   l,
				Args:   v.Args,
			})
		}
		for _, o := range k.Outputs {
			rk.Outputs = append(rk.Outputs, RowTensor{
				Name:  o.Name,
				DType: DType(o.DType),
				Kind:  RowKind(o.Kind),
				Value: o.Value,
			})
		}
		if withCUDA {
			src, err := k.CUDA()
			if err != nil {
				return Step{}, err
			}
			rk.CUDASource = hopper.Prelude + src
		}
		return Step{Row: rk}, nil
	case pk.Tile != nil:
		g := *pk.Tile
		gm, err := toTileBridgeGEMM("tile_gemm", g.Name, g.N, g.K, g.GLU, g.Out, g.Epilogue, g.Extras)
		if err != nil {
			return Step{}, err
		}
		if withCUDA {
			src, err := g.CUDA()
			if err != nil {
				return Step{}, err
			}
			gm.CUDASource = hopper.Prelude + src
			gm.Grid, gm.Block = g.Grid(m, 1)
			gm.SharedBytes = g.SharedBytes()
		}
		return Step{GEMM: gm}, nil
	case pk.GEMM != nil:
		g := *pk.GEMM
		gm, err := toTileBridgeGEMM("skinny_gemm", g.Name, g.N, g.K, false, g.Out, g.Epilogue, g.Extras)
		if err != nil {
			return Step{}, err
		}
		if withCUDA {
			src, err := g.CUDA()
			if err != nil {
				return Step{}, err
			}
			grid, block, err := g.Grid()
			if err != nil {
				return Step{}, err
			}
			gm.CUDASource = hopper.Prelude + src
			gm.Grid, gm.Block = grid, block
			gm.WorkspaceBytes = g.WorkspaceBytes()
		}
		return Step{GEMM: gm}, nil
	default:
		return Step{}, fmt.Errorf("tilebridge: unsupported PlanKernel step (need Row, Tile, or GEMM)")
	}
}

func toTileBridgeGEMM(kind, name string, n, k int, glu bool, out hopper.DType, epilogue hopper.Lambda, extras []hopper.Operand) (*GEMM, error) {
	ep := Identity
	if !epilogue.IsZero() {
		var err error
		ep, err = ParseLambda(epilogue.String())
		if err != nil {
			return nil, err
		}
	}
	gm := &GEMM{
		Kind:     kind,
		Name:     name,
		N:        n,
		K:        k,
		GLU:      glu,
		Out:      DType(out),
		Epilogue: ep,
	}
	for _, ex := range extras {
		gm.Extras = append(gm.Extras, Operand{
			Name:      ex.Name,
			DType:     DType(ex.DType),
			Broadcast: Broadcast(ex.Broadcast),
		})
	}
	return gm, nil
}
