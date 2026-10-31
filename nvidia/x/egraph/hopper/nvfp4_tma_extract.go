package hopper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// ErrPreparedNVFP4TMA reports an unsupported or incomplete prepared-operation
// contract. Callers may retain their existing implementation on this error.
var ErrPreparedNVFP4TMA = errors.New("hopper: unsupported prepared NVFP4 TMA contract")

const opPreparedNVFP4TMA = "prepared_nvfp4_tma"

// NVFP4TMAEpilogue is the explicit arithmetic policy of a prepared operation.
type NVFP4TMAEpilogue string

const (
	// NVFP4TMAIdentity stores the accumulated product as BF16.
	NVFP4TMAIdentity NVFP4TMAEpilogue = "identity"
	// NVFP4TMAFastSwiGLU applies silu(gate)*up with CUDA __expf, then BF16.
	// It is deliberately not equated with the generic precise SiLU term.
	NVFP4TMAFastSwiGLU NVFP4TMAEpilogue = "swiglu-fast-exp"
)

// PreparedNVFP4TMA is an already prepared routed GEMM operation, not a raw
// GroupedMatMul. It requires BF16 activations in expert-sorted, BM-padded
// storage, two-word repacked NVFP4 weights, compact quad scales, and global
// scales already multiplied by 256. Outputs are BF16; padded rows are
// unspecified. FastScales is an input precondition, not a runtime guess.
//
// Rows is allocation capacity, not a host read of npost. Device tile values
// and their extents must pass the caller's existing runtime contract checks.
// Different BM values describe different prepared layouts and are not merged.
// Target must be sm_90a; the zero value is invalid.
type PreparedNVFP4TMA struct {
	N, K, BM, Rows, Experts int
	Target                  string
	Epilogue                NVFP4TMAEpilogue
	FastScales              bool
}

// NVFP4TMASchedule supplies one legal physical choice. Kernel N, K, BM, GLU
// and FastScales must match the operation. Grid is an explicit positive
// persistent CTA count supplied by the caller, never inferred from a device
// name or borrowed from an earlier measurement.
type NVFP4TMASchedule struct {
	Kernel NVFP4TMAGEMM
	Grid   int
}

type preparedNVFP4TMAChoice struct {
	Operation PreparedNVFP4TMA
	Schedule  NVFP4TMASchedule
}

// PreparedNVFP4TMALowering binds an extracted operation to emitted source.
// SourceSHA256 names source bytes, while OperationSHA256 names the tensor
// schema and arithmetic contract. Neither identifies input values, proves
// native qualification, or supplies an estimated execution duration.
type PreparedNVFP4TMALowering struct {
	Operation       PreparedNVFP4TMA
	Schedule        NVFP4TMASchedule
	Inputs          []Tensor
	OperationSHA256 string
	SourceSHA256    string
	Source          string
}

func preparedNVFP4TMASchema(op PreparedNVFP4TMA) ([]Tensor, error) {
	if op.Target != "sm_90a" || op.Epilogue != NVFP4TMAIdentity && op.Epilogue != NVFP4TMAFastSwiGLU {
		return nil, fmt.Errorf("%w: target or arithmetic policy", ErrPreparedNVFP4TMA)
	}
	g := NVFP4TMAGEMM{Name: "prepared_contract", N: op.N, K: op.K, BM: op.BM,
		GLU: op.Epilogue == NVFP4TMAFastSwiGLU, FastScales: op.FastScales}
	if _, err := g.norm(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrPreparedNVFP4TMA, err)
	}
	if op.BM != 64 && op.BM != 128 && op.BM != 256 || op.Rows <= 0 || op.Rows > math.MaxInt32 || op.Rows%op.BM != 0 || op.Experts <= 0 || op.Experts > math.MaxInt32 {
		return nil, fmt.Errorf("%w: padded row or expert extent", ErrPreparedNVFP4TMA)
	}
	if op.Experts > math.MaxInt/(op.N*op.K) || op.Rows > math.MaxInt/op.K || op.Rows > math.MaxInt/op.N {
		return nil, fmt.Errorf("%w: allocation extent overflows int", ErrPreparedNVFP4TMA)
	}
	tiles, columns := uint64(op.Rows/op.BM), uint64(op.N/128)
	if tiles > math.MaxInt32/columns || tiles*columns > math.MaxUint32/uint64(op.K/64) {
		return nil, fmt.Errorf("%w: tile or continuous k-step counter overflow", ErrPreparedNVFP4TMA)
	}
	return []Tensor{
		{DType: BF16, Shape: []int{op.Rows, op.K}},
		{DType: U8, Shape: []int{op.Experts, op.N / 128, op.K / 64, 4096}},
		{DType: U8, Shape: []int{op.Experts, op.N / 128, op.K / 64, 512}},
		{DType: F32, Shape: []int{op.Experts}},
		{DType: I32, Shape: []int{op.Rows / op.BM}},
		{DType: I32, Shape: []int{op.Rows / op.BM}},
		{DType: I32, Shape: []int{1}},
	}, nil
}

func checkPreparedNVFP4TMAInputs(op PreparedNVFP4TMA, inputs []Tensor) error {
	schema, err := preparedNVFP4TMASchema(op)
	if err != nil {
		return err
	}
	if len(inputs) != len(schema) {
		return fmt.Errorf("%w: seven prepared inputs required", ErrPreparedNVFP4TMA)
	}
	seen := make(map[string]bool)
	for i, input := range inputs {
		if !cIdent(input.Name) || seen[input.Name] || input.DType != schema[i].DType || !slices.Equal(input.Shape, schema[i].Shape) {
			return fmt.Errorf("%w: input %d schema or name mismatch", ErrPreparedNVFP4TMA, i)
		}
		seen[input.Name] = true
	}
	return nil
}

func checkedNVFP4TMASchedule(op PreparedNVFP4TMA, schedule NVFP4TMASchedule) (NVFP4TMASchedule, error) {
	g, err := schedule.Kernel.norm()
	if err != nil {
		return NVFP4TMASchedule{}, fmt.Errorf("%w: %v", ErrPreparedNVFP4TMA, err)
	}
	if g.N != op.N || g.K != op.K || g.BM != op.BM || g.FastScales != op.FastScales || g.GLU != (op.Epilogue == NVFP4TMAFastSwiGLU) || schedule.Grid <= 0 || schedule.Grid > math.MaxInt32 {
		return NVFP4TMASchedule{}, fmt.Errorf("%w: schedule changes the prepared operation", ErrPreparedNVFP4TMA)
	}
	schedule.Kernel = g
	return schedule, nil
}

// AddPreparedNVFP4TMAChoices adds at most 32 schedule alternatives for one
// exact prepared operation. Inputs are ordered x, words, compact scales,
// global_scales_256, eids, bvalid, npost. All choices use those same tensor
// leaves. No rewrite from raw/grouped or precise-SiLU terms is installed.
//
// The result is for ExtractTopK or Tuner.SelectRankedWithExpressionEvaluator
// with PreparedNVFP4TMAEvaluator. Existing Optimize, Compile, schedule defaults
// and generic tuner backends are unchanged; they do not support this term.
func (b *Builder) AddPreparedNVFP4TMAChoices(op PreparedNVFP4TMA, inputs []Tensor, schedules []NVFP4TMASchedule) (egraph.Id, error) {
	if b == nil || b.G == nil {
		return 0, fmt.Errorf("%w: nil builder", ErrPreparedNVFP4TMA)
	}
	if err := checkPreparedNVFP4TMAInputs(op, inputs); err != nil {
		return 0, err
	}
	if len(schedules) == 0 || len(schedules) > 32 {
		return 0, fmt.Errorf("%w: one to 32 explicit choices required", ErrPreparedNVFP4TMA)
	}
	checked := make([]NVFP4TMASchedule, len(schedules))
	for i, schedule := range schedules {
		var err error
		checked[i], err = checkedNVFP4TMASchedule(op, schedule)
		if err != nil {
			return 0, err
		}
	}
	children := make([]egraph.Id, len(inputs))
	for i, input := range inputs {
		children[i] = b.Input(input.Name, input.DType, input.Shape...)
	}
	var root egraph.Id
	for i, schedule := range checked {
		id := b.G.Add(egraph.ENode{Op: opPreparedNVFP4TMA, Data: preparedNVFP4TMAChoice{op, schedule}, Children: slices.Clone(children)})
		if i == 0 {
			root = id
		} else {
			root, _ = b.G.Union(root, id)
		}
	}
	if err := b.G.RebuildChecked(); err != nil {
		return 0, fmt.Errorf("prepared nvfp4 tma choices: %w", err)
	}
	return root, nil
}

// LowerPreparedNVFP4TMA emits one extracted prepared operation. It rejects
// unknown operations, non-leaf inputs, schema drift, changed arithmetic and
// changed padding. It does not compile CUDA, resolve driver symbols, create
// tensor maps or execute a device operation.
func LowerPreparedNVFP4TMA(expr *egraph.Expr) (PreparedNVFP4TMALowering, error) {
	if expr == nil || expr.Op != opPreparedNVFP4TMA {
		return PreparedNVFP4TMALowering{}, fmt.Errorf("%w: operation is not an explicit prepared term", ErrPreparedNVFP4TMA)
	}
	choice, ok := expr.Data.(preparedNVFP4TMAChoice)
	if !ok || len(expr.Children) != 7 {
		return PreparedNVFP4TMALowering{}, fmt.Errorf("%w: missing typed choice or inputs", ErrPreparedNVFP4TMA)
	}
	inputs := make([]Tensor, 7)
	for i, child := range expr.Children {
		if child == nil || len(child.Children) != 0 {
			return PreparedNVFP4TMALowering{}, fmt.Errorf("%w: prepared inputs must be leaves", ErrPreparedNVFP4TMA)
		}
		input, ok := child.Data.(Tensor)
		if !ok || child.Op != "t:"+input.Name {
			return PreparedNVFP4TMALowering{}, fmt.Errorf("%w: input is not a tensor leaf", ErrPreparedNVFP4TMA)
		}
		inputs[i] = Tensor{Name: input.Name, DType: input.DType, Shape: slices.Clone(input.Shape)}
	}
	if err := checkPreparedNVFP4TMAInputs(choice.Operation, inputs); err != nil {
		return PreparedNVFP4TMALowering{}, err
	}
	schedule, err := checkedNVFP4TMASchedule(choice.Operation, choice.Schedule)
	if err != nil {
		return PreparedNVFP4TMALowering{}, err
	}
	source, err := schedule.Kernel.Source()
	if err != nil {
		return PreparedNVFP4TMALowering{}, err
	}
	identity, err := json.Marshal(struct {
		Operation PreparedNVFP4TMA
		Inputs    []Tensor
	}{choice.Operation, inputs})
	if err != nil {
		return PreparedNVFP4TMALowering{}, fmt.Errorf("prepared nvfp4 tma identity: %w", err)
	}
	opSum, sourceSum := sha256.Sum256(identity), sha256.Sum256([]byte(source))
	return PreparedNVFP4TMALowering{Operation: choice.Operation, Schedule: schedule, Inputs: inputs,
		OperationSHA256: hex.EncodeToString(opSum[:]), SourceSHA256: hex.EncodeToString(sourceSum[:]), Source: source}, nil
}

// PreparedNVFP4TMAEvaluator adapts a checked workload callback to the existing
// expression-choice tuner. run must perform its actual compilation, setup,
// numerical/graph/health gates and timing for the exact supplied lowering.
// A nonpositive duration is rejected. No catalogue duration or heuristic cost
// is supplied by this adapter. workloadID passed to the tuner must bind the
// actual inputs, weights, device, code and checks; OperationSHA256 alone is
// only a schema identity and is insufficient.
//
// Return a context cancellation/deadline error on terminal device health loss
// so the existing tuner stops rather than tries another candidate. Tensor-map
// setup and configure must succeed before graph capture; all launch status
// returns must be checked. Driver/context ownership remains with the caller.
func PreparedNVFP4TMAEvaluator(run func(context.Context, PreparedNVFP4TMALowering) (time.Duration, error)) egraph.ExpressionWorkloadEvaluator {
	return func(ctx context.Context, expr *egraph.Expr) (egraph.TuningCandidate, time.Duration, error) {
		if ctx == nil || run == nil {
			return egraph.TuningCandidate{}, 0, fmt.Errorf("prepared nvfp4 tma evaluator: context and workload callback required")
		}
		if err := ctx.Err(); err != nil {
			return egraph.TuningCandidate{}, 0, err
		}
		lowering, err := LowerPreparedNVFP4TMA(expr)
		if err != nil {
			return egraph.TuningCandidate{}, 0, err
		}
		duration, err := run(ctx, lowering)
		if err != nil {
			return egraph.TuningCandidate{}, 0, err
		}
		if err := ctx.Err(); err != nil {
			return egraph.TuningCandidate{}, 0, err
		}
		if duration <= 0 {
			return egraph.TuningCandidate{}, 0, fmt.Errorf("prepared nvfp4 tma evaluator: non-positive measured duration")
		}
		candidate := egraph.TuningCandidate{Expr: expr, BlockDim: 384, GridDim: lowering.Schedule.Grid,
			SharedMem: lowering.Schedule.Kernel.SharedBytes(), KernelName: lowering.Schedule.Kernel.Name, CUDASource: lowering.Source}
		return candidate, duration, nil
	}
}
