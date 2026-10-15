package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"maps"
	"math"
	"slices"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaquant"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// NormQuantWorkload specifies the row-major RMSNorm-to-Q8_1 computation.
// MaterializeNorm requests the normalized float32 values for another consumer.
type NormQuantWorkload struct {
	Rows, Cols                int
	Input, Weight             []float32
	Epsilon                   float32
	Weighted, MaterializeNorm bool
}

// NormQuantCompiler binds legal fused/unfused alternatives to private inputs.
// Its declared policy is exact packed Q8_1 bytes and, when requested, exact
// normalized float32 values. It does not claim every CUDA target satisfies
// that policy: the evaluator must check outputs before timing either arm.
type NormQuantCompiler struct {
	preparation                 []PreparationPhase
	encodedInput, encodedWeight []byte
	dev                         *cuda.Device
	target                      string
	w                           NormQuantWorkload
	bindings                    WorkloadBindings
	references                  map[string][]float32
	candidates                  []cudaquant.NormQuantCandidate
	checkLease                  func() bool
}

// NewNormQuantCompiler copies and validates the workload without calling CUDA.
// checkLease must verify a live hardware lease. Keep dev open until all returned
// executables close. Epsilon, weighting, output consumers, input contents and
// target are part of the workload identity.
func NewNormQuantCompiler(dev *cuda.Device, target string, w NormQuantWorkload, checkLease func() bool) (*NormQuantCompiler, error) {
	if dev == nil || checkLease == nil {
		return nil, fmt.Errorf("cudasearch: norm quant requires device and lease checker")
	}
	major, minor, suffix, err := cudakernel.ParseArch(target)
	if err != nil || target != fmt.Sprintf("sm_%d%d%s", major, minor, suffix) || (suffix != "" && suffix != "a" && suffix != "f") {
		return nil, fmt.Errorf("cudasearch: invalid norm quant physical target %q", target)
	}
	candidates, err := cudaquant.RMSNormQ8_1Candidates(w.Rows, w.Cols, w.Weighted, w.MaterializeNorm)
	if err != nil {
		return nil, err
	}
	blocks, err := cudaquant.RMSNormQ8_1(w.Input, w.Weight, w.Rows, w.Cols, w.Epsilon, w.Weighted)
	if err != nil {
		return nil, err
	}
	start := time.Now()
	w.Input = slices.Clone(w.Input)
	if w.Weighted {
		w.Weight = slices.Clone(w.Weight)
	} else {
		w.Weight = nil
	}
	encodedInput, encodedWeight := floatBytes(w.Input), floatBytes(w.Weight)
	encoding := PreparationPhase{Name: "input_copy_and_encode", WallNs: time.Since(start).Nanoseconds(), InputBytes: int64(len(encodedInput) + len(encodedWeight)), OutputBytes: int64(len(encodedInput) + len(encodedWeight)), Complete: true}
	shape, err := cudashape.New(w.Rows, w.Cols)
	if err != nil {
		return nil, err
	}
	c := &NormQuantCompiler{dev: dev, target: target, w: w, candidates: candidates, checkLease: checkLease}
	c.encodedInput, c.encodedWeight = encodedInput, encodedWeight
	c.preparation = []PreparationPhase{encoding, {Name: "external_prepack", WallNs: cudatrial.Unmeasured}}
	c.bindings = WorkloadBindings{
		Name:  fmt.Sprintf("norm-q8/%s/%d/%d/eps%08x/weighted%t/norm%t", target, w.Rows, w.Cols, math.Float32bits(w.Epsilon), w.Weighted, w.MaterializeNorm),
		Shape: shape, DType: cudair.Float32, Layout: "row_major", InputNames: []string{"input"},
		InputIdentities: map[string]string{"input": floatDigest(w.Input)}, NumericalPolicy: NumericalPolicyExact, Objective: ObjectiveMinLatency,
	}
	if w.Weighted {
		c.bindings.InputNames = append(c.bindings.InputNames, "weight")
		c.bindings.WeightIdentities = map[string]string{"weight": floatDigest(w.Weight)}
	}
	packed := make([]byte, len(blocks)*36)
	for i, b := range blocks {
		binary.LittleEndian.PutUint16(packed[i*36:], b.D)
		binary.LittleEndian.PutUint16(packed[i*36+2:], b.S)
		for j, q := range b.Quants {
			packed[i*36+4+j] = byte(q)
		}
	}
	c.references = map[string][]float32{"q8_bytes": byteValues(packed)}
	if w.MaterializeNorm {
		norm := make([]float32, len(w.Input))
		for r := 0; r < w.Rows; r++ {
			var ss float64
			for _, v := range w.Input[r*w.Cols : (r+1)*w.Cols] {
				ss += float64(v) * float64(v)
			}
			inv := float32(1 / math.Sqrt(ss/float64(w.Cols)+float64(w.Epsilon)))
			for col := 0; col < w.Cols; col++ {
				v := w.Input[r*w.Cols+col] * inv
				if w.Weighted {
					v *= w.Weight[col]
				}
				norm[r*w.Cols+col] = v
			}
		}
		c.references["norm"] = norm
	}
	return c, nil
}

func floatBytes(values []float32) []byte {
	b := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}
func floatDigest(values []float32) string {
	return fmt.Sprintf("%x", sha256.Sum256(floatBytes(values)))
}
func byteValues(b []byte) []float32 {
	v := make([]float32, len(b))
	for i, b := range b {
		v[i] = float32(b)
	}
	return v
}

// Bindings returns detached workload identities for Compile and the evaluator.
func (c *NormQuantCompiler) Bindings() WorkloadBindings {
	b := c.bindings
	b.InputNames = slices.Clone(b.InputNames)
	b.InputIdentities = maps.Clone(b.InputIdentities)
	b.WeightIdentities = maps.Clone(b.WeightIdentities)
	return b
}

// References returns detached exact-policy outputs. q8_bytes encodes each
// packed byte as an exactly representable float32 integer for the evaluator.
func (c *NormQuantCompiler) References() map[string][]float32 {
	out := make(map[string][]float32, len(c.references))
	for name, values := range c.references {
		out[name] = slices.Clone(values)
	}
	return out
}

// Preparation returns detached host copy/encoding measurements. Upstream
// preparation is unknown; normalization and quantization belong to device time.
func (c *NormQuantCompiler) Preparation() []PreparationPhase { return slices.Clone(c.preparation) }

func (c *NormQuantCompiler) admit(expr *egraph.Expr, b WorkloadBindings) (cudaquant.NormQuantCandidate, error) {
	if b.Key() != c.bindings.Key() {
		return cudaquant.NormQuantCandidate{}, fmt.Errorf("%w: norm quant workload differs", ErrMissingBinding)
	}
	if expr != nil && expr.Data == nil && len(expr.Children) == 0 {
		for _, candidate := range c.candidates {
			if expr.Op == candidate.Name {
				return candidate, nil
			}
		}
	}
	return cudaquant.NormQuantCandidate{}, fmt.Errorf("cudasearch: unsupported norm quant expression or output consumers")
}

// Compile accepts legal childless norm_quant_fused or norm_quant_unfused
// expressions with nil Data. It compiles and loads real artifacts and uploads
// inputs, requiring hardware authority. A nonnil result must be closed even on
// error. Device-event timing covers the complete kernel sequence (GPU busy),
// excluding allocation, input conversion, transfer and output verification.
func (c *NormQuantCompiler) Compile(ctx context.Context, expr *egraph.Expr, b WorkloadBindings) (Executable, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	candidate, err := c.admit(expr, b)
	if err != nil {
		return nil, err
	}
	if !c.checkLease() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	major, minor, err := c.dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	archMajor, archMinor, _, _ := cudakernel.ParseArch(c.target)
	if major != archMajor || minor != archMinor {
		return nil, fmt.Errorf("cudasearch: norm quant device target mismatch")
	}
	var artifacts []*cudakernel.Artifact
	if candidate.Name == "norm_quant_fused" {
		a, err := cudaquant.RMSNormQ8_1Artifact(c.target)
		if err != nil {
			return nil, err
		}
		artifacts = []*cudakernel.Artifact{a}
	} else {
		a, err := cudaquant.RMSNormQ8_1UnfusedArtifacts(c.target)
		if err != nil {
			return nil, err
		}
		artifacts = a[:]
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !c.checkLease() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	x := &sequenceExecutable{memory: searchMemoryBudget(ctx), dev: c.dev, checkLease: c.checkLease, closed: true, buffers: make(map[string]cudamem.Slice[byte]), poison: []string{"norm", "output"}, outputs: []sequenceOutput{{buffer: "output", name: "q8_bytes", bytes: true}}}
	x.preparation = c.Preparation()
	if c.w.MaterializeNorm {
		x.outputs = append(x.outputs, sequenceOutput{buffer: "norm", name: "norm"})
	}
	for _, l := range candidate.Launches {
		x.launches = append(x.launches, sequenceLaunch{grid: l.Grid, block: l.Block})
	}
	x.identity = c.bindings.Key() + "/" + candidate.Name
	if err = x.load(artifacts); err != nil {
		return x, err
	}
	for _, name := range []string{"input", "weight", "norm", "output"} {
		layout, ok := candidate.Buffers[name]
		if !ok {
			continue
		}
		if err = x.addBuffer(name, layout); err != nil {
			return x, err
		}
	}
	if err = x.upload("input", c.encodedInput); err != nil {
		return x, err
	}
	if c.w.Weighted {
		if err = x.upload("weight", c.encodedWeight); err != nil {
			return x, err
		}
	}
	weighted := uint32(0)
	if c.w.Weighted {
		weighted = 1
	}
	for i, launch := range candidate.Launches {
		args := []any{x.buffers[launch.Input], x.buffers["weight"].Ptr(), x.buffers[launch.Output], uint32(c.w.Rows), uint32(c.w.Cols), c.w.Epsilon, weighted}
		if _, err = artifacts[i].Schema.Pack(args...); err != nil {
			return x, err
		}
		x.stages[i].args = args
	}
	x.closed = false
	return x, nil
}
