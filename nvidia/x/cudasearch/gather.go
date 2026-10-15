package cudasearch

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// GatherCompiler binds scalar, tile16 and tile32 gathers to immutable expert
// rows and a route distribution. Its declared output policy is exact copied
// bytes, including signed zeros and NaN payloads; the evaluator must check it
// on the actual device before timing. Routes are not sorted or rewritten.
type GatherCompiler struct {
	dev                *cuda.Device
	target             string
	width, expertCount int
	experts, routes    []byte
	reference          []float32
	bindings           WorkloadBindings
	candidates         []cudamoe.GatherCandidate
	checkLease         func() bool
	preparation        []PreparationPhase
}

// NewGatherCompiler validates and copies an expert-major float32 matrix and
// int32 routes, without calling CUDA. width is the number of floats per expert;
// expertCount is the number of rows. Every route must name an existing expert.
// checkLease must verify a live hardware lease. Keep dev open until all returned
// executables close. Input encoding is outside measured kernel execution.
func NewGatherCompiler(dev *cuda.Device, target string, width, expertCount int, experts []float32, routes []int32, checkLease func() bool) (*GatherCompiler, error) {
	if dev == nil || checkLease == nil {
		return nil, fmt.Errorf("cudasearch: gather requires device and lease checker")
	}
	major, minor, suffix, err := cudakernel.ParseArch(target)
	if err != nil || target != fmt.Sprintf("sm_%d%d%s", major, minor, suffix) || (suffix != "" && suffix != "a" && suffix != "f") {
		return nil, fmt.Errorf("cudasearch: invalid gather physical target %q", target)
	}
	candidates, err := cudamoe.GatherCandidates(len(routes), width, expertCount)
	if err != nil {
		return nil, err
	}
	size, err := candidates[0].Layouts[0].FootprintBytes()
	if err != nil {
		return nil, err
	}
	if len(experts) != size/4 {
		return nil, fmt.Errorf("cudasearch: gather expert extent mismatch")
	}
	for _, r := range routes {
		if r < 0 || int64(r) >= int64(expertCount) {
			return nil, fmt.Errorf("cudasearch: gather route %d out of range", r)
		}
	}
	start := time.Now()
	c := &GatherCompiler{dev: dev, target: target, width: width, expertCount: expertCount, experts: floatBytes(experts), routes: make([]byte, len(routes)*4), candidates: candidates, checkLease: checkLease}
	for i, r := range routes {
		binary.LittleEndian.PutUint32(c.routes[i*4:], uint32(r))
	}
	c.preparation = []PreparationPhase{{Name: "input_encode", WallNs: time.Since(start).Nanoseconds(), InputBytes: int64(len(c.experts) + len(c.routes)), OutputBytes: int64(len(c.experts) + len(c.routes)), Complete: true}, {Name: "external_expert_prepack", WallNs: cudatrial.Unmeasured}, {Name: "external_routing", WallNs: cudatrial.Unmeasured}}
	outputBytes, err := candidates[0].Layouts[1].FootprintBytes()
	if err != nil {
		return nil, err
	}
	output := make([]byte, outputBytes)
	for i, r := range routes {
		copy(output[i*width*4:(i+1)*width*4], c.experts[int(r)*width*4:(int(r)+1)*width*4])
	}
	c.reference = byteValues(output)
	shape, err := cudashape.New(len(routes), width)
	if err != nil {
		return nil, err
	}
	c.bindings = WorkloadBindings{Name: fmt.Sprintf("gather/%s/tokens%d/width%d/experts%d", target, len(routes), width, expertCount), Shape: shape, DType: cudair.Float32, Layout: "row_major", InputNames: []string{"experts", "routes"}, InputIdentities: map[string]string{"routes": fmt.Sprintf("%x", sha256.Sum256(c.routes))}, WeightIdentities: map[string]string{"experts": fmt.Sprintf("%x", sha256.Sum256(c.experts))}, NumericalPolicy: NumericalPolicyExact, Objective: ObjectiveMinLatency}
	return c, nil
}

// Bindings returns detached identities for Compile and the evaluator.
func (c *GatherCompiler) Bindings() WorkloadBindings {
	b := c.bindings
	b.InputNames = slices.Clone(b.InputNames)
	b.InputIdentities = maps.Clone(b.InputIdentities)
	b.WeightIdentities = maps.Clone(b.WeightIdentities)
	return b
}

// References returns each output byte as an exactly representable float32
// integer under the key gather_bytes, for the evaluator's exact policy.
func (c *GatherCompiler) References() map[string][]float32 {
	return map[string][]float32{"gather_bytes": slices.Clone(c.reference)}
}

// Preparation reports host encoding separately from unknown upstream expert
// packing and route selection. Returned metadata is detached from the compiler.
func (c *GatherCompiler) Preparation() []PreparationPhase { return slices.Clone(c.preparation) }

func (c *GatherCompiler) admit(expr *egraph.Expr, b WorkloadBindings) (cudamoe.GatherCandidate, error) {
	if b.Key() != c.bindings.Key() {
		return cudamoe.GatherCandidate{}, fmt.Errorf("%w: gather workload differs", ErrMissingBinding)
	}
	if expr != nil && expr.Data == nil && len(expr.Children) == 0 {
		for _, candidate := range c.candidates {
			if expr.Op == candidate.Name {
				return candidate, nil
			}
		}
	}
	return cudamoe.GatherCandidate{}, fmt.Errorf("cudasearch: unsupported gather expression")
}

// Compile accepts legal childless gather_scalar, gather16 or gather32
// expressions with nil Data. It compiles real artifacts, loads them and uploads
// private buffers, requiring hardware authority. A nonnil result owns resources
// even on error and must be closed. Run reports GPU-busy event time for the
// kernel, excluding input encoding, transfers and output verification.
func (c *GatherCompiler) Compile(ctx context.Context, expr *egraph.Expr, b WorkloadBindings) (Executable, error) {
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
	am, an, _, _ := cudakernel.ParseArch(c.target)
	if major != am || minor != an {
		return nil, fmt.Errorf("cudasearch: gather device target mismatch")
	}
	var artifact *cudakernel.Artifact
	if candidate.Tile == 0 {
		artifact, err = cudamoe.GatherScalarArtifact(c.target)
	} else {
		artifact, err = cudamoe.GatherArtifact(candidate.Tile, c.target)
	}
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !c.checkLease() {
		return nil, cudatrial.ErrHardwareLockRequired
	}
	x := &sequenceExecutable{memory: searchMemoryBudget(ctx), dev: c.dev, checkLease: c.checkLease, closed: true, identity: c.bindings.Key() + "/" + candidate.Name,
		buffers: make(map[string]cudamem.Slice[byte]), poison: []string{"output"}, outputs: []sequenceOutput{{buffer: "output", name: "gather_bytes", bytes: true}}, launches: []sequenceLaunch{{grid: candidate.Grid, block: candidate.Block}}}
	x.preparation = c.Preparation()
	if err := x.load([]*cudakernel.Artifact{artifact}); err != nil {
		return x, err
	}
	for i, name := range []string{"experts", "output", "routes"} {
		if err := x.addBuffer(name, candidate.Layouts[i]); err != nil {
			return x, err
		}
	}
	if err := x.upload("experts", c.experts); err != nil {
		return x, err
	}
	if err := x.upload("routes", c.routes); err != nil {
		return x, err
	}
	args := []any{x.buffers["experts"], x.buffers["output"], x.buffers["routes"], int32(len(c.routes) / 4), int32(c.width), int32(c.expertCount)}
	if _, err := artifact.Schema.Pack(args...); err != nil {
		return x, err
	}
	x.stages[0].args = args
	x.closed = false
	return x, nil
}
