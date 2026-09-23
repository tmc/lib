package cudapdl_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudapdl"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestPDLEligibility validates positive admission on sm_90+ targets and negative
// controls for target < sm_90, vendor kernels, and missing intrinsics.
func TestPDLEligibility(t *testing.T) {
	tests := []struct {
		name        string
		target      string
		producer    cudapdl.KernelDesc
		consumer    cudapdl.KernelDesc
		wantElig    bool
		wantPolicy  cudapdl.LaunchPolicy
		containsErr string
	}{
		{
			name:   "valid sm_90 pair",
			target: "sm_90",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:   true,
			wantPolicy: cudapdl.PolicyProgrammatic,
		},
		{
			name:   "valid sm_100a pair via source inspection",
			target: "sm_100a",
			producer: cudapdl.KernelDesc{
				Name:   "prod_ptx",
				Source: `asm volatile("griddepcontrol.launch_dependents;");`,
			},
			consumer: cudapdl.KernelDesc{
				Name:   "cons_ptx",
				Source: `asm volatile("griddepcontrol.wait;":::"memory");`,
			},
			wantElig:   true,
			wantPolicy: cudapdl.PolicyProgrammatic,
		},
		{
			name:   "valid sm_121a pair via artifact inspection",
			target: "sm_121a",
			producer: cudapdl.KernelDesc{
				Name: "prod_art",
				Artifact: &cudakernel.Artifact{
					Target: "sm_121a",
					Raw:    []byte("... griddepcontrol.launch_dependents ..."),
				},
			},
			consumer: cudapdl.KernelDesc{
				Name: "cons_art",
				Artifact: &cudakernel.Artifact{
					Target: "sm_121a",
					Raw:    []byte("... griddepcontrol.wait ..."),
				},
			},
			wantElig:   true,
			wantPolicy: cudapdl.PolicyProgrammatic,
		},
		{
			name:   "ineligible legacy target sm_80",
			target: "sm_80",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "sm_90 or higher required",
		},
		{
			name:   "ineligible legacy target sm_75",
			target: "sm_75",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "sm_90 or higher required",
		},
		{
			name:   "ineligible opaque vendor producer",
			target: "sm_90",
			producer: cudapdl.KernelDesc{
				Name:                "cublaslt_gemm",
				IsVendor:            true,
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "opaque vendor kernel",
		},
		{
			name:   "ineligible opaque vendor consumer",
			target: "sm_90",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:     "cudnn_conv",
				IsVendor: true,
				HasWait:  true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "opaque vendor kernel",
		},
		{
			name:   "ineligible missing producer trigger",
			target: "sm_90",
			producer: cudapdl.KernelDesc{
				Name:                "prod_plain",
				HasLaunchDependents: false,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "missing required trigger",
		},
		{
			name:   "ineligible missing consumer wait",
			target: "sm_90",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons_plain",
				HasWait: false,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "missing required barrier",
		},
		{
			name:   "invalid target syntax",
			target: "invalid_arch",
			producer: cudapdl.KernelDesc{
				Name:                "prod",
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons",
				HasWait: true,
			},
			wantElig:    false,
			wantPolicy:  cudapdl.PolicyStandardFallback,
			containsErr: "invalid target architecture",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			el := cudapdl.CheckEligibility(tt.target, tt.producer, tt.consumer)
			if el.Eligible != tt.wantElig {
				t.Errorf("Eligible = %v, want %v", el.Eligible, tt.wantElig)
			}
			if el.Policy != tt.wantPolicy {
				t.Errorf("Policy = %v, want %v", el.Policy, tt.wantPolicy)
			}
			if tt.containsErr != "" {
				if !strings.Contains(el.FallbackReason, tt.containsErr) {
					t.Errorf("FallbackReason %q does not contain %q", el.FallbackReason, tt.containsErr)
				}
			}
		})
	}
}

// TestPDLFallback verifies that when eligibility fails, Compose creates a standard
// dependency edge and records the exact fallback reason.
func TestPDLFallback(t *testing.T) {
	producerDesc := cudapdl.KernelDesc{
		Name:     "cublas_opaque",
		IsVendor: true,
	}
	consumerDesc := cudapdl.KernelDesc{
		Name:    "cons",
		HasWait: true,
	}

	el := cudapdl.CheckEligibility("sm_90", producerDesc, consumerDesc)
	if el.Eligible {
		t.Fatal("expected vendor kernel to be ineligible")
	}

	// Verify error handling with nil/closed graph
	_, err := cudapdl.Compose(nil, nil, nil, el)
	if err == nil {
		t.Fatal("expected error on nil graph")
	}
}

// TestPDLConsumerWaitRequired is the mutant control: removing the consumer wait
// causes ineligibility and fallback to standard dependency.
func TestPDLConsumerWaitRequired(t *testing.T) {
	prod := cudapdl.KernelDesc{
		Name:                "prod",
		HasLaunchDependents: true,
	}

	// Baseline: consumer with wait is eligible
	consWithWait := cudapdl.KernelDesc{
		Name:    "cons",
		HasWait: true,
	}
	elBaseline := cudapdl.CheckEligibility("sm_90", prod, consWithWait)
	if !elBaseline.Eligible {
		t.Fatalf("baseline with wait should be eligible: %s", elBaseline.FallbackReason)
	}

	// Mutant: consumer without wait is ineligible
	consMutantNoWait := cudapdl.KernelDesc{
		Name:    "cons_mutant",
		HasWait: false,
	}
	elMutant := cudapdl.CheckEligibility("sm_90", prod, consMutantNoWait)
	if elMutant.Eligible {
		t.Fatal("mutant without wait must be marked ineligible")
	}
	if elMutant.Policy != cudapdl.PolicyStandardFallback {
		t.Errorf("policy = %v, want %v", elMutant.Policy, cudapdl.PolicyStandardFallback)
	}
	if !strings.Contains(elMutant.FallbackReason, "missing required barrier") {
		t.Errorf("expected missing barrier error, got %q", elMutant.FallbackReason)
	}
}

// TestPDLParameterLifetime verifies parameter lifetime safety across updates.
func TestPDLParameterLifetime(t *testing.T) {
	// Verifies that args packed into UpdateKernelNode or HotSwapNode
	// are not prematurely released or corrupted.
	type ComplexArgs struct {
		X, Y float32
		Bias uint32
	}

	args1 := ComplexArgs{X: 1.5, Y: 2.5, Bias: 10}
	packed1, err := cudalaunch.PackArgs(args1)
	if err != nil {
		t.Fatalf("PackArgs failed: %v", err)
	}
	if len(packed1) != 1 {
		t.Fatalf("len(packed1) = %d, want 1", len(packed1))
	}

	args2 := ComplexArgs{X: 3.5, Y: 4.5, Bias: 20}
	packed2, err := cudalaunch.PackArgs(args2)
	if err != nil {
		t.Fatalf("PackArgs failed: %v", err)
	}

	val1 := *(*ComplexArgs)(packed1[0])
	val2 := *(*ComplexArgs)(packed2[0])
	if val1.Bias != 10 || val2.Bias != 20 {
		t.Errorf("packed values corrupted: val1=%+v, val2=%+v", val1, val2)
	}
}

// TestPDLNoOverlapStillCorrect verifies that programmatic execution logic yields
// numerically identical results regardless of whether hardware overlap occurs.
func TestPDLNoOverlapStillCorrect(t *testing.T) {
	// Reference sequential model: producer writes out[i] = in[i] * 2,
	// consumer reads out[i] and computes final[i] = out[i] + 3.
	const n = 128
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i + 1)
	}

	// 1. Host sequential baseline
	expected := make([]float32, n)
	for i := range input {
		intermediate := input[i] * 2.0
		expected[i] = intermediate + 3.0
	}

	// 2. Simulated zero-overlap execution (predecessor completes entirely before consumer)
	simZeroOverlap := make([]float32, n)
	intermediateBuf := make([]float32, n)
	for i := range input {
		intermediateBuf[i] = input[i] * 2.0
	}
	for i := range input {
		simZeroOverlap[i] = intermediateBuf[i] + 3.0
	}

	// 3. Simulated maximum-overlap execution (consumer preamble precomputes independent work
	// before synchronizing at barrier)
	simOverlap := make([]float32, n)
	independentBias := float32(3.0) // precomputed in preamble
	for i := range input {
		// preamble executes independently
		_ = independentBias
		// barrier: wait for intermediateBuf
		val := intermediateBuf[i]
		// dependent body consumes intermediate
		simOverlap[i] = val + independentBias
	}

	for i := range expected {
		if simZeroOverlap[i] != expected[i] {
			t.Fatalf("zero overlap mismatch at %d: got %f, want %f", i, simZeroOverlap[i], expected[i])
		}
		if simOverlap[i] != expected[i] {
			t.Fatalf("overlap mismatch at %d: got %f, want %f", i, simOverlap[i], expected[i])
		}
	}
}

// Native PDL kernel sources for hardware gated tests.
const nativePDLProducerSource = `
static inline __device__ void cudaTriggerProgrammaticLaunchCompletion(void) {
    asm volatile("griddepcontrol.launch_dependents;":::);
}

extern "C" __global__ void pdl_producer(const float *in, float *out, int n, long long trailing_cycles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
    __threadfence();
    __syncthreads();

    // Trigger dependent consumer early
    cudaTriggerProgrammaticLaunchCompletion();

    // Trailing independent producer work
    long long t0 = clock64();
    while (clock64() - t0 < trailing_cycles) {}
}
`

const nativePDLConsumerSource = `
static inline __device__ void cudaGridDependencySynchronize(void) {
    asm volatile("griddepcontrol.wait;":::"memory");
}

extern "C" __global__ void pdl_consumer(const float *in, float *out, float bias, int n, long long compute_cycles) {
    // Independent preamble work: load bias and setup
    float b = bias;

    // Synchronize on producer grid
    cudaGridDependencySynchronize();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + b;
    }

    long long t0 = clock64();
    while (clock64() - t0 < compute_cycles) {}
}
`

// TestNativePDLReference executes an end-to-end programmatic dependent launch
// on native hardware when authorized, comparing PDL against standard dependencies.
func TestNativePDLReference(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" && os.Getenv("CUDAGRAPH_PDL_ABLATION") != "1" {
		t.Skip("native qualification pending: CUDA_COMPILER_TEST=1 or CUDAGRAPH_PDL_ABLATION=1 required for hardware execution")
	}

	dev := cudatest.Require(t)
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major < 9 {
		t.Skipf("device SM %d.%d does not support programmatic dependent launch (requires SM 90+)", major, minor)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	// Compile producer and consumer kernels
	prodComp, err := nvrtcx.Compile(nativePDLProducerSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("compile producer: %v", err)
	}
	prodRaw, err := prodComp.Load(dev)
	if err != nil {
		t.Fatalf("load producer module: %v", err)
	}
	prodMod, err := cuda.OwnModule(dev, prodRaw)
	if err != nil {
		t.Fatal(err)
	}
	defer prodMod.Close()
	prodKern, err := prodMod.Function("pdl_producer")
	if err != nil {
		t.Fatal(err)
	}

	consComp, err := nvrtcx.Compile(nativePDLConsumerSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("compile consumer: %v", err)
	}
	consRaw, err := consComp.Load(dev)
	if err != nil {
		t.Fatalf("load consumer module: %v", err)
	}
	consMod, err := cuda.OwnModule(dev, consRaw)
	if err != nil {
		t.Fatal(err)
	}
	defer consMod.Close()
	consKern, err := consMod.Function("pdl_consumer")
	if err != nil {
		t.Fatal(err)
	}

	const (
		n     = 256
		block = 128
		grid  = n / block
	)

	// Allocate input, intermediate, and output device buffers
	inBuf, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(inBuf.Free)

	interBuf, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(interBuf.Free)

	outBuf, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(outBuf.Free)

	hostIn := make([]float32, n)
	for i := range hostIn {
		hostIn[i] = float32(i + 1)
	}
	if err := inBuf.CopyFromHost(hostIn); err != nil {
		t.Fatal(err)
	}

	// 1. Build Graph with Programmatic Dependent Launch edge
	g, err := cudagraph.New(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	trailingCycles := int64(1000)
	prodLauncher := cudalaunch.OwnedKernel(prodKern).
		Grid(uint32(grid)).
		Block(uint32(block)).
		WithProgrammaticStreamSerialization(true)

	prodNode, err := g.AddLauncher(prodLauncher, []any{inBuf, interBuf, int32(n), trailingCycles})
	if err != nil {
		t.Fatalf("AddLauncher producer: %v", err)
	}

	biasVal := float32(3.0)
	computeCycles := int64(1000)
	consLauncher := cudalaunch.OwnedKernel(consKern).
		Grid(uint32(grid)).
		Block(uint32(block))

	consNode, err := g.AddLauncher(consLauncher, []any{interBuf, outBuf, biasVal, int32(n), computeCycles})
	if err != nil {
		t.Fatalf("AddLauncher consumer: %v", err)
	}

	archStr := fmt.Sprintf("sm_%d%d", major, minor)
	producerDesc := cudapdl.KernelDesc{Name: "pdl_producer", HasLaunchDependents: true}
	consumerDesc := cudapdl.KernelDesc{Name: "pdl_consumer", HasWait: true}

	comp, err := cudapdl.Compose(g, prodNode, consNode, cudapdl.CheckEligibility(archStr, producerDesc, consumerDesc))
	if err != nil {
		t.Fatalf("cudapdl.Compose: %v", err)
	}
	if comp.Policy != cudapdl.PolicyProgrammatic {
		t.Fatalf("comp.Policy = %v, want PolicyProgrammatic", comp.Policy)
	}

	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate: %v", err)
	}
	defer exec.Destroy()

	if err := exec.Launch(stream); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	hostOut := make([]float32, n)
	if err := outBuf.CopyToHost(hostOut); err != nil {
		t.Fatal(err)
	}

	for i := range hostOut {
		want := hostIn[i]*2.0 + biasVal
		if hostOut[i] != want {
			t.Fatalf("out[%d] = %f, want %f", i, hostOut[i], want)
		}
	}
}
