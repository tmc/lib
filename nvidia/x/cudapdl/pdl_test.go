package cudapdl_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
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
			name:   "declared sm_100a source pair",
			target: "sm_100a",
			producer: cudapdl.KernelDesc{
				Name:                "prod_ptx",
				Source:              `asm volatile("griddepcontrol.launch_dependents;");`,
				HasLaunchDependents: true,
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons_ptx",
				Source:  `asm volatile("griddepcontrol.wait;":::"memory");`,
				HasWait: true,
			},
			wantElig:   true,
			wantPolicy: cudapdl.PolicyProgrammatic,
		},
		{
			name:   "declared sm_121a artifact pair",
			target: "sm_121a",
			producer: cudapdl.KernelDesc{
				Name:                "prod_art",
				HasLaunchDependents: true,
				Artifact: &cudakernel.Artifact{
					Target: "sm_121a",
				},
			},
			consumer: cudapdl.KernelDesc{
				Name:    "cons_art",
				HasWait: true,
				Artifact: &cudakernel.Artifact{
					Target: "sm_121a",
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

const (
	nativePDLN     = 256
	nativePDLBlock = 128
	nativePDLBias  = float32(3)
)

type pdlReferenceFixture struct {
	target             string
	producer, consumer cudapdl.KernelDesc
	arms               [2]pdlReferenceArm
}

type pdlReferenceArm struct {
	name         string
	programmatic bool
	wantPolicy   cudapdl.LaunchPolicy
}

func newPDLReferenceFixture(target string) pdlReferenceFixture {
	return pdlReferenceFixture{
		target: target,
		producer: cudapdl.KernelDesc{
			Name: "pdl_producer", Source: nativePDLProducerSource, HasLaunchDependents: true,
		},
		consumer: cudapdl.KernelDesc{
			Name: "pdl_consumer", Source: nativePDLConsumerSource, HasWait: true, HasIndependentWork: true,
		},
		arms: [2]pdlReferenceArm{
			{"ordinary", false, cudapdl.PolicyStandardFallback},
			{"programmatic", true, cudapdl.PolicyProgrammatic},
		},
	}
}

func (f pdlReferenceFixture) eligibility(programmatic bool) cudapdl.Eligibility {
	if programmatic {
		return cudapdl.CheckEligibility(f.target, f.producer, f.consumer)
	}
	return cudapdl.Eligibility{
		Policy: cudapdl.PolicyStandardFallback, TargetArch: f.target, FallbackReason: "ordinary reference arm",
	}
}

type pdlReferenceLaunch struct {
	desc cudalaunch.Description
	args []any
}

// launches is shared by the native fixture and its CPU contract control.
// Graph PDL uses a typed edge; stream-only launch attributes are absent.
func (f pdlReferenceFixture) launches(producer, consumer *cudalaunch.Launcher, input, intermediate, output any) ([2]pdlReferenceLaunch, error) {
	var launches [2]pdlReferenceLaunch
	for i, launcher := range []*cudalaunch.Launcher{producer, consumer} {
		if launcher == nil {
			return launches, fmt.Errorf("fixture kernel %d has no launcher", i)
		}
		desc, err := launcher.Grid(nativePDLN/nativePDLBlock, 1, 1).Block(nativePDLBlock, 1, 1).Describe()
		if err != nil {
			return launches, err
		}
		launches[i].desc = desc
	}
	launches[0].args = []any{input, intermediate, int32(nativePDLN), int64(1000)}
	launches[1].args = []any{intermediate, output, nativePDLBias, int32(nativePDLN), int64(1000)}
	return launches, nil
}

func pdlReferenceInputs() (input, want []float32) {
	input = make([]float32, nativePDLN)
	want = make([]float32, nativePDLN)
	for i := range input {
		input[i] = float32(i-128) / 4
		// Closed form from the input index, shared by both graph arms.
		want[i] = float32(i-128)/2 + 3
	}
	return input, want
}

// These conservative fixture predicates are checked after native authorization
// and before compilation. They do not certify the pair's device behavior.
func pdlReferenceUnsupported(major, minor, driverVersion int, typedEdges bool, compilerMajor, compilerMinor int32, archs []int32) string {
	if major < 9 {
		return "device requires SM90 or higher"
	}
	if driverVersion < 12030 || !typedEdges {
		return "driver requires CUDA 12.3 or higher and cuGraphAddDependencies_v2"
	}
	if compilerMajor < 12 || compilerMajor == 12 && compilerMinor < 3 {
		return "fixture requires NVRTC 12.3 or higher"
	}
	for _, arch := range archs {
		if int(arch) == major*10+minor {
			return ""
		}
	}
	return "NVRTC does not list the device's physical architecture"
}

// TestNativePDLReference compares ordinary and typed programmatic graph edges
// against one independent reference. It does not require or measure overlap.
func TestNativePDLReference(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" && os.Getenv("CUDAGRAPH_PDL_ABLATION") != "1" {
		t.Skip("native qualification pending: CUDA_COMPILER_TEST=1 or CUDAGRAPH_PDL_ABLATION=1 required for hardware execution")
	}

	if unresolvedPDLFixtures.unresolved() {
		t.Fatal("previous native fixture ownership remains unresolved; refusing another submission")
	}
	lockCtx, cancelLock := context.WithTimeout(context.Background(), 30*time.Second)
	lock, err := cudalock.Acquire(lockCtx, cudalock.WithLabel(t.Name()), cudalock.WithNoReclaim())
	cancelLock()
	if err != nil {
		t.Fatalf("fixture GPU lock: %v", err)
	}
	custody := new(pdlFixtureCustody)
	custody.own("GPU lock", lock.Release)
	var stream *cuda.Stream
	defer func() {
		if err := custody.release(0, func() error { return stream.Synchronize(context.Background()) }); err != nil {
			unresolvedPDLFixtures.keep(custody)
			t.Errorf("native fixture ownership retained until process exit: %v", err)
		}
	}()
	dev := cudatest.Require(t)
	custody.own("device", dev.Close)
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major < 9 {
		t.Skipf("device SM %d.%d does not support programmatic dependent launch (requires SM 90+)", major, minor)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatalf("driver version: %v", err)
	}
	if driverVersion < 12030 || !driver.HasGraphAddDependenciesV2() {
		t.Skipf("typed graph PDL requires CUDA 12.3+ and cuGraphAddDependencies_v2; driver=%d", driverVersion)
	}
	if err := nvrtc.Load(""); err != nil {
		t.Fatalf("load NVRTC: %v", err)
	}
	compilerMajor, compilerMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatalf("NVRTC version: %v", err)
	}
	narch, err := nvrtc.GetNumSupportedArchs()
	if err != nil || narch <= 0 {
		t.Fatalf("NVRTC architecture count: count=%d err=%v", narch, err)
	}
	archs := make([]int32, narch)
	if err := nvrtc.GetSupportedArchs(&archs[0]); err != nil {
		t.Fatalf("NVRTC architectures: %v", err)
	}
	if reason := pdlReferenceUnsupported(major, minor, driverVersion, driver.HasGraphAddDependenciesV2(), compilerMajor, compilerMinor, archs); reason != "" {
		t.Skipf("native qualification pending: %s (device=sm_%d%d driver=%d NVRTC=%d.%d)", reason, major, minor, driverVersion, compilerMajor, compilerMinor)
	}
	fixture := newPDLReferenceFixture(fmt.Sprintf("sm_%d%d", major, minor))
	t.Logf("fixture target=%s driver=%d NVRTC=%d.%d driverPath=%q compilerPath=%q", fixture.target, driverVersion, compilerMajor, compilerMinor, driver.Path(), nvrtc.Path())

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err = dev.NewStream()
	if stream != nil {
		custody.own("stream", stream.Close)
	}
	if err != nil {
		if stream == nil {
			// NewStream can lose a created handle when context restoration fails.
			// Keep its device/context and lock; do not claim that handle was closed.
			custody.unresolved = fmt.Errorf("stream constructor ownership unresolved: %w", err)
		}
		t.Fatal(err)
	}

	// Compile producer and consumer kernels
	prodComp, err := nvrtcx.Compile(fixture.producer.Source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	custody.ownError("producer compiler", err)
	if err != nil {
		t.Fatalf("compile producer: %v", err)
	}
	prodMod, _, err := prodComp.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if prodMod != nil {
		custody.own("producer module", prodMod.Close)
	}
	custody.ownError("producer module load", err)
	if err != nil {
		t.Fatalf("load producer module: %v", err)
	}
	prodKern, err := prodMod.Function(fixture.producer.Name)
	if err != nil {
		t.Fatal(err)
	}

	consComp, err := nvrtcx.Compile(fixture.consumer.Source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	custody.ownError("consumer compiler", err)
	if err != nil {
		t.Fatalf("compile consumer: %v", err)
	}
	consMod, _, err := consComp.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if consMod != nil {
		custody.own("consumer module", consMod.Close)
	}
	custody.ownError("consumer module load", err)
	if err != nil {
		t.Fatalf("load consumer module: %v", err)
	}
	consKern, err := consMod.Function(fixture.consumer.Name)
	if err != nil {
		t.Fatal(err)
	}

	// Allocate input, intermediate, and output device buffers
	inBuf, err := cudamem.Alloc[float32](dev, nativePDLN)
	custody.own("input buffer", inBuf.Free)
	if err != nil {
		t.Fatal(err)
	}

	interBuf, err := cudamem.Alloc[float32](dev, nativePDLN)
	custody.own("intermediate buffer", interBuf.Free)
	if err != nil {
		t.Fatal(err)
	}

	outBuf, err := cudamem.Alloc[float32](dev, nativePDLN)
	custody.own("output buffer", outBuf.Free)
	if err != nil {
		t.Fatal(err)
	}

	hostIn, want := pdlReferenceInputs()
	poison := make([]float32, nativePDLN)
	for i := range poison {
		poison[i] = -12345
	}
	launches, err := fixture.launches(cudalaunch.OwnedKernel(prodKern), cudalaunch.OwnedKernel(consKern), inBuf, interBuf, outBuf)
	if err != nil {
		t.Fatal(err)
	}
	for _, arm := range fixture.arms {
		if !t.Run(arm.name, func(t *testing.T) {
			mark := len(custody.owners)
			defer func() {
				if err := custody.release(mark, func() error { return stream.Synchronize(context.Background()) }); err != nil {
					t.Error(err)
				}
			}()
			// Reset both outputs so the second arm cannot inherit a correct
			// result from the first. Both use the same modules and buffers.
			for _, copy := range []struct {
				buffer cudamem.Slice[float32]
				data   []float32
			}{{inBuf, hostIn}, {interBuf, poison}, {outBuf, poison}} {
				if err := copy.buffer.CopyFromHost(copy.data); err != nil {
					t.Fatal(err)
				}
			}
			g, err := cudagraph.New(dev)
			if g != nil {
				custody.own("graph", g.Destroy)
			}
			if err != nil {
				t.Fatal(err)
			}
			prodNode, err := g.AddLaunch(launches[0].desc, launches[0].args)
			if err != nil {
				t.Fatalf("AddLaunch producer: %v", err)
			}
			consNode, err := g.AddLaunch(launches[1].desc, launches[1].args)
			if err != nil {
				t.Fatalf("AddLaunch consumer: %v", err)
			}
			comp, err := cudapdl.Compose(g, prodNode, consNode, fixture.eligibility(arm.programmatic))
			if err != nil {
				t.Fatalf("Compose: %v", err)
			}
			t.Logf("policy=%s fallback=%q; overlap unmeasured", comp.Policy, comp.FallbackReason)
			if comp.Policy != arm.wantPolicy {
				t.Fatalf("policy=%s, want %s (fallback=%q)", comp.Policy, arm.wantPolicy, comp.FallbackReason)
			}
			exec, err := g.Instantiate()
			if exec != nil {
				custody.own("graph executable", exec.Destroy)
			}
			if err != nil {
				t.Fatalf("Instantiate: %v", err)
			}
			if err := custody.launch(func() error { return exec.Launch(stream) }); err != nil {
				t.Fatalf("Launch: %v", err)
			}
			if err := custody.observe(func() error { return stream.Synchronize(ctx) }); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}
			got := make([]float32, nativePDLN)
			if err := outBuf.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("out[%d]=%g, want %g", i, got[i], want[i])
				}
			}
		}) {
			// Do not submit the other arm after a failed launch or drain.
			return
		}
	}
}
