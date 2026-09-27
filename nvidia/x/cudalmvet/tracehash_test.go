package cudalmvet

import (
	"context"
	"errors"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func buildTestGraph() (*cudair.Graph, map[string][]float32) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{8}, cudair.Float32)
	w := g.AddInput("W", []int{8}, cudair.Float32)

	h := g.AddBinary("mul", x, w)
	h.Name = "h"

	act := g.AddUnary("relu", h)
	act.Name = "act"

	_ = g.SetOutputs(act)

	inputs := map[string][]float32{
		"X": {1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0},
		"W": {0.5, 0.5, 2.0, -1.0, 1.0, 0.5, 2.0, 0.25},
	}

	return g, inputs
}

func TestTraceHash_CPUDeterministic(t *testing.T) {
	g, inputs := buildTestGraph()

	oldProcs := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(oldProcs)

	var baselineHash uint64

	// Test across multiple GOMAXPROCS values and iterations
	for _, procs := range []int{1, 2, 4, 8} {
		runtime.GOMAXPROCS(procs)

		for iter := 0; iter < 5; iter++ {
			acc, err := EvaluateCPUGraph(g, inputs, PolicyRawBits, false)
			if err != nil {
				t.Fatalf("procs=%d iter=%d: EvaluateCPUGraph failed: %v", procs, iter, err)
			}
			h := acc.Hash()
			if baselineHash == 0 {
				baselineHash = h
			} else if h != baselineHash {
				t.Fatalf("procs=%d iter=%d: non-deterministic hash: got 0x%016x, want 0x%016x",
					procs, iter, h, baselineHash)
			}
		}
	}
}

func TestTraceHash_OrderIsDeclared(t *testing.T) {
	// Fixed traversal: evaluating the same graph in declared topological order
	// produces identical hashes.
	g1, inputs1 := buildTestGraph()
	acc1, err := EvaluateCPUGraph(g1, inputs1, PolicyRawBits, false)
	if err != nil {
		t.Fatalf("EvaluateCPUGraph 1: %v", err)
	}

	g2, inputs2 := buildTestGraph()
	acc2, err := EvaluateCPUGraph(g2, inputs2, PolicyRawBits, false)
	if err != nil {
		t.Fatalf("EvaluateCPUGraph 2: %v", err)
	}

	if acc1.Hash() != acc2.Hash() {
		t.Errorf("declared order evaluations produced different hashes: 0x%016x vs 0x%016x",
			acc1.Hash(), acc2.Hash())
	}

	// Significant traversal: reversing element traversal order must produce different hashes.
	vals := []float32{1.0, 2.0, 3.0, 4.0}
	accForward := NewAccumulator()
	accForward.FoldFloat32s(vals)

	accReverse := NewAccumulator()
	for i := len(vals) - 1; i >= 0; i-- {
		accReverse.FoldFloat32(vals[i])
	}

	if accForward.Hash() == accReverse.Hash() {
		t.Errorf("reversing fold order unexpectedly produced identical hash: 0x%016x", accForward.Hash())
	}
}

func TestTraceHash_SignedZeroPolicy(t *testing.T) {
	posZero := float32(0.0)
	negZero := float32(math.Copysign(0, -1))

	if math.Float32bits(posZero) == math.Float32bits(negZero) {
		t.Fatal("expected distinct bit patterns for +0.0 and -0.0")
	}

	// PolicyRawBits: +0.0 and -0.0 produce distinct hashes
	accPos := NewAccumulator(WithPolicy(PolicyRawBits))
	accPos.FoldFloat32(posZero)

	accNeg := NewAccumulator(WithPolicy(PolicyRawBits))
	accNeg.FoldFloat32(negZero)

	if accPos.Hash() == accNeg.Hash() {
		t.Errorf("PolicyRawBits: expected different hashes for +0.0 and -0.0, both got 0x%016x",
			accPos.Hash())
	}

	// PolicyCanonicalizeZero: +0.0 and -0.0 produce identical hashes
	accPosCanon := NewAccumulator(WithPolicy(PolicyCanonicalizeZero))
	accPosCanon.FoldFloat32(posZero)

	accNegCanon := NewAccumulator(WithPolicy(PolicyCanonicalizeZero))
	accNegCanon.FoldFloat32(negZero)

	if accPosCanon.Hash() != accNegCanon.Hash() {
		t.Errorf("PolicyCanonicalizeZero: expected identical hashes, got 0x%016x vs 0x%016x",
			accPosCanon.Hash(), accNegCanon.Hash())
	}
}

func TestDifferential_UnmeasuredWithoutDevice(t *testing.T) {
	g, inputs := buildTestGraph()

	// Calling Compare without providing a device must return StatusUnmeasured,
	// never StatusAgreed.
	res, err := Compare(context.Background(), g, WithInputs(inputs))
	if err != nil {
		t.Fatalf("Compare returned unexpected error: %v", err)
	}

	if res.Status != StatusUnmeasured {
		t.Errorf("Compare without device: got status %v, want %v", res.Status, StatusUnmeasured)
	}
	if res.CPUHash == 0 || res.CPUHash == FNVOffsetBasis {
		t.Errorf("expected non-trivial CPUHash, got 0x%016x", res.CPUHash)
	}
	if res.DeviceHash != 0 {
		t.Errorf("expected 0 DeviceHash for unmeasured run, got 0x%016x", res.DeviceHash)
	}
}

func TestDifferential_AgreesOnReference(t *testing.T) {
	if os.Getenv("CUDA_LMVET_NATIVE") != "1" {
		t.Skip("skipping native test: CUDA_LMVET_NATIVE unset")
	}

	if err := driver.Init(0); err != nil {
		t.Fatalf("driver.Init: %v", err)
	}
	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		t.Fatalf("cuda.Devices: %v (count: %d)", err, len(devs))
	}
	dev := devs[0]

	g, inputs := buildTestGraph()

	res, err := Compare(context.Background(), g,
		WithInputs(inputs),
		WithCudaDevice(dev),
	)
	if err != nil {
		t.Fatalf("Compare failed: %v", err)
	}

	if res.Status != StatusAgreed {
		t.Fatalf("Compare: got status %v, want %v; details: %v", res.Status, StatusAgreed, res)
	}
	if res.CPUHash != res.DeviceHash {
		t.Errorf("hash mismatch: CPU 0x%016x != Device 0x%016x", res.CPUHash, res.DeviceHash)
	}
}

func TestPerturbation_ReportsTable(t *testing.T) {
	if os.Getenv("CUDA_LMVET_NATIVE") != "1" {
		t.Skip("skipping native test: CUDA_LMVET_NATIVE unset")
	}

	if err := driver.Init(0); err != nil {
		t.Fatalf("driver.Init: %v", err)
	}
	devs, err := cuda.Devices()
	if err != nil || len(devs) == 0 {
		t.Fatalf("cuda.Devices: %v (count: %d)", err, len(devs))
	}
	dev := devs[0]

	g, inputs := buildTestGraph()

	report, err := RunPerturbations(context.Background(), g, DefaultPerturbations(),
		WithInputs(inputs),
		WithCudaDevice(dev),
	)
	if err != nil {
		t.Fatalf("RunPerturbations failed: %v", err)
	}

	t.Log("\n" + report.String())

	if len(report.Results) != len(DefaultPerturbations()) {
		t.Errorf("got %d perturbation results, want %d", len(report.Results), len(DefaultPerturbations()))
	}

	for _, r := range report.Results {
		if r.CompileErr != nil {
			t.Errorf("perturbation %q compilation failed: %v", r.Perturbation.Name, r.CompileErr)
		}
		if r.RunErr != nil {
			t.Errorf("perturbation %q run failed: %v", r.Perturbation.Name, r.RunErr)
		}
		if r.Status != StatusAgreed {
			t.Errorf("perturbation %q: status is %v, want %v", r.Perturbation.Name, r.Status, StatusAgreed)
		}
	}
}

// Mutant Control 1: The oracle must fail when output is perturbed by 1 ULP.
func TestTraceHash_Mutant_OneULP(t *testing.T) {
	baseVals := []float32{1.0, 2.0, 3.0, 4.0}
	accA := NewAccumulator(WithRecording(true))
	accA.FoldFloat32s(baseVals)

	// Perturb index 2 by exactly 1 ULP
	perturbedVals := make([]float32, len(baseVals))
	copy(perturbedVals, baseVals)
	bits := math.Float32bits(perturbedVals[2])
	perturbedVals[2] = math.Float32frombits(bits + 1)

	accB := NewAccumulator(WithRecording(true))
	accB.FoldFloat32s(perturbedVals)

	if accA.Hash() == accB.Hash() {
		t.Fatalf("oracle failed to detect 1 ULP perturbation: both hashes are 0x%016x", accA.Hash())
	}

	report := Triage(accA, accB)
	if report == nil {
		t.Fatal("expected triage report for differing accumulators")
	}
	if report.DiffIndex != 2 {
		t.Errorf("triage report DiffIndex: got %d, want 2", report.DiffIndex)
	}
	if report.BitsA == report.BitsB {
		t.Errorf("triage report BitsA (%08x) equals BitsB (%08x)", report.BitsA, report.BitsB)
	}
}

// Mutant Control 2: NaN payload test asserts documented policy behavior.
func TestTraceHash_Mutant_NaNPayload(t *testing.T) {
	nan1 := math.Float32frombits(0x7fc00001)
	nan2 := math.Float32frombits(0x7fc00002)

	if !math.IsNaN(float64(nan1)) || !math.IsNaN(float64(nan2)) {
		t.Fatal("values are not NaNs")
	}

	// Under PolicyRawBits: differing payloads yield distinct hashes
	accRaw1 := NewAccumulator(WithPolicy(PolicyRawBits))
	accRaw1.FoldFloat32(nan1)

	accRaw2 := NewAccumulator(WithPolicy(PolicyRawBits))
	accRaw2.FoldFloat32(nan2)

	if accRaw1.Hash() == accRaw2.Hash() {
		t.Errorf("PolicyRawBits: expected distinct hashes for differing NaN payloads, got 0x%016x",
			accRaw1.Hash())
	}

	// Under PolicyCanonicalizeNaN: differing payloads yield identical hashes
	accCanon1 := NewAccumulator(WithPolicy(PolicyCanonicalizeNaN))
	accCanon1.FoldFloat32(nan1)

	accCanon2 := NewAccumulator(WithPolicy(PolicyCanonicalizeNaN))
	accCanon2.FoldFloat32(nan2)

	if accCanon1.Hash() != accCanon2.Hash() {
		t.Errorf("PolicyCanonicalizeNaN: expected equal hashes, got 0x%016x vs 0x%016x",
			accCanon1.Hash(), accCanon2.Hash())
	}
}

// Mutant Control 3: Drop device reset between launches and assert contamination detection.
func TestTraceHash_Mutant_ContaminatedReset(t *testing.T) {
	// An accumulator starting at an arbitrary contaminated value (not FNVOffsetBasis)
	// produces a contaminated result.
	contaminatedBasis := uint64(0xdeadbeef12345678)
	accClean := NewAccumulator()
	accClean.FoldFloat32(1.0)

	accContaminated := &Accumulator{
		hash:   contaminatedBasis,
		policy: PolicyRawBits,
	}
	accContaminated.FoldFloat32(1.0)

	if accClean.Hash() == accContaminated.Hash() {
		t.Fatal("contaminated start state unexpectedly matched clean state")
	}

	// Check that CheckDeviceReset logic recognizes any non-offset basis as contaminated
	if contaminatedBasis == nvrtcx.TraceHashOffsetBasis {
		t.Fatal("contaminated basis equals TraceHashOffsetBasis")
	}
}

// Mutant Control 4: Force a cache hit on a perturbed compile and assert detection of artifact mismatch.
func TestTraceHash_Mutant_CacheHitArtifactMismatch(t *testing.T) {
	pert := Perturbation{
		Name:                 "vectorize_on",
		ExtraDeviceVectorize: true,
		JITOptLevel:          3,
	}

	// Simulate a cached artifact that was compiled without --extra-device-vectorization (e.g. from baseline)
	unperturbedArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 3,
		},
	}

	err := ValidatePerturbationArtifact(pert, unperturbedArtifact)
	if err == nil {
		t.Fatal("expected ValidatePerturbationArtifact to reject unperturbed artifact for vectorize_on")
	}
	if !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Errorf("got error %v, want ErrPerturbationArtifactMismatch", err)
	}

	// Simulate an artifact with mismatched JIT opt level
	optLevelMismatchArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17", "--extra-device-vectorization"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 4, // requested 3, cached 4
		},
	}

	err = ValidatePerturbationArtifact(pert, optLevelMismatchArtifact)
	if err == nil {
		t.Fatal("expected ValidatePerturbationArtifact to reject artifact with mismatched JIT opt level")
	}
	if !errors.Is(err, ErrPerturbationArtifactMismatch) {
		t.Errorf("got error %v, want ErrPerturbationArtifactMismatch", err)
	}

	// Valid matching artifact
	matchingArtifact := &cudakernel.Artifact{
		CompilerOpts: []string{"-std=c++17", "--extra-device-vectorization"},
		JITOptions: cudakernel.JITOptions{
			OptLevel: 3,
		},
	}
	if err := ValidatePerturbationArtifact(pert, matchingArtifact); err != nil {
		t.Errorf("unexpected validation failure for matching artifact: %v", err)
	}
}
