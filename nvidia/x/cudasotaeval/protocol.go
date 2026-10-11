package cudasotaeval

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"runtime"
	"time"
)

var (
	// ErrProtocolFrozen indicates an attempt to modify an already frozen protocol.
	ErrProtocolFrozen = errors.New("cudasotaeval: protocol is frozen and cannot be modified")

	// ErrProtocolNotFrozen indicates qualification attempted on an unfrozen protocol.
	ErrProtocolNotFrozen = errors.New("cudasotaeval: qualification requires a frozen protocol")

	// ErrInvalidProtocol indicates invalid or incomplete protocol parameters.
	ErrInvalidProtocol = errors.New("cudasotaeval: invalid protocol specification")

	// ErrGateFailed indicates an evaluation acceptance gate failed.
	ErrGateFailed = errors.New("cudasotaeval: acceptance gate failed")
)

// DefaultBudgets returns conservative, preregistered qualification thresholds.
func DefaultBudgets() PreregisteredBudgets {
	return PreregisteredBudgets{
		MaxThroughputDegradationPct: 5.0,   // Max 5% continuous throughput degradation
		MaxRSSGrowthMB:              64.0,  // Max 64MB memory RSS growth
		MaxTailLatencyP99Pct:        10.0,  // Max 10% p99 latency inflation
		MaxIncorrectJoins:           0,     // Zero incorrect exact joins allowed
		MinNoFaultRejectionPct:      100.0, // 100% false positive rejection on normal runs
		MinHeldOutDiagnosisPct:      95.0,  // Minimum 95% accuracy on held-out target faults
	}
}

// DefaultHardwareClosure returns default host hardware metadata for test environment.
func DefaultHardwareClosure() HardwareClosure {
	return HardwareClosure{
		GPUModel:      "NVIDIA GB10 (simulated/host-qualified)",
		GPUUUID:       "GPU-12345678-abcd-ef01-2345-6789abcdef01",
		Architecture:  "sm_121a",
		SMCount:       128,
		DriverVersion: "570.86.16",
		CUDAVersion:   "12.8",
		HostOS:        runtime.GOOS,
		HostArch:      runtime.GOARCH,
		KernelRelease: "6.8.0-generic",
		MIGMode:       "disabled",
		ClocksLocked:  true,
		PowerLimitW:   700,
	}
}

// DefaultWorkloads returns representative evaluation workloads.
func DefaultWorkloads() []WorkloadSpec {
	return []WorkloadSpec{
		{
			Name:             "inference-batch",
			Concurrency:      4,
			WarmupIterations: 4,
			TestIterations:   20,
			BatchSize:        16,
			InputDim:         1024,
		},
		{
			Name:             "launch-heavy",
			Concurrency:      8,
			WarmupIterations: 4,
			TestIterations:   20,
			BatchSize:        4,
			InputDim:         256,
		},
		{
			Name:             "long-kernel",
			Concurrency:      2,
			WarmupIterations: 2,
			TestIterations:   10,
			BatchSize:        64,
			InputDim:         4096,
		},
	}
}

// DefaultProtocol creates a preregistered evaluation protocol.
func DefaultProtocol(phase Phase) Protocol {
	if phase == "" {
		phase = PhasePilot
	}
	return Protocol{
		Name:      "cudasotaeval-submission-delay-v1",
		Phase:     phase,
		Frozen:    false,
		Workloads: DefaultWorkloads(),
		Hardware:  DefaultHardwareClosure(),
		Budgets:   DefaultBudgets(),
		Sampling: SamplingConfig{
			BufferCapacity: 65536,
			Window:         10 * time.Second,
			RateHz:         1000,
		},
		ExecutionOrder:    "interleaved_abba",
		UncertaintyMethod: "bootstrap_95ci",
		StoppingRules: []string{
			"stop immediately if any incorrect exact join occurs",
			"stop if ring buffer overflow exceeds 5% in qualification",
			"stop if hardware clock drift exceeds 1ms",
		},
		ExclusionRules: []string{
			"exclude runs with detected non-participant processes on GPU lane",
			"exclude warm-up phase iterations from cost accounting",
			"retain failed and withheld diagnoses explicitly (never exclude)",
		},
	}
}

// Freeze locks the protocol parameters prior to qualification runs.
func (p *Protocol) Freeze() error {
	if p.Frozen {
		return ErrProtocolFrozen
	}
	if err := p.Validate(); err != nil {
		return fmt.Errorf("freeze: %w", err)
	}
	p.Frozen = true
	p.FrozenTimestamp = time.Now().UTC()
	p.FrozenDigest = protocolDigest(*p)
	return nil
}

func protocolDigest(p Protocol) string {
	p.FrozenDigest = ""
	data, _ := json.Marshal(p)
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

// Validate verifies protocol structural correctness and completeness.
func (p *Protocol) Validate() error {
	if p.Name == "" {
		return fmt.Errorf("%w: missing protocol name", ErrInvalidProtocol)
	}
	if p.Phase != PhasePilot && p.Phase != PhaseQualification {
		return fmt.Errorf("%w: invalid phase %q", ErrInvalidProtocol, p.Phase)
	}
	names := make(map[string]bool)
	for _, tool := range p.Tools {
		if tool.Name == "" || tool.Version == "" || tool.Configuration == "" || tool.Lineage == "" || names[tool.Name] {
			return fmt.Errorf("%w: incomplete or duplicate tool", ErrInvalidProtocol)
		}
		if tool.Support != "measured" && tool.Support != "unknown" && tool.Support != "unavailable" {
			return fmt.Errorf("%w: invalid support", ErrInvalidProtocol)
		}
		names[tool.Name] = true
	}
	for name, value := range p.PropertyLimits {
		if name != "attribution" && name != "diagnosis" && name != "cost" && name != "reproducibility" {
			return fmt.Errorf("%w: unknown property", ErrInvalidProtocol)
		}
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf("%w: invalid property limit", ErrInvalidProtocol)
		}
	}
	if len(p.Workloads) == 0 {
		return fmt.Errorf("%w: no workloads defined", ErrInvalidProtocol)
	}
	for i, w := range p.Workloads {
		if w.Name == "" || w.Concurrency <= 0 || w.TestIterations <= 0 {
			return fmt.Errorf("%w: workload[%d] missing name or invalid iterations", ErrInvalidProtocol, i)
		}
	}
	if p.Hardware.GPUModel == "" || p.Hardware.DriverVersion == "" || p.Hardware.CUDAVersion == "" {
		return fmt.Errorf("%w: incomplete hardware closure", ErrInvalidProtocol)
	}
	for _, value := range []float64{p.Budgets.MaxThroughputDegradationPct, p.Budgets.MaxRSSGrowthMB, p.Budgets.MaxTailLatencyP99Pct, p.Budgets.MinNoFaultRejectionPct, p.Budgets.MinHeldOutDiagnosisPct} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf("%w: non-finite or negative budget", ErrInvalidProtocol)
		}
	}
	if p.Budgets.MinHeldOutDiagnosisPct > 100 {
		return fmt.Errorf("%w: held-out percentage exceeds 100", ErrInvalidProtocol)
	}
	if p.Frozen && (p.FrozenTimestamp.IsZero() || p.FrozenDigest == "" || p.FrozenDigest != protocolDigest(*p)) {
		return fmt.Errorf("%w: frozen parameters changed", ErrInvalidProtocol)
	}
	if p.Budgets.MaxIncorrectJoins != 0 {
		return fmt.Errorf("%w: invalid negative max incorrect joins", ErrInvalidProtocol)
	}
	if p.Budgets.MinNoFaultRejectionPct <= 0 || p.Budgets.MinNoFaultRejectionPct > 100 {
		return fmt.Errorf("%w: invalid min no-fault rejection percentage", ErrInvalidProtocol)
	}
	if len(p.StoppingRules) == 0 {
		return fmt.Errorf("%w: missing stopping rules", ErrInvalidProtocol)
	}
	if len(p.ExclusionRules) == 0 {
		return fmt.Errorf("%w: missing exclusion rules", ErrInvalidProtocol)
	}
	return nil
}
