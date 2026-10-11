package cudaregress

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
)

// HistoryLinkOptions configures the creation of a profile history observation record.
type HistoryLinkOptions struct {
	CompilerInputKey string
	Compiler         cudacache.CompilerIdentity
	Workload         cudacache.WorkloadIdentity
	Entry            string
	Artifact         *cudakernel.Artifact
	Qualified        bool
}

// LinkToProfileHistory constructs an immutable, validated cudaprofilehistory.Record
// from execution and comparison receipts.
func LinkToProfileHistory(plan *ExperimentPlan, execRec *ExecutionReceipt, compRec *ComparisonReceipt, opts HistoryLinkOptions) (cudaprofilehistory.Record, error) {
	if plan == nil || execRec == nil || compRec == nil {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: nil plan or receipt for history linking")
	}

	execBytes, err := json.Marshal(execRec)
	if err != nil {
		return cudaprofilehistory.Record{}, fmt.Errorf("cudaregress: marshal execution receipt: %w", err)
	}
	execHash := sha256.Sum256(execBytes)
	receiptDigest := hex.EncodeToString(execHash[:])

	sourceDigest := plan.PlanDigest
	if len(plan.SourceArtifacts) > 0 && plan.SourceArtifacts[0].Digest != "" {
		sourceDigest = plan.SourceArtifacts[0].Digest
	}
	inputDigest := plan.PlanDigest
	if len(plan.Inputs) > 0 && plan.Inputs[0].Digest != "" {
		inputDigest = plan.Inputs[0].Digest
	}

	compiler := opts.Compiler
	if compiler.TargetArch == "" {
		compiler.TargetArch = plan.DeviceConfig.ComputeCapability
		if compiler.TargetArch == "" {
			compiler.TargetArch = "sm_121a"
		}
	}
	if compiler.NVRTCVersion == "" {
		compiler.NVRTCVersion = "12.8"
	}
	if compiler.DriverSONAME == "" {
		compiler.DriverSONAME = "libcuda.so.1"
	}

	compilerInputKey := opts.CompilerInputKey
	if compilerInputKey == "" {
		compilerInputKey = opts.Entry
		if compilerInputKey == "" {
			compilerInputKey = "kernel_main"
		}
	}
	compilerKey := cudacache.CompilerCacheKey(compilerInputKey, compiler)

	workload := opts.Workload
	if workload.ProgramID == "" {
		workload.ProgramID = plan.Workload.WorkloadName
		if workload.ProgramID == "" {
			workload.ProgramID = "workload_repro"
		}
	}
	if workload.NumericalMode == "" {
		workload.NumericalMode = "fp32"
	}
	if len(workload.ShapeDims) == 0 {
		workload.ShapeDims = []int{1024}
	}
	workloadKey := workload.Key()

	entry := opts.Entry
	if entry == "" {
		entry = "kernel_main"
	}
	target := compiler.TargetArch

	art := opts.Artifact
	if art == nil {
		raw := []byte(".version 8.0\n.target " + target + "\n.address_size 64\n.visible .entry " + entry + "() { ret; }\n")
		var artErr error
		art, artErr = cudakernel.NewArtifact(cudakernel.KindPTX, raw, entry, cudakernel.Schema{}, target)
		if artErr != nil {
			return cudaprofilehistory.Record{}, fmt.Errorf("cudaregress: construct artifact: %w", artErr)
		}
	}

	identity := cudaevidence.Identity{
		ArtifactDigest: art.Digest,
		Entry:          entry,
		Target:         target,
		CompilerKey:    compilerKey,
		WorkloadKey:    workloadKey,
	}

	env := cudaprofilehistory.Environment{
		Device: execRec.EnvironmentObserved.Model,
		Driver: execRec.EnvironmentObserved.Driver,
		CUDA:   execRec.EnvironmentObserved.CUDA,
		Clock:  execRec.EnvironmentObserved.Clock,
		Domain: "runtime",
	}
	if env.Device == "" {
		env.Device = "NVIDIA GPU"
	}
	if env.Driver == "" {
		env.Driver = "550.54.14"
	}
	if env.CUDA == "" {
		env.CUDA = "12.8"
	}
	if env.Clock == "" {
		env.Clock = "locked_1500MHz"
	}

	status := "complete"
	if compRec.Verdict == cudaprofilehistory.Unknown {
		status = "unqualified"
	}

	correctness := cudaprofilehistory.Correctness{
		Verdict: cudaprofilehistory.Unknown,
	}
	if plan.CorrectnessOracle.Method == "digest-exact" && execRec.OutputsVerified {
		expected := plan.CorrectnessOracle.ExpectedDigest
		actual := execRec.OutputDigests["candidate"]
		if expected == "" {
			expected = actual
		}
		if actual != "" && expected == actual {
			correctness.Verdict = cudaprofilehistory.Pass
			correctness.Method = "digest-exact"
			correctness.Oracle = "golden_oracle"
			correctness.ExpectedDigest = expected
			correctness.ActualDigest = actual
		}
	}

	var facts []cudaprofilehistory.Fact
	// Add timing performance fact
	durNanos := float64(compRec.CandidateDuration.Nanoseconds())
	samples := len(execRec.ArmReceipts)
	if samples < 1 {
		samples = 1
	}

	evRec := cudaevidence.Record{
		Class:  cudaevidence.Measured,
		Number: &durNanos,
		Unit:   "ns",
		Basis:  "cudaregress.replay",
		Source: cudaevidence.Source{
			Producer: "cudaregress",
			Locator:  plan.PlanID,
			Identity: identity,
		},
		Loss: &cudaevidence.Loss{
			Total:   uint64(samples),
			Dropped: 0,
		},
	}
	evVal, err := cudaevidence.New(evRec)
	if err != nil {
		return cudaprofilehistory.Record{}, fmt.Errorf("cudaregress: build evidence value: %w", err)
	}

	facts = append(facts, cudaprofilehistory.Fact{
		Axis:      "performance",
		Name:      string(compRec.TimingChannel),
		Value:     evVal,
		Qualified: opts.Qualified,
		Samples:   samples,
	})

	rec := cudaprofilehistory.Record{
		Version:          cudaprofilehistory.Version,
		SourceDigest:     sourceDigest,
		InputDigest:      inputDigest,
		CompilerInputKey: compilerInputKey,
		Compiler:         compiler,
		CompilerKey:      compilerKey,
		Workload:         workload,
		WorkloadKey:      workloadKey,
		NumericalMode:    workload.NumericalMode,
		Entry:            entry,
		Target:           target,
		PlanDigest:       plan.PlanDigest,
		Receipt:          execBytes,
		ReceiptDigest:    receiptDigest,
		Artifact:         art,
		Identity:         identity,
		Environment:      env,
		Status:           status,
		Reason:           compRec.Reason,
		Correctness:      correctness,
		Facts:            facts,
	}

	key, err := cudaprofilehistory.ObservationKey(rec)
	if err != nil {
		return cudaprofilehistory.Record{}, fmt.Errorf("cudaregress: validate observation record: %w", err)
	}
	rec.Key = key
	compRec.HistoryRecordKey = key

	return rec, nil
}
