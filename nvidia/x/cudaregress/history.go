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

	if err := ValidatePlan(plan); err != nil {
		return cudaprofilehistory.Record{}, err
	}
	if execRec.ExecutionID == "" || execRec.PlanID != plan.PlanID || execRec.PlanDigest != plan.PlanDigest || compRec.PlanID != plan.PlanID || compRec.ExecutionID != execRec.ExecutionID || compRec.TimingChannel != plan.TimingScope.Channel {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: history receipt identity mismatch")
	}
	if opts.Artifact == nil {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: history requires the executed artifact")
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
	compilerInputKey := opts.CompilerInputKey
	if compilerInputKey == "" || compiler.NVRTCVersion == "" || compiler.DriverSONAME == "" || compiler.TargetArch == "" {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: history requires compiler identity")
	}
	compilerKey := cudacache.CompilerCacheKey(compilerInputKey, compiler)
	workload := opts.Workload
	workloadKey := workload.Key()
	entry := opts.Entry
	target := compiler.TargetArch
	art := opts.Artifact

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
		Domain: execRec.EnvironmentObserved.Domain,
	}

	if env.Device == "" || env.Driver == "" || env.CUDA == "" || env.Clock == "" || env.Domain == "" {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: history requires observed environment identity")
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
	samples := 0
	for _, arm := range execRec.ArmReceipts {
		if arm.Name == compRec.CandidateArm || (compRec.CandidateArm == "" && arm.Name == "candidate") {
			if !arm.Complete || arm.Err != nil {
				return cudaprofilehistory.Record{}, errors.New("cudaregress: incomplete candidate history observation")
			}
			samples++
		}
	}
	if samples == 0 || execRec.Partial || !execRec.OutputsVerified {
		return cudaprofilehistory.Record{}, errors.New("cudaregress: history requires verified candidate observations")
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
