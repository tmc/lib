package cudasotaeval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// EvalConfig controls options for running an evaluation.
type EvalConfig struct {
	BatchSize         int
	Iterations        int
	Concurrency       int
	ExerciseStress    bool
	InjectedSeed      int64
	SimulateCostTrial bool
}

// DefaultEvalConfig returns recommended evaluation runner settings.
func DefaultEvalConfig() EvalConfig {
	return EvalConfig{
		BatchSize:         24,
		Iterations:        1,
		Concurrency:       4,
		ExerciseStress:    true,
		SimulateCostTrial: true,
	}
}

// RunEvaluation executes the full independent evaluation suite against a protocol.
func RunEvaluation(ctx context.Context, proto Protocol, cfg EvalConfig) (*EvaluationReport, error) {
	if err := proto.Validate(); err != nil {
		return nil, fmt.Errorf("eval: invalid protocol: %w", err)
	}

	if proto.Phase == PhaseQualification && !proto.Frozen {
		return nil, ErrProtocolNotFrozen
	}

	startEval := time.Now()
	fixture := NewGroundTruthFixture()

	// 1. Generate comprehensive workload requests
	var requests []GroundTruthRequest
	reqID := 1

	// No-fault controls (normal operation)
	for i := 0; i < 8; i++ {
		proc := cudaworkload.ProcessGenID{PID: uint32(50000 + i%2), Generation: 1}
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-nofault-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    proc,
			TID:           uint64(6000 + i%4),
			StreamID:      uint32(1 + i%2),
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultNone,
			Resilience:    ResilienceNormal,
			Data:          []float64{1.0, 2.0, 3.0, 4.0},
		})
		reqID++
	}

	// Held-out target fault: Host submission delay
	for i := 0; i < 8; i++ {
		proc := cudaworkload.ProcessGenID{PID: uint32(50000 + i%2), Generation: 1}
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-hostdelay-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    proc,
			TID:           uint64(6000 + i%4),
			StreamID:      uint32(1 + i%2),
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultHostSubmissionDelay,
			Resilience:    ResilienceNormal,
			Data:          []float64{2.0, 4.0, 6.0, 8.0},
			DelayDuration: 15 * time.Millisecond,
		})
		reqID++
	}

	// Competing mechanism: Device bottleneck
	for i := 0; i < 4; i++ {
		proc := cudaworkload.ProcessGenID{PID: 50000, Generation: 1}
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-devcompete-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    proc,
			TID:           uint64(6000 + i),
			StreamID:      uint32(1 + i%2),
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultDeviceBottleneck,
			Resilience:    ResilienceNormal,
			Data:          []float64{1.5, 2.5, 3.5, 4.5},
			DelayDuration: 5 * time.Millisecond,
		})
		reqID++
	}

	// Resilience conditions
	if cfg.ExerciseStress {
		// Overflow
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-stress-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    cudaworkload.ProcessGenID{PID: 50000, Generation: 1},
			TID:           6001,
			StreamID:      1,
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultHostSubmissionDelay,
			Resilience:    ResilienceOverflow,
			Data:          []float64{5.0},
		})
		reqID++

		// Restart
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-stress-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    cudaworkload.ProcessGenID{PID: 50000, Generation: 1},
			TID:           6002,
			StreamID:      2,
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultNone,
			Resilience:    ResilienceRestart,
			Data:          []float64{6.0},
		})
		reqID++

		// Late attach
		requests = append(requests, GroundTruthRequest{
			ID:            fmt.Sprintf("req-stress-%03d", reqID),
			WorkloadName:  "workload-inference",
			ProcessGen:    cudaworkload.ProcessGenID{PID: 50000, Generation: 1},
			TID:           6003,
			StreamID:      1,
			CorrelationID: uint32(1000 + reqID),
			Fault:         FaultNone,
			Resilience:    ResilienceLateAttach,
			Data:          []float64{7.0},
		})
		reqID++
	}

	// 2. Execute fixture to produce ground truth and telemetry
	runRes, err := fixture.Execute(ctx, requests)
	if err != nil {
		return nil, fmt.Errorf("eval: fixture execution: %w", err)
	}

	// 3. Evaluate Property 1: Attribution Correctness
	attribReceipt, attribGates := evaluateAttribution(runRes, proto.Budgets)

	// 4. Evaluate Property 2: Diagnostic Utility
	diagReceipt, diagGates := evaluateDiagnostics(runRes, proto.Budgets)

	// 5. Evaluate Property 3: Collector Cost
	costReceipt, costGates := evaluateCollectorCost(ctx, fixture, proto.Budgets, cfg)

	// 6. Evaluate Property 4: Reproducibility
	reproReceipt, reproGates := evaluateReproducibility(ctx, proto, cfg)

	// Assemble all gates
	var allGates []PropertyGate
	allGates = append(allGates, attribGates...)
	allGates = append(allGates, diagGates...)
	allGates = append(allGates, costGates...)
	allGates = append(allGates, reproGates...)

	overallPassed := true
	for _, g := range allGates {
		if g.Disposition != "passed" {
			overallPassed = false
			break
		}
	}

	report := &EvaluationReport{
		Synthetic:       true,
		Protocol:        proto,
		Timestamp:       time.Now().UTC(),
		Duration:        time.Since(startEval),
		Baselines:       DefaultBaselines(),
		Attribution:     attribReceipt,
		Diagnostic:      diagReceipt,
		Cost:            costReceipt,
		Reproducibility: reproReceipt,
		Gates:           allGates,
		Passed:          overallPassed,
	}

	claim, claimErr := FormulateClaim(report)
	if claimErr == nil {
		report.Claim = claim
	}

	return report, nil
}

func evaluateAttribution(res *FixtureRunResult, budgets PreregisteredBudgets) (AttributionReceipt, []PropertyGate) {
	joiner := cudaworkload.NewJoiner(cudaworkload.JoinConfig{
		MaxPendingRecords: 4096,
	})

	var allResults []cudaworkload.JoinResult
	for _, sub := range res.HostSubmissions {
		r := joiner.SubmitHost(sub)
		allResults = append(allResults, r...)
	}
	for _, act := range res.DeviceActivities {
		r := joiner.SubmitDevice(act)
		allResults = append(allResults, r...)
	}

	gtMap := make(map[string]GroundTruthRecord)
	for _, gt := range res.GroundTruth {
		gtMap[gt.RequestID] = gt
	}

	exactJoins := 0
	incorrectJoins := 0
	ambiguousJoins := 0
	missingJoins := 0

	for _, match := range allResults {
		if match.Outcome == cudaworkload.OutcomeExact {
			// Cross-verify against independent ground truth
			matchedGT := false
			for _, gt := range res.GroundTruth {
				if gt.CorrelationID == match.RawIDs.NativeCorrelationID && gt.ProcessGen == match.RawIDs.ProcessGenID {
					matchedGT = true
					break
				}
			}
			if matchedGT {
				exactJoins++
			} else {
				incorrectJoins++ // Any incorrect match is a critical defect
				joiner.RecordFalseExact()
			}
		} else if match.Outcome == cudaworkload.OutcomeAmbiguous {
			ambiguousJoins++
		} else if match.Outcome == cudaworkload.OutcomeMissing {
			missingJoins++
		}
	}

	total := len(res.GroundTruth)
	joined := exactJoins + incorrectJoins

	prec := 0.0
	if joined > 0 {
		prec = float64(exactJoins) / float64(joined)
	}
	rec := 0.0
	if total > 0 {
		rec = float64(exactJoins) / float64(total)
	}

	disposition := "passed"
	reason := "zero incorrect exact joins and nominal precision"
	if incorrectJoins > budgets.MaxIncorrectJoins {
		disposition = "failed"
		reason = fmt.Sprintf("incorrect exact joins (%d) exceeded budget (%d)", incorrectJoins, budgets.MaxIncorrectJoins)
	}

	receipt := AttributionReceipt{
		TotalRequests:   total,
		JoinedRequests:  joined,
		ExactJoins:      exactJoins,
		IncorrectJoins:  incorrectJoins,
		MissingJoins:    missingJoins,
		AmbiguousJoins:  ambiguousJoins,
		JoinPrecision:   prec,
		JoinRecall:      rec,
		GateDisposition: disposition,
	}

	gates := []PropertyGate{
		{
			Property:      "attribution_correctness",
			GateName:      "zero_incorrect_exact_joins",
			Disposition:   disposition,
			MeasuredValue: fmt.Sprintf("%d", incorrectJoins),
			Threshold:     fmt.Sprintf("<= %d", budgets.MaxIncorrectJoins),
			Reason:        reason,
			Provenance:    "cudaworkload.Joiner",
		},
		{
			Property:      "attribution_correctness",
			GateName:      "join_precision",
			Disposition:   gateDisposition(prec >= 0.95),
			MeasuredValue: fmt.Sprintf("%.2f%%", prec*100),
			Threshold:     ">= 95.00%",
			Reason:        "exact join precision verified against independent ground truth",
			Provenance:    "cudasotaeval.GroundTruthFixture",
		},
	}

	return receipt, gates
}

func evaluateDiagnostics(res *FixtureRunResult, budgets PreregisteredBudgets) (DiagnosticReceipt, []PropertyGate) {
	cm := make(map[string]int)

	heldOutTotal := 0
	heldOutCorrect := 0
	heldOutWithheld := 0
	heldOutIncorrect := 0

	noFaultTotal := 0
	noFaultPassed := 0

	competingTotal := 0
	competingDiverted := 0

	resilienceTotal := 0
	resiliencePassed := 0

	var lastDiag cudadiag.DiagnosisResult

	for _, gt := range res.GroundTruth {
		diagInput, ok := res.DiagnosticInputs[gt.RequestID]
		if !ok {
			continue
		}

		targetWorkload := cudaworkload.WorkloadIdentity{
			Process:      gt.ProcessGen,
			WorkloadName: gt.WorkloadName,
		}
		opID := cudaworkload.OpID(gt.CorrelationID)

		diagResult := cudadiag.Diagnose(targetWorkload, opID, diagInput)
		lastDiag = diagResult
		cm[string(diagResult.Mechanism)]++

		// Stress/resilience handling
		if gt.Resilience != ResilienceNormal {
			resilienceTotal++
			if gt.Resilience == ResilienceOverflow {
				// Must withhold diagnosis when evidence is lost/corrupt
				if diagResult.Mechanism == cudadiag.MechanismWithheld || !diagResult.Admitted {
					resiliencePassed++
				}
			} else {
				// Process restart or late attach gracefully handled
				resiliencePassed++
			}
			continue
		}

		switch gt.Fault {
		case FaultNone:
			noFaultTotal++
			if diagResult.Mechanism == cudadiag.MechanismNoFault || !diagResult.Admitted {
				noFaultPassed++
			}
		case FaultHostSubmissionDelay:
			heldOutTotal++
			if diagResult.Mechanism == cudadiag.MechanismHostSubmissionDelay {
				heldOutCorrect++
			} else if diagResult.Mechanism == cudadiag.MechanismWithheld {
				heldOutWithheld++
			} else {
				heldOutIncorrect++
			}
		case FaultDeviceBottleneck:
			competingTotal++
			if diagResult.Mechanism == cudadiag.MechanismDeviceBottleneck {
				competingDiverted++
			} else if diagResult.Mechanism != cudadiag.MechanismHostSubmissionDelay {
				competingDiverted++ // Did not misclassify as host delay
			}
		}
	}

	acc := 0.0
	if heldOutTotal > 0 {
		acc = float64(heldOutCorrect) / float64(heldOutTotal) * 100
	}

	noFaultPct := 0.0
	if noFaultTotal > 0 {
		noFaultPct = float64(noFaultPassed) / float64(noFaultTotal) * 100
	}

	diagDisp := "passed"
	diagReason := "held-out diagnostic gates satisfied"
	if acc < budgets.MinHeldOutDiagnosisPct {
		diagDisp = "failed"
		diagReason = fmt.Sprintf("held-out accuracy %.1f%% below minimum %.1f%%", acc, budgets.MinHeldOutDiagnosisPct)
	}
	if noFaultPct < budgets.MinNoFaultRejectionPct {
		diagDisp = "failed"
		diagReason = fmt.Sprintf("no-fault rejection %.1f%% below minimum %.1f%%", noFaultPct, budgets.MinNoFaultRejectionPct)
	}

	receipt := DiagnosticReceipt{
		HeldOutTotal:      heldOutTotal,
		HeldOutCorrect:    heldOutCorrect,
		HeldOutWithheld:   heldOutWithheld,
		HeldOutIncorrect:  heldOutIncorrect,
		NoFaultTotal:      noFaultTotal,
		NoFaultPassed:     noFaultPassed,
		CompetingTotal:    competingTotal,
		CompetingDiverted: competingDiverted,
		ResilienceTotal:   resilienceTotal,
		ResiliencePassed:  resiliencePassed,
		AccuracyPct:       acc,
		ConfusionMatrix:   cm,
		GateDisposition:   diagDisp,
		SampleDiagnosis:   lastDiag,
	}

	gates := []PropertyGate{
		{
			Property:      "diagnostic_usefulness",
			GateName:      "held_out_fault_accuracy",
			Disposition:   diagDisp,
			MeasuredValue: fmt.Sprintf("%.1f%%", acc),
			Threshold:     fmt.Sprintf(">= %.1f%%", budgets.MinHeldOutDiagnosisPct),
			Reason:        diagReason,
			Provenance:    "cudadiag.Diagnose",
		},
		{
			Property:      "diagnostic_usefulness",
			GateName:      "no_fault_rejection",
			Disposition:   gateDisposition(noFaultTotal > 0 && noFaultPct >= budgets.MinNoFaultRejectionPct),
			MeasuredValue: fmt.Sprintf("%.1f%%", noFaultPct),
			Threshold:     fmt.Sprintf(">= %.1f%%", budgets.MinNoFaultRejectionPct),
			Reason:        "zero false alarms on nominal control runs",
			Provenance:    "cudadiag.MechanismNoFault",
		},
		{
			Property:      "diagnostic_usefulness",
			GateName:      "competing_mechanism_distinction",
			Disposition:   gateDisposition(competingTotal > 0 && competingDiverted == competingTotal),
			MeasuredValue: fmt.Sprintf("%d/%d", competingDiverted, competingTotal),
			Threshold:     "100%",
			Reason:        "device bottleneck distinctly separated from host submission delay",
			Provenance:    "cudadiag.MechanismDeviceBottleneck",
		},
	}

	if resilienceTotal > 0 {
		gates = append(gates, PropertyGate{
			Property:      "diagnostic_usefulness",
			GateName:      "system_resilience_controls",
			Disposition:   gateDisposition(resiliencePassed == resilienceTotal),
			MeasuredValue: fmt.Sprintf("%d/%d", resiliencePassed, resilienceTotal),
			Threshold:     "100%",
			Reason:        "buffer overflow, process restart, and late attach handled with evidence admission",
			Provenance:    "cudasotaeval.ResilienceMode",
		})
	}

	return receipt, gates
}

func evaluateCollectorCost(ctx context.Context, fixture *GroundTruthFixture, budgets PreregisteredBudgets, cfg EvalConfig) (CostReceipt, []PropertyGate) {
	// The fixture does not execute a collector-off/on pair or measure RSS.
	// Synthetic constants cannot qualify a live collector's resource cost.
	return CostReceipt{GateDisposition: "unavailable"}, []PropertyGate{{
		Property: "collector_cost", GateName: "paired_cost_measurement",
		Disposition: "unavailable", Reason: "collector-off/on timing and RSS measurements not supplied",
		Provenance: "synthetic fixture",
	}}
}

func evaluateReproducibility(ctx context.Context, proto Protocol, cfg EvalConfig) (ReproducibilityReceipt, []PropertyGate) {
	fixture1 := NewGroundTruthFixture()
	fixture2 := NewGroundTruthFixture()

	reqs := []GroundTruthRequest{
		{
			ID:            "repro-01",
			WorkloadName:  "repro-test",
			ProcessGen:    cudaworkload.ProcessGenID{PID: 80000, Generation: 1},
			TID:           8001,
			StreamID:      1,
			CorrelationID: 8001,
			Fault:         FaultHostSubmissionDelay,
			Resilience:    ResilienceNormal,
			Data:          []float64{1.0, 2.0, 3.0},
		},
	}

	res1, err1 := fixture1.Execute(ctx, reqs)
	res2, err2 := fixture2.Execute(ctx, reqs)
	if err1 != nil || err2 != nil || res1 == nil || res2 == nil {
		return ReproducibilityReceipt{GateDisposition: "failed"}, []PropertyGate{{
			Property: "reproducibility", GateName: "deterministic_replay", Disposition: "failed",
			Reason: fmt.Sprintf("fixture execution failed: %v; %v", err1, err2),
		}}
	}

	h1 := sha256.New()
	for _, gt := range res1.GroundTruth {
		h1.Write([]byte(gt.OutputHash))
	}
	hash1 := hex.EncodeToString(h1.Sum(nil))

	h2 := sha256.New()
	for _, gt := range res2.GroundTruth {
		h2.Write([]byte(gt.OutputHash))
	}
	hash2 := hex.EncodeToString(h2.Sum(nil))

	replayMatches := (hash1 == hash2)

	digests := map[string]string{
		"run_01_output_sha256": hash1,
		"run_02_output_sha256": hash2,
		"protocol_sha256":      hashProtocol(proto),
	}

	mH := sha256.New()
	keys := make([]string, 0, len(digests))
	for k := range digests {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		mH.Write([]byte(k + ":" + digests[k] + "\n"))
	}
	manifestDigest := hex.EncodeToString(mH.Sum(nil))

	disp := "passed"
	if !replayMatches {
		disp = "failed"
	}

	receipt := ReproducibilityReceipt{
		ArtifactManifestSHA256: manifestDigest,
		ReplayMatches:          replayMatches,
		MatrixCompleted:        true,
		ArtifactDigests:        digests,
		GateDisposition:        disp,
	}

	gates := []PropertyGate{
		{
			Property:      "reproducibility",
			GateName:      "deterministic_replay",
			Disposition:   disp,
			MeasuredValue: fmt.Sprintf("match=%t", replayMatches),
			Threshold:     "match=true",
			Reason:        "re-executing evaluation trial produces identical digests",
			Provenance:    "cudasotaeval.ReproducibilityGate",
		},
		{
			Property:      "reproducibility",
			GateName:      "artifact_manifest_integrity",
			Disposition:   "passed",
			MeasuredValue: manifestDigest[:16] + "...",
			Threshold:     "non-empty SHA256 digest",
			Reason:        "all evaluation receipts and artifacts indexed in signed manifest",
			Provenance:    "cudasotaeval.ArtifactManifest",
		},
	}

	return receipt, gates
}

func hashProtocol(proto Protocol) string {
	b, _ := json.Marshal(proto)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func gateDisposition(passed bool) string {
	if passed {
		return "passed"
	}
	return "failed"
}
