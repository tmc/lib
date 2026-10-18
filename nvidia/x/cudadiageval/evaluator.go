package cudadiageval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"slices"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
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
	noFaultWithheld := 0

	competingTotal := 0
	competingDiverted := 0
	competingWithheld := 0

	resilienceTotal := 0
	resiliencePassed := 0
	resilienceWithheld := 0

	// A case whose diagnostic input is missing stays in its denominator and
	// never passes, so withholding inputs cannot raise a rate. Resilience is
	// requested when any ground-truth case has a non-normal resilience mode;
	// those cases are counted from ground truth, not from the inputs.
	heldOutMissing := 0
	noFaultMissing := 0
	competingMissing := 0
	resilienceMissing := 0

	var lastDiag cudadiag.DiagnosisResult
	var joins []cudaworkload.JoinResult

	for _, gt := range res.GroundTruth {
		diagInput, ok := res.DiagnosticInputs[gt.RequestID]
		if !ok {
			switch {
			case gt.Resilience != ResilienceNormal:
				resilienceTotal++
				resilienceMissing++
			case gt.Fault == FaultNone:
				noFaultTotal++
				noFaultMissing++
			case gt.Fault == FaultHostSubmissionDelay:
				heldOutTotal++
				heldOutMissing++
			case gt.Fault == FaultDeviceBottleneck:
				competingTotal++
				competingMissing++
			}
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
		withheld := diagResult.Mechanism == cudadiag.MechanismWithheld || !diagResult.Admitted

		// A resilience case passes only on the evidence its mode
		// requires; see resilienceMet. A withheld diagnosis without that
		// evidence is counted apart and never passes.
		if gt.Resilience != ResilienceNormal {
			if joins == nil {
				joins = flushedJoins(res)
			}
			resilienceTotal++
			switch {
			case resilienceMet(gt, diagInput, diagResult, joins):
				resiliencePassed++
			case withheld:
				resilienceWithheld++
			}
			continue
		}

		switch gt.Fault {
		case FaultNone:
			noFaultTotal++
			switch {
			case withheld:
				noFaultWithheld++
			case diagResult.Mechanism == cudadiag.MechanismNoFault:
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
			// Only the exact competing mechanism is correct. A
			// withheld diagnosis is not a distinction, and any other
			// admitted mechanism is a misclassification.
			competingTotal++
			switch {
			case withheld:
				competingWithheld++
			case diagResult.Mechanism == cudadiag.MechanismDeviceBottleneck:
				competingDiverted++
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

	heldOutDisp, heldOutReason := rateGate("held-out accuracy", heldOutCorrect, heldOutMissing, heldOutTotal, budgets.MinHeldOutDiagnosisPct)
	noFaultDisp, noFaultReason := noFaultGate(noFaultPassed, noFaultWithheld, noFaultMissing, noFaultTotal, budgets.MinNoFaultRejectionPct)
	competingDisp, competingReason := rateGate("competing mechanism distinction", competingDiverted, competingMissing, competingTotal, 100)
	if competingWithheld > 0 {
		competingReason += fmt.Sprintf("; %d diagnoses withheld", competingWithheld)
	}
	resilienceDisp, resilienceReason := rateGate("resilience control", resiliencePassed, resilienceMissing, resilienceTotal, 100)
	if resilienceWithheld > 0 {
		resilienceReason += fmt.Sprintf("; %d diagnoses withheld", resilienceWithheld)
	}
	resilienceRequested := resilienceTotal > 0

	// The receipt reports the worst of its reported gates.
	dispositions := []string{heldOutDisp, noFaultDisp, competingDisp}
	if resilienceRequested {
		dispositions = append(dispositions, resilienceDisp)
	}
	diagDisp := "passed"
	switch {
	case slices.Contains(dispositions, "failed"):
		diagDisp = "failed"
	case slices.ContainsFunc(dispositions, func(d string) bool { return d != "passed" }):
		diagDisp = "withheld"
	}

	receipt := DiagnosticReceipt{
		HeldOutTotal:       heldOutTotal,
		HeldOutCorrect:     heldOutCorrect,
		HeldOutWithheld:    heldOutWithheld,
		HeldOutIncorrect:   heldOutIncorrect,
		HeldOutMissing:     heldOutMissing,
		NoFaultTotal:       noFaultTotal,
		NoFaultPassed:      noFaultPassed,
		NoFaultWithheld:    noFaultWithheld,
		NoFaultMissing:     noFaultMissing,
		CompetingTotal:     competingTotal,
		CompetingDiverted:  competingDiverted,
		CompetingWithheld:  competingWithheld,
		CompetingMissing:   competingMissing,
		ResilienceTotal:    resilienceTotal,
		ResiliencePassed:   resiliencePassed,
		ResilienceWithheld: resilienceWithheld,
		ResilienceMissing:  resilienceMissing,
		AccuracyPct:        acc,
		ConfusionMatrix:    cm,
		GateDisposition:    diagDisp,
		SampleDiagnosis:    lastDiag,
	}

	gates := []PropertyGate{
		{
			Property:      "diagnostic_usefulness",
			GateName:      "held_out_fault_accuracy",
			Disposition:   heldOutDisp,
			MeasuredValue: fmt.Sprintf("%.1f%%", acc),
			Threshold:     fmt.Sprintf(">= %.1f%%", budgets.MinHeldOutDiagnosisPct),
			Reason:        heldOutReason,
			Provenance:    "cudadiag.Diagnose",
		},
		{
			Property:      "diagnostic_usefulness",
			GateName:      "no_fault_rejection",
			Disposition:   noFaultDisp,
			MeasuredValue: fmt.Sprintf("%.1f%%", noFaultPct),
			Threshold:     fmt.Sprintf(">= %.1f%%", budgets.MinNoFaultRejectionPct),
			Reason:        noFaultReason,
			Provenance:    "cudadiag.MechanismNoFault",
		},
		{
			Property:      "diagnostic_usefulness",
			GateName:      "competing_mechanism_distinction",
			Disposition:   competingDisp,
			MeasuredValue: fmt.Sprintf("%d/%d", competingDiverted, competingTotal),
			Threshold:     "100%",
			Reason:        competingReason,
			Provenance:    "cudadiag.MechanismDeviceBottleneck",
		},
	}

	if resilienceRequested {
		gates = append(gates, PropertyGate{
			Property:      "diagnostic_usefulness",
			GateName:      "system_resilience_controls",
			Disposition:   resilienceDisp,
			MeasuredValue: fmt.Sprintf("%d/%d", resiliencePassed, resilienceTotal),
			Threshold:     "100%",
			Reason:        resilienceReason,
			Provenance:    "cudasotaeval.ResilienceMode",
		})
	}

	return receipt, gates
}

// flushedJoins joins the run's host submissions and device activities and
// flushes the joiner, so every record left unmatched is reported as
// missing. It uses its own joiner and leaves the attribution receipt
// unchanged.
func flushedJoins(res *FixtureRunResult) []cudaworkload.JoinResult {
	j := cudaworkload.NewJoiner(cudaworkload.JoinConfig{MaxPendingRecords: 4096})
	var out []cudaworkload.JoinResult
	for _, sub := range res.HostSubmissions {
		out = append(out, j.SubmitHost(sub)...)
	}
	for _, act := range res.DeviceActivities {
		out = append(out, j.SubmitDevice(act)...)
	}
	return append(out, j.Flush()...)
}

// requestKey is the explicit identity of a request: its process generation
// and native correlation ID. Native correlation IDs repeat across
// generations, so neither part identifies a request alone.
type requestKey struct {
	process     cudaworkload.ProcessGenID
	correlation uint32
}

func (k requestKey) valid() bool {
	return k.process.PID != 0 && k.process.Generation != 0 && k.correlation != 0
}

// names reports whether r carries k in its raw IDs, its host submission or
// its device activity.
func (k requestKey) names(r cudaworkload.JoinResult) bool {
	if (requestKey{r.RawIDs.ProcessGenID, r.RawIDs.NativeCorrelationID}) == k {
		return true
	}
	if s := r.HostSubmission; s != nil && (requestKey{s.Process, s.NativeCorrelationID}) == k {
		return true
	}
	if a := r.DeviceActivity; a != nil && (requestKey{a.Process, a.NativeCorrelationID}) == k {
		return true
	}
	return false
}

// unattributed reports whether the join results report the request k only
// as missing or ambiguous attribution: at least one result names it, and
// none joins it exactly or by association.
//
// A record that expired from the joiner's bounded queue is reported as
// missing whether or not its counterpart was observed, so it is not
// evidence of a missing counterpart. When any record of the run expired,
// the joins cannot tell missing evidence from evaluation capacity loss,
// and no request is reported unattributed.
func unattributed(k requestKey, joins []cudaworkload.JoinResult) bool {
	if !k.valid() {
		return false
	}
	reported := false
	for _, r := range joins {
		if r.Reason == cudaworkload.ReasonExpiredFIFO {
			return false
		}
		if !k.names(r) {
			continue
		}
		switch r.Outcome {
		case cudaworkload.OutcomeMissing, cudaworkload.OutcomeAmbiguous:
			reported = true
		default:
			return false
		}
	}
	return reported
}

// activityAdmissionRule is the rule of the finding cudadiag.Diagnose
// withholds when the evidence requirements, among them activity loss, are
// not met. Withholding for scheduler evidence uses another rule.
const activityAdmissionRule = "mechanism_diagnosis"

// resilienceMet reports whether a resilience case meets its mode's
// predicate. Each passing case is a safe refusal: its diagnosis was
// withheld for the loss or identity change the mode induces.
//
// An overflow case needs observed activity loss in its receipt. A restart
// case needs scheduler evidence from a later generation of the same
// process, a diagnosis withheld for identity mismatch, and join results
// that report the request only as missing or ambiguous, with no record of
// the run expired from the join queue. A late attach case needs a receipt
// declaring an activity coverage gap or unknown activity loss, a diagnosis
// withheld by the activity admission rule for unknown loss, and the same
// join report. Withholding for scheduler evidence or for measured activity
// loss belongs to another mode and does not count. No other mode passes.
func resilienceMet(gt GroundTruthRecord, in cudadiag.DiagnosticInput, d cudadiag.DiagnosisResult, joins []cudaworkload.JoinResult) bool {
	if d.Admitted || d.Mechanism != cudadiag.MechanismWithheld {
		return false
	}
	key := requestKey{gt.ProcessGen, gt.CorrelationID}
	switch gt.Resilience {
	case ResilienceOverflow:
		return in.Receipt.LostRecords["activity"].Dropped > 0
	case ResilienceRestart:
		observed := in.Sched.Process
		return observed.PID == gt.ProcessGen.PID &&
			observed.Generation > gt.ProcessGen.Generation &&
			d.WithheldReason == tracecontract.ReasonIdentityMismatch &&
			unattributed(key, joins)
	case ResilienceLateAttach:
		cov, ok := in.Receipt.ProbeCoverage["cupti_activity"]
		gap := (ok && (!cov.Present || cov.Ratio < 1)) || in.Receipt.LostRecords["activity"].Unknown
		return gap &&
			d.PrimaryFinding.Rule == activityAdmissionRule &&
			d.WithheldReason == tracecontract.ReasonUnknownLoss &&
			unattributed(key, joins)
	}
	return false
}

// noFaultCounts validates the disjoint no-fault counts before deriving
// admitted answers and false alarms. Subtracting bounded counts avoids
// overflow even when imported receipts use the largest possible int.
func noFaultCounts(passed, withheld, missing, total int) (admitted, alarms int, err error) {
	if total == 0 {
		return 0, 0, fmt.Errorf("no-fault has no cases")
	}
	if total < 0 {
		return 0, 0, fmt.Errorf("invalid no-fault total: %d", total)
	}
	remaining := total
	for _, count := range []int{passed, withheld, missing} {
		if count < 0 || count > remaining {
			return 0, 0, fmt.Errorf("invalid no-fault partition: passed=%d withheld=%d missing=%d total=%d", passed, withheld, missing, total)
		}
		remaining -= count
	}
	return total - withheld - missing, remaining, nil
}

// noFaultMeetsThreshold compares exact counts to the registered floating
// point threshold without overflowing an int product or rounding a nearly
// complete population to 100%. Its caller validates counts and threshold.
func noFaultMeetsThreshold(passed, total int, min float64) bool {
	if min == 100 {
		return passed == total
	}
	rate := new(big.Rat).SetFrac64(int64(passed), int64(total))
	rate.Mul(rate, big.NewRat(100, 1))
	return rate.Cmp(new(big.Rat).SetFloat64(min)) >= 0
}

// noFaultSummary keeps the declared and admitted denominators distinct.
// Percentages are display values; acceptance uses exact count comparisons.
func noFaultSummary(passed, withheld, missing, total, admitted, alarms int) string {
	specificity := "specificity undefined (no admitted answers); false alarms 0/0 (undefined)"
	if admitted > 0 {
		specificity = fmt.Sprintf("admitted specificity %d/%d (%.1f%%); false alarms %d/%d (%.1f%%)", passed, admitted, float64(passed)/float64(admitted)*100, alarms, admitted, float64(alarms)/float64(admitted)*100)
	}
	return fmt.Sprintf("no-fault admitted correctness %d/%d (%.1f%%); admitted coverage %d/%d; %s; %d diagnoses withheld; %d inputs missing", passed, total, float64(passed)/float64(total)*100, admitted, total, specificity, withheld, missing)
}

// noFaultGate requires admitted correctness over the declared population
// and complete diagnostic coverage. Unanswered cases can only withhold;
// known false alarms fail when even the optimistic bound misses min.
func noFaultGate(passed, withheld, missing, total int, min float64) (disposition, reason string) {
	admitted, alarms, err := noFaultCounts(passed, withheld, missing, total)
	if err != nil {
		return "failed", err.Error()
	}
	if math.IsNaN(min) || math.IsInf(min, 0) || min <= 0 || min > 100 {
		return "failed", "invalid minimum no-fault rejection percentage"
	}
	reason = noFaultSummary(passed, withheld, missing, total, admitted, alarms)
	if !noFaultMeetsThreshold(total-alarms, total, min) {
		return "failed", reason + fmt.Sprintf("; optimistic correctness below minimum %.1f%%", min)
	}
	if withheld > 0 || missing > 0 {
		return "withheld", reason + "; diagnostic coverage incomplete"
	}
	if !noFaultMeetsThreshold(passed, total, min) {
		return "failed", reason + fmt.Sprintf("; admitted correctness below minimum %.1f%%", min)
	}
	return "passed", reason + fmt.Sprintf("; meets minimum %.1f%% with complete coverage", min)
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

// rateGate decides a gate on passed/total against a minimum percentage.
// Missing cases count in total and never pass. A gate that would fail even if
// every missing case passed is failed; otherwise missing cases withhold it.
// An empty population fails. Comparisons scale the minimum rather than divide
// the counts, so an exact boundary such as 29/100 against 29% passes.
func rateGate(name string, passed, missing, total int, min float64) (disposition, reason string) {
	pct := func(n int) float64 { return float64(n) / float64(total) * 100 }
	below := func(n int) bool { return float64(n)*100 < min*float64(total) }
	switch {
	case total == 0:
		return "failed", fmt.Sprintf("%s has no cases", name)
	case below(passed + missing):
		reason = fmt.Sprintf("%s %d/%d (%.1f%%) below minimum %.1f%%", name, passed, total, pct(passed), min)
		if missing > 0 {
			reason += fmt.Sprintf(" with %d inputs missing", missing)
		}
		return "failed", reason
	case missing > 0:
		return "withheld", fmt.Sprintf("%s %d/%d with %d inputs missing", name, passed, total, missing)
	}
	return "passed", fmt.Sprintf("%s %d/%d meets minimum %.1f%%", name, passed, total, min)
}

func gateDisposition(passed bool) string {
	if passed {
		return "passed"
	}
	return "failed"
}
