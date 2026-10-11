//go:build darwin

package metaltensor

import (
	"fmt"
	"sort"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/egraph"
)

// TensorTuningOptions bounds candidate search and controls device measurements.
type TensorTuningOptions struct {
	Search egraph.CandidateSearchConfig
	// WarmupRuns must be nonnegative. Zero selects one required untimed run so
	// pipeline compilation does not enter the steady-state latency score.
	WarmupRuns      int
	MeasurementRuns int
}

// TensorTuningResult contains the first hard-validated measured winner and the
// measurements collected during search. The caller owns Executor and must close
// it when the selected plan will no longer be executed.
type TensorTuningResult struct {
	SelectedID       string
	SelectedScoreNS  float64
	SelectedPlan     *TensorExecutionPlan
	Executor         *TensorPlanExecutor
	CandidateResults []egraph.CandidateResult
	FinalistResults  []egraph.CandidateFinalistResult
	SearchStats      egraph.CandidateSearchStats
}

// TensorPlanValidator performs caller-owned hard validation of a candidate's
// output. Returning nil accepts the candidate; returning an error rejects it.
type TensorPlanValidator func(plan *TensorExecutionPlan, output apple.MTLBuffer) error

// TuneTensorCandidates measures lowered candidates on dev, then validates
// finalists with validate. It returns a reusable executor containing only the
// winner's compiled pipelines. Scores are median end-to-end execution latency
// in nanoseconds, including intermediate allocation and command submission.
// Candidate deadlines are checked between synchronous executions; a committed
// Metal command buffer always runs to a terminal status before its buffers are
// released.
func TuneTensorCandidates(dev *metal.Device, space *TensorCandidateSpace, inputs map[TensorBufferRef]apple.MTLBuffer, options TensorTuningOptions, validate TensorPlanValidator) (*TensorTuningResult, error) {
	if space == nil || space.snapshot == nil {
		return nil, fmt.Errorf("tensor tuning: nil candidate space")
	}
	if validate == nil {
		return nil, fmt.Errorf("tensor tuning: nil finalist validator")
	}
	if options.WarmupRuns < 0 || options.MeasurementRuns <= 0 {
		return nil, fmt.Errorf("tensor tuning: warmup runs must be nonnegative and measurement runs must be positive")
	}
	if options.WarmupRuns == 0 {
		options.WarmupRuns = 1
	}
	if options.Search.Finalists <= 0 {
		return nil, fmt.Errorf("tensor tuning: finalist count must be positive")
	}
	options.Search.ScoreDirection = egraph.CandidateScoreAscending
	search, err := space.NewSearch(options.Search)
	if err != nil {
		return nil, fmt.Errorf("tensor tuning: create candidate search: %w", err)
	}

	for {
		candidate, nextErr := search.Next()
		if nextErr != nil {
			return nil, fmt.Errorf("tensor tuning: get candidate: %w", nextErr)
		}
		if candidate == nil {
			break
		}
		plan, lowerErr := lowerTensorCandidateExpr(candidate.Expr, space.dtype.String(), nil, "tensor candidate")
		if lowerErr != nil {
			if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateInvalid, Reason: lowerErr.Error()}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report invalid candidate: %w", err)
			}
			continue
		}
		executor, executorErr := NewTensorPlanExecutor(dev)
		if executorErr != nil {
			return nil, executorErr
		}
		invalid := ""
		timedOut := false
		for i := 0; i < options.WarmupRuns; i++ {
			if tensorCandidateExpired(candidate) {
				timedOut = true
				break
			}
			result, runErr := executor.Execute(plan, inputs)
			if runErr != nil {
				if tensorCandidateExpired(candidate) {
					timedOut = true
				} else {
					invalid = runErr.Error()
				}
				break
			}
			_ = result.Close()
			if tensorCandidateExpired(candidate) {
				timedOut = true
				break
			}
		}
		if timedOut {
			_ = executor.Close()
			if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateTimedOut, Reason: "candidate deadline expired during warmup"}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report timed out candidate: %w", err)
			}
			continue
		}
		if invalid != "" {
			_ = executor.Close()
			if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateInvalid, Reason: invalid}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report unexecutable candidate: %w", err)
			}
			continue
		}
		samples := make([]float64, 0, options.MeasurementRuns)
		for i := 0; i < options.MeasurementRuns; i++ {
			if tensorCandidateExpired(candidate) {
				timedOut = true
				break
			}
			started := time.Now()
			result, runErr := executor.Execute(plan, inputs)
			elapsed := time.Since(started)
			if runErr != nil {
				if tensorCandidateExpired(candidate) {
					timedOut = true
				} else {
					invalid = runErr.Error()
				}
				break
			}
			_ = result.Close()
			if tensorCandidateExpired(candidate) {
				timedOut = true
				break
			}
			samples = append(samples, float64(elapsed.Nanoseconds()))
		}
		if timedOut {
			_ = executor.Close()
			if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateTimedOut, Reason: "candidate deadline expired during measurement"}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report timed out measurement: %w", err)
			}
			continue
		}
		if invalid != "" {
			_ = executor.Close()
			if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateInvalid, Reason: invalid}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report failed candidate: %w", err)
			}
			continue
		}
		sort.Float64s(samples)
		score := samples[len(samples)/2]
		if len(samples)%2 == 0 {
			score = (samples[len(samples)/2-1] + samples[len(samples)/2]) / 2
		}
		if err := search.Report(candidate.ID, egraph.CandidateReport{Kind: egraph.CandidateMeasured, Score: score}); err != nil {
			_ = executor.Close()
			return nil, fmt.Errorf("tensor tuning: report measurement: %w", err)
		}
		if err := executor.Close(); err != nil {
			return nil, fmt.Errorf("tensor tuning: release measured candidate pipelines: %w", err)
		}
	}

	result := &TensorTuningResult{
		CandidateResults: search.Results(),
		SearchStats:      search.Stats(),
	}
	for {
		finalist, nextErr := search.NextFinalist()
		if nextErr != nil {
			return nil, fmt.Errorf("tensor tuning: get finalist: %w", nextErr)
		}
		if finalist == nil {
			break
		}
		plan, lowerErr := LowerTensorExecutionPlan(finalist.Expr, nil)
		if lowerErr != nil {
			if err := search.ReportFinalist(finalist.ID, egraph.CandidateFinalistReport{Reason: lowerErr.Error()}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report unlowerable finalist: %w", err)
			}
			continue
		}
		executor, executorErr := NewTensorPlanExecutor(dev)
		if executorErr != nil {
			return nil, executorErr
		}
		execution, runErr := executor.Execute(plan, inputs)
		if runErr != nil {
			_ = executor.Close()
			if err := search.ReportFinalist(finalist.ID, egraph.CandidateFinalistReport{Reason: runErr.Error()}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report unexecutable finalist: %w", err)
			}
			continue
		}
		validationErr := validate(plan, execution.Output)
		_ = execution.Close()
		if validationErr != nil {
			_ = executor.Close()
			if err := search.ReportFinalist(finalist.ID, egraph.CandidateFinalistReport{Reason: validationErr.Error()}); err != nil {
				return nil, fmt.Errorf("tensor tuning: report rejected finalist: %w", err)
			}
			continue
		}
		if err := search.ReportFinalist(finalist.ID, egraph.CandidateFinalistReport{Accepted: true}); err != nil {
			_ = executor.Close()
			return nil, fmt.Errorf("tensor tuning: report accepted finalist: %w", err)
		}
		result.SelectedID = finalist.ID
		result.SelectedScoreNS = finalist.Score
		result.SelectedPlan = plan
		result.Executor = executor
		break
	}
	result.FinalistResults = search.FinalistResults()
	result.SearchStats = search.Stats()
	return result, nil
}

func tensorCandidateExpired(candidate *egraph.SearchCandidate) bool {
	return candidate != nil && !candidate.Deadline.IsZero() && !time.Now().Before(candidate.Deadline)
}
