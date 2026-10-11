package metaltensor

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

// TensorBucketCandidate is a measured, hard-validated candidate for one
// tensor shape bucket. RawScore preserves the evaluator's score direction;
// LatticeScore is finite and ascending for use with CandidateLattice.
type TensorBucketCandidate struct {
	Bucket     egraph.BucketContext
	RawScore   float64
	FinalistID string
	Expr       *egraph.Expr
}

// TensorBucketEvaluator measures one candidate. The context is canceled at
// the candidate deadline. Expr fields are defensive copies, but Data payloads
// are shared and must be treated as immutable.
type TensorBucketEvaluator func(context.Context, egraph.BucketContext, *egraph.SearchCandidate) egraph.CandidateReport

// TensorBucketValidator performs hard validation for the whole bucket,
// including binding runtime dimensions. Representative-point measurements
// only rank candidates and do not establish correctness throughout a bucket.
// Expr fields are defensive copies, but Data payloads are shared and must be
// treated as immutable.
type TensorBucketValidator func(context.Context, egraph.BucketContext, *egraph.CandidateFinalist) error

// TensorBucketSearchStats reports candidate search and callback outcomes.
type TensorBucketSearchStats struct {
	Search          egraph.CandidateSearchStats
	EvaluatorCalls  int
	EvaluatorPanics int
	ValidatorCalls  int
	ValidatorPanics int
	Accepted        int
	Rejected        int
	TimedOut        int
	FinalistResults []egraph.CandidateFinalistResult
}

// SearchTensorBucketCandidates measures and validates candidates from one
// prepared bucket space. It reports every yielded candidate and finalist once,
// records callback panics as rejections, and continues after rejected
// finalists. Callers can combine the returned ascending LatticeScore values
// across buckets with egraph.CandidateLattice.
func SearchTensorBucketCandidates(
	ctx context.Context,
	space *TensorBucketCandidateSpace,
	config egraph.CandidateSearchConfig,
	evaluate TensorBucketEvaluator,
	validate TensorBucketValidator,
) ([]egraph.CandidateLatticeFinalist[TensorBucketCandidate], TensorBucketSearchStats, error) {
	var stats TensorBucketSearchStats
	if ctx == nil {
		return nil, stats, fmt.Errorf("tensor bucket search: nil context")
	}
	if space == nil {
		return nil, stats, fmt.Errorf("tensor bucket search: nil candidate space")
	}
	if evaluate == nil {
		return nil, stats, fmt.Errorf("tensor bucket search: nil evaluator")
	}
	if validate == nil {
		return nil, stats, fmt.Errorf("tensor bucket search: nil validator")
	}
	if err := ctx.Err(); err != nil {
		return nil, stats, fmt.Errorf("tensor bucket search: %w", err)
	}
	search, err := space.NewSearch(config)
	if err != nil {
		return nil, stats, fmt.Errorf("tensor bucket search: create search: %w", err)
	}
	bucket := space.BucketContext()
	for {
		if err := ctx.Err(); err != nil {
			return nil, stats, fmt.Errorf("tensor bucket search: %w", err)
		}
		candidate, err := search.Next()
		if err != nil {
			return nil, stats, fmt.Errorf("tensor bucket search: next candidate: %w", err)
		}
		if candidate == nil {
			break
		}
		report, panicked := callTensorBucketEvaluator(ctx, candidate.Deadline, evaluate, cloneBucketContext(bucket), cloneSearchCandidate(candidate))
		stats.EvaluatorCalls++
		if panicked {
			stats.EvaluatorPanics++
			report = egraph.CandidateReport{Kind: egraph.CandidateRejected, Reason: "evaluator panicked"}
		}
		if err := search.Report(candidate.ID, report); err != nil {
			return nil, stats, fmt.Errorf("tensor bucket search: report candidate %s: %w", candidate.ID, err)
		}
	}

	var finalists []egraph.CandidateLatticeFinalist[TensorBucketCandidate]
	for {
		if err := ctx.Err(); err != nil {
			return finalists, stats, fmt.Errorf("tensor bucket finalist validation: %w", err)
		}
		finalist, err := search.NextFinalist()
		if err != nil {
			return finalists, stats, fmt.Errorf("tensor bucket search: next finalist: %w", err)
		}
		if finalist == nil {
			break
		}
		stats.ValidatorCalls++
		validationErr, panicked := callTensorBucketValidator(ctx, finalist.Deadline, validate, cloneBucketContext(bucket), cloneCandidateFinalist(finalist))
		report := egraph.CandidateFinalistReport{}
		if panicked {
			stats.ValidatorPanics++
			report = egraph.CandidateFinalistReport{Reason: "validator panicked"}
		} else if validationErr != nil {
			report = egraph.CandidateFinalistReport{Reason: strings.TrimSpace(validationErr.Error())}
			if report.Reason == "" {
				report.Reason = "validator rejected finalist"
			}
		} else {
			report.Accepted = true
		}
		if !finalist.Deadline.IsZero() && !time.Now().Before(finalist.Deadline) {
			report.Accepted = false
			report.TimedOut = true
			report.Reason = "candidate timeout expired during finalist validation"
		}
		if err := search.ReportFinalist(finalist.ID, report); err != nil {
			return finalists, stats, fmt.Errorf("tensor bucket search: report finalist %s: %w", finalist.ID, err)
		}
		// ReportFinalist applies its own deadline check. Read the recorded result
		// so a deadline that expires between our check and the report cannot
		// leak a timed-out finalist into the lattice.
		reports := search.FinalistResults()
		if len(reports) == 0 {
			return finalists, stats, fmt.Errorf("tensor bucket search: finalist %s has no recorded report", finalist.ID)
		}
		recorded := reports[len(reports)-1].Report
		if recorded.Accepted {
			rawScore := finalist.Score
			latticeScore := rawScore
			if config.ScoreDirection == egraph.CandidateScoreDescending {
				latticeScore = -rawScore
			}
			if math.IsNaN(latticeScore) || math.IsInf(latticeScore, 0) {
				return finalists, stats, fmt.Errorf("tensor bucket search: finalist %s has non-finite normalized score", finalist.ID)
			}
			finalists = append(finalists, egraph.CandidateLatticeFinalist[TensorBucketCandidate]{
				Score: latticeScore,
				Value: TensorBucketCandidate{Bucket: cloneBucketContext(bucket), RawScore: rawScore, FinalistID: finalist.ID, Expr: cloneExpr(finalist.Expr)},
			})
		}
	}
	stats.Search = search.Stats()
	stats.FinalistResults = search.FinalistResults()
	for _, result := range stats.FinalistResults {
		if result.Report.Accepted {
			stats.Accepted++
		} else {
			stats.Rejected++
		}
		if result.Report.TimedOut {
			stats.TimedOut++
		}
	}
	return finalists, stats, nil
}

func callTensorBucketEvaluator(parent context.Context, deadline time.Time, fn TensorBucketEvaluator, bucket egraph.BucketContext, candidate *egraph.SearchCandidate) (report egraph.CandidateReport, panicked bool) {
	ctx, cancel := callbackContext(parent, deadline)
	defer cancel()
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	return fn(ctx, bucket, candidate), false
}

func callTensorBucketValidator(parent context.Context, deadline time.Time, fn TensorBucketValidator, bucket egraph.BucketContext, finalist *egraph.CandidateFinalist) (err error, panicked bool) {
	ctx, cancel := callbackContext(parent, deadline)
	defer cancel()
	defer func() {
		if recover() != nil {
			err = nil
			panicked = true
		}
	}()
	return fn(ctx, bucket, finalist), false
}

func callbackContext(parent context.Context, deadline time.Time) (context.Context, context.CancelFunc) {
	if deadline.IsZero() {
		return context.WithCancel(parent)
	}
	return context.WithDeadline(parent, deadline)
}

func cloneBucketContext(bucket egraph.BucketContext) egraph.BucketContext {
	copy := egraph.BucketContext{Index: bucket.Index}
	if bucket.BucketIndices != nil {
		copy.BucketIndices = make(map[string]int, len(bucket.BucketIndices))
		for key, value := range bucket.BucketIndices {
			copy.BucketIndices[key] = value
		}
	}
	if bucket.Intervals != nil {
		copy.Intervals = make(map[string]egraph.DimensionInterval, len(bucket.Intervals))
		for key, value := range bucket.Intervals {
			copy.Intervals[key] = value
		}
	}
	return copy
}

func cloneSearchCandidate(candidate *egraph.SearchCandidate) *egraph.SearchCandidate {
	if candidate == nil {
		return nil
	}
	copy := *candidate
	copy.Expr = cloneExpr(candidate.Expr)
	if candidate.EarlyStop != nil {
		hint := *candidate.EarlyStop
		copy.EarlyStop = &hint
	}
	copy.Choice = cloneCandidateChoice(candidate.Choice)
	return &copy
}

func cloneCandidateFinalist(finalist *egraph.CandidateFinalist) *egraph.CandidateFinalist {
	if finalist == nil {
		return nil
	}
	copy := *finalist
	copy.Expr = cloneExpr(finalist.Expr)
	copy.Choice = cloneCandidateChoice(finalist.Choice)
	return &copy
}

func cloneCandidateChoice(choice egraph.CandidateChoice) egraph.CandidateChoice {
	if choice == nil {
		return nil
	}
	copy := make(egraph.CandidateChoice, len(choice))
	for id, index := range choice {
		copy[id] = index
	}
	return copy
}

func cloneExpr(expr *egraph.Expr) *egraph.Expr {
	return cloneExprMemo(expr, make(map[*egraph.Expr]*egraph.Expr))
}

func cloneExprMemo(expr *egraph.Expr, seen map[*egraph.Expr]*egraph.Expr) *egraph.Expr {
	if expr == nil {
		return nil
	}
	if copy, ok := seen[expr]; ok {
		return copy
	}
	copy := &egraph.Expr{Op: expr.Op, Sort: expr.Sort, Data: expr.Data}
	seen[expr] = copy
	copy.Children = make([]*egraph.Expr, len(expr.Children))
	for i, child := range expr.Children {
		copy.Children[i] = cloneExprMemo(child, seen)
	}
	return copy
}
