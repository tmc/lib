package cudasearch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudareg"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// RegisterOption binds a published schedule to an executable identity. Identity
// must identify the actual image, entry, target and complete launch/schedule,
// not just the expression. The compiler must return that exact identity.
// Neither the identity nor the resource facts authenticate a compiler run.
type RegisterOption struct {
	Identity  string
	Candidate cudareg.Candidate
}

// RegisterSearchLimits bounds a finite register search. Both limits must be
// positive. MaxResourceBytes bounds conservative accounting for the copied
// options, including compiler argument copies; it is not a process heap limit.
// Executable work, buffers and timing remain subject to ExecutableEvaluator's
// contracts, CompileTimeout and the caller's context and hardware lease.
type RegisterSearchLimits struct {
	MaxCandidates    int
	MaxResourceBytes int
}

// RegisterSearchResult owns the best measured plan so far and its detached
// register evidence. Evaluations counts attempts, including a failed attempt.
// A nonnil result may accompany an error; it is not a promotion approval.
type RegisterSearchResult struct {
	Winner      *CandidatePlan
	Registers   cudareg.Candidate
	Evaluations int
}

// Close releases the winning plan and its register metadata. It does not close
// the evaluator or its borrowed baseline. It is safe to call more than once.
func (r *RegisterSearchResult) Close() {
	if r != nil {
		r.Winner.Close()
		r.Winner = nil
		r.Registers = cudareg.Candidate{}
	}
}

// SearchRegisters evaluates caller-supplied finite register alternatives using
// the existing numerical checks, ABBA timing and executable cleanup. It does
// not enumerate schedules or use register counts as latency estimates. Ties
// retain the first option. A failure or cancellation stops further evaluation
// and returns the best admitted result, if any, with the error.
//
// All options are checked and copied before compilation. Only canonical
// ProtocolFeasible publications with known allocation and numeric target are
// admitted. This does not establish suffix support, occupancy or native
// acceptance; the original facts and target caveats are retained in Registers.
// Compile receives a detached option and must implement its actual schedule,
// honor the context, and return any partial owner with its error. Its identity
// must equal the option identity. Failed cleanup remains owned by e until
// e.Close succeeds; result.Close does not release that owner.
//
// Like Evaluate, this method is not safe for concurrent use or mutation of e.
// Expressions and their payloads follow the same borrowing contract as Confirm.
// Compile is restored before return. Held-out confirmation remains separate.
func (e *ExecutableEvaluator) SearchRegisters(ctx context.Context, expr *egraph.Expr, options []RegisterOption, limits RegisterSearchLimits, compile func(context.Context, *egraph.Expr, WorkloadBindings, RegisterOption) (Executable, error)) (*RegisterSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e == nil || compile == nil || limits.MaxCandidates <= 0 || limits.MaxResourceBytes <= 0 || len(options) == 0 || len(options) > limits.MaxCandidates {
		return nil, fmt.Errorf("cudasearch: invalid register search bounds or compiler")
	}
	// Check the size before PublishCandidate copies any caller-controlled map
	// or slice. Charge three copies: input snapshot, compiler argument, winner.
	remaining := limits.MaxResourceBytes
	for _, option := range options {
		if err := registerOptionSize(ctx, option, &remaining); err != nil {
			return nil, err
		}
	}
	snapshots := make([]RegisterOption, 0, len(options))
	identities := make(map[string]bool, len(options))
	for _, option := range options {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if strings.TrimSpace(option.Identity) == "" || strings.ContainsRune(option.Identity, 0) || identities[option.Identity] {
			return nil, fmt.Errorf("cudasearch: missing or duplicate register executable identity")
		}
		identities[option.Identity] = true
		canonical, err := cudareg.PublishCandidate(option.Candidate.Resources, option.Candidate.Schedule)
		if err != nil {
			return nil, fmt.Errorf("cudasearch: register option %q: %w", option.Identity, err)
		}
		if canonical.Status != cudareg.ProtocolFeasible || !reflect.DeepEqual(canonical, option.Candidate) {
			return nil, fmt.Errorf("cudasearch: register option %q is not a canonical feasible publication", option.Identity)
		}
		detachRegisterStrings(&canonical)
		snapshots = append(snapshots, RegisterOption{Identity: strings.Clone(option.Identity), Candidate: canonical})
	}
	if _, err := confirmationExpr(ctx, expr); err != nil {
		return nil, err
	}
	previous := e.Compile
	defer func() { e.Compile = previous }()
	result := &RegisterSearchResult{}
	for _, option := range snapshots {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		selection, err := confirmationExpr(ctx, expr)
		if err != nil {
			return result, err
		}
		e.Compile = func(ctx context.Context, expr *egraph.Expr, workload WorkloadBindings) (Executable, error) {
			publication, err := cudareg.PublishCandidate(option.Candidate.Resources, option.Candidate.Schedule)
			if err != nil {
				return nil, err
			}
			x, err := compile(ctx, expr, workload, RegisterOption{Identity: option.Identity, Candidate: publication})
			if x != nil && x.Identity() != option.Identity {
				err = errors.Join(err, fmt.Errorf("cudasearch: register executable identity mismatch"))
			}
			return x, err
		}
		result.Evaluations++
		plan, err := e.evaluate(ctx, nil, 0, selection, nil)
		if err != nil {
			return result, err
		}
		if err := admitSearchCandidate(plan); err != nil {
			plan.Close()
			return result, err
		}
		if result.Winner != nil {
			if err := admitComparableCandidate(plan, result.Winner); err != nil {
				plan.Close()
				return result, err
			}
		}
		if result.Winner == nil || plan.Cost < result.Winner.Cost {
			result.Winner.Close()
			result.Winner, result.Registers = plan, option.Candidate
		} else {
			plan.Close()
		}
	}
	return result, nil
}

// PublishCandidate copies containers but borrows immutable strings. Detach
// those too: a short substring must not retain an unaccounted caller buffer.
// c's maps, slices and Resources already belong to this search.
func detachRegisterStrings(c *cudareg.Candidate) {
	c.Status = cudareg.PublicationStatus(strings.Clone(string(c.Status)))
	c.Reason = strings.Clone(c.Reason)
	c.TargetReason = strings.Clone(c.TargetReason)
	c.Schedule.Target.Name = strings.Clone(c.Schedule.Target.Name)
	groups := make(map[string]cudareg.Budget, len(c.Schedule.Groups))
	for name, budget := range c.Schedule.Groups {
		groups[strings.Clone(name)] = budget
	}
	c.Schedule.Groups = groups
	for i := range c.Schedule.Steps {
		c.Schedule.Steps[i].Role = strings.Clone(c.Schedule.Steps[i].Role)
	}
	a := &c.TargetEvidence
	a.Target, a.Feature = strings.Clone(a.Target), strings.Clone(a.Feature)
	a.Source, a.Diagnostic = strings.Clone(a.Source), strings.Clone(a.Diagnostic)
	if r := c.Resources; r != nil {
		r.ImageDigest, r.Entry = strings.Clone(r.ImageDigest), strings.Clone(r.Entry)
		r.Target, r.DecodedTarget = strings.Clone(r.Target), strings.Clone(r.DecodedTarget)
		for _, f := range []*cudareg.Fact{&r.Allocated, &r.PeakLive, &r.Cap, &r.Modeled} {
			f.Status = cudareg.FactStatus(strings.Clone(string(f.Status)))
			f.Reason = strings.Clone(f.Reason)
		}
	}
}

// Account bounded metadata before copying it. The fixed charges exceed the
// inline struct/map-entry storage; strings count their complete byte lengths.
func registerOptionSize(ctx context.Context, o RegisterOption, remaining *int) error {
	charge := func(n int) error {
		if n > *remaining/3 {
			return fmt.Errorf("cudasearch: register resource byte limit exceeded")
		}
		*remaining -= 3 * n
		return nil
	}
	if err := charge(2048); err != nil {
		return err
	}
	a := o.Candidate.TargetEvidence
	for _, s := range []string{o.Identity, o.Candidate.Reason, o.Candidate.TargetReason, o.Candidate.Schedule.Target.Name, string(o.Candidate.Status), a.Target, a.Feature, a.Source, a.Diagnostic} {
		if err := charge(len(s)); err != nil {
			return err
		}
	}
	if r := o.Candidate.Resources; r != nil {
		for _, s := range []string{r.ImageDigest, r.Entry, r.Target, r.DecodedTarget, r.Allocated.Reason, r.PeakLive.Reason, r.Cap.Reason, r.Modeled.Reason, string(r.Allocated.Status), string(r.PeakLive.Status), string(r.Cap.Status), string(r.Modeled.Status)} {
			if err := charge(len(s)); err != nil {
				return err
			}
		}
	}
	for range o.Candidate.Schedule.Target.LegalCounts {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := charge(8); err != nil {
			return err
		}
	}
	for name := range o.Candidate.Schedule.Groups {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := charge(256); err != nil {
			return err
		}
		if err := charge(len(name)); err != nil {
			return err
		}
	}
	for _, step := range o.Candidate.Schedule.Steps {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := charge(128); err != nil {
			return err
		}
		if err := charge(len(step.Role)); err != nil {
			return err
		}
	}
	return nil
}
