package cudaflightrecorder

import (
	"context"
	"errors"
	"fmt"
)

// ErrEscalationQueueFull indicates that no bounded pending-job slot is available.
// The request was not admitted and will not produce a completion result.
var ErrEscalationQueueFull = errors.New("cudaflightrecorder: escalation queue full")

// EscalationResult is the terminal result of one asynchronously submitted trigger.
// A failed capture may have both a receipt and an error.
type EscalationResult struct {
	Receipt *IncidentReceipt
	Err     error
}

type escalationJob struct {
	ctx     context.Context
	request TriggerRequest
	result  chan EscalationResult
}

// TriggerAsync submits a trigger without waiting for resource sampling, native
// acquisition, or snapshot export. At most Recorder.Config.MaxQueuedJobs requests
// are pending, in addition to one executing request. Queue admission does not
// imply capture admission: the worker evaluates the normal Trigger budgets.
// Request details are copied; the caller must not mutate them during submission.
// Each accepted request receives exactly one buffered result and a closed channel,
// even when stopped or canceled. The receiver need not consume it for progress.
// Call Stop and Wait when finished to release the lazily started worker.
func (ec *EscalationCoordinator) TriggerAsync(ctx context.Context, req TriggerRequest) (<-chan EscalationResult, error) {
	if ctx == nil {
		return nil, errors.New("cudaflightrecorder: nil trigger context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	req, err := copyTriggerRequest(req)
	if err != nil {
		return nil, err
	}
	ec.mu.Lock()
	defer ec.mu.Unlock()
	if ec.closed {
		return nil, fmt.Errorf("%w: coordinator is stopped", ErrEscalationRefused)
	}
	if ec.jobs == nil {
		ec.jobs = make(chan escalationJob, ec.recorder.cfg.MaxQueuedJobs)
		go ec.runJobs(ec.jobs)
	}
	result := make(chan EscalationResult, 1)
	job := escalationJob{ctx: ctx, request: req, result: result}
	select {
	case ec.jobs <- job:
		return result, nil
	default:
		return nil, ErrEscalationQueueFull
	}
}

func (ec *EscalationCoordinator) runJobs(jobs <-chan escalationJob) {
	defer close(ec.jobsDone)
	for job := range jobs {
		receipt, err := ec.Trigger(job.ctx, job.request)
		job.result <- EscalationResult{Receipt: receipt, Err: err}
		close(job.result)
	}
}

// Wait waits for all accepted asynchronous jobs to finish after Stop. It does
// not retry retained native cleanup or wait for separately invoked synchronous
// Trigger calls. Providers and writers must cooperate with context cancellation.
// The caller may time out without preventing later worker completion.
func (ec *EscalationCoordinator) Wait(ctx context.Context) error {
	if ctx == nil {
		return errors.New("cudaflightrecorder: nil wait context")
	}
	ec.mu.Lock()
	stopped := ec.closed
	done := ec.jobsDone
	ec.mu.Unlock()
	if !stopped {
		return errors.New("cudaflightrecorder: stop before waiting for escalation jobs")
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func copyTriggerRequest(req TriggerRequest) (TriggerRequest, error) {
	if len(req.TenantID) > 1024 || len(req.Reason) > 1024 || len(req.MetricName) > 1024 || len(req.Details) > 64 {
		return TriggerRequest{}, errors.New("cudaflightrecorder: trigger metadata exceeds bounds")
	}
	details := make(map[string]string, len(req.Details))
	for key, value := range req.Details {
		if len(key) > 4096 || len(value) > 4096 {
			return TriggerRequest{}, errors.New("cudaflightrecorder: trigger detail exceeds bounds")
		}
		details[key] = value
	}
	req.Details = details
	return req, nil
}

func copyIncidentReceipt(receipt IncidentReceipt) IncidentReceipt {
	details := make(map[string]string, len(receipt.Trigger.Details))
	for key, value := range receipt.Trigger.Details {
		details[key] = value
	}
	receipt.Trigger.Details = details
	receipt.Trigger.PreTrigger.MissingPreTriggerFamilies = append([]string(nil), receipt.Trigger.PreTrigger.MissingPreTriggerFamilies...)
	return receipt
}
