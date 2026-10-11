package cudaflightrecorder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// IncidentCapture starts additional acquisition for an admitted incident. The
// coordinator has reserved its cooperating device lease before calling Capture.
// A provider retaining native cleanup ownership must retain releaseLease and
// report RetainedOwnership; it must release the lease only after cleanup succeeds.
// A provider must not synchronize or otherwise alter application GPU work.
type IncidentCapture interface {
	Capture(ctx context.Context, recorder *Recorder, request TriggerRequest, releaseLease func()) (CaptureReport, error)
}

// CaptureReport describes acquisition separately from snapshot serialization.
// Started means native startup succeeded, not that complete GPU work was seen.
type CaptureReport struct {
	Provider          string        `json:"provider"`
	Started           bool          `json:"started"`
	Closed            bool          `json:"closed"`
	RetainedOwnership bool          `json:"retained_ownership"`
	Duration          time.Duration `json:"duration"`
	Records           int           `json:"records"`
	PendingBuffers    int           `json:"pending_buffers"`
	RefusedBuffers    uint64        `json:"refused_buffers"`
	IncompleteRecords uint64        `json:"incomplete_records"`
	DiscardedRecords  uint64        `json:"discarded_records"`
	DroppedRecords    uint64        `json:"dropped_records"`
	CallbackErrors    uint64        `json:"callback_errors"`
	Error             string        `json:"error,omitempty"`
}

// WithIncidentCapture configures additional acquisition. Without this option,
// Trigger only snapshots observations already in the recorder.
func WithIncidentCapture(provider IncidentCapture) EscalationOption {
	return func(ec *EscalationCoordinator) { ec.captureProvider = provider }
}

// CUPTIActivityCaptureConfig bounds a process-wide activity session.
// ExclusiveOwnership acknowledges that the application owns activity callback
// registration and enable/disable APIs, including other profiling libraries.
// Device leases coordinate cooperating collectors only; they cannot establish
// this process-wide ownership. CollectionWindow must leave time for draining and
// snapshot serialization within the coordinator's MaxCaptureDuration.
type CUPTIActivityCaptureConfig struct {
	Activities         cuptiactivity.Config
	CollectionWindow   time.Duration
	ExclusiveOwnership bool
}

type incidentActivitySession interface {
	Close() error
	Result() cuptiactivity.Result
}

// CUPTIActivityCapture starts and drains real CUPTI activity acquisition. It
// preserves cuptiactivity's version/layout admission checks. It collects activity
// for the current process, including other device ordinals, and does not attach
// to another process. Application work must already be running. Close does not
// synchronize that work; incomplete records are reported, never silently repaired.
// A capture with losses returns an error and retains its partial raw evidence.
// Do not copy a CUPTIActivityCapture after use.
type CUPTIActivityCapture struct {
	mu      sync.Mutex
	config  CUPTIActivityCaptureConfig
	start   func(cuptiactivity.Config) (incidentActivitySession, error)
	pending incidentActivitySession
	release func()
}

// NewCUPTIActivityCapture validates the ownership and memory contract. Native
// capability and record-layout admission happen during Capture before collection.
func NewCUPTIActivityCapture(config CUPTIActivityCaptureConfig) (*CUPTIActivityCapture, error) {
	if !config.ExclusiveOwnership {
		return nil, errors.New("cudaflightrecorder: exclusive process-wide CUPTI activity ownership not established")
	}
	if config.CollectionWindow <= 0 {
		return nil, errors.New("cudaflightrecorder: activity collection window must be positive")
	}
	a := config.Activities
	if len(a.Kinds) == 0 || a.MaxRecords <= 0 || a.MaxBuffers <= 0 || a.BufferBytes <= 0 || a.BufferBytes%8 != 0 {
		return nil, errors.New("cudaflightrecorder: invalid activity collection bounds")
	}
	config.Activities.Kinds = append(config.Activities.Kinds[:0:0], config.Activities.Kinds...)
	return &CUPTIActivityCapture{config: config, start: func(c cuptiactivity.Config) (incidentActivitySession, error) { return cuptiactivity.Start(c) }}, nil
}

// Capture starts acquisition, waits a bounded window, drains the owned session,
// and attaches raw records and loss status as incident-activity.json. Host clock
// calibration is inherited from Recorder; unknown alignment remains unknown.
// On cleanup failure, call Close explicitly to retry cleanup and release the
// retained cooperating device lease. There is no automatic cleanup retry.
func (p *CUPTIActivityCapture) Capture(ctx context.Context, rec *Recorder, req TriggerRequest, releaseLease func()) (report CaptureReport, captureErr error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	report.Provider = "cupti_activity"
	if p.pending != nil {
		return report, errors.New("cudaflightrecorder: previous activity session still owns cleanup")
	}
	if rec == nil || releaseLease == nil {
		return report, errors.New("cudaflightrecorder: missing activity capture dependency")
	}
	if err := ctx.Err(); err != nil {
		return report, err
	}
	rec.mu.Lock()
	running, capacity := rec.running, len(rec.records)
	rec.mu.Unlock()
	if !running {
		return report, errors.New("cudaflightrecorder: activity capture requires a running recorder")
	}
	if p.config.Activities.MaxRecords > capacity {
		return report, errors.New("cudaflightrecorder: activity record bound exceeds recorder capacity")
	}
	if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) <= p.config.CollectionWindow {
		return report, errors.New("cudaflightrecorder: activity window leaves no snapshot time")
	}
	started := time.Now()
	session, startErr := p.start(p.config.Activities)
	report.Started = startErr == nil && session != nil
	if session == nil {
		if startErr == nil {
			startErr = errors.New("activity startup returned no session")
		}
		return report, fmt.Errorf("start incident activity: %w", startErr)
	}
	if startErr == nil {
		timer := time.NewTimer(p.config.CollectionWindow)
		select {
		case <-timer.C:
		case <-ctx.Done():
			captureErr = ctx.Err()
		}
		timer.Stop()
	} else {
		captureErr = fmt.Errorf("start incident activity: %w", startErr)
	}
	var closeErr error
	if startErr == nil {
		closeErr = session.Close()
	}
	// A session returned with a startup error already failed native rollback.
	// Retain it for an explicit Close call rather than automatically retrying.
	if closeErr != nil || startErr != nil {
		p.pending, p.release = session, releaseLease
		report.RetainedOwnership = true
		if closeErr != nil {
			captureErr = errors.Join(captureErr, fmt.Errorf("close incident activity: %w", closeErr))
		}
	}
	result := session.Result()
	report.Closed = result.Closed
	report.Duration = time.Since(started)
	report.Records = len(result.Records)
	report.PendingBuffers, report.RefusedBuffers = result.PendingBuffers, result.RefusedBuffers
	report.IncompleteRecords, report.DiscardedRecords = result.IncompleteRecords, result.DiscardedRecords
	report.DroppedRecords, report.CallbackErrors = result.DroppedRecords, result.CallbackErrors
	captureErr = errors.Join(captureErr, result.FirstError)
	if !result.Closed || result.PendingBuffers != 0 || result.RefusedBuffers != 0 || result.IncompleteRecords != 0 || result.DiscardedRecords != 0 || result.DroppedRecords != 0 || result.CallbackErrors != 0 {
		captureErr = errors.Join(captureErr, errors.New("cudaflightrecorder: incomplete incident activity capture"))
	}
	if captureErr != nil {
		report.Error = captureErr.Error()
	}
	raw, err := json.Marshal(struct {
		Report  CaptureReport          `json:"capture"`
		Records []cuptiactivity.Record `json:"records"`
	}{report, result.Records})
	if err != nil {
		captureErr = errors.Join(captureErr, fmt.Errorf("encode incident activity: %w", err))
	} else {
		rec.AttachFile("incident-activity.json", raw)
	}
	for _, record := range result.Records {
		rec.PushActivityRecord(record)
	}
	return report, captureErr
}

// Close retries only cleanup retained after a failed Capture. It releases the
// cooperating device lease when native cleanup succeeds and is otherwise safe
// to call repeatedly. Callers explicitly decide whether a cleanup retry is safe.
func (p *CUPTIActivityCapture) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.pending == nil {
		return nil
	}
	if err := p.pending.Close(); err != nil {
		return err
	}
	p.pending = nil
	p.release()
	p.release = nil
	return nil
}
