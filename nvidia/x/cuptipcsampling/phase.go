package cuptipcsampling

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// PhaseLimits specifies explicit bounds for bounded phase capture.
// Every limit must be positive, following the package policy of avoiding implicit defaults.
// The zero PhaseLimits is not usable.
type PhaseLimits struct {
	// MaxBatches bounds the total number of batches drained during the phase,
	// across intermediate reads/handoffs and the final drain.
	MaxBatches uint

	// DrainTimeout is the maximum duration to wait for background drain
	// completion and final buffer draining after workload synchronization.
	DrainTimeout time.Duration

	// PollInterval is the polling/drain check cadence when intermediate draining
	// is performed without a concurrent consumer. If zero, polling is disabled.
	PollInterval time.Duration
}

// PhaseConfig configures bounded phase PC sampling collection.
// It joins the hardware PC sampling Config, PhaseLimits, and an optional consumer.
// The zero PhaseConfig is not usable.
type PhaseConfig struct {
	Config   Config
	Limits   PhaseLimits
	Consumer func(Batch) error // Optional batch consumer; returned batches are retained too.
	// Modules supplies already observed module lifecycle metadata. Nil disables
	// metadata reannouncement. MetadataLimits must be positive when Modules is set.
	Modules        *cudacapture.ModuleTracker
	MetadataLimits cudacapture.ReannouncementLimits
	// MetadataConsumer receives detached start and terminal snapshots. Failure
	// is reported separately and never changes PC sample accounting.
	MetadataConsumer func(cudacapture.Reannouncement) error
}

// PhaseReport details the outcome of a bounded phase capture, separately reporting
// samples, dropped samples, remaining PC records, consumer failure,
// and cancellation status.
//
// Native output counts what it left behind in PC records, not samples, so
// the report has no count of remaining samples. RemainingPCs adds the count
// reported by a read after Stop to the count reported by the final batch;
// the two may describe the same records, so it is not a count of distinct
// records left behind.
type PhaseReport struct {
	// MetadataStart is captured before Start; MetadataEnd after the final drain
	// and Close attempt. Nil means metadata was not requested or not reached.
	MetadataStart  *cudacapture.Reannouncement
	MetadataEnd    *cudacapture.Reannouncement
	MetadataErr    error   // Snapshot or delivery failure, independent of PC sample loss.
	Info           Info    // Diagnostic and session metadata
	Batches        []Batch // All drained batches, including those delivered to Consumer
	TotalSamples   uint64  // Sum of all valid PC total samples observed
	DroppedSamples uint64  // Sum of dropped samples reported by hardware or buffers
	RemainingPCs   uint64  // Sum of PC record counts reported remaining after Stop and by the final batch
	BatchesDrained uint    // Total batches successfully drained
	FinalDrained   bool    // A read after Stop reported no remaining PC records
	DrainLimited   bool    // MaxBatches stopped the final drain; records may remain, count unknown
	FinalBatch     bool    // Batches ends with the session's post-disable final batch
	ConsumerErr    error   // Consumer failure reported separately
	WorkloadErr    error   // Error returned by the workload execution function
	Canceled       bool    // True if the phase context was canceled
	HardwareFull   bool    // True if hardware buffer overflow was reported
}

// Loss reports whether samples were dropped, hardware buffer overflow
// occurred, or PC records remained or may remain uncollected.
func (r PhaseReport) Loss() bool {
	return r.DroppedSamples > 0 || r.RemainingPCs > 0 || r.HardwareFull || r.DrainLimited
}

// phaseController coordinates the phase capture lifecycle with separate dependency tracking.
type phaseController struct {
	ctx      driver.Context
	cfg      PhaseConfig
	starter  func(driver.Context, Config) (*Session, error)
	syncFn   func() error
	workload func(context.Context) error

	mu          sync.Mutex
	session     *Session
	batches     []Batch
	report      PhaseReport
	consumerErr error
}

func (c *phaseController) validate() error {
	if c.ctx == 0 {
		return errors.New("cuptipcsampling: invalid context")
	}
	if c.cfg.Limits.MaxBatches == 0 || c.cfg.Limits.DrainTimeout <= 0 {
		return errors.New("cuptipcsampling: invalid phase limits")
	}
	if c.cfg.Modules != nil && (c.cfg.MetadataLimits.MaxEntries == 0 || c.cfg.MetadataLimits.MaxBytes == 0) {
		return errors.New("cuptipcsampling: invalid metadata limits")
	}
	if c.cfg.Modules == nil && c.cfg.MetadataConsumer != nil {
		return errors.New("cuptipcsampling: metadata consumer without module tracker")
	}
	if c.workload == nil {
		return errors.New("cuptipcsampling: nil workload function")
	}
	if c.syncFn == nil {
		return errors.New("cuptipcsampling: nil synchronize function")
	}
	if err := validateConfig(c.ctx, c.cfg.Config); err != nil {
		return err
	}
	if c.cfg.Limits.MaxBatches > c.cfg.Config.MaxReads {
		return errors.New("cuptipcsampling: phase max batches exceeds session max reads")
	}
	return nil
}

func (c *phaseController) recordBatch(b Batch) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.report.BatchesDrained++
	if b.DroppedSamples > 0 {
		c.report.DroppedSamples += b.DroppedSamples
	}
	if b.HardwareBufferFull != 0 {
		c.report.HardwareFull = true
	}
	if b.Valid {
		c.report.TotalSamples += b.TotalSamples
	}

	c.batches = append(c.batches, clonePhaseBatch(b))
	if c.cfg.Consumer != nil {
		if err := c.cfg.Consumer(clonePhaseBatch(b)); err != nil {
			if c.consumerErr == nil {
				c.consumerErr = fmt.Errorf("cuptipcsampling: consumer failure: %w", err)
			}
			return err
		}
	}
	return nil
}

// CapturePhase executes a workload under bounded PC sampling collection.
// It composes the full lifecycle:
// 1. Session start with caller limits and exclusivity enforcement.
// 2. Optional concurrent or periodic background draining into the consumer.
// 3. Workload execution under ctx with cancellation observation.
// 4. Workload completion and device synchronization.
// 5. Native sampling range stop.
// 6. Consumer completion and final buffer drain.
// 7. Session cleanup and release of native resources.
//
// Resources are strictly preserved while native collection or drain consumers
// are active; cleanup retries if disable fails.
//
// Remaining PC records, dropped samples, consumer failure, and cancellation
// are reported separately in the returned [PhaseReport].
func CapturePhase(ctx context.Context, cuCtx driver.Context, cfg PhaseConfig, syncWorkload func() error, workload func(context.Context) error) (PhaseReport, error) {
	return capturePhase(ctx, cuCtx, cfg, Start, syncWorkload, workload)
}

// PhaseCapture is an alias for [CapturePhase].
func PhaseCapture(ctx context.Context, cuCtx driver.Context, cfg PhaseConfig, syncWorkload func() error, workload func(context.Context) error) (PhaseReport, error) {
	return CapturePhase(ctx, cuCtx, cfg, syncWorkload, workload)
}

func capturePhase(
	ctx context.Context,
	cuCtx driver.Context,
	cfg PhaseConfig,
	starter func(driver.Context, Config) (*Session, error),
	syncWorkload func() error,
	workload func(context.Context) error,
) (PhaseReport, error) {
	pc := &phaseController{
		ctx:      cuCtx,
		cfg:      cfg,
		starter:  starter,
		syncFn:   syncWorkload,
		workload: workload,
	}

	if err := pc.validate(); err != nil {
		return PhaseReport{}, err
	}

	if err := pc.metadata(false); err != nil {
		return pc.report, err
	}
	session, startErr := pc.starter(cuCtx, cfg.Config)
	if startErr != nil {
		if session != nil {
			_ = session.Close()
		}
		return pc.report, fmt.Errorf("cuptipcsampling: start session: %w", startErr)
	}
	pc.session = session

	var (
		stopDrain  func() error
		pollCancel context.CancelFunc
		pollDone   chan struct{}
	)

	// Set up intermediate draining: either concurrent ring buffer drain or periodic polling.
	if cfg.Config.BufferCount >= 2 && cfg.Consumer != nil {
		var dcErr error
		stopDrain, dcErr = session.DrainConcurrent(func(b Batch) error {
			return pc.recordBatch(b)
		})
		if dcErr != nil {
			_ = session.Close()
			return pc.report, fmt.Errorf("cuptipcsampling: start concurrent drain: %w", dcErr)
		}
	} else if cfg.Limits.PollInterval > 0 {
		pollCtx, cancel := context.WithCancel(context.Background())
		pollCancel = cancel
		pollDone = make(chan struct{})
		go func() {
			defer close(pollDone)
			ticker := time.NewTicker(cfg.Limits.PollInterval)
			defer ticker.Stop()
			for {
				select {
				case <-pollCtx.Done():
					return
				case <-ticker.C:
					pc.mu.Lock()
					drained := pc.report.BatchesDrained
					maxB := pc.cfg.Limits.MaxBatches
					pc.mu.Unlock()

					if drained >= maxB {
						return
					}
					b, err := session.Read()
					if err != nil {
						return
					}
					if err := pc.recordBatch(b); err != nil {
						return
					}
				}
			}
		}()
	}

	// Phase execution step 3: Workload execution under ctx.
	var (
		workloadErr error
		canceled    bool
	)

	workloadErr = pc.workload(ctx)
	if ctx.Err() != nil {
		canceled = true
	}

	// Phase execution step 4: Workload completion and device synchronization.
	var syncErr error
	if syncWorkload != nil {
		syncErr = syncWorkload()
	}

	// Stop polling if active before stopping sampling
	if pollCancel != nil {
		pollCancel()
		<-pollDone
	}

	// Phase execution step 5: Native sampling range stop.
	stopErr := session.Stop()

	// Wait for concurrent drain consumer to stop and complete if active.
	if stopDrain != nil {
		// Enforce DrainTimeout when stopping consumer.
		drainCh := make(chan error, 1)
		go func() {
			drainCh <- stopDrain()
		}()

		var cErr error
		select {
		case cErr = <-drainCh:
		case <-time.After(cfg.Limits.DrainTimeout):
			cErr = errors.New("drain timeout exceeded waiting for consumer")
		}

		if cErr != nil && pc.consumerErr == nil {
			pc.consumerErr = fmt.Errorf("cuptipcsampling: drain consumer: %w", cErr)
		}
	}

	// Phase execution step 6: Final drain of any remaining batches from session.
	// When concurrent draining with a ring buffer was used, stopDrain has already
	// closed the ring buffer and consumed queued batches. For non-concurrent draining,
	// drain any remaining batches from the session until the limit or EOF is reached.
	// The concurrent path reads no remaining count after Stop, so it never
	// sets FinalDrained.
	var finalDrainErr error
	if stopDrain == nil {
		var remainingAfterStop uint64 // zero until a read after Stop
		for {
			pc.mu.Lock()
			drained := pc.report.BatchesDrained
			maxB := pc.cfg.Limits.MaxBatches
			pc.mu.Unlock()

			if drained >= maxB {
				// Limit reached. A further Read would move records out of the
				// native buffer only to discard them. Records may remain, and
				// a remaining count from a read before Stop says nothing about
				// them, so only a count read after Stop is reported.
				pc.report.DrainLimited = true
				pc.report.RemainingPCs = remainingAfterStop
				break
			}
			b, err := session.Read()
			if err != nil {
				finalDrainErr = err
				// Even on error, Read may return diagnostic loss counters
				if b.DroppedSamples > 0 {
					pc.report.DroppedSamples += b.DroppedSamples
				}
				if b.HardwareBufferFull != 0 {
					pc.report.HardwareFull = true
				}
				if b.RemainingPCs > 0 {
					pc.report.RemainingPCs = b.RemainingPCs
				}
				break
			}
			if err := pc.recordBatch(b); err != nil {
				break
			}
			remainingAfterStop = b.RemainingPCs
			if b.RemainingPCs == 0 {
				pc.report.FinalDrained = true
				break
			}
		}
	}

	// Phase execution step 7: Session cleanup and release of native resources.
	// Resources are not destroyed while native collection or consumer was active;
	// Close disables native sampling and finalizes buffers.
	closeErr := session.Close()

	// Retrieve final batch if available.
	if finalBatch, ok := session.Final(); ok {
		if finalBatch.DroppedSamples > 0 {
			pc.report.DroppedSamples += finalBatch.DroppedSamples
		}
		if finalBatch.RemainingPCs > 0 {
			pc.report.RemainingPCs += finalBatch.RemainingPCs
		}
		if finalBatch.HardwareBufferFull != 0 {
			pc.report.HardwareFull = true
		}
		if finalBatch.Valid {
			pc.report.TotalSamples += finalBatch.TotalSamples
		}
		// A final batch may carry a decode failure. Preserve that evidence too;
		// validity controls sample accounting, not retention or delivery.
		pc.batches = append(pc.batches, clonePhaseBatch(finalBatch))
		if cfg.Consumer != nil {
			if err := cfg.Consumer(clonePhaseBatch(finalBatch)); err != nil && pc.consumerErr == nil {
				pc.consumerErr = fmt.Errorf("cuptipcsampling: consumer final batch: %w", err)
			}
		}
		pc.report.BatchesDrained++
		pc.report.FinalBatch = true
	}

	pc.report.Info = session.Info()
	pc.report.Batches = pc.batches
	pc.report.WorkloadErr = workloadErr
	pc.report.ConsumerErr = pc.consumerErr
	pc.report.Canceled = canceled

	// Metadata observes terminal lifecycle even when native cleanup is pending.
	_ = pc.metadata(true)
	// Assemble composite error if any operational stage encountered failure.
	var combinedErr error
	if workloadErr != nil {
		combinedErr = errors.Join(combinedErr, fmt.Errorf("workload execution: %w", workloadErr))
	}
	if syncErr != nil {
		combinedErr = errors.Join(combinedErr, fmt.Errorf("workload synchronize: %w", syncErr))
	}
	if stopErr != nil {
		combinedErr = errors.Join(combinedErr, fmt.Errorf("stop sampling: %w", stopErr))
	}
	if finalDrainErr != nil {
		combinedErr = errors.Join(combinedErr, fmt.Errorf("final drain: %w", finalDrainErr))
	}
	if closeErr != nil {
		combinedErr = errors.Join(combinedErr, fmt.Errorf("close session: %w", closeErr))
	}

	return pc.report, errors.Join(combinedErr, pc.report.MetadataErr)
}

func (c *phaseController) metadata(terminal bool) error {
	if c.cfg.Modules == nil {
		return nil
	}
	r, err := c.cfg.Modules.Reannounce(c.cfg.MetadataLimits)
	if err == nil {
		if terminal {
			c.report.MetadataEnd = &r
		} else {
			c.report.MetadataStart = &r
		}
		if c.cfg.MetadataConsumer != nil {
			err = c.cfg.MetadataConsumer(r.Clone())
		}
	}
	if err != nil {
		err = fmt.Errorf("cuptipcsampling: metadata reannouncement: %w", err)
		c.report.MetadataErr = errors.Join(c.report.MetadataErr, err)
	}
	return err
}

func clonePhaseBatch(b Batch) Batch {
	b.PCs = append([]PC(nil), b.PCs...)
	for i := range b.PCs {
		b.PCs[i].Reasons = append([]StallSamples(nil), b.PCs[i].Reasons...)
	}
	return b
}
