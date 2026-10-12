package nvtelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// Collector coordinates scheduled sampling from a hardware or fake Sampler.
type Collector struct {
	mu              sync.Mutex
	sampler         Sampler
	clock           Clock
	config          CollectorConfig
	writer          *Writer
	runID           string
	seq             uint64
	missedTicks     uint64
	lostSamples     uint64
	samples         []Sample
	phases          []PhaseMarker
	currentPhase    string
	phaseStart      int64
	prevSampleStart int64
	startTime       int64
	startTimeU      int64
	endTime         int64
	endTimeU        int64
	running         bool
	finished        bool
	terminalErr     error
	finishErr       error
	stopCh          chan struct{}
	doneCh          chan struct{}
	sampleHook      func(Sample)
}

// NewCollector creates an unstarted Collector.
func NewCollector(runID string, sampler Sampler, clock Clock, cfg CollectorConfig, w *Writer) *Collector {
	if clock == nil {
		clock = RealClock{}
	}
	if cfg.IntervalNanos <= 0 {
		cfg.IntervalNanos = int64(100 * time.Millisecond)
	}
	return &Collector{
		runID:   runID,
		sampler: sampler,
		clock:   clock,
		config:  cfg,
		writer:  w,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}
}

// SetPhase transitions the collector into a new named phase.
func (c *Collector) SetPhase(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.finished {
		return errors.New("collector already finished")
	}
	now := c.clock.NowMonotonic()
	if c.currentPhase != "" {
		marker := PhaseMarker{
			Name:                c.currentPhase,
			StartMonotonicNanos: c.phaseStart,
			EndMonotonicNanos:   now,
		}
		c.phases = append(c.phases, marker)
		if c.writer != nil {
			if err := c.writer.WritePhase(&marker); err != nil {
				return fmt.Errorf("write phase marker: %w", err)
			}
		}
	}

	c.currentPhase = name
	c.phaseStart = now
	return nil
}

// SetSampleHook configures an optional callback invoked for each collected sample.
func (c *Collector) SetSampleHook(hook func(Sample)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleHook = hook
}

// Run executes the scheduled collection loop until ctx is canceled or Stop is called.
func (c *Collector) Run(ctx context.Context) (*Manifest, error) {
	c.mu.Lock()
	if c.running {
		c.mu.Unlock()
		return nil, errors.New("collector already running")
	}
	if c.finished {
		c.mu.Unlock()
		return nil, errors.New("collector already finished (restart not supported)")
	}
	c.running = true
	c.startTime = c.clock.NowMonotonic()
	c.startTimeU = c.clock.NowUnix()
	c.mu.Unlock()

	if err := c.InitStream(); err != nil {
		c.mu.Lock()
		c.running = false
		c.finished = true
		c.terminalErr = err
		c.mu.Unlock()
		close(c.doneCh)
		return nil, err
	}

	interval := time.Duration(c.config.IntervalNanos)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Initial sample immediately
	if err := c.collectOnce(ctx); err != nil {
		c.mu.Lock()
		c.terminalErr = err
		m, finishErr := c.finishLocked()
		c.finishErr = finishErr
		c.mu.Unlock()
		close(c.doneCh)
		if finishErr != nil {
			return m, fmt.Errorf("%w: finish error: %v", err, finishErr)
		}
		return m, err
	}

	for {
		select {
		case <-ctx.Done():
			c.mu.Lock()
			m, finishErr := c.finishLocked()
			c.finishErr = finishErr
			c.mu.Unlock()
			close(c.doneCh)
			if finishErr != nil {
				return m, fmt.Errorf("finish stream on cancel: %w", finishErr)
			}
			return m, ctx.Err()
		case <-c.stopCh:
			c.mu.Lock()
			m, finishErr := c.finishLocked()
			c.finishErr = finishErr
			err := c.terminalErr
			c.mu.Unlock()
			close(c.doneCh)
			if err != nil {
				return m, err
			}
			return m, finishErr
		case <-ticker.C:
			if err := c.collectOnce(ctx); err != nil {
				c.mu.Lock()
				c.terminalErr = err
				m, finishErr := c.finishLocked()
				c.finishErr = finishErr
				c.mu.Unlock()
				close(c.doneCh)
				if ctx.Err() != nil {
					return m, ctx.Err()
				}
				if finishErr != nil {
					return m, fmt.Errorf("%w: finish error: %v", err, finishErr)
				}
				return m, err
			}

			if c.config.MaxDurationNanos > 0 {
				elapsed := c.clock.NowMonotonic() - c.startTime
				if elapsed >= c.config.MaxDurationNanos {
					c.mu.Lock()
					m, finishErr := c.finishLocked()
					c.finishErr = finishErr
					err := c.terminalErr
					c.mu.Unlock()
					close(c.doneCh)
					if err != nil {
						return m, err
					}
					return m, finishErr
				}
			}
		}
	}
}

// finishLocked finalizes end time, closes final phase, writes final completeness counters, and builds manifest.
// It is idempotent; subsequent calls return the existing manifest without rewriting phases or stream records.
func (c *Collector) finishLocked() (*Manifest, error) {
	if c.finished {
		return c.buildManifestLocked(), c.finishErr
	}
	c.finished = true
	c.running = false
	c.endTime = c.clock.NowMonotonic()
	c.endTimeU = c.clock.NowUnix()

	var phaseErr error
	if c.currentPhase != "" {
		marker := PhaseMarker{
			Name:                c.currentPhase,
			StartMonotonicNanos: c.phaseStart,
			EndMonotonicNanos:   c.endTime,
		}
		c.phases = append(c.phases, marker)
		if c.writer != nil {
			if err := c.writer.WritePhase(&marker); err != nil {
				phaseErr = fmt.Errorf("write final phase: %w", err)
			}
		}
		c.currentPhase = ""
	}

	c.finishErr = phaseErr
	manifest := c.buildManifestLocked()
	if c.writer != nil {
		if err := c.writer.WriteManifest(manifest); err != nil && phaseErr == nil {
			phaseErr = fmt.Errorf("write final manifest: %w", err)
			c.finishErr = phaseErr
			manifest = c.buildManifestLocked()
		}
	}
	return manifest, phaseErr
}

// Stop requests the collection loop to terminate gracefully. It is safe to call multiple times.
func (c *Collector) Stop() (*Manifest, error) {
	c.mu.Lock()
	if c.finished {
		m := c.buildManifestLocked()
		err := c.terminalErr
		if err == nil {
			err = c.finishErr
		}
		c.mu.Unlock()
		return m, err
	}
	if !c.running {
		m, err := c.finishLocked()
		if c.terminalErr != nil {
			err = c.terminalErr
		}
		c.mu.Unlock()
		return m, err
	}
	select {
	case <-c.stopCh:
	default:
		close(c.stopCh)
	}
	c.mu.Unlock()

	<-c.doneCh
	c.mu.Lock()
	m := c.buildManifestLocked()
	err := c.terminalErr
	if err == nil {
		err = c.finishErr
	}
	c.mu.Unlock()
	return m, err
}

// InitStream writes the initial manifest header to the stream sink.
func (c *Collector) InitStream() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.config.SchemaVersion != 0 && c.config.SchemaVersion != SchemaVersionV2 && c.config.SchemaVersion != SchemaVersionV3 && c.config.SchemaVersion != SchemaVersionV4 && c.config.SchemaVersion != SchemaVersionV5 {
		return ErrUnsupportedSchema
	}
	if c.writer == nil {
		return nil
	}
	if c.startTime == 0 {
		c.startTime = c.clock.NowMonotonic()
		c.startTimeU = c.clock.NowUnix()
	}
	monoDomain, monoEpoch, wallDomain, provenance := c.clockMeta()
	buildRev, buildMod := currentBuildInfo()
	var (
		validFrom  int64
		validUntil int64
		calBounds  int64
		hasCal     bool
	)
	if cal, ok := c.clock.(CalibratedClock); ok {
		validFrom, validUntil, calBounds, hasCal = cal.Calibration()
	}
	manifest := &Manifest{
		RunID:                       c.runID,
		SchemaVersion:               c.streamVersion(),
		Receipt:                     c.config.Receipt,
		CollectorVersion:            CollectorVersion,
		MetricRegistryVersion:       DefaultMetricRegistryVersion,
		ClockMonotonicDomain:        monoDomain,
		ClockMonotonicEpochID:       monoEpoch,
		ClockWallDomain:             wallDomain,
		ClockProvenance:             provenance,
		ClockMappingValidFrom:       validFrom,
		ClockMappingValidUntil:      validUntil,
		ClockCalibrationBoundsNanos: calBounds,
		HasClockCalibration:         hasCal,
		BuildRevision:               buildRev,
		BuildModified:               buildMod,
		Devices:                     c.sampler.Devices(),
		Config:                      c.config,
		StartTimeMonotonic:          c.startTime,
		StartTimeUnix:               c.startTimeU,
	}
	return c.writer.WriteManifest(manifest)
}

// Step performs a single collection pass without ticker scheduling (for fake clocks).
func (c *Collector) Step(ctx context.Context) error {
	return c.collectOnce(ctx)
}

func (c *Collector) collectOnce(ctx context.Context) error {
	c.mu.Lock()
	if c.finished {
		c.mu.Unlock()
		return errors.New("collector already finished")
	}
	c.mu.Unlock()
	start := c.clock.NowMonotonic()
	c.mu.Lock()
	if c.startTime == 0 {
		c.startTime = start
		c.startTimeU = c.clock.NowUnix()
	}
	c.mu.Unlock()

	readings, err := c.sampler.Sample(ctx)
	readings = cloneReadings(readings)
	if err == nil && c.streamVersion() >= SchemaVersionV3 {
		// Copy versioned pointer-backed evidence without changing the sampler's values.
		var data []byte
		data, err = json.Marshal(readings)
		if err == nil {
			err = json.Unmarshal(data, &readings)
		}
		if err == nil {
			for i := range readings {
				r := &readings[i]
				if r.Number != nil && (r.IsInt || r.IntValue != 0 || r.Value != 0) {
					err = fmt.Errorf("%w: conflicting sampler scalar representations", ErrMalformedStream)
					break
				}
				if r.Number == nil && r.IsAvailable() && (r.Observation == nil || r.Observation.Histogram == nil) {
					if r.IsInt {
						n := r.IntValue
						r.Number = &Number{Uint: &n}
					} else {
						n := r.Value
						r.Number = &Number{Float: &n}
					}
				}
				r.Value, r.IntValue, r.IsInt = 0, 0, false
			}
		}
	}
	end := c.clock.NowMonotonic()

	queryDur := end - start
	if queryDur > c.config.IntervalNanos && c.config.IntervalNanos > 0 {
		missed := queryDur / c.config.IntervalNanos
		c.mu.Lock()
		c.missedTicks += uint64(missed)
		c.mu.Unlock()
	}

	if err != nil {
		c.mu.Lock()
		if c.terminalErr == nil {
			c.terminalErr = err
		}
		c.mu.Unlock()
		return err
	}

	c.mu.Lock()
	if c.finished {
		c.mu.Unlock()
		return errors.New("collector already finished")
	}
	c.seq++
	var discontinuity bool
	var discontinuityReason string
	if c.seq > 1 && start < c.prevSampleStart {
		discontinuity = true
		discontinuityReason = "negative_monotonic_step"
	}
	if end < start {
		discontinuity = true
		discontinuityReason = "negative_query_duration"
	}
	c.prevSampleStart = start

	for i := range readings {
		if !readings[i].HasQueryBounds && readings[i].QueryStartNanos == 0 && readings[i].QueryEndNanos == 0 {
			readings[i].QueryStartNanos = start
			readings[i].QueryEndNanos = end
			readings[i].HasQueryBounds = true
		}
	}

	sample := Sample{
		Seq:                      c.seq,
		QueryStartNanos:          start,
		QueryEndNanos:            end,
		LagNanos:                 queryDur,
		ClockDiscontinuity:       discontinuity,
		ClockDiscontinuityReason: discontinuityReason,
		Readings:                 readings,
	}

	// In-memory buffer budget enforcement: do not append unbounded samples
	if c.config.MaxBufferedRecords > 0 && len(c.samples) >= c.config.MaxBufferedRecords {
		if c.config.BudgetPolicy == BudgetExceededDrop || c.config.DropOnFull {
			c.lostSamples++
			c.mu.Unlock()
			if c.writer != nil {
				if err := c.writer.WriteSample(&sample); err != nil {
					c.mu.Lock()
					if c.terminalErr == nil {
						c.terminalErr = err
					}
					c.mu.Unlock()
					return err
				}
			}
			c.notifySample(sample)
			return nil
		}
		c.terminalErr = ErrBudgetExceeded
		c.mu.Unlock()
		return ErrBudgetExceeded
	}

	c.samples = append(c.samples, sample)
	c.mu.Unlock()

	if c.writer != nil {
		if err := c.writer.WriteSample(&sample); err != nil {
			c.mu.Lock()
			if c.terminalErr == nil {
				c.terminalErr = err
			}
			c.mu.Unlock()
			return err
		}
	}

	c.notifySample(sample)
	return nil
}

func (c *Collector) notifySample(sample Sample) {
	c.mu.Lock()
	hook := c.sampleHook
	c.mu.Unlock()
	if hook != nil {
		copySample := sample
		copySample.Readings = cloneReadings(sample.Readings)
		hook(copySample)
	}
}

func (c *Collector) buildManifestLocked() *Manifest {
	var lost uint64
	if c.writer != nil {
		lost = c.writer.LostRecords()
	}
	if c.lostSamples > lost {
		lost = c.lostSamples
	}

	phasesCopy := make([]PhaseMarker, len(c.phases))
	copy(phasesCopy, c.phases)

	isComplete := c.finished && c.terminalErr == nil && c.finishErr == nil && lost == 0
	var reason string
	if !isComplete {
		switch {
		case !c.finished:
			reason = "in_progress"
		case c.terminalErr != nil:
			reason = "terminal_error: " + c.terminalErr.Error()
		case c.finishErr != nil:
			reason = "finish_error: " + c.finishErr.Error()
		case lost > 0:
			reason = "lost_records"
		default:
			reason = "incomplete"
		}
	}

	monoDomain, monoEpoch, wallDomain, provenance := c.clockMeta()
	buildRev, buildMod := currentBuildInfo()
	var (
		validFrom  int64
		validUntil int64
		calBounds  int64
		hasCal     bool
	)
	if cal, ok := c.clock.(CalibratedClock); ok {
		validFrom, validUntil, calBounds, hasCal = cal.Calibration()
	}
	return &Manifest{
		RunID:                       c.runID,
		SchemaVersion:               c.streamVersion(),
		Receipt:                     c.config.Receipt,
		LostRecordsByFamily:         c.streamFamilyLoss(),
		CollectorVersion:            CollectorVersion,
		MetricRegistryVersion:       DefaultMetricRegistryVersion,
		ClockMonotonicDomain:        monoDomain,
		ClockMonotonicEpochID:       monoEpoch,
		ClockWallDomain:             wallDomain,
		ClockProvenance:             provenance,
		ClockMappingValidFrom:       validFrom,
		ClockMappingValidUntil:      validUntil,
		ClockCalibrationBoundsNanos: calBounds,
		HasClockCalibration:         hasCal,
		BuildRevision:               buildRev,
		BuildModified:               buildMod,
		Devices:                     c.sampler.Devices(),
		Config:                      c.config,
		StartTimeMonotonic:          c.startTime,
		StartTimeUnix:               c.startTimeU,
		EndTimeMonotonic:            c.endTime,
		EndTimeUnix:                 c.endTimeU,
		Phases:                      phasesCopy,
		TotalSamples:                uint64(len(c.samples)),
		MissedTicks:                 c.missedTicks,
		LostRecords:                 lost,
		IsComplete:                  isComplete,
		IncompleteReason:            reason,
	}
}

func (c *Collector) streamVersion() int {
	if c.config.SchemaVersion == 0 {
		return SchemaVersionV2
	}
	return c.config.SchemaVersion
}

func (c *Collector) streamFamilyLoss() map[string]tracecontract.Loss {
	if c.streamVersion() < SchemaVersionV3 || c.writer == nil {
		return nil
	}
	return c.writer.FamilyLoss()
}

func (c *Collector) clockMeta() (string, string, string, string) {
	if meta, ok := c.clock.(ClockMetadata); ok {
		return meta.MonotonicDomain(), meta.MonotonicEpochID(), meta.WallDomain(), meta.Provenance()
	}
	return "unknown", "", "unknown", "unknown"
}

// Manifest returns the current manifest summary.
func (c *Collector) Manifest() *Manifest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buildManifestLocked()
}

// Samples returns a copy of collected samples.
func (c *Collector) Samples() []Sample {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Sample, len(c.samples))
	copy(out, c.samples)
	for i := range out {
		out[i].Readings = cloneReadings(out[i].Readings)
	}
	return out
}

func currentBuildInfo() (string, bool) {
	if info, ok := debug.ReadBuildInfo(); ok {
		var rev string
		var modified bool
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" {
				rev = setting.Value
			}
			if setting.Key == "vcs.modified" && setting.Value == "true" {
				modified = true
			}
		}
		return rev, modified
	}
	return "", false
}
