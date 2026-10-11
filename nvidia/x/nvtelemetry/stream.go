package nvtelemetry

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
)

// Stream errors.
var (
	ErrBudgetExceeded          = errors.New("telemetry budget exceeded")
	ErrUnsupportedSchema       = errors.New("unsupported schema version")
	ErrTruncatedRecord         = errors.New("truncated telemetry record")
	ErrMalformedStream         = errors.New("malformed telemetry stream")
	ErrIdentityMismatch        = errors.New("device identity mismatch in stream")
	ErrUnidentifiableDevice    = errors.New("unidentifiable device: missing or unassigned device UUID")
	ErrBitmaskAveragingRefused = errors.New("refusing to average bitmask or event metric")
)

// Canonical metric unit expectations.
var expectedMetricUnits = map[string]string{
	"imex_domain_status":      "code",
	"imex_daemon_status":      "code",
	MetricGPUUtilization:      UnitPercent,
	MetricMemoryUtilization:   UnitPercent,
	MetricMemoryUsedBytes:     UnitBytes,
	MetricMemoryTotalBytes:    UnitBytes,
	MetricPowerUsageMW:        UnitMilliwatts,
	MetricTotalEnergyMJ:       UnitMillijoules,
	MetricTemperatureC:        UnitCelsius,
	MetricSMClockMHz:          UnitMHz,
	MetricMemoryClockMHz:      UnitMHz,
	MetricClocksEventReasons:  UnitBitmask,
	MetricSMActivity:          UnitPercent,
	MetricTensorActivity:      UnitPercent,
	MetricDRAMActivity:        UnitPercent,
	MetricPCIeRxBytesPerSec:   UnitBytesPerSec,
	MetricPCIeTxBytesPerSec:   UnitBytesPerSec,
	MetricNVLinkRxBytesPerSec: UnitBytesPerSec,
	MetricNVLinkTxBytesPerSec: UnitBytesPerSec,
	MetricPCIERxBytes:         UnitBytes,
	MetricPCIETxBytes:         UnitBytes,
}

// StreamRecord is the envelope for lines in a telemetry JSONL stream.
type StreamRecord struct {
	Manifest *Manifest    `json:"manifest,omitempty"`
	Sample   *Sample      `json:"sample,omitempty"`
	Phase    *PhaseMarker `json:"phase,omitempty"`
}

// Writer writes streaming telemetry records to an io.Writer with budget enforcement.
type Writer struct {
	mu           sync.Mutex
	w            io.Writer
	config       CollectorConfig
	bytesWritten int
	recordsCount int
	lostRecords  uint64
}

// NewWriter creates a telemetry stream writer.
func NewWriter(w io.Writer, cfg CollectorConfig) *Writer {
	return &Writer{
		w:      w,
		config: cfg,
	}
}

// WriteManifest writes the manifest header to the stream.
func (w *Writer) WriteManifest(m *Manifest) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if m.SchemaVersion <= 0 || m.SchemaVersion > SchemaVersionV2 {
		return fmt.Errorf("%w: got %d (supported: 1..%d)", ErrUnsupportedSchema, m.SchemaVersion, SchemaVersionV2)
	}
	for _, d := range m.Devices {
		if strings.TrimSpace(d.UUID) == "" || d.Status == DeviceStatusUnidentifiable {
			return fmt.Errorf("%w: %w", ErrIdentityMismatch, ErrUnidentifiableDevice)
		}
	}
	rec := StreamRecord{Manifest: m}
	return w.writeRecordLocked(rec)
}

// WriteSample writes a sample line to the stream.
func (w *Writer) WriteSample(s *Sample) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.config.MaxRecordBudget > 0 && w.recordsCount >= w.config.MaxRecordBudget {
		if w.config.BudgetPolicy == BudgetExceededDrop || w.config.DropOnFull {
			w.lostRecords++
			return nil
		}
		return ErrBudgetExceeded
	}
	rec := StreamRecord{Sample: s}
	err := w.writeRecordLocked(rec)
	if err == nil {
		w.recordsCount++
	}
	return err
}

// WritePhase writes a phase transition marker to the stream.
func (w *Writer) WritePhase(p *PhaseMarker) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	rec := StreamRecord{Phase: p}
	return w.writeRecordLocked(rec)
}

func (w *Writer) writeRecordLocked(rec StreamRecord) error {
	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal stream record: %w", err)
	}
	data = append(data, '\n')
	if w.config.MaxByteBudget > 0 && w.bytesWritten+len(data) > w.config.MaxByteBudget {
		if w.config.BudgetPolicy == BudgetExceededDrop || w.config.DropOnFull {
			w.lostRecords++
			return nil
		}
		return ErrBudgetExceeded
	}

	n, err := w.w.Write(data)
	w.bytesWritten += n
	if err != nil {
		return fmt.Errorf("write stream record: %w", err)
	}
	if n < len(data) {
		return io.ErrShortWrite
	}
	return nil
}

// RecordDrop increments the dropped record counter.
func (w *Writer) RecordDrop() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lostRecords++
	return nil
}

// LostRecords returns the count of records dropped due to budget exhaustion.
func (w *Writer) LostRecords() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lostRecords
}

// BytesWritten returns total bytes successfully written to the stream.
func (w *Writer) BytesWritten() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.bytesWritten
}

// ReadStream reads a full JSONL telemetry stream, returning manifest, phases, and samples.
// It strictly validates schema version, device identities, canonical units, monotonic ordering,
// and detects truncated records.
func ReadStream(r io.Reader, maxBytes int64) (*Manifest, []PhaseMarker, []Sample, error) {
	if maxBytes <= 0 {
		maxBytes = 64 << 20 // default 64 MiB
	}
	lr := io.LimitReader(r, maxBytes+1)
	scanner := bufio.NewScanner(lr)
	// Allow lines up to 4 MiB
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 4<<20)

	var manifest *Manifest
	var phases []PhaseMarker
	var samples []Sample
	var totalRead int64
	scanner.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		advance, token, err := bufio.ScanLines(data, atEOF)
		totalRead += int64(advance)
		if totalRead > maxBytes {
			return 0, nil, ErrBudgetExceeded
		}
		return advance, token, err
	})

	allowedDevices := make(map[string]bool)
	var completed bool
	var haveSample bool
	var lastSeq uint64
	var lastSampleStart int64 = -1
	var lastPhaseEnd int64 = -1

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}

		var rec StreamRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) || strings.Contains(err.Error(), "unexpected end of JSON input") {
				return nil, nil, nil, fmt.Errorf("%w at line %d: %v", ErrTruncatedRecord, lineNum, err)
			}
			return nil, nil, nil, fmt.Errorf("%w at line %d: %v", ErrMalformedStream, lineNum, err)
		}

		fields := 0
		if rec.Manifest != nil {
			fields++
		}
		if rec.Sample != nil {
			fields++
		}
		if rec.Phase != nil {
			fields++
		}
		if fields != 1 || completed || (manifest == nil && rec.Manifest == nil) {
			return nil, nil, nil, fmt.Errorf("%w: invalid record ordering or envelope at line %d", ErrMalformedStream, lineNum)
		}

		// Manifest processing
		if rec.Manifest != nil {
			if rec.Manifest.SchemaVersion <= 0 || rec.Manifest.SchemaVersion > SchemaVersionV2 {
				return nil, nil, nil, fmt.Errorf("%w: got %d (supported: 1..%d)", ErrUnsupportedSchema, rec.Manifest.SchemaVersion, SchemaVersionV2)
			}
			if rec.Manifest.HasClockCalibration && rec.Manifest.ClockCalibrationBoundsNanos < 0 {
				return nil, nil, nil, fmt.Errorf("%w: negative clock calibration bounds %d", ErrMalformedStream, rec.Manifest.ClockCalibrationBoundsNanos)
			}
			if rec.Manifest.HasClockCalibration && rec.Manifest.ClockMappingValidUntil < rec.Manifest.ClockMappingValidFrom {
				return nil, nil, nil, fmt.Errorf("%w: reversed clock mapping validity", ErrMalformedStream)
			}
			if manifest == nil {
				manifest = rec.Manifest
				manifest.IsComplete = false
				manifest.IncompleteReason = "missing_completion_manifest"
				for _, d := range manifest.Devices {
					if strings.TrimSpace(d.UUID) == "" || d.Status == DeviceStatusUnidentifiable {
						return nil, nil, nil, fmt.Errorf("%w: %w", ErrIdentityMismatch, ErrUnidentifiableDevice)
					}
					if allowedDevices[d.UUID] {
						return nil, nil, nil, ErrIdentityMismatch
					}
					allowedDevices[d.UUID] = true
				}
			} else {
				completed = true
				if rec.Manifest.EndTimeMonotonic < manifest.StartTimeMonotonic || (rec.Manifest.IsComplete && (rec.Manifest.LostRecords != 0 || rec.Manifest.IncompleteReason != "" || rec.Manifest.TotalSamples != uint64(len(samples)))) {
					return nil, nil, nil, fmt.Errorf("%w: inconsistent completion manifest", ErrMalformedStream)
				}
				// Trailing manifest with completeness counters: validate identity consistency
				if rec.Manifest.RunID != manifest.RunID {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest run ID %q does not match %q", ErrIdentityMismatch, rec.Manifest.RunID, manifest.RunID)
				}
				if rec.Manifest.SchemaVersion != manifest.SchemaVersion {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest schema version %d does not match initial %d", ErrIdentityMismatch, rec.Manifest.SchemaVersion, manifest.SchemaVersion)
				}
				if rec.Manifest.ClockMonotonicDomain != manifest.ClockMonotonicDomain {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest clock monotonic domain %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.ClockMonotonicDomain, manifest.ClockMonotonicDomain)
				}
				if rec.Manifest.ClockMonotonicEpochID != manifest.ClockMonotonicEpochID {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest clock epoch %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.ClockMonotonicEpochID, manifest.ClockMonotonicEpochID)
				}
				if rec.Manifest.ClockWallDomain != manifest.ClockWallDomain {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest clock wall domain %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.ClockWallDomain, manifest.ClockWallDomain)
				}
				if rec.Manifest.ClockProvenance != manifest.ClockProvenance {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest clock provenance %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.ClockProvenance, manifest.ClockProvenance)
				}
				if rec.Manifest.HasClockCalibration != manifest.HasClockCalibration || rec.Manifest.ClockCalibrationBoundsNanos != manifest.ClockCalibrationBoundsNanos || rec.Manifest.ClockMappingValidFrom != manifest.ClockMappingValidFrom || rec.Manifest.ClockMappingValidUntil != manifest.ClockMappingValidUntil {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest clock calibration differs", ErrIdentityMismatch)
				}
				if rec.Manifest.MetricRegistryVersion != manifest.MetricRegistryVersion {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest metric registry version %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.MetricRegistryVersion, manifest.MetricRegistryVersion)
				}
				if rec.Manifest.CollectorVersion != manifest.CollectorVersion {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest collector version %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.CollectorVersion, manifest.CollectorVersion)
				}
				if rec.Manifest.BuildRevision != manifest.BuildRevision {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest build revision %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.BuildRevision, manifest.BuildRevision)
				}
				if rec.Manifest.BuildModified != manifest.BuildModified {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest build modified %v does not match initial %v", ErrIdentityMismatch, rec.Manifest.BuildModified, manifest.BuildModified)
				}
				if rec.Manifest.BinaryDigest != manifest.BinaryDigest {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest binary digest %q does not match initial %q", ErrIdentityMismatch, rec.Manifest.BinaryDigest, manifest.BinaryDigest)
				}
				if len(rec.Manifest.Devices) != len(manifest.Devices) {
					return nil, nil, nil, fmt.Errorf("%w: trailing manifest device count %d does not match %d", ErrIdentityMismatch, len(rec.Manifest.Devices), len(manifest.Devices))
				}
				for i, d := range rec.Manifest.Devices {
					if d.UUID != manifest.Devices[i].UUID {
						return nil, nil, nil, fmt.Errorf("%w: trailing manifest device[%d] UUID %q does not match %q", ErrIdentityMismatch, i, d.UUID, manifest.Devices[i].UUID)
					}
				}

				manifest.EndTimeMonotonic = rec.Manifest.EndTimeMonotonic
				manifest.EndTimeUnix = rec.Manifest.EndTimeUnix
				manifest.TotalSamples = rec.Manifest.TotalSamples
				manifest.MissedTicks = rec.Manifest.MissedTicks
				manifest.LostRecords = rec.Manifest.LostRecords
				manifest.IsComplete = rec.Manifest.IsComplete
				manifest.IncompleteReason = rec.Manifest.IncompleteReason
				if !rec.Manifest.IsComplete && manifest.IncompleteReason == "" {
					manifest.IncompleteReason = "producer_marked_incomplete"
				}
				if len(rec.Manifest.Phases) > len(manifest.Phases) {
					manifest.Phases = rec.Manifest.Phases
				}
			}
		}

		// Phase marker processing
		if rec.Phase != nil {
			p := *rec.Phase
			if p.EndMonotonicNanos < p.StartMonotonicNanos {
				return nil, nil, nil, fmt.Errorf("%w: phase %q end %d before start %d", ErrMalformedStream, p.Name, p.EndMonotonicNanos, p.StartMonotonicNanos)
			}
			if lastPhaseEnd >= 0 && p.StartMonotonicNanos < lastPhaseEnd {
				return nil, nil, nil, fmt.Errorf("%w: phase %q start %d overlaps or out-of-order with previous end %d", ErrMalformedStream, p.Name, p.StartMonotonicNanos, lastPhaseEnd)
			}
			lastPhaseEnd = p.EndMonotonicNanos
			phases = append(phases, p)
		}

		// Sample processing
		if rec.Sample != nil {
			s := *rec.Sample

			// Monotonic sequence and timestamp ordering check
			if haveSample && s.Seq <= lastSeq {
				return nil, nil, nil, fmt.Errorf("%w: sample seq %d is not strictly greater than previous seq %d", ErrMalformedStream, s.Seq, lastSeq)
			}
			haveSample = true
			lastSeq = s.Seq
			if manifest.SchemaVersion >= SchemaVersionV2 {
				if s.QueryEndNanos < s.QueryStartNanos {
					return nil, nil, nil, fmt.Errorf("%w: sample seq %d inverted query bounds [%d, %d]", ErrMalformedStream, s.Seq, s.QueryStartNanos, s.QueryEndNanos)
				}
			} else {
				if s.QueryEndNanos != 0 && s.QueryEndNanos < s.QueryStartNanos {
					return nil, nil, nil, fmt.Errorf("%w: sample seq %d negative duration", ErrMalformedStream, s.Seq)
				}
			}

			if lastSampleStart >= 0 && s.QueryStartNanos < lastSampleStart {
				return nil, nil, nil, fmt.Errorf("%w: sample seq %d timestamp %d is before previous timestamp %d", ErrMalformedStream, s.Seq, s.QueryStartNanos, lastSampleStart)
			}
			lastSampleStart = s.QueryStartNanos

			// Device identity and unit validation
			for _, r := range s.Readings {
				if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived && r.Evidence != EvidenceUnavailable && !(manifest.SchemaVersion == SchemaVersionLegacy && r.Evidence == "") {
					return nil, nil, nil, fmt.Errorf("%w: invalid reading evidence", ErrMalformedStream)
				}
				if err := ValidateReadingSource(r.Source); err != nil {
					return nil, nil, nil, fmt.Errorf("%w: source: %v", ErrMalformedStream, err)
				}
				if err := validateManagedValue(r); err != nil {
					return nil, nil, nil, fmt.Errorf("%w: source: %v", ErrMalformedStream, err)
				}
				if r.Source != nil {
					if r.HasSourceTimestamp || r.SourceTimestamp != 0 || r.SourceTimestampDomain != "" {
						return nil, nil, nil, fmt.Errorf("%w: managed exposition cannot establish sensor time", ErrMalformedStream)
					}
					if r.Source.Producer != r.Backend || (r.Source.Labels["UUID"] != "" && r.Source.Labels["UUID"] != r.DeviceUUID) {
						return nil, nil, nil, fmt.Errorf("%w: managed source identity contradiction", ErrIdentityMismatch)
					}
					if r.Evidence != EvidenceMeasured && r.Evidence != EvidenceDerived && r.Evidence != EvidenceUnavailable {
						return nil, nil, nil, fmt.Errorf("%w: invalid managed evidence", ErrMalformedStream)
					}
					if (r.Evidence == EvidenceUnavailable) != (r.UnavailableReason != "") {
						return nil, nil, nil, fmt.Errorf("%w: managed status contradiction", ErrMalformedStream)
					}
					kind, _ := LookupMetricKind(r.Name)
					if r.Source.CounterReset && kind != KindCounter {
						return nil, nil, nil, fmt.Errorf("%w: reset on non-counter", ErrMalformedStream)
					}
				}
				if !allowedDevices[r.DeviceUUID] {
					return nil, nil, nil, fmt.Errorf("%w: reading contains undeclared device UUID %q at sample seq %d", ErrIdentityMismatch, r.DeviceUUID, s.Seq)
				}
				if r.HasWallClockUncertainty && r.WallClockUncertaintyNanos < 0 {
					return nil, nil, nil, fmt.Errorf("%w: negative wall clock uncertainty %d on metric %q at sample seq %d", ErrMalformedStream, r.WallClockUncertaintyNanos, r.Name, s.Seq)
				}
				if r.HasQueryBounds {
					if r.QueryEndNanos < r.QueryStartNanos {
						return nil, nil, nil, fmt.Errorf("%w: query end %d before query start %d on metric %q at sample seq %d", ErrMalformedStream, r.QueryEndNanos, r.QueryStartNanos, r.Name, s.Seq)
					}
					if manifest.SchemaVersion >= SchemaVersionV2 {
						if r.QueryStartNanos < s.QueryStartNanos || r.QueryEndNanos > s.QueryEndNanos {
							return nil, nil, nil, fmt.Errorf("%w: reading query bounds [%d, %d] outside sample pass bounds [%d, %d] on metric %q at sample seq %d", ErrMalformedStream, r.QueryStartNanos, r.QueryEndNanos, s.QueryStartNanos, s.QueryEndNanos, r.Name, s.Seq)
						}
					} else if s.QueryStartNanos != 0 || s.QueryEndNanos != 0 {
						if r.QueryStartNanos < s.QueryStartNanos || (s.QueryEndNanos != 0 && r.QueryEndNanos > s.QueryEndNanos) {
							return nil, nil, nil, fmt.Errorf("%w: reading query bounds [%d, %d] outside sample pass bounds [%d, %d] on metric %q at sample seq %d", ErrMalformedStream, r.QueryStartNanos, r.QueryEndNanos, s.QueryStartNanos, s.QueryEndNanos, r.Name, s.Seq)
						}
					}
				}
				if expectedUnit, ok := expectedMetricUnits[r.Name]; ok {
					if r.Unit != expectedUnit {
						return nil, nil, nil, fmt.Errorf("%w: metric %q has unexpected unit %q, expected %q", ErrMalformedStream, r.Name, r.Unit, expectedUnit)
					}
				}
			}

			samples = append(samples, s)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, nil, fmt.Errorf("read stream scanner: %w", err)
	}

	if manifest == nil {
		return nil, nil, nil, fmt.Errorf("%w: missing manifest header", ErrTruncatedRecord)
	}

	if !manifest.IsComplete && manifest.IncompleteReason == "" {
		manifest.IncompleteReason = "missing_completion_manifest"
	}

	return manifest, phases, samples, nil
}

// ErrIncompleteStream indicates a stream terminated without its final completeness record.
var ErrIncompleteStream = errors.New("incomplete telemetry stream: missing final completeness manifest")

// ReadStreamStrict reads a stream and returns ErrIncompleteStream if the final completeness record is missing.
func ReadStreamStrict(r io.Reader, maxBytes int64) (*Manifest, []PhaseMarker, []Sample, error) {
	manifest, phases, samples, err := ReadStream(r, maxBytes)
	if err != nil {
		return nil, nil, nil, err
	}
	if !manifest.IsComplete {
		return manifest, phases, samples, fmt.Errorf("%w (%s)", ErrIncompleteStream, manifest.IncompleteReason)
	}
	return manifest, phases, samples, nil
}
