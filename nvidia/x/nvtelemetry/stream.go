package nvtelemetry

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
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
	schema       int
	familyLoss   map[string]tracecontract.Loss
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
	if m == nil || m.SchemaVersion <= 0 || m.SchemaVersion > SchemaVersionV5 {
		return ErrUnsupportedSchema
	}
	if err := validateVersionedRecord(StreamRecord{Manifest: m}, m.SchemaVersion); err != nil {
		return err
	}
	for _, d := range m.Devices {
		if strings.TrimSpace(d.UUID) == "" || d.Status == DeviceStatusUnidentifiable {
			return fmt.Errorf("%w: %w", ErrIdentityMismatch, ErrUnidentifiableDevice)
		}
	}
	rec := StreamRecord{Manifest: m}
	if err := w.writeRecordLocked(rec); err != nil {
		return err
	}
	w.schema = m.SchemaVersion
	return nil
}

// WriteSample writes a sample line to the stream.
func (w *Writer) WriteSample(s *Sample) (err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if s == nil || w.w == nil {
		return ErrMalformedStream
	}
	version := w.schema
	if version == 0 {
		version = SchemaVersionV2
	}
	if err := validateVersionedRecord(StreamRecord{Sample: s}, version); err != nil {
		return err
	}
	before := w.lostRecords
	defer func() {
		if version < SchemaVersionV3 {
			return
		}
		if w.familyLoss == nil {
			w.familyLoss = make(map[string]tracecontract.Loss)
		}
		for _, r := range s.Readings {
			if r.Observation == nil || r.Observation.Family == "" {
				continue
			}
			family := r.Observation.Family
			loss := w.familyLoss[family]
			if loss.Total == math.MaxUint64 {
				loss.Unknown = true
			} else {
				loss.Total++
			}
			if w.lostRecords > before {
				if loss.Dropped == math.MaxUint64 {
					loss.Unknown = true
				} else {
					loss.Dropped++
				}
			}
			if err != nil {
				loss.Unknown = true
			}
			w.familyLoss[family] = loss
		}
	}()
	if w.config.MaxRecordBudget > 0 && w.recordsCount >= w.config.MaxRecordBudget {
		if w.config.BudgetPolicy == BudgetExceededDrop || w.config.DropOnFull {
			w.lostRecords++
			return nil
		}
		return ErrBudgetExceeded
	}
	rec := StreamRecord{Sample: s}
	err = w.writeRecordLocked(rec)
	if err == nil {
		w.recordsCount++
	}
	return err
}

// WritePhase writes a phase transition marker to the stream.
func (w *Writer) WritePhase(p *PhaseMarker) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if p == nil || w.w == nil {
		return ErrMalformedStream
	}
	version := w.schema
	if version == 0 {
		version = SchemaVersionV2
	}
	rec := StreamRecord{Phase: p}
	if err := validateVersionedRecord(rec, version); err != nil {
		return err
	}
	return w.writeRecordLocked(rec)
}

func (w *Writer) writeRecordLocked(rec StreamRecord) error {
	if w.w == nil {
		return ErrMalformedStream
	}
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

// FamilyLoss returns a copy of V3 attempted and dropped reading counts. A
// Counts use canonical readings, independently of producer receipt units.
// A missing family is unassessed, not measured zero loss. Overflow saturates
// the affected count and marks it unknown.
func (w *Writer) FamilyLoss() map[string]tracecontract.Loss {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make(map[string]tracecontract.Loss, len(w.familyLoss))
	for family, loss := range w.familyLoss {
		out[family] = loss
	}
	return out
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
	if maxBytes > 64<<20 {
		return nil, nil, nil, ErrBudgetExceeded
	}
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
		if err := decodeStreamJSON(line, &rec); err != nil {
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
		version := SchemaVersionV2
		if manifest != nil {
			version = manifest.SchemaVersion
		}
		if rec.Manifest != nil {
			version = rec.Manifest.SchemaVersion
		}
		if version >= SchemaVersionV3 {
			if version >= SchemaVersionV4 {
				if err := rejectDuplicateJSONKeys(line); err != nil {
					return nil, nil, nil, fmt.Errorf("%w: %v", ErrMalformedStream, err)
				}
			}
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			var strict StreamRecord
			if err := d.Decode(&strict); err != nil {
				return nil, nil, nil, fmt.Errorf("%w: %v", ErrMalformedStream, err)
			}
		}
		if err := versionedJSONKeys(line, version); err != nil {
			return nil, nil, nil, err
		}
		if err := validateVersionedRecord(rec, version); err != nil {
			return nil, nil, nil, err
		}
		if rec.Sample != nil && len(samples) >= 65536 || rec.Phase != nil && len(phases) >= 65536 {
			return nil, nil, nil, ErrBudgetExceeded
		}

		// Manifest processing
		if rec.Manifest != nil {
			if rec.Manifest.SchemaVersion <= 0 || rec.Manifest.SchemaVersion > SchemaVersionV5 {
				return nil, nil, nil, ErrUnsupportedSchema
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
				if version >= SchemaVersionV3 && !reflect.DeepEqual(rec.Manifest.Receipt, manifest.Receipt) {
					return nil, nil, nil, fmt.Errorf("%w: capture receipt changed", ErrIdentityMismatch)
				}
				if version >= SchemaVersionV3 && !reflect.DeepEqual(rec.Manifest.Devices, manifest.Devices) {
					return nil, nil, nil, fmt.Errorf("%w: device scope changed", ErrIdentityMismatch)
				}
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

				if version >= SchemaVersionV3 {
					if err := validateFamilyCounts(rec.Manifest.LostRecordsByFamily, samples); err != nil {
						return nil, nil, nil, err
					}
				}

				manifest.EndTimeMonotonic = rec.Manifest.EndTimeMonotonic
				manifest.EndTimeUnix = rec.Manifest.EndTimeUnix
				manifest.TotalSamples = rec.Manifest.TotalSamples
				manifest.MissedTicks = rec.Manifest.MissedTicks
				manifest.LostRecords = rec.Manifest.LostRecords
				manifest.LostRecordsByFamily = rec.Manifest.LostRecordsByFamily
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
				managedReading := r
				if manifest.SchemaVersion >= SchemaVersionV3 && r.Number != nil {
					if r.Number.Uint != nil {
						managedReading.IsInt = true
						managedReading.IntValue = *r.Number.Uint
						managedReading.Value = float64(*r.Number.Uint)
					}
					if r.Number.Float != nil {
						managedReading.Value = *r.Number.Float
					}
					if r.Number.Int != nil && r.Source != nil {
						return nil, nil, nil, fmt.Errorf("%w: signed managed mapping unsupported", ErrMalformedStream)
					}
				}
				if err := validateManagedValue(managedReading); err != nil {
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

// decodeStreamJSON checks keys before decoding into concrete integer fields.
// Map keys remain case sensitive; struct aliases follow encoding/json matching.
func decodeStreamJSON(raw []byte, rec *StreamRecord) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	if err := checkStreamJSON(d, 0, reflect.TypeOf(*rec)); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return errors.New("trailing stream json")
	}
	return json.Unmarshal(raw, rec)
}

func checkStreamJSON(d *json.Decoder, depth int, typ reflect.Type) error {
	if depth > 128 {
		return errors.New("stream json nesting limit exceeded")
	}
	t, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	for typ != nil && typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			t, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := t.(string)
			if !ok {
				return errors.New("invalid stream json key")
			}
			name := key
			var next reflect.Type
			if typ != nil && typ.Kind() == reflect.Struct {
				// Prefer an exact tag match, as encoding/json does.
				for i := 0; i < typ.NumField(); i++ {
					f := typ.Field(i)
					if f.PkgPath != "" {
						continue
					}
					tag := strings.Split(f.Tag.Get("json"), ",")[0]
					if tag == "-" {
						continue
					}
					if tag == "" {
						tag = f.Name
					}
					if tag == key {
						name, next = tag, f.Type
						break
					}
					if next == nil && strings.EqualFold(tag, key) {
						name, next = tag, f.Type
					}
				}
			} else if typ != nil && typ.Kind() == reflect.Map {
				next = typ.Elem()
			}
			if seen[name] {
				return fmt.Errorf("duplicate stream json key %q", key)
			}
			seen[name] = true
			if err := checkStreamJSON(d, depth+1, next); err != nil {
				return err
			}
		}
	case '[':
		var next reflect.Type
		if typ != nil && (typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array) {
			next = typ.Elem()
		}
		for d.More() {
			if err := checkStreamJSON(d, depth+1, next); err != nil {
				return err
			}
		}
	default:
		return errors.New("invalid stream json delimiter")
	}
	_, err = d.Token()
	return err
}

// validateFamilyCounts compares only completion counts, in canonical reading
// units. Producer receipt totals may count different records and are independent.
func validateFamilyCounts(losses map[string]tracecontract.Loss, samples []Sample) error {
	retained := make(map[string]uint64)
	for _, s := range samples {
		for _, r := range s.Readings {
			if r.Observation != nil && r.Observation.Family != "" {
				retained[r.Observation.Family]++
			}
		}
	}
	for family, loss := range losses {
		if !loss.Unknown && (loss.Dropped > loss.Total || loss.Total-loss.Dropped != retained[family]) {
			return fmt.Errorf("%w: inconsistent canonical reading counts for family %q", ErrMalformedStream, family)
		}
	}
	return nil
}

func validateVersionedRecord(rec StreamRecord, version int) error {
	if version < SchemaVersionLegacy || version > SchemaVersionV5 {
		return ErrUnsupportedSchema
	}
	if version < SchemaVersionV5 && rec.Sample != nil {
		for _, r := range rec.Sample.Readings {
			if o := r.Observation; o != nil {
				h := o.Histogram
				if o.OTLPScopeName != nil || o.OTLPClock != nil || o.OTLPSourceTime != nil || h != nil && (h.Count != nil || h.Sum != nil || h.Min != nil || h.Max != nil || h.Temporality != "" || h.Flags != 0) {
					return fmt.Errorf("%w: V5 field in older reading", ErrMalformedStream)
				}
			}
		}
	}
	if version < SchemaVersionV3 {
		if rec.Manifest != nil && (rec.Manifest.Receipt != nil || rec.Manifest.LostRecordsByFamily != nil) {
			return fmt.Errorf("%w: V3 metadata in legacy manifest", ErrMalformedStream)
		}
		if rec.Sample != nil {
			for _, r := range rec.Sample.Readings {
				if r.Number != nil || r.Observation != nil {
					return fmt.Errorf("%w: V3 reading in legacy sample", ErrMalformedStream)
				}
			}
		}
		return nil
	}
	if err := boundedMetadata(reflect.ValueOf(rec), 0, version == SchemaVersionV5); err != nil {
		return fmt.Errorf("%w: %v", ErrMalformedStream, err)
	}
	if m := rec.Manifest; m != nil {
		if m.Receipt != nil {
			if _, err := m.Receipt.JSON(); err != nil {
				return fmt.Errorf("%w: %v", ErrMalformedStream, err)
			}
		}
		for family, loss := range m.LostRecordsByFamily {
			if family == "" || (!loss.Unknown && loss.Dropped > loss.Total) {
				return fmt.Errorf("%w: invalid family loss", ErrMalformedStream)
			}
		}
	}
	if rec.Sample == nil {
		return nil
	}
	for _, r := range rec.Sample.Readings {
		if version == SchemaVersionV5 {
			switch r.Evidence {
			case EvidenceMeasured, EvidenceDerived, EvidenceUnavailable:
			default:
				return fmt.Errorf("%w: invalid V5 evidence", ErrMalformedStream)
			}
		}
		if version < SchemaVersionV4 && r.Observation != nil && r.Observation.Observer.SamplingCoverage != nil {
			return fmt.Errorf("%w: V4 field in older observer", ErrMalformedStream)
		}
		if r.Value != 0 || r.IntValue != 0 || r.IsInt {
			return fmt.Errorf("%w: legacy scalar in V3 reading", ErrMalformedStream)
		}
		if r.Evidence == EvidenceUnavailable {
			if r.Number != nil || r.UnavailableReason == "" || (r.Observation != nil && r.Observation.Histogram != nil && !(version == SchemaVersionV5 && noRecordedHistogram(r))) {
				return fmt.Errorf("%w: unavailable V3 value", ErrMalformedStream)
			}
		} else {
			if r.UnavailableReason != "" {
				return fmt.Errorf("%w: available V3 value has unavailable reason", ErrMalformedStream)
			}
			if r.Number == nil && (r.Observation == nil || r.Observation.Histogram == nil) {
				return fmt.Errorf("%w: missing V3 number", ErrMalformedStream)
			}
		}
		if r.Number != nil {
			if err := validNumber(r.Number); err != nil {
				return fmt.Errorf("%w: %v", ErrMalformedStream, err)
			}
		}
		if o := r.Observation; o != nil {
			switch o.SourceClass {
			case "", "native", "managed", "imported", "synthetic", "modeled":
			default:
				return fmt.Errorf("%w: unknown source class", ErrMalformedStream)
			}
			switch o.Kind {
			case "", "gauge", "counter", "histogram", "host_call", "kernel", "sched_running", "sched_runnable", "sched_blocked":
			default:
				return fmt.Errorf("%w: unknown observation kind", ErrMalformedStream)
			}
			for _, hash := range []string{o.RawSHA256, o.TransformSHA256} {
				if hash != "" && !streamHash(hash) {
					return fmt.Errorf("%w: invalid observation digest", ErrMalformedStream)
				}
			}
			if o.UncertaintyNS != nil && *o.UncertaintyNS < 0 {
				return fmt.Errorf("%w: negative observation uncertainty", ErrMalformedStream)
			}
			if o.Interval != nil && (o.Interval.Start < 0 || o.Interval.End < o.Interval.Start) {
				return fmt.Errorf("%w: invalid observation interval", ErrMalformedStream)
			}
			if o.Denominator != nil {
				if err := validNumber(o.Denominator); err != nil {
					return fmt.Errorf("%w: denominator: %v", ErrMalformedStream, err)
				}
			}
			if o.CounterReset && o.Kind != "counter" {
				return fmt.Errorf("%w: reset on non-counter", ErrMalformedStream)
			}
			if o.Histogram != nil {
				if o.Kind != "histogram" || r.Number != nil {
					return fmt.Errorf("%w: conflicting histogram value", ErrMalformedStream)
				}
				var err error
				if version == SchemaVersionV5 {
					err = validHistogramV5(o.Histogram)
				} else {
					err = validHistogram(o.Histogram)
				}
				if err != nil {
					return fmt.Errorf("%w: %v", ErrMalformedStream, err)
				}
			}
			if version == SchemaVersionV5 {
				if err := validOTLPObservation(r); err != nil {
					return fmt.Errorf("%w: %v", ErrMalformedStream, err)
				}
			}
			if version >= SchemaVersionV4 {
				if o.Observer.SamplingCoverage == nil {
					return fmt.Errorf("%w: missing V4 observer sampling coverage state", ErrMalformedStream)
				}
				if err := validSamplingCoverage(o.Observer, o.Interval); err != nil {
					return fmt.Errorf("%w: observer sampling coverage: %v", ErrMalformedStream, err)
				}
			}
		}
	}
	return nil
}

func validSamplingCoverage(observer Observer, interval *tracecontract.Interval) error {
	coverage := observer.SamplingCoverage
	if coverage == nil {
		return errors.New("absent descriptor")
	}
	validBasis := func(basis string) bool {
		switch basis {
		case "", "time", "event", "pc_sample", "range":
			return true
		default:
			return false
		}
	}
	if !validBasis(coverage.Basis) {
		return errors.New("unknown basis")
	}
	switch coverage.Status {
	case "measured":
		if observer.ID == "" || observer.Version == "" || observer.Mode == "" || observer.Pass == "" || observer.Population == "" {
			return errors.New("measured coverage requires observer identity")
		}
		if coverage.Basis == "" || coverage.Scope == "" || coverage.Unit == "" || coverage.Numerator == nil || coverage.Denominator == nil {
			return errors.New("measured coverage requires basis, scope, unit and operands")
		}
		numerator, ok := coverageInteger(coverage.Numerator)
		if !ok {
			return errors.New("numerator must be a nonnegative integer")
		}
		denominator, ok := coverageInteger(coverage.Denominator)
		if !ok || denominator == 0 || numerator > denominator {
			return errors.New("denominator must be positive and at least numerator")
		}
		if coverage.Basis == "time" {
			if interval == nil || interval.Start < 0 || interval.End <= interval.Start || coverage.Unit != "ns" {
				return errors.New("time coverage requires a positive nanosecond interval")
			}
			span := uint64(interval.End - interval.Start)
			if denominator != span {
				return errors.New("time denominator must equal observation interval")
			}
		}
	case "unknown", "unavailable", "not_applicable":
		if coverage.Numerator != nil || coverage.Denominator != nil || coverage.Unit != "" {
			return errors.New("non-measured coverage cannot have operands or unit")
		}
		if coverage.Status == "unavailable" && coverage.Reason == "" {
			return errors.New("unavailable coverage requires reason")
		}
	default:
		return errors.New("unknown status")
	}
	return nil
}

func coverageInteger(number *Number) (uint64, bool) {
	if number == nil {
		return 0, false
	}
	if number.Uint != nil && number.Float == nil && number.Int == nil {
		return *number.Uint, true
	}
	if number.Int != nil && number.Float == nil && number.Uint == nil && *number.Int >= 0 {
		return uint64(*number.Int), true
	}
	return 0, false
}

func validNumber(n *Number) error {
	if n == nil {
		return errors.New("absent number")
	}
	count := 0
	if n.Float != nil {
		count++
		if math.IsNaN(*n.Float) || math.IsInf(*n.Float, 0) {
			return errors.New("nonfinite number")
		}
	}
	if n.Int != nil {
		count++
	}
	if n.Uint != nil {
		count++
	}
	if count != 1 {
		return errors.New("number requires exactly one arm")
	}
	return nil
}

func validHistogram(h *Histogram) error {
	if len(h.Bounds) == 0 || len(h.Bounds) != len(h.Counts) || len(h.Bounds) > 256 {
		return errors.New("invalid histogram size")
	}
	var previous *big.Rat
	for i, bound := range h.Bounds {
		if i > 0 && h.Counts[i] < h.Counts[i-1] {
			return errors.New("noncumulative histogram counts")
		}
		if bound == "+Inf" && i == len(h.Bounds)-1 {
			continue
		}
		value, ok := boundedDecimal(bound)
		if !ok || strings.ContainsAny(bound, "/ ") || (previous != nil && value.Cmp(previous) <= 0) {
			return errors.New("invalid histogram boundary")
		}
		previous = value
	}
	return nil
}

func streamHash(hash string) bool {
	if len(hash) != 64 {
		return false
	}
	for _, c := range hash {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func boundedV3(v reflect.Value, depth int) error { return boundedMetadata(v, depth, false) }

func boundedMetadata(v reflect.Value, depth int, v5 bool) error {
	if depth > 32 {
		return errors.New("V3 metadata depth limit")
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return nil
		}
		return boundedMetadata(v.Elem(), depth+1, v5)
	}
	switch v.Kind() {
	case reflect.String:
		if v.Len() > 4096 {
			return errors.New("V3 string limit")
		}
	case reflect.Struct:
		if v5 && v.Type() == reflect.TypeOf(Histogram{}) {
			h := v.Interface().(Histogram)
			if len(h.Bounds) > 4097 || len(h.Counts) > 4097 {
				return errors.New("V5 histogram collection limit")
			}
			for _, b := range h.Bounds {
				if len(b) > 4096 {
					return errors.New("V5 bound string limit")
				}
			}
			return nil
		}
		for i := 0; i < v.NumField(); i++ {
			if err := boundedMetadata(v.Field(i), depth+1, v5); err != nil {
				return err
			}
		}
	case reflect.Slice, reflect.Array:
		if v.Len() > 4096 {
			return errors.New("V3 collection limit")
		}
		for i := 0; i < v.Len(); i++ {
			if err := boundedMetadata(v.Index(i), depth+1, v5); err != nil {
				return err
			}
		}
	case reflect.Map:
		if v.Len() > 256 {
			return errors.New("V3 map limit")
		}
		iter := v.MapRange()
		for iter.Next() {
			if err := boundedMetadata(iter.Key(), depth+1, v5); err != nil {
				return err
			}
			if err := boundedMetadata(iter.Value(), depth+1, v5); err != nil {
				return err
			}
		}
	}
	return nil
}

// Presence matters even for null or zero: old-version readers must not accept
// contract-critical V3 fields, and V3 must not carry a second legacy scalar.
func versionedJSONKeys(raw []byte, version int) error {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return err
	}
	for key, data := range envelope {
		if strings.EqualFold(key, "manifest") && version < SchemaVersionV3 {
			var fields map[string]json.RawMessage
			json.Unmarshal(data, &fields)
			for key := range fields {
				if strings.EqualFold(key, "receipt") || strings.EqualFold(key, "lost_records_by_family") {
					return fmt.Errorf("%w: V3 field in legacy manifest", ErrMalformedStream)
				}
			}
		}
		if !strings.EqualFold(key, "sample") {
			continue
		}
		var sample map[string]json.RawMessage
		json.Unmarshal(data, &sample)
		for key, data := range sample {
			if !strings.EqualFold(key, "readings") {
				continue
			}
			var readings []map[string]json.RawMessage
			json.Unmarshal(data, &readings)
			for _, fields := range readings {
				for key, readingData := range fields {
					if version < SchemaVersionV3 && (strings.EqualFold(key, "number") || strings.EqualFold(key, "observation")) {
						return fmt.Errorf("%w: V3 field in legacy reading", ErrMalformedStream)
					}
					if version >= SchemaVersionV3 && (strings.EqualFold(key, "value") || strings.EqualFold(key, "int_value") || strings.EqualFold(key, "is_int")) {
						return fmt.Errorf("%w: legacy scalar in V3 reading", ErrMalformedStream)
					}
					if strings.EqualFold(key, "observation") {
						var observation map[string]json.RawMessage
						if err := json.Unmarshal(readingData, &observation); err == nil {
							for observationKey, observerData := range observation {
								switch strings.ToLower(observationKey) {
								case "otlp_scope_name", "otlp_clock_metadata", "otlp_source_time":
									if version < SchemaVersionV5 {
										return fmt.Errorf("%w: V5 key in older observation", ErrMalformedStream)
									}
									if observationKey != strings.ToLower(observationKey) {
										return fmt.Errorf("%w: noncanonical V5 key", ErrMalformedStream)
									}
									if bytes.Equal(bytes.TrimSpace(observerData), []byte("null")) {
										return fmt.Errorf("%w: null V5 metadata", ErrMalformedStream)
									}
									var typ reflect.Type
									if observationKey == "otlp_clock_metadata" {
										typ = reflect.TypeOf(OTLPClockMetadata{})
									}
									if observationKey == "otlp_source_time" {
										typ = reflect.TypeOf(OTLPSourceTime{})
									}
									if typ != nil {
										if err := requiredJSONFields(observerData, typ); err != nil {
											return fmt.Errorf("%w: %v", ErrMalformedStream, err)
										}
									}

								case "histogram":
									var histogram map[string]json.RawMessage
									if json.Unmarshal(observerData, &histogram) == nil {
										for k := range histogram {
											switch strings.ToLower(k) {
											case "count", "sum", "min", "max", "temporality", "flags":
												if version < SchemaVersionV5 {
													return fmt.Errorf("%w: V5 key in older histogram", ErrMalformedStream)
												}
												if k != strings.ToLower(k) {
													return fmt.Errorf("%w: noncanonical V5 histogram key", ErrMalformedStream)
												}
												if bytes.Equal(bytes.TrimSpace(histogram[k]), []byte("null")) {
													return fmt.Errorf("%w: null V5 histogram field", ErrMalformedStream)
												}
											}
										}
									}
								}

								if !strings.EqualFold(observationKey, "observer") {
									continue
								}
								var observer map[string]json.RawMessage
								if err := json.Unmarshal(observerData, &observer); err == nil {
									for observerKey, coverageData := range observer {
										if !strings.EqualFold(observerKey, "sampling_coverage") {
											continue
										}
										if version < SchemaVersionV4 {
											return fmt.Errorf("%w: V4 field in older observer", ErrMalformedStream)
										}
										if observerKey != "sampling_coverage" {
											return fmt.Errorf("%w: noncanonical sampling coverage key", ErrMalformedStream)
										}
										var fields map[string]json.RawMessage
										if err := json.Unmarshal(coverageData, &fields); err == nil {
											for field := range fields {
												switch field {
												case "status", "basis", "scope", "numerator", "denominator", "unit", "reason":
												default:
													return fmt.Errorf("%w: noncanonical sampling coverage field", ErrMalformedStream)
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return nil
}

func rejectDuplicateJSONKeys(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := consumeJSONValue(d); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func consumeJSONValue(d *json.Decoder) error {
	token, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			keyToken, err := d.Token()
			if err != nil {
				return err
			}
			key, ok := keyToken.(string)
			if !ok {
				return errors.New("object key is not a string")
			}
			folded := strings.ToLower(key)
			if seen[folded] {
				return errors.New("duplicate or case-folded JSON key")
			}
			seen[folded] = true
			if err := consumeJSONValue(d); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	case '[':
		for d.More() {
			if err := consumeJSONValue(d); err != nil {
				return err
			}
		}
		_, err = d.Token()
		return err
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delim)
	}
}

func noRecordedHistogram(r Reading) bool {
	h := r.Observation.Histogram
	return r.Evidence == EvidenceUnavailable && r.UnavailableReason == "no_recorded_value" && h.Flags == 1
}

func validHistogramV5(h *Histogram) error {
	if h.Count == nil || len(h.Bounds) != len(h.Counts) || len(h.Bounds) > 4097 || h.Temporality != "delta" && h.Temporality != "cumulative" || h.Flags > 1 {
		return errors.New("invalid V5 histogram shape")
	}
	for _, x := range []*float64{h.Sum, h.Min, h.Max} {
		if x != nil && (math.IsNaN(*x) || math.IsInf(*x, 0)) {
			return errors.New("nonfinite histogram summary")
		}
	}
	if h.Min != nil && h.Max != nil && *h.Min > *h.Max || *h.Count == 0 && (h.Sum != nil && *h.Sum != 0 || h.Min != nil || h.Max != nil) {
		return errors.New("invalid histogram summary")
	}
	if h.Flags == 1 && (*h.Count != 0 || len(h.Bounds) != 0 || h.Sum != nil || h.Min != nil || h.Max != nil) {
		return errors.New("invalid no-recorded histogram")
	}
	if len(h.Bounds) == 0 {
		return nil
	}
	if h.Bounds[len(h.Bounds)-1] != "+Inf" || h.Counts[len(h.Counts)-1] != *h.Count {
		return errors.New("missing histogram terminal count")
	}
	var previous float64
	for i, b := range h.Bounds {
		if i > 0 && h.Counts[i] < h.Counts[i-1] {
			return errors.New("noncumulative histogram counts")
		}
		if i == len(h.Bounds)-1 {
			break
		}
		v, err := histogramBound(b)
		if err != nil || i > 0 && v <= previous || b != strconv.FormatFloat(v, 'g', -1, 64) {
			return errors.New("noncanonical histogram bound")
		}
		previous = v
	}
	return nil
}

func histogramBound(s string) (float64, error) {
	if len(s) == 0 || len(s) > 4096 {
		return 0, errors.New("invalid histogram bound")
	}
	// A JSON number grammar excludes rational strings and nonfinite spellings.
	if !json.Valid([]byte(s)) || s[0] == '"' || s[0] == '[' || s[0] == '{' || s == "null" || s == "true" || s == "false" {
		return 0, errors.New("invalid histogram bound")
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, errors.New("nonfinite histogram bound")
	}
	if v == 0 {
		mantissa := strings.SplitN(strings.ToLower(s), "e", 2)[0]
		if strings.ContainsAny(mantissa, "123456789") {
			return 0, errors.New("histogram bound underflow")
		}
		return 0, nil
	}
	return v, nil
}

const otlpScopeName = "github.com/tmc/lib/nvidia/x/nvtelemetry"

func validOTLPClock(c OTLPClockMetadata) bool {
	return exportText(c.ClockDomain) && c.ClockCalibrationBoundsNanos >= 0 && (!c.HasClockCalibration || c.ClockDomain != "") && (c.HasClockCalibration || c.ClockCalibrationBoundsNanos == 0)
}

func validOTLPObservation(r Reading) error {
	o := r.Observation
	if o.OTLPScopeName != nil || o.OTLPClock != nil || o.OTLPSourceTime != nil {
		if o.Kind != "histogram" || o.Histogram == nil {
			return errors.New("OTLP metadata on non-histogram")
		}
	}
	if o.Histogram != nil {
		if o.Histogram.Flags == 1 && !noRecordedHistogram(r) || o.Histogram.Flags == 0 && r.Evidence == EvidenceUnavailable {
			return errors.New("histogram flags/evidence mismatch")
		}
		if o.SourceClass == "imported" {
			if o.OTLPClock == nil || o.OTLPScopeName == nil || r.TimestampMonotonic != 0 || r.TimestampUnix <= 0 || o.ClockEpoch != "unix" {
				return errors.New("missing imported OTLP provenance")
			}
			if o.Interval != nil && (o.Interval.Clock != "unix" || o.Interval.Start <= 0 || o.Interval.End != r.TimestampUnix) {
				return errors.New("invalid imported interval")
			}
		}
	}
	if o.OTLPScopeName != nil && (o.SourceClass != "imported" || *o.OTLPScopeName != "" && *o.OTLPScopeName != otlpScopeName) {
		return errors.New("invalid OTLP scope provenance")
	}
	if o.OTLPClock != nil && !validOTLPClock(*o.OTLPClock) {
		return errors.New("invalid OTLP clock provenance")
	}
	if o.OTLPSourceTime != nil {
		if o.SourceClass != "imported" {
			return errors.New("source time on unprojected reading")
		}
		if err := validOTLPSourceTime(r); err != nil {
			return err
		}
	}
	return nil
}

func checkedUnixTime(v, offset int64) (int64, error) {
	if offset > 0 && v > math.MaxInt64-offset || offset < 0 && v < math.MinInt64-offset {
		return 0, errors.New("Unix mapping overflow")
	}
	t := v + offset
	if t <= 0 {
		return 0, errors.New("nonpositive mapped Unix time")
	}
	return t, nil
}

func validOTLPSourceTime(r Reading) error {
	s := r.Observation.OTLPSourceTime
	for _, v := range []string{s.SourceClass, s.SourceClockEpoch, s.MappingDomain, s.MappingEpoch, s.MappingProvenance} {
		if v == "" || !exportText(v) {
			return errors.New("incomplete source-time assertion")
		}
	}
	if s.SourceClass != "native" && s.SourceClass != "managed" && s.SourceClass != "synthetic" && s.SourceClass != "modeled" || s.SourceMonotonicEndNanos == 0 || s.SourceClockEpoch != s.MappingEpoch || s.MappingValidFromNanos > s.MappingValidUntilNanos || s.SourceMonotonicEndNanos < s.MappingValidFromNanos || s.SourceMonotonicEndNanos > s.MappingValidUntilNanos {
		return errors.New("invalid source-time assertion")
	}
	end, err := checkedUnixTime(s.SourceMonotonicEndNanos, s.MappingUnixOffsetNanos)
	if err != nil || end != r.TimestampUnix {
		return errors.New("source-time end mismatch")
	}
	if !s.SourceIntervalPresent {
		if s.SourceIntervalClock != "" || s.SourceIntervalStartNanos != 0 || s.SourceIntervalEndNanos != 0 || r.Observation.Interval != nil {
			return errors.New("source-time interval presence mismatch")
		}
	} else {
		start, err := checkedUnixTime(s.SourceIntervalStartNanos, s.MappingUnixOffsetNanos)
		i := r.Observation.Interval
		if err != nil || i == nil || s.SourceIntervalClock != s.MappingDomain || s.SourceIntervalStartNanos <= 0 || s.SourceIntervalEndNanos != s.SourceMonotonicEndNanos || s.SourceIntervalStartNanos < s.MappingValidFromNanos || s.SourceIntervalStartNanos > s.MappingValidUntilNanos || start != i.Start || i.End != end || start > end {
			return errors.New("source-time interval mismatch")
		}
	}
	return nil
}

func requiredJSONFields(raw []byte, typ reflect.Type) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if len(fields) != typ.NumField() {
		return errors.New("incomplete V5 metadata object")
	}
	for i := 0; i < typ.NumField(); i++ {
		key := typ.Field(i).Tag.Get("json")
		value, ok := fields[key]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return errors.New("missing or null V5 metadata field")
		}
	}
	return nil
}
