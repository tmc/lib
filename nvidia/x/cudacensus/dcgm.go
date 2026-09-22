package cudacensus

import (
	"fmt"
	"math"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

// dcgmFields maps the ratio fields to the same fraction-valued counters as GPM.
// DMMA has no corresponding field in the current binding and remains absent.
var dcgmFields = map[Counter]dcgm.FieldID{
	GraphicsActive:    dcgm.FieldProfGrEngineActive,
	SMActive:          dcgm.FieldProfSmActive,
	SMOccupancy:       dcgm.FieldProfSmOccupancy,
	DRAMBandwidthUtil: dcgm.FieldProfDramActive,
	FP64Active:        dcgm.FieldProfPipeFP64Active,
	FP32Active:        dcgm.FieldProfPipeFP32Active,
	FP16Active:        dcgm.FieldProfPipeFP16Active,
	IntActive:         dcgm.FieldProfPipeIntActive,
	AnyTensorActive:   dcgm.FieldProfPipeTensorActive,
	HMMATensorActive:  dcgm.FieldProfPipeTensorHmmaActive,
	IMMATensorActive:  dcgm.FieldProfPipeTensorImmaActive,
	DFMATensorActive:  dcgm.FieldProfPipeTensorDfmaActive,
}

// DCGMReader reads cached activity ratios for one whole GPU. The caller owns
// the DCGM connection and must watch Fields before reading. This reader does
// not start a host engine, install watches, or alter device settings. It is not
// safe for concurrent use. The zero value is not usable.
type DCGMReader struct {
	want   []Counter
	fields []uint16
	index  []int
	values []dcgm.FieldValueV1
	times  []time.Time
	maxAge time.Duration
	query  func(*uint16, uint32, *dcgm.FieldValueV1) error
	now    func() time.Time
}

// DCGMSample includes the host engine's per-field sample times, in reading
// order. SampleTimes is borrowed until the reader's next Read; copy it to retain
// evidence. Readings uses the caller's destination. IntervalNs is -1: cached
// timestamps do not establish the integration interval of an activity ratio.
type DCGMSample struct {
	Sample
	SampleTimes []time.Time
}

// NewDCGMReader prepares reusable buffers without querying the device. maxAge
// must be positive and bounds cached sample age against host observation time.
// gpuID names a whole GPU, not a MIG instance. Unsupported mappings remain
// unavailable; a mapping alone does not prove device support.
func NewDCGMReader(h dcgm.Handle, gpuID int32, want []Counter, maxAge time.Duration) (*DCGMReader, error) {
	if h == 0 || gpuID < 0 || maxAge <= 0 {
		return nil, fmt.Errorf("cudacensus: DCGM reader requires a handle, nonnegative GPU ID and positive maximum age")
	}
	if err := validateCounters(want); err != nil {
		return nil, err
	}
	s := &DCGMReader{want: append([]Counter(nil), want...), index: make([]int, len(want)), times: make([]time.Time, len(want)), maxAge: maxAge, now: time.Now}
	for i, c := range want {
		s.index[i] = -1
		if f, ok := dcgmFields[c]; ok {
			s.index[i] = len(s.fields)
			s.fields = append(s.fields, uint16(f))
		}
	}
	s.values = make([]dcgm.FieldValueV1, len(s.fields))
	s.query = func(fields *uint16, count uint32, values *dcgm.FieldValueV1) error {
		return h.GetLatestValuesForFields(gpuID, fields, count, values)
	}
	return s, nil
}

// Fields returns the DCGM fields the caller must watch, in query order.
// The returned slice belongs to the caller and omits unmapped counters.
func (s *DCGMReader) Fields() []dcgm.FieldID {
	fields := make([]dcgm.FieldID, len(s.fields))
	for i, f := range s.fields {
		fields[i] = dcgm.FieldID(f)
	}
	return fields
}

// Read fills dst with one result per requested counter. Missing, blank, failed,
// stale, future-dated and invalid fields are unavailable, never a carried-over
// value. Successful fields retain their own timestamps; fields sampled at
// different times are not claimed to be one simultaneous observation.
func (s *DCGMReader) Read(dst []Reading) (DCGMSample, error) {
	if s == nil || s.query == nil {
		return DCGMSample{}, fmt.Errorf("cudacensus: uninitialized DCGM reader")
	}
	if len(dst) < len(s.want) {
		return DCGMSample{}, fmt.Errorf("cudacensus: destination holds %d readings, need %d", len(dst), len(s.want))
	}
	dst = dst[:len(s.want)]
	for i, c := range s.want {
		dst[i] = Reading{Counter: c, Reason: "DCGM query unavailable"}
		s.times[i] = time.Time{}
	}
	for i := range s.values {
		// Clearing the entire block prevents partial native writes from retaining
		// the last sample. Field ID zero cannot match any requested profile field.
		s.values[i] = dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, Status: int32(dcgm.ErrNoData)}
		s.values[i].Value[0] = math.Float64bits(dcgm.Fp64Blank)
	}
	if len(s.fields) > 0 {
		if err := s.query(&s.fields[0], uint32(len(s.fields)), &s.values[0]); err != nil {
			return DCGMSample{}, fmt.Errorf("cudacensus: DCGM latest fields: %w", err)
		}
	}
	now := s.now()
	for i, idx := range s.index {
		if idx < 0 {
			dst[i].Reason = "no DCGM field mapping"
			continue
		}
		v := &s.values[idx]
		if v.FieldId != s.fields[idx] {
			dst[i].Reason = "missing or mismatched DCGM field"
			continue
		}
		s.times[i] = v.Time()
		if err := v.Err(); err != nil {
			dst[i].Reason = err.Error()
			continue
		}
		value, ok := v.Float64()
		if !ok || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			dst[i].Reason = "blank, invalid or non-ratio DCGM field"
			continue
		}
		if s.times[i].IsZero() || s.times[i].After(now) || now.Sub(s.times[i]) > s.maxAge {
			dst[i].Reason = "missing, future or stale DCGM timestamp"
			continue
		}
		dst[i] = Reading{Counter: s.want[i], Value: value, Available: true}
	}
	return DCGMSample{Sample: Sample{Readings: dst, IntervalNs: -1, TakenAt: now}, SampleTimes: s.times}, nil
}
