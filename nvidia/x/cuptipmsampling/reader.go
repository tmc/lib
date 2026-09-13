package cuptipmsampling

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"time"
)

// Sample represents one decoded hardware Performance Monitor (PM) observation
// over a time or cycle interval.
type Sample struct {
	Index          uint               // 0-based sample sequence index
	StartTimestamp uint64             // Nanoseconds in CUPTI clock domain
	EndTimestamp   uint64             // Nanoseconds in CUPTI clock domain
	DurationNs     uint64             // Duration in nanoseconds (EndTimestamp - StartTimestamp)
	Values         map[string]float64 // Metric name to measured counter value
}

// SamplerConfig provides high-level configuration for periodic PM counter sampling
// in diagnostic tools without manual buffer sizing or low-level boilerplate.
type SamplerConfig struct {
	// DeviceIndex is the 0-based CUDA device ordinal.
	DeviceIndex uint32

	// Interval specifies the sampling period (defaults to 100µs if zero).
	Interval time.Duration

	// MetricNames specifies the hardware performance metrics to sample.
	MetricNames []string

	// HardwareBufferSize is the size of the GPU hardware buffer in bytes (defaults to 64MB).
	HardwareBufferSize uint

	// MaxSamples is the maximum number of samples retained in the counter data image (defaults to 65536).
	MaxSamples uint32

	// TriggerMode selects time- or cycle-based periodic triggers (defaults to TriggerModeTimeInterval).
	TriggerMode TriggerMode

	// AppendMode selects overflow policy (defaults to AppendModeKeepLatest).
	AppendMode AppendMode
}

// Config converts SamplerConfig into the low-level Config required by NewSession,
// applying safe defaults for zero fields. configImage must be provided from
// profiler host configuration.
func (c SamplerConfig) Config(configImage []byte) Config {
	interval := uint64(c.Interval.Nanoseconds())
	if interval == 0 {
		interval = uint64(100 * time.Microsecond)
	}
	hwBuf := c.HardwareBufferSize
	if hwBuf == 0 {
		hwBuf = 64 * 1024 * 1024
	}
	maxSamples := c.MaxSamples
	if maxSamples == 0 {
		maxSamples = 65536
	}
	trigger := c.TriggerMode
	if trigger == 0 {
		trigger = TriggerModeTimeInterval
	}
	appendMode := c.AppendMode
	if appendMode == 0 {
		appendMode = AppendModeKeepLatest
	}

	return Config{
		DeviceIndex:        c.DeviceIndex,
		ConfigImage:        configImage,
		HardwareBufferSize: hwBuf,
		SamplingInterval:   interval,
		TriggerMode:        trigger,
		AppendMode:         appendMode,
		MaxSamples:         maxSamples,
		MetricNames:        append([]string(nil), c.MetricNames...),
	}
}

// Reader provides an idiomatic, streaming reader over decoded PM counter samples.
type Reader struct {
	samples []Sample
	pos     int
}

// NewReader creates a Reader over the provided slice of decoded samples.
func NewReader(samples []Sample) *Reader {
	return &Reader{
		samples: append([]Sample(nil), samples...),
	}
}

// ReadSample returns the next Sample, or io.EOF when all samples are consumed.
func (r *Reader) ReadSample() (*Sample, error) {
	if r == nil || r.pos >= len(r.samples) {
		return nil, io.EOF
	}
	s := &r.samples[r.pos]
	r.pos++
	return s, nil
}

// All returns all remaining unread samples, advancing the reader to the end.
func (r *Reader) All() []Sample {
	if r == nil || r.pos >= len(r.samples) {
		return nil
	}
	remaining := r.samples[r.pos:]
	r.pos = len(r.samples)
	return append([]Sample(nil), remaining...)
}

// Len returns the total number of samples.
func (r *Reader) Len() int {
	if r == nil {
		return 0
	}
	return len(r.samples)
}

// Remaining returns the number of unread samples.
func (r *Reader) Remaining() int {
	if r == nil || r.pos >= len(r.samples) {
		return 0
	}
	return len(r.samples) - r.pos
}

// Reset resets the read position to the beginning.
func (r *Reader) Reset() {
	if r != nil {
		r.pos = 0
	}
}

const (
	// mockMagic identifies serialized PM sample fixtures.
	mockMagic   = 0x504d5350 // "PMSP"
	mockVersion = 1
)

// EncodeMockCounterData encodes decoded samples into a portable binary fixture
// for offline testing and diagnostics.
func EncodeMockCounterData(samples []Sample) ([]byte, error) {
	var buf []byte
	// Header: magic (4), version (2), count (4)
	buf = binary.BigEndian.AppendUint32(buf, mockMagic)
	buf = binary.BigEndian.AppendUint16(buf, mockVersion)
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(samples)))

	for _, s := range samples {
		buf = binary.BigEndian.AppendUint32(buf, uint32(s.Index))
		buf = binary.BigEndian.AppendUint64(buf, s.StartTimestamp)
		buf = binary.BigEndian.AppendUint64(buf, s.EndTimestamp)
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(s.Values)))
		for k, v := range s.Values {
			kBytes := []byte(k)
			buf = binary.BigEndian.AppendUint16(buf, uint16(len(kBytes)))
			buf = append(buf, kBytes...)
			buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(v))
		}
	}
	return buf, nil
}

// DecodeMockCounterData decodes a binary PM sample fixture.
func DecodeMockCounterData(data []byte) ([]Sample, error) {
	if len(data) < 10 {
		return nil, errors.New("cuptipmsampling: buffer too small for fixture header")
	}
	magic := binary.BigEndian.Uint32(data[0:4])
	if magic != mockMagic {
		return nil, fmt.Errorf("cuptipmsampling: invalid fixture magic 0x%08x", magic)
	}
	version := binary.BigEndian.Uint16(data[4:6])
	if version != mockVersion {
		return nil, fmt.Errorf("cuptipmsampling: unsupported fixture version %d", version)
	}
	count := int(binary.BigEndian.Uint32(data[6:10]))
	offset := 10

	samples := make([]Sample, 0, count)
	for i := 0; i < count; i++ {
		if offset+24 > len(data) {
			return nil, errors.New("cuptipmsampling: unexpected EOF reading sample")
		}
		index := uint(binary.BigEndian.Uint32(data[offset : offset+4]))
		start := binary.BigEndian.Uint64(data[offset+4 : offset+12])
		end := binary.BigEndian.Uint64(data[offset+12 : offset+20])
		numVals := int(binary.BigEndian.Uint32(data[offset+20 : offset+24]))
		offset += 24

		var duration uint64
		if end >= start {
			duration = end - start
		}

		values := make(map[string]float64, numVals)
		for j := 0; j < numVals; j++ {
			if offset+2 > len(data) {
				return nil, errors.New("cuptipmsampling: unexpected EOF reading metric name length")
			}
			nameLen := int(binary.BigEndian.Uint16(data[offset : offset+2]))
			offset += 2
			if offset+nameLen+8 > len(data) {
				return nil, errors.New("cuptipmsampling: unexpected EOF reading metric value")
			}
			name := string(data[offset : offset+nameLen])
			offset += nameLen
			val := math.Float64frombits(binary.BigEndian.Uint64(data[offset : offset+8]))
			offset += 8
			values[name] = val
		}

		samples = append(samples, Sample{
			Index:          index,
			StartTimestamp: start,
			EndTimestamp:   end,
			DurationNs:     duration,
			Values:         values,
		})
	}

	return samples, nil
}
