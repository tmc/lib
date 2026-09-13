package cuptipcsampling

import (
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

// PMSample represents one decoded hardware Performance Monitor (PM) counter observation.
type PMSample = cuptipmsampling.Sample

// PMReader provides a streaming reader interface over decoded PM samples.
type PMReader = cuptipmsampling.Reader

// SamplerConfig specifies high-level configuration for hardware PM counter sampling.
type SamplerConfig = cuptipmsampling.SamplerConfig

// NewPMReader creates a streaming PM counter reader over decoded samples.
func NewPMReader(samples []PMSample) *PMReader {
	return cuptipmsampling.NewReader(samples)
}

// DecodeMockPMSamples decodes a serialized PM sample fixture for offline testing.
func DecodeMockPMSamples(data []byte) ([]PMSample, error) {
	return cuptipmsampling.DecodeMockCounterData(data)
}

// EncodeMockPMSamples encodes PM samples into a portable fixture.
func EncodeMockPMSamples(samples []PMSample) ([]byte, error) {
	return cuptipmsampling.EncodeMockCounterData(samples)
}
