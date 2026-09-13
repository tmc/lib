package cuptipmsampling_test

import (
	"errors"
	"io"
	"reflect"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
)

func TestSamplerConfig_Config(t *testing.T) {
	tests := []struct {
		name     string
		input    cuptipmsampling.SamplerConfig
		img      []byte
		wantHw   uint
		wantSamp uint64
		wantMax  uint32
	}{
		{
			name:     "defaults",
			input:    cuptipmsampling.SamplerConfig{},
			img:      []byte{1, 2, 3},
			wantHw:   64 * 1024 * 1024,
			wantSamp: uint64(100 * time.Microsecond),
			wantMax:  65536,
		},
		{
			name: "custom",
			input: cuptipmsampling.SamplerConfig{
				DeviceIndex:        1,
				Interval:           500 * time.Microsecond,
				HardwareBufferSize: 16 * 1024 * 1024,
				MaxSamples:         10000,
				MetricNames:        []string{"sm__throughput"},
				TriggerMode:        cuptipmsampling.TriggerModeSysclkInterval,
				AppendMode:         cuptipmsampling.AppendModeKeepOldest,
			},
			img:      []byte{4, 5, 6},
			wantHw:   16 * 1024 * 1024,
			wantSamp: uint64(500 * time.Microsecond),
			wantMax:  10000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.input.Config(tt.img)
			if cfg.HardwareBufferSize != tt.wantHw {
				t.Errorf("HardwareBufferSize = %d, want %d", cfg.HardwareBufferSize, tt.wantHw)
			}
			if cfg.SamplingInterval != tt.wantSamp {
				t.Errorf("SamplingInterval = %d, want %d", cfg.SamplingInterval, tt.wantSamp)
			}
			if cfg.MaxSamples != tt.wantMax {
				t.Errorf("MaxSamples = %d, want %d", cfg.MaxSamples, tt.wantMax)
			}
			if !reflect.DeepEqual(cfg.ConfigImage, tt.img) {
				t.Errorf("ConfigImage mismatch")
			}
		})
	}
}

func TestReader(t *testing.T) {
	samples := []cuptipmsampling.Sample{
		{
			Index:          0,
			StartTimestamp: 1000,
			EndTimestamp:   2000,
			DurationNs:     1000,
			Values:         map[string]float64{"sm_busy": 82.5},
		},
		{
			Index:          1,
			StartTimestamp: 2000,
			EndTimestamp:   3000,
			DurationNs:     1000,
			Values:         map[string]float64{"sm_busy": 91.0, "dram_busy": 45.2},
		},
	}

	r := cuptipmsampling.NewReader(samples)
	if got := r.Len(); got != 2 {
		t.Fatalf("r.Len() = %d, want 2", got)
	}
	if got := r.Remaining(); got != 2 {
		t.Fatalf("r.Remaining() = %d, want 2", got)
	}

	s1, err := r.ReadSample()
	if err != nil {
		t.Fatalf("ReadSample 1 failed: %v", err)
	}
	if s1.Index != 0 || s1.Values["sm_busy"] != 82.5 {
		t.Errorf("s1 unexpected: %+v", s1)
	}
	if got := r.Remaining(); got != 1 {
		t.Errorf("Remaining = %d, want 1", got)
	}

	s2, err := r.ReadSample()
	if err != nil {
		t.Fatalf("ReadSample 2 failed: %v", err)
	}
	if s2.Index != 1 || s2.Values["dram_busy"] != 45.2 {
		t.Errorf("s2 unexpected: %+v", s2)
	}

	_, err = r.ReadSample()
	if !errors.Is(err, io.EOF) {
		t.Errorf("ReadSample at EOF returned %v, want io.EOF", err)
	}

	r.Reset()
	if got := r.Remaining(); got != 2 {
		t.Errorf("Remaining after Reset = %d, want 2", got)
	}
	all := r.All()
	if len(all) != 2 {
		t.Errorf("All() returned %d samples, want 2", len(all))
	}
	if got := r.Remaining(); got != 0 {
		t.Errorf("Remaining after All = %d, want 0", got)
	}
}

func TestMockCounterDataRoundtrip(t *testing.T) {
	orig := []cuptipmsampling.Sample{
		{
			Index:          0,
			StartTimestamp: 100,
			EndTimestamp:   200,
			DurationNs:     100,
			Values:         map[string]float64{"sm__throughput": 75.5, "dram__throughput": 60.25},
		},
		{
			Index:          1,
			StartTimestamp: 200,
			EndTimestamp:   350,
			DurationNs:     150,
			Values:         map[string]float64{"sm__throughput": 88.0},
		},
	}

	encoded, err := cuptipmsampling.EncodeMockCounterData(orig)
	if err != nil {
		t.Fatalf("EncodeMockCounterData failed: %v", err)
	}

	decoded, err := cuptipmsampling.DecodeMockCounterData(encoded)
	if err != nil {
		t.Fatalf("DecodeMockCounterData failed: %v", err)
	}

	if len(decoded) != len(orig) {
		t.Fatalf("decoded length = %d, want %d", len(decoded), len(orig))
	}
	for i := range orig {
		if decoded[i].Index != orig[i].Index ||
			decoded[i].StartTimestamp != orig[i].StartTimestamp ||
			decoded[i].EndTimestamp != orig[i].EndTimestamp ||
			decoded[i].DurationNs != orig[i].DurationNs {
			t.Errorf("sample %d mismatch: got %+v, want %+v", i, decoded[i], orig[i])
		}
		if !reflect.DeepEqual(decoded[i].Values, orig[i].Values) {
			t.Errorf("sample %d values mismatch: got %+v, want %+v", i, decoded[i].Values, orig[i].Values)
		}
	}
}
