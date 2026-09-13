package cuptipmsampling

import (
	"testing"
)

func TestVersionQualification(t *testing.T) {
	tests := []struct {
		name    string
		version uint32
		want    bool
	}{
		{name: "CUDA 12.6 API version", version: 24, want: true},
		{name: "CUDA 12.8 API version", version: 26, want: true},
		{name: "CUDA 12.9 API version", version: 28, want: true},
		{name: "CUDA 13.0 API version", version: 130001, want: true},
		{name: "CUDA 13.1 API version", version: 130101, want: true},
		{name: "CUDA 13.2 header version", version: 130200, want: true},
		{name: "CUDA 13.2 runtime version", version: 130202, want: true},
		{name: "CUDA 13.3 API version", version: 130301, want: true},
		{name: "CUDA 13.4 API version", version: 130400, want: true},
		{name: "zero version", version: 0, want: false},
		{name: "unmeasured 130100", version: 130100, want: false},
		{name: "unmeasured 130201", version: 130201, want: false},
		{name: "unmeasured 130500", version: 130500, want: false},
		{name: "unmeasured CUDA 14", version: 140000, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := qualifiedVersion(tt.version); got != tt.want {
				t.Errorf("qualifiedVersion(%d) = %v, want %v", tt.version, got, tt.want)
			}
		})
	}
}

func TestSetConfigPrefixSize(t *testing.T) {
	if got, want := setConfigStructSize(24), uint(60); got != want {
		t.Errorf("setConfigStructSize(24) = %d, want %d", got, want)
	}
	if got, want := setConfigStructSize(130400), uint(64); got != want {
		t.Errorf("setConfigStructSize(130400) = %d, want %d", got, want)
	}
}

func TestEnums(t *testing.T) {
	t.Run("TriggerMode", func(t *testing.T) {
		if got, want := TriggerModeSysclkInterval.String(), "SysclkInterval"; got != want {
			t.Errorf("TriggerModeSysclkInterval.String() = %q, want %q", got, want)
		}
		if got, want := TriggerModeTimeInterval.String(), "TimeInterval"; got != want {
			t.Errorf("TriggerModeTimeInterval.String() = %q, want %q", got, want)
		}
		if got, want := TriggerMode(99).String(), "TriggerMode(99)"; got != want {
			t.Errorf("TriggerMode(99).String() = %q, want %q", got, want)
		}
	})

	t.Run("AppendMode", func(t *testing.T) {
		if got, want := AppendModeKeepOldest.String(), "KeepOldest"; got != want {
			t.Errorf("AppendModeKeepOldest.String() = %q, want %q", got, want)
		}
		if got, want := AppendModeKeepLatest.String(), "KeepLatest"; got != want {
			t.Errorf("AppendModeKeepLatest.String() = %q, want %q", got, want)
		}
		if got, want := AppendMode(99).String(), "AppendMode(99)"; got != want {
			t.Errorf("AppendMode(99).String() = %q, want %q", got, want)
		}
	})

	t.Run("DecodeStopReason", func(t *testing.T) {
		if got, want := DecodeStopReasonOther.String(), "Other"; got != want {
			t.Errorf("DecodeStopReasonOther.String() = %q, want %q", got, want)
		}
		if got, want := DecodeStopReasonCounterDataFull.String(), "CounterDataFull"; got != want {
			t.Errorf("DecodeStopReasonCounterDataFull.String() = %q, want %q", got, want)
		}
		if got, want := DecodeStopReasonEndOfRecords.String(), "EndOfRecords"; got != want {
			t.Errorf("DecodeStopReasonEndOfRecords.String() = %q, want %q", got, want)
		}
		if got, want := DecodeStopReason(99).String(), "DecodeStopReason(99)"; got != want {
			t.Errorf("DecodeStopReason(99).String() = %q, want %q", got, want)
		}
	})
}

func TestNewSessionValidation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{
			name: "missing sampling interval",
			cfg: Config{
				SamplingInterval: 0,
				MaxSamples:       100,
				ConfigImage:      []byte{1, 2, 3},
				MetricNames:      []string{"metric_a"},
			},
			wantErr: true,
		},
		{
			name: "missing max samples",
			cfg: Config{
				SamplingInterval: 1000,
				MaxSamples:       0,
				ConfigImage:      []byte{1, 2, 3},
				MetricNames:      []string{"metric_a"},
			},
			wantErr: true,
		},
		{
			name: "empty config image",
			cfg: Config{
				SamplingInterval: 1000,
				MaxSamples:       100,
				ConfigImage:      nil,
				MetricNames:      []string{"metric_a"},
			},
			wantErr: true,
		},
		{
			name: "empty metric names",
			cfg: Config{
				SamplingInterval: 1000,
				MaxSamples:       100,
				ConfigImage:      []byte{1, 2, 3},
				MetricNames:      nil,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSession(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Fatalf("NewSession() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
