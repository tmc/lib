package metalprof

import "testing"

func TestReplayCounterObservations(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		want        int
		wantErr     bool
	}{
		{"absent", `{}`, 0, false},
		{"zero", `{"counter_archive":{"total_samples":0}}`, 1, false},
		{"partial", `{"counter_archive":{"total_samples":10,"attributed_samples":2,"machine_wide_samples":3}}`, 3, false},
		{"overcount", `{"counter_archive":{"total_samples":10,"attributed_samples":8,"machine_wide_samples":3}}`, 0, true},
		{"overflow", `{"counter_archive":{"total_samples":9223372036854775807,"attributed_samples":9223372036854775807,"machine_wide_samples":1}}`, 0, true},
		{"negative", `{"counter_archive":{"total_samples":-1}}`, 0, true},
		{"fraction", `{"counter_archive":{"total_samples":1.5}}`, 0, true},
		{"mismatches", `{"counter_archive":{"blobs":1,"stride_mismatches":2}}`, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			obs, err := replayCounterObservations([]byte(tt.input), "test")
			if (err != nil) != tt.wantErr || len(obs) != tt.want {
				t.Fatalf("got %d observations, error %v", len(obs), err)
			}
		})
	}
}
