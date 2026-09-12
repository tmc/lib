package metalprof

import "testing"

func TestReplayTimelineObservations(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		want        int
		wantErr     bool
	}{
		{"absent", `{}`, 0, false},
		{"null", `{"timeline":{"command_buffer_active_time_ns":null}}`, 0, false},
		{"zero", `{"timeline":{"command_buffer_active_time_ns":0}}`, 1, false},
		{"negative", `{"timeline":{"command_buffer_active_time_ns":-1}}`, 0, true},
		{"overflow", `{"timeline":{"command_buffer_active_time_ns":9223372036854775808}}`, 0, true},
		{"fraction", `{"timeline":{"command_buffer_active_time_ns":0.5}}`, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			obs, err := replayTimelineObservations([]byte(tt.input), "test")
			if (err != nil) != tt.wantErr || len(obs) != tt.want {
				t.Fatalf("got %d observations, error %v", len(obs), err)
			}
			for _, o := range obs {
				if o.Source == "" || !o.NonAdditive || o.Scope != "run" {
					t.Fatalf("missing provenance: %+v", o)
				}
			}
		})
	}
}
