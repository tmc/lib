package metalprof

import "testing"

func TestReplayPipelinePresence(t *testing.T) {
	for _, tt := range []struct {
		name, input string
		want        int
		wantErr     bool
	}{
		{"unrecorded zero", `{"pipelines":[{"temporary_register_count":0}]}`, 0, false},
		{"recorded zero", `{"pipelines":[{"recorded_statistics":["Temporary register count"],"temporary_register_count":0}]}`, 1, false},
		{"missing value", `{"pipelines":[{"recorded_statistics":["Temporary register count"]}]}`, 0, true},
		{"fractional count", `{"pipelines":[{"recorded_statistics":["Temporary register count"],"temporary_register_count":1.5}]}`, 0, true},
		{"negative count", `{"pipelines":[{"recorded_statistics":["Temporary register count"],"temporary_register_count":-1}]}`, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			obs, err := replayPipelineObservations([]byte(tt.input), "test")
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error %v", err, tt.wantErr)
			}
			if len(obs) != tt.want {
				t.Fatalf("observations = %d, want %d", len(obs), tt.want)
			}
			for _, o := range obs {
				if !o.NonAdditive || o.Value.IntVal != 0 {
					t.Fatalf("unexpected observation: %+v", o)
				}
			}
		})
	}
}
