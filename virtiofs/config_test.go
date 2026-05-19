package virtiofs

import "testing"

func TestConfigValidate(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  Config
		ok   bool
	}{
		{name: "default", cfg: DefaultConfig("host"), ok: true},
		{name: "empty tag", cfg: DefaultConfig(""), ok: false},
		{name: "long tag", cfg: Config{Tag: "0123456789012345678901234567890123456", NumQueues: 1, QueueSize: 128}, ok: false},
		{name: "zero queues", cfg: Config{Tag: "host", QueueSize: 128}, ok: false},
		{name: "non power queue", cfg: Config{Tag: "host", NumQueues: 1, QueueSize: 96}, ok: false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if (err == nil) != tt.ok {
				t.Fatalf("Validate() = %v, ok %v", err, tt.ok)
			}
		})
	}
}
