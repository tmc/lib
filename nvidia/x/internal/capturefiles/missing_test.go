package capturefiles

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestMissingModuleCode(t *testing.T) {
	tests := []struct {
		name   string
		status string
		want   string
	}{
		{
			name:   "no status",
			status: ``,
			want:   "",
		},
		{
			name:   "every module stored",
			status: `{"status":{"modules":[{"module_id":21,"size":10,"file":"m.cubin","stored":true}]}}`,
			want:   "",
		},
		{
			name: "agent dropped modules",
			status: `{"status":{"modules":[
				{"module_id":21,"size":3000000,"file":"m.cubin","stored":true},
				{"module_id":23,"size":120980400,"stored":false,"reason":"over per-module budget"},
				{"module_id":80,"size":2000000,"stored":false,"reason":"over total module budget"},
				{"module_id":84,"size":1000000,"stored":false,"reason":"over total module budget"}]}}`,
			want: "3 modules, 124.0 MB: 2 over total module budget (3.0 MB), 1 over per-module budget (121.0 MB)",
		},
		{
			name:   "capture omitted a cubin",
			status: `{"omitted_files":[{"name":"kernels.cubin","bytes":999,"reason":"exceeds capture file budget of 67108864 bytes"},{"name":"capture.journal","bytes":5,"reason":"exceeds capture file budget"}]}`,
			want:   "1 module, 0.0 MB: 1 exceeds capture file budget of 67108864 bytes (0.0 MB)",
		},
		{
			name:   "dropped for another reason",
			status: `{"status":{"modules":[{"module_id":23,"size":10,"stored":false,"reason":"unreadable"}]}}`,
			want:   "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{Files: map[string][]byte{}}
			if tt.status != "" {
				cap.Files["capture-status.json"] = []byte(tt.status)
			}
			if got := DescribeMissingCode(MissingModuleCode(cap)); got != tt.want {
				t.Errorf("DescribeMissingCode = %q, want %q", got, tt.want)
			}
		})
	}
}
