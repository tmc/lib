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
			want: "3 loaded modules, 124.0 MB: 2 over total module budget (3.0 MB), 1 over per-module budget (121.0 MB)",
		},
		{
			name:   "capture omitted a cubin",
			status: `{"omitted_files":[{"name":"kernels.cubin","bytes":999,"reason":"exceeds capture file budget of 67108864 bytes"},{"name":"capture.journal","bytes":5,"reason":"exceeds capture file budget"}]}`,
			want:   "1 loaded module, 0.0 MB: 1 exceeds capture file budget of 67108864 bytes (0.0 MB)",
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

func TestDescribeMissingLaunches(t *testing.T) {
	missing := []MissingCode{
		{Module: 23, Bytes: 121e6, Reason: "over per-module budget"},
		{Module: 87, Bytes: 1e6, Reason: "over total module budget"},
		{Module: 88, Bytes: 2e6, Reason: "over total module budget"},
	}
	tests := []struct {
		name     string
		missing  []MissingCode
		launches *ModuleLaunches
		want     string
	}{
		{"no timeline", missing, nil, ""},
		{"no launches", missing, &ModuleLaunches{}, ""},
		{
			name:     "none ran",
			missing:  missing,
			launches: &ModuleLaunches{Total: 5, ByModule: map[uint32]int{21: 5}},
			want:     "no launch names any of those modules",
		},
		{
			name:     "some ran",
			missing:  missing,
			launches: &ModuleLaunches{Total: 9, ByModule: map[uint32]int{21: 5, 87: 2, 88: 2}},
			want:     "2 of those modules (3.0 MB) ran captured kernel launches",
		},
		{
			name:     "launches name no module",
			missing:  missing,
			launches: &ModuleLaunches{Total: 3904, ByModule: map[uint32]int{87: 5, 88: 5}, Unnamed: 3739, Graph: 3495},
			want:     "2 of those modules (3.0 MB) ran captured kernel launches; 3739 of 3904 launches (3495 in CUDA graphs) name no module and may have run from the others",
		},
		{
			name:     "no named launch reached them",
			missing:  missing,
			launches: &ModuleLaunches{Total: 10, ByModule: map[uint32]int{21: 4}, Unnamed: 6},
			want:     "no launch names any of those modules; 6 of 10 launches name no module and may have run from any of them",
		},
		{
			name:     "one module ran",
			missing:  missing[:1],
			launches: &ModuleLaunches{Total: 4, ByModule: map[uint32]int{23: 4}},
			want:     "it ran captured kernel launches",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DescribeMissingLaunches(tt.missing, tt.launches); got != tt.want {
				t.Errorf("DescribeMissingLaunches = %q, want %q", got, tt.want)
			}
		})
	}
}
