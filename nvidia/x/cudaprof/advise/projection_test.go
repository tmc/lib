package advise

import (
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// decodeCapture is a GB10 decode step: n kernels of d ns separated by gaps
// of g ns.
func decodeCapture(n int, d, g uint64) *cudacapture.Capture {
	var samples []*profile.Sample
	t := uint64(1_000_000)
	for i := 0; i < n; i++ {
		samples = append(samples, microKernelSample("matvec_q4", 7, t, t+d, i+1))
		t += d + g
	}
	return &cudacapture.Capture{
		Metadata: []byte(`{"device":"NVIDIA GB10","arch":"sm_121a"}`),
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample:     samples,
		},
	}
}

func TestProjectTo(t *testing.T) {
	tests := []struct {
		name   string
		cap    *cudacapture.Capture
		target string
		remedy string
	}{
		// 300 kernels of 200 us with 4 us gaps: on GB300 each kernel is
		// about 7 us, longer than the gap.
		{"long kernels hide gaps", decodeCapture(300, 200_000, 4_000), "gb300", "pdl"},
		// 300 kernels of 20 us with 4 us gaps: about 0.7 us on GB300.
		{"short kernels need a persistent kernel", decodeCapture(300, 20_000, 4_000), "sm_103a", "persistent"},
		// Gaps of 10 ns stay negligible anywhere.
		{"no gaps", decodeCapture(300, 200_000, 10), "B200", "none"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Report{}
			if err := r.ProjectTo(tt.cap, tt.target); err != nil {
				t.Fatal(err)
			}
			p := r.Target
			if p.Remedy != tt.remedy {
				t.Errorf("remedy = %s, want %s; projection %+v", p.Remedy, tt.remedy, p)
			}
			if p.SourceProduct != "GB10" || p.Launches != 300 {
				t.Errorf("source %s launches %d, want GB10, 300", p.SourceProduct, p.Launches)
			}
			if want := int64(float64(p.ComputeNanos) * 273 / 8000); p.FloorNanos != want {
				t.Errorf("floor = %d, want %d", p.FloorNanos, want)
			}
			if len(r.Items) != 1 || r.Items[0].EvidenceLevel != EvidenceModeled || !strings.Contains(r.Items[0].ActionableRecommendation, "This is a model") {
				t.Errorf("items = %+v, want one modeled finding", r.Items)
			}
		})
	}
}

func TestProjectToUnknown(t *testing.T) {
	r := &Report{}
	if err := r.ProjectTo(decodeCapture(3, 10, 10), "sm_999"); err == nil {
		t.Error("unknown target: no error")
	}
	c := decodeCapture(3, 10, 10)
	c.Metadata = []byte(`{"arch":"sm_90a"}`)
	if err := r.ProjectTo(c, "gb300"); err == nil {
		t.Error("source without a product: no error")
	}
}
