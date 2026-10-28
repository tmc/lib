package advise_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// Exercise the public serialized-envelope and Analyze APIs. A compiler-only
// transport must not turn absent samples into proof of PC sampling; real
// sample-only and timed fixtures retain their different instrument limits.
func TestCompilerOnlyAdviceHeader(t *testing.T) {
	for _, tt := range []struct {
		name, channel, unit   string
		value                 int64
		noSamples, sampleOnly bool
	}{
		{"compiler-only", "samples", "count", 0, true, false},
		{"pc-only", "pc_samples", "count", 17, false, true},
		{"timed", "kernel_time", "nanoseconds", 1000, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := &profile.Profile{SampleType: []*profile.ValueType{{Type: tt.channel, Unit: tt.unit}}}
			if !tt.noSamples {
				fn := &profile.Function{ID: 1, Name: "k"}
				loc := &profile.Location{ID: 1, Address: 0x10, Line: []profile.Line{{Function: fn, Line: 1}}}
				p.Function = []*profile.Function{fn}
				p.Location = []*profile.Location{loc}
				p.Sample = []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{tt.value}, Label: map[string][]string{"kernel": {"k"}, "subtype": {"kernel"}}}}
			}
			cap := &cudacapture.Capture{Profile: p, Code: cudacapture.CodeFiles{SASS: "kernel.sass"}, CodeUnavailable: map[string]string{"source": "fixture has no source", "ptx": "fixture has no PTX", "cubin": "fixture has no CUBIN"}, Files: map[string][]byte{"kernel.sass": []byte(".section .text.k,\"ax\",@progbits\n/*0000*/ IMAD.WIDE R4, R1, 0x10, R2 ;\n/*0010*/ EXIT ;\n")}}
			var buf bytes.Buffer
			if err := cudacapture.Write(&buf, cap); err != nil {
				t.Fatal(err)
			}
			retained, err := cudacapture.Read(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			report, err := advise.Analyze(retained)
			if err != nil {
				t.Fatal(err)
			}
			if report.SampleOnly != tt.sampleOnly {
				t.Fatalf("SampleOnly=%v, want %v", report.SampleOnly, tt.sampleOnly)
			}
			md := report.Markdown()
			pcClaim := strings.Contains(md, "This capture is PC samples only")
			if pcClaim != tt.sampleOnly {
				t.Fatalf("PC-sample claim=%v, want %v:\n%s", pcClaim, tt.sampleOnly, md)
			}
			if tt.noSamples && !strings.Contains(md, "This input retains no profile samples") {
				t.Fatalf("missing absent-sample caveat:\n%s", md)
			}
			if tt.sampleOnly && !strings.Contains(md, "not a share of elapsed time") {
				t.Fatalf("lost PC-sampling limit:\n%s", md)
			}
			if !tt.noSamples && strings.Contains(md, "This input retains no profile samples") {
				t.Fatalf("falsely claimed samples absent:\n%s", md)
			}
			data, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			var record map[string]any
			if err := json.Unmarshal(data, &record); err != nil {
				t.Fatal(err)
			}
			absent, _ := record["profile_samples_absent"].(bool)
			if absent != tt.noSamples {
				t.Fatalf("JSON profile_samples_absent=%v, want %v", absent, tt.noSamples)
			}
		})
	}
}
