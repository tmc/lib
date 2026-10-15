package advise_test

import (
	"encoding/json"
	"fmt"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

func ExampleAnalyzeWithEvidence() {
	// An incomplete source is refused, even if the capture carries a number.
	cap := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}}}
	_, err := advise.AnalyzeWithEvidence(cap, cudaevidence.Source{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleAnalyze_classifiedShare() {
	// Compiler-only input has no profile or complete artifact/compiler identity.
	cap := &cudacapture.Capture{Files: map[string][]byte{"remarks.yaml": []byte(`--- !Failure
Pass: Memory
Name: RemarkMemoryTmaLoadFailed
Function: kernel
Args:
  - Remark: example finding
...
`)}}
	report, err := advise.Analyze(cap)
	if err != nil {
		fmt.Println(err)
		return
	}
	text, err := report.JSON()
	if err != nil {
		fmt.Println(err)
		return
	}
	var decoded advise.Report
	if err := json.Unmarshal([]byte(text), &decoded); err != nil {
		fmt.Println(err)
		return
	}
	r := decoded.Items[0].ClassifiedShare.Record()
	fmt.Println(r.Number == nil, r.Reason)
	// Output: true profile unavailable
}
