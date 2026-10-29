package advise_test

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

func ExampleAnalyzeCounters() {
	// These identities are CPU example placeholders, not admitted hardware evidence.
	digest := strings.Repeat("a", 64)
	ref := advise.CounterReference{Locator: "example-receipt", SHA256: digest}
	action := advise.CounterAction{
		ID: "example-action", Kernel: "example-kernel", AcquisitionID: "example-acquisition",
		Identity: advise.CounterIdentity{
			SourceSelectionSHA256: digest, CandidateSourceSHA256: digest,
			CaseConfigSHA256: digest, ImageSHA256: digest, DeviceUUID: "CPU_EXAMPLE",
			RawReport: ref, TypedCensus: ref, Admission: ref,
		},
		Descriptors: []cudacapture.MetricDescriptor{{
			ID: "duration", Name: "gpu__time_duration.sum", NumericType: "uint64",
			Unit: "ns", Scope: "ACTION", Denominator: "action", Aggregation: "sum",
		}},
		Observations: []cudacapture.Observation{{
			DescriptorID: "duration", EntityID: "example-action", RunID: "example-acquisition",
			Availability: "MEASURED", Value: json.RawMessage(`143616`),
		}},
	}
	report, err := advise.AnalyzeCounters([]advise.CounterAction{action})
	if err != nil {
		panic(err)
	}
	fmt.Println(report.Mode, report.Findings[0].Item.NoRuntimeAttribution)
	// Output: COUNTER_ONLY true
}
