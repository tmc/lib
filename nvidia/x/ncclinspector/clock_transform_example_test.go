package ncclinspector_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func exampleClockSpec() ncclinspector.ClockTransformSpec {
	return ncclinspector.ClockTransformSpec{
		Source:     ncclinspector.ClockIdentity{Host: "worker", PID: 1, Domain: ncclinspector.ClockDomainHostMonotonic, BootEpoch: "boot", ProcessEpoch: "process"},
		Reference:  ncclinspector.ClockIdentity{Host: "reference", Domain: ncclinspector.ClockDomainHostMonotonic, BootEpoch: "reference-boot", ProcessEpoch: "reference-process"},
		SourceUnit: "nanoseconds", ReferenceUnit: "nanoseconds", Anchors: []ncclinspector.ClockAnchor{{SourceNS: 0, ReferenceMinNS: 100, ReferenceMaxNS: 104}},
		RateMinPPB: 1_000_000_000, RateMaxPPB: 1_000_000_000, DriftKnown: true, UncertaintyKnown: true, ValidFromNS: 0, ValidUntilNS: 1000,
		Method: "supplied calibration bounds", VerificationRef: "constructed-verification.json",
	}
}

func ExampleNewClockTransform() {
	spec := exampleClockSpec()
	transform, err := ncclinspector.NewClockTransform(spec)
	fmt.Println(err == nil, len(transform.Steps()))
	// Output: true 1
}

func ExampleClockTransform_Mapping() {
	transform, _ := ncclinspector.NewClockTransform(exampleClockSpec())
	raw := ncclinspector.BoundedTimestamp{Nanoseconds: 20, Domain: ncclinspector.ClockDomainHostMonotonic, Epoch: "process", BootEpoch: "boot"}
	mapped, err := raw.Aligned(transform.Mapping())
	fmt.Println(mapped.Nanoseconds, mapped.UncertaintyNS, err)
	// Output: 122 2 <nil>
}

func ExampleClockTransform_Steps() {
	transform, _ := ncclinspector.NewClockTransform(exampleClockSpec())
	step := transform.Steps()[0]
	fmt.Println(step.Method, step.VerificationRef)
	// Output: supplied calibration bounds constructed-verification.json
}

func ExampleComposeClockTransforms() {
	spec := exampleClockSpec()
	first, _ := ncclinspector.NewClockTransform(spec)
	spec.Source = spec.Reference
	spec.Reference.ProcessEpoch = "second-reference"
	spec.Anchors = []ncclinspector.ClockAnchor{{SourceNS: 100, ReferenceMinNS: 200, ReferenceMaxNS: 200}}
	second, _ := ncclinspector.NewClockTransform(spec)
	combined, err := ncclinspector.ComposeClockTransforms(first, second, "constructed-composition.json")
	fmt.Println(len(combined.Steps()), err)
	// Output: 2 <nil>
}

func ExampleCollectiveObservation_WithCompletion() {
	transform, _ := ncclinspector.NewClockTransform(exampleClockSpec())
	record := ncclinspector.Record{Header: ncclinspector.Header{ID: "0x1234567890abcdef", Ranks: 2}, Metadata: ncclinspector.Metadata{Hostname: "worker", PID: 1}, Collective: &ncclinspector.Collective{Name: "AllReduce"}}
	obs, _ := ncclinspector.ObservationFromRecord(record, transform.Mapping())
	completion := ncclinspector.BoundedTimestamp{Nanoseconds: 20, Domain: ncclinspector.ClockDomainHostMonotonic, Epoch: "process", BootEpoch: "boot"}
	obs, err := obs.WithCompletion(completion) // supplied fact, not record publication
	fmt.Println(obs.HasCompletion, obs.Completion.Nanoseconds, err)
	// Output: true 122 <nil>
}

func ExampleClockIdentity() {
	id := exampleClockSpec().Source
	fmt.Println(id.Host, id.BootEpoch, id.ProcessEpoch)
	// Output: worker boot process
}

func ExampleClockAnchor() {
	anchor := exampleClockSpec().Anchors[0]
	fmt.Println(anchor.SourceNS, anchor.ReferenceMinNS, anchor.ReferenceMaxNS)
	// Output: 0 100 104
}

func ExampleClockTransformSpec() {
	spec := exampleClockSpec()
	fmt.Println(spec.SourceUnit, spec.RateMinPPB, spec.RateMaxPPB)
	// Output: nanoseconds 1000000000 1000000000
}

func ExampleClockTransform() {
	transform, _ := ncclinspector.NewClockTransform(exampleClockSpec())
	fmt.Println(transform.Mapping().EvidenceRef)
	// Output: constructed-verification.json
}
