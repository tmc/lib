package tracecontract

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func Example() {
	member, _ := Digest("perfetto_manifest.json", "manifest", "", []byte("manifest"))
	fmt.Println(member.Path, member.Bytes)
	// Output: perfetto_manifest.json 8
}

func ExampleSource() {
	source := Source{CaptureSHA256: strings.Repeat("0", 64), Journal: "process-1", Kind: "activity", Sequence: 7}
	text, _ := source.String()
	fmt.Println(text)
	// Output: {"capture_sha256":"0000000000000000000000000000000000000000000000000000000000000000","journal":"process-1","kind":"activity","sequence":7}
}

func ExampleSource_String() {
	source := Source{CaptureSHA256: strings.Repeat("0", 64), Journal: "process-1", Kind: "api", Sequence: 8}
	_, err := source.String()
	fmt.Println(err)
	// Output: <nil>
}

func ExampleTrack() {
	track := Track{UUID: 1, Name: "recorded GPU union", Role: "gpu_union", Clock: "arm:monotonic", Scope: "token:0", Provenance: "observed", Aggregation: "gpu_union", Canonical: true}
	fmt.Println(track.Role, track.Canonical)
	// Output: gpu_union true
}

func ExampleTrack_Write() {
	var buffer bytes.Buffer
	writer := perfetto.NewWriter(&buffer)
	track := Track{UUID: 1, Name: "device activity", Role: "raw_activity", Clock: "capture:journal:monotonic", Provenance: "observed", Aggregation: "none"}
	err := track.Write(writer, 0)
	fmt.Println(err, buffer.Len() > 0)
	// Output: <nil> true
}

func ExampleValidateTracks() {
	tracks := []Track{{UUID: 1, Name: "device", Role: "device", Clock: "monotonic", Provenance: "observed", Aggregation: "none"}, {UUID: 2, Parent: 1, Name: "token", Role: "token", Clock: "monotonic", Scope: "token:0", Provenance: "observed", Aggregation: "elapsed", Canonical: true}}
	fmt.Println(ValidateTracks(tracks))
	// Output: <nil>
}

func ExampleGate() {
	gate := Gate{Disposition: "unchecked", Reason: "no independent workload invariant", Provenance: "caller options"}
	fmt.Println(gate.Disposition)
	// Output: unchecked
}

func ExampleBackend() {
	backend := Backend{Name: "native"}
	fmt.Println(backend.Name)
	// Output: native
}

func receiptFixture() Receipt {
	r := Receipt{Version: 1, Command: "trace summary", InputPath: "capture.pprof", InputSHA256: strings.Repeat("0", 64), Kind: "capture", Backends: []Backend{{Name: "native"}}, Gates: map[string]Gate{}, Partial: true}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		r.Gates[name] = Gate{Disposition: "unchecked", Reason: "no independent qualification supplied", Provenance: "caller options"}
	}
	return r
}

func ExampleReceipt() {
	receipt := receiptFixture()
	fmt.Println(receipt.Version, receipt.Partial)
	// Output: 1 true
}

func ExampleReceipt_JSON() {
	_, err := receiptFixture().JSON()
	fmt.Println(err)
	// Output: <nil>
}

func ExampleReceipt_Write() {
	var buffer bytes.Buffer
	writer := perfetto.NewWriter(&buffer)
	writer.WriteTrackDescriptor(1, "evidence receipt", 0)
	err := receiptFixture().Write(writer, 1, 0)
	fmt.Println(err, buffer.Len() > 0)
	// Output: <nil> true
}

func ExampleDigest() {
	member, _ := Digest("perfetto_manifest.json", "manifest", "", []byte("manifest"))
	fmt.Println(member.Path, member.Bytes, len(member.SHA256))
	// Output: perfetto_manifest.json 8 64
}

func ExampleMember() {
	member, _ := Digest("base.pftrace", "trace", strings.Repeat("0", 64), []byte("trace"))
	fmt.Println(member.Kind, member.Bytes)
	// Output: trace 5
}

func bindingFixture() (Bindings, map[string][]byte) {
	members := map[string][]byte{"perfetto_manifest.json": []byte("manifest"), ".alignment.json": []byte("receipt")}
	binding := Bindings{Version: 1, Schema: "cudaprof.bindings.v1"}
	for name, data := range members {
		member, _ := Digest(name, "metadata", "", data)
		binding.Members = append(binding.Members, member)
	}
	return binding, members
}

func ExampleBindings() {
	bindings, _ := bindingFixture()
	fmt.Println(bindings.Schema)
	// Output: cudaprof.bindings.v1
}

func ExampleBindings_JSON() {
	bindings, _ := bindingFixture()
	_, err := bindings.JSON()
	fmt.Println(err)
	// Output: <nil>
}

func ExampleBindings_Verify() {
	bindings, members := bindingFixture()
	fmt.Println(bindings.Verify(members))
	// Output: <nil>
}

func ExampleProbeCoverage() {
	coverage := ProbeCoverage{Present: true, Ratio: 1.0}
	fmt.Println(coverage.Present, coverage.Ratio)
	// Output: true 1
}

func ExampleLoss() {
	loss := Loss{Dropped: 0, Total: 100, Unknown: false}
	fmt.Println(loss.Dropped, loss.Total, loss.Unknown)
	// Output: 0 100 false
}

func ExampleCaptureBounds() {
	bounds := CaptureBounds{Start: 1000, End: 5000, Clock: "monotonic"}
	fmt.Println(bounds.Start, bounds.End, bounds.Clock)
	// Output: 1000 5000 monotonic
}

func ExampleRequirement() {
	req := Requirement{Name: "complete_activity", Family: "activity", MaxLoss: 0}
	fmt.Println(req.Name, req.Family, req.MaxLoss)
	// Output: complete_activity activity 0
}

func ExampleAdmission() {
	admission := Admission{Admitted: true, Status: "admitted", ReasonCode: ReasonAdmitted}
	fmt.Println(admission.Admitted, admission.Status, admission.ReasonCode)
	// Output: true admitted admitted
}

func ExampleFinding() {
	finding := Finding{
		ID:    "finding-001",
		Rule:  "launch_count",
		Class: FindingObserved,
		Admission: Admission{
			Admitted:   true,
			Status:     "admitted",
			ReasonCode: ReasonAdmitted,
		},
	}
	fmt.Println(finding.ID, finding.Rule, finding.Class, finding.Admission.Admitted)
	// Output: finding-001 launch_count observed true
}

func ExampleAdmit() {
	receipt := receiptFixture()
	receipt.LostRecords = map[string]Loss{"activity": {Dropped: 0, Total: 10}}
	ev := EvidenceContext{Receipt: receipt}
	reqs := []Requirement{{Name: "activity_loss", Family: "activity", MaxLoss: 0}}
	val := 10.0
	finding := Admit("f1", "launch_count", FindingObserved, "observed 10 launches", &val, "launches", reqs, ev)
	fmt.Println(finding.Class, finding.Admission.Admitted, *finding.Value)
	// Output: observed true 10
}

func ExampleWriteFindings() {
	var buf bytes.Buffer
	writer := perfetto.NewWriter(&buf)
	findings := []Finding{{
		ID:        "f1",
		Rule:      "launch_count",
		Class:     FindingObserved,
		Admission: Admission{Admitted: true, Status: "admitted", ReasonCode: ReasonAdmitted},
	}}
	err := WriteFindings(writer, 1, 0, findings)
	fmt.Println(err, buf.Len() > 0)
	// Output: <nil> true
}
