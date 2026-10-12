package cudadiageval

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

func exampleRegistration() (Registration, QualificationLedger) {
	p := DefaultProtocol(PhaseQualification)
	// The default protocol is a rehearsal, not native hardware evidence.
	if err := p.Freeze(); err != nil {
		panic(err)
	}
	c := QualificationClosure{Candidate: strings.Repeat("a", 40), Tree: strings.Repeat("b", 40)}
	l := QualificationLedger{Version: 1, Cells: []QualificationCell{{ID: "native", CapabilityID: "nvml", Capability: "unknown", Gate: "native", Status: "pending", Closure: c, SourcePaths: []string{"x/source.go"}, Consumers: []string{"x/consumer.go"}, Owner: "prof", NextAction: "supply authentic evidence"}}}
	var b bytes.Buffer
	if err := WriteQualificationLedger(&b, l); err != nil {
		panic(err)
	}
	cb, _ := json.Marshal(c)
	fact := RegistrationFact{State: "unknown", Reason: "discovery not run", Owner: "prof", NextAction: "supply facts"}
	contract := RegistrationContract{State: "unknown", Reason: "not registered", Owner: "prof", NextAction: "supply policy"}
	r := Registration{Version: 1, ID: "example", Task: "offline registration", Mode: "property", Owner: "caller", Protocol: p, Source: c, PrerequisiteLedgerSHA256: registrationSHA(b.Bytes()), SelectedCells: []RegistrationCell{{Applicability: "selected", CellID: "native", CapabilityID: "nvml", Gate: "native", ClosureSHA256: registrationSHA(cb), Reason: "receipt missing", Owner: "prof", NextAction: "supply evidence"}}, TargetRows: []RegistrationTarget{{ID: "native-unknown", Applicability: "selected", Host: fact, Device: fact, Stack: fact, Topology: fact, Workload: fact, Deployment: fact, Observer: fact, Owner: "prof", NextAction: "supply facts"}}, Workload: RegistrationWorkload{Contract: contract}, Oracle: RegistrationOracle{Contract: contract}, Clock: RegistrationClock{Contract: contract}, Loss: RegistrationLoss{Contract: contract}, Observer: RegistrationObserver{Contract: contract}, Analysis: RegistrationAnalysis{Contract: contract}, Stopping: RegistrationStopping{Contract: contract}, MissingData: RegistrationMissingData{Contract: contract}, Limits: RegistrationLimits{MaxDurationNanos: 1e9, MaxRecords: 1, MaxBytes: 1024}}
	return r, l
}

func ExampleVerifyRegistration() {
	r, l := exampleRegistration()
	r, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	if err != nil {
		fmt.Println(err)
		return
	}
	root, err := os.MkdirTemp("", "registration-example-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(root)
	report, err := VerifyRegistration(context.Background(), r, l, root)
	fmt.Println(err, report.MeasurementAcceptance)
	fmt.Println(report.Cells[0].ReceiptConsistency, report.Cells[0].MeasurementAcceptance)
	// Output:
	// <nil> unverified
	// false unverified
}

func ExampleReadRegistration() {
	r, _ := exampleRegistration()
	var b bytes.Buffer
	if err := WriteRegistration(&b, r); err != nil {
		fmt.Println(err)
		return
	}
	read, err := ReadRegistration(&b)
	fmt.Println(err, read.ID)
	// Output:
	// <nil> example
}

func ExampleWriteRegistration() {
	r, _ := exampleRegistration()
	var b bytes.Buffer
	fmt.Println(WriteRegistration(&b, r), b.Len() > 0)
	// Output:
	// <nil> true
}

func ExampleRegistration_Freeze() {
	r, _ := exampleRegistration()
	frozen, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	fmt.Println(err, frozen.Digest != "")
	// Output:
	// <nil> true
}

func ExampleRegistration_Validate() {
	fmt.Println((Registration{}).Validate())
	// Output:
	// registration: invalid version or identity
}

func ExampleCheckRegistrationSuccessor() {
	r, _ := exampleRegistration()
	r, err := r.Freeze(time.Date(2026, 10, 11, 0, 0, 0, 0, time.UTC))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(CheckRegistrationSuccessor(r, r))
	// Output:
	// <nil>
}
