package cudadiageval

import (
	"bytes"
	"context"
	"fmt"
	"strings"
)

func pendingLedgerExample() QualificationLedger {
	return QualificationLedger{Version: 1, Cells: []QualificationCell{{ID: "nvml-h100-native", CapabilityID: "nvml-samples", Capability: "unknown", Gate: "native", Status: "pending", Closure: QualificationClosure{Candidate: strings.Repeat("1", 40), Tree: strings.Repeat("2", 40)}, SourcePaths: []string{"nvidia/x/nvtelemetry/nvml_sampler.go"}, Consumers: []string{"nvidia/x/cmd/nvtelemetry/main.go"}, Owner: "native-qualification", NextAction: "admit the exact executable and H100 target"}}}
}
func ExampleQualificationLedger_Validate() {
	fmt.Println(pendingLedgerExample().Validate())
	// Output: <nil>
}
func ExampleWriteQualificationLedger() {
	var output bytes.Buffer
	err := WriteQualificationLedger(&output, pendingLedgerExample())
	fmt.Println(err, output.Len() > 0)
	// Output: <nil> true
}
func ExampleReadQualificationLedger() {
	var output bytes.Buffer
	if err := WriteQualificationLedger(&output, pendingLedgerExample()); err != nil {
		fmt.Println(err)
		return
	}
	ledger, err := ReadQualificationLedger(&output)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(ledger.Cells[0].Status)
	// Output: pending
}
func ExampleVerifyQualificationLedger() {
	results, err := VerifyQualificationLedger(context.Background(), ".", pendingLedgerExample())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(results[0].Gate, results[0].Status, results[0].Accepted)
	// Output: native pending false
}
