package xidcatalog_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
)

func ExampleLookupRaw() {
	entry, known := xidcatalog.LookupRaw(154)
	fmt.Println(entry.Name, known)
	// Output: GPU_RECOVERY_ACTION_CHANGED true
}

func ExampleLookupProduct() {
	entry, coverage := xidcatalog.LookupProduct(154, "L4")
	fmt.Println(entry.Code, coverage == xidcatalog.UncoveredProduct)
	// Output: 154 true
}

func ExampleDecodeRules() {
	rules := xidcatalog.DecodeRules(144)
	fmt.Println(len(rules), rules[0].Row, rules[0].Cells[5])
	// Output: 6 2 IGNORE
}

func ExampleDecodeSources() {
	cells := xidcatalog.DecodeSources()
	fmt.Println(cells[0].Sheet, cells[0].Coordinate, cells[0].Value)
	// Output: Xid 144-150 Decode A1 Xid
}

func ExampleDecode() {
	result := xidcatalog.Decode(145, xidcatalog.B100, "575.0", 0x86, 1)
	fmt.Println(result.Outcome == xidcatalog.OneDecodeMatch, result.Matches[0].Rule.Row, result.Matches[0].Bucket)
	// Output: true 23 RESET_GPU
}

func ExampleRecognizeRecoveryText() {
	state, known := xidcatalog.RecognizeRecoveryText("Future action")
	fmt.Println(state == xidcatalog.UnknownRecoveryText, known)
	// Output: true false
}

func ExampleParseRecoveryTransition() {
	event, err := xidcatalog.ParseRecoveryTransition("NVRM: Xid (PCI:0008:01:00): 154, GPU recovery action changed from 0x0 (None) to 0x1 (GPU Reset Required)")
	fmt.Println(event.PCI, event.From.Hex, event.To.Text, err)
	// Output: 0008:01:00 0x0 GPU Reset Required <nil>
}

func ExampleApplicability() {
	_, state := xidcatalog.LookupProduct(1, xidcatalog.H100)
	fmt.Println(state == xidcatalog.NotApplicable)
	// Output: true
}

func ExampleDecodeRule() {
	rule := xidcatalog.DecodeRules(144)[0]
	fmt.Println(rule.Code, rule.Row)
	// Output: 144 2
}

func ExampleSourceCell() {
	cell := xidcatalog.DecodeSources()[0]
	fmt.Println(cell.Coordinate, cell.Value)
	// Output: A1 Xid
}

func ExampleDecodeOutcome() {
	result := xidcatalog.Decode(144, xidcatalog.B100, "unknown", 1, 1)
	fmt.Println(result.Outcome == xidcatalog.UnknownDriver)
	// Output: true
}

func ExampleDecodeVariant() {
	result := xidcatalog.Decode(145, xidcatalog.B100, "575.0", 0x86, 1)
	fmt.Println(result.Matches[0].Variant == xidcatalog.SecondaryDecode)
	// Output: true
}

func ExampleDecodeMatch() {
	result := xidcatalog.Decode(144, xidcatalog.B100, "575.0", 1, 1)
	match := result.Matches[0]
	fmt.Println(match.StatusConstrained, match.Bucket)
	// Output: true IGNORE
}

func ExampleDecodeResult() {
	result := xidcatalog.Decode(144, xidcatalog.B100, "575.0", 1, 1)
	fmt.Println(result.Driver, result.DriverMajor)
	// Output: 575.0 575
}

func ExampleRecoveryText() {
	state, known := xidcatalog.RecognizeRecoveryText("None")
	fmt.Println(known, state == xidcatalog.RecoveryNone)
	// Output: true true
}

func ExampleRecoveryToken() {
	transition, _ := xidcatalog.ParseRecoveryTransition("Xid 154 GPU recovery action changed from 0x0 (None) to 0xcd (Future action)")
	token := transition.To
	fmt.Println(token.Hex, token.Value, token.Recognized)
	// Output: 0xcd 205 false
}

func ExampleRecoveryTransition() {
	transition, _ := xidcatalog.ParseRecoveryTransition("Xid 154 GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)")
	fmt.Println(transition.XID, transition.PCI == "", transition.To.Text)
	// Output: 154 true Node Reboot Required
}

func ExampleMaxRecoveryMessageBytes() {
	fmt.Println(xidcatalog.MaxRecoveryMessageBytes)
	// Output: 4096
}
