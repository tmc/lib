package cudalowering

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

func ExampleVerifyInferredSharedConversions() {
	_, inference, _ := nvvmx.InferAddressSpaces([]byte(slotIR))
	report, err := VerifyInferredSharedConversions([]byte(slotIR), inference, []byte(slotPTX), []byte(slotSASS(slotPTX)))
	fmt.Println(err, report.Slots[0].Verified, report.Divergences[0].Slot)
	// Output: <nil> true @kernel/%b
}
func ExampleVerifyCompiledSharedConversions() {
	_, err := VerifyCompiledSharedConversions(&nvvmx.Result{}, []byte("not compiled SASS"))
	fmt.Println(err != nil)
	// Output: true
}
func ExampleSlotEvidence() {
	var evidence SlotEvidence
	fmt.Println(evidence.Verified)
	// Output: false
}
func ExampleSlotUseEvidence() {
	var evidence SlotUseEvidence
	fmt.Println(len(evidence.SharedPCs), len(evidence.ConversionPCs))
	// Output: 0 0
}
