package cudacensus_test

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func ExampleInstructionMix_Total() {
	mix := cudacensus.InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP32: 3, cudaroofline.FP16Tensor: 1,
	}}
	fmt.Println(mix.Total())
	// Output: 4
}

func ExampleInstructionMix_Dominant() {
	mix := cudacensus.InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP32: 3, cudaroofline.FP16Tensor: 1,
	}}
	p, share, err := mix.Dominant()
	fmt.Println(p, share, err)
	// Output: fp32 0.75 <nil>
}

func ExampleAttributeFromMix() {
	mix := cudacensus.InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP32: math.MaxUint64, cudaroofline.FP16Tensor: 2,
	}}
	attr, err := cudacensus.AttributeFromMix(mix)
	fmt.Println(attr.Known, attr.Method, err)
	// Output: false unattributed cudacensus: instruction mix total overflows uint64
}
