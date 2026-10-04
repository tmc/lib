package cudarecipe_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudarecipe"
)

func ExampleCompiledResources_CorroboratedSpills() {
	n := uint64(4)
	r := cudarecipe.CompiledResources{InstructionsKnown: true, SpillStoreBytes: cudarecipe.ResourceFact{Value: &n, Category: "compiler reported"}, LocalStores: []uint64{32}}
	fmt.Println(r.CorroboratedSpills())
	// Output: true
}

func ExampleCompiledResources_CheckRegisterResidency() {
	r := new(cudarecipe.CompiledResources)
	err := r.CheckRegisterResidency(cudarecipe.RegisterResidencyClaim{}, cudarecipe.LoadedResources{})
	fmt.Println(err)
	// Output: register residency: invalid identity
}

func ExampleInspector_InspectCompiledResources() {
	i, _ := cudarecipe.NewInspector(".")
	_, err := i.InspectCompiledResources(nil, cudarecipe.CompiledResourceInput{})
	fmt.Println(err)
	// Output: compiled resources: inspector and artifact required
}
