package cudainspect_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudainspect"
)

func ExampleCompiledResources_CorroboratedSpills() {
	n := uint64(4)
	r := cudainspect.CompiledResources{InstructionsKnown: true, SpillStoreBytes: cudainspect.ResourceFact{Value: &n, Category: "compiler reported"}, LocalStores: []uint64{32}}
	fmt.Println(r.CorroboratedSpills())
	// Output: true
}

func ExampleCompiledResources_CheckRegisterResidency() {
	r := new(cudainspect.CompiledResources)
	err := r.CheckRegisterResidency(cudainspect.RegisterResidencyClaim{}, cudainspect.LoadedResources{})
	fmt.Println(err)
	// Output: register residency: invalid identity
}

func ExampleInspector_InspectCompiledResources() {
	i, _ := cudainspect.NewInspector(".")
	_, err := i.InspectCompiledResources(nil, cudainspect.CompiledResourceInput{})
	fmt.Println(err)
	// Output: compiled resources: inspector and artifact required
}
