package cudakernel_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleNewArtifactFromSpec() {
	a, err := cudakernel.NewArtifactFromSpec(cudakernel.ArtifactSpec{
		Kind: cudakernel.KindPTX, Raw: []byte("fixture ptx"), EntryPoint: "first", Target: "sm_80",
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	s := a.Spec()
	s.EntryPoint = "second"
	b, err := cudakernel.NewArtifactFromSpec(s)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.EntryPoint(), b.EntryPoint(), a.Digest() == b.Digest(), *a == *b)
	// Output: first second true false
}
