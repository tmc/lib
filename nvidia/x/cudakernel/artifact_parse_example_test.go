package cudakernel_test

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func ExampleParseArtifact() {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "sm_80")
	if err != nil {
		fmt.Println(err)
		return
	}
	data, err := json.Marshal(a)
	if err != nil {
		fmt.Println(err)
		return
	}
	b, err := cudakernel.ParseArtifact(data)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(b.EntryPoint(), b.Digest() == a.Digest())

	extra := strings.Replace(string(data), `"kind":`, `"extra":1,"kind":`, 1)
	_, err = cudakernel.ParseArtifact([]byte(extra))
	fmt.Println(err)
	// Output:
	// k true
	// cudakernel: parse artifact: json: unknown field "extra"
}

func ExampleArtifact_Size() {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "sm_80")
	if err != nil {
		fmt.Println(err)
		return
	}
	var zero cudakernel.Artifact
	fmt.Println(a.Size(), zero.Size())
	// Output: 11 0
}
