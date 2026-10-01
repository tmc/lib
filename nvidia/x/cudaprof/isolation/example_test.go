package isolation_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
	"strings"
)

func Example() {
	result := isolation.Evaluate(nil, "", isolation.Binding{})
	fmt.Println(result.State)
	// Output: unchecked
}

func ExampleRead() {
	receipt, hash, err := isolation.Read(strings.NewReader(`{"version":1}`))
	fmt.Println(receipt.Version, len(hash), err)
	// Output: 1 64 <nil>
}

func ExampleEvaluate() {
	result := isolation.Evaluate(nil, "", isolation.Binding{})
	fmt.Println(result.State, result.Reasons[0])
	// Output: unchecked no bound observation receipt
}

func ExampleRequire() {
	result := isolation.Evaluate(nil, "", isolation.Binding{})
	fmt.Println(isolation.Require(result, false))
	fmt.Println(isolation.Require(result, true))
	// Output:
	// <nil>
	// gpu isolation: unchecked
}

func ExampleReadManifest() {
	hash := strings.Repeat("0", 64)
	raw := fmt.Sprintf(`{"version":1,"role":"gpu-isolation-binding","binding":{},"sources":[{"attachment":"receipt.json","sha256":"%s"}],"hops":[{"input_capture_sha256":"%s","added_receipt_sha256":"%s"}],"consumer":"example"}`, hash, hash, hash)
	manifest, _, err := isolation.ReadManifest(strings.NewReader(raw))
	fmt.Println(manifest.Version, len(manifest.Sources), err)
	// Output: 1 1 <nil>
}
