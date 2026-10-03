package cudaprof_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func ExampleCaptureManifest_Complete() {
	var manifest cudaprof.CaptureManifest
	fmt.Println(manifest.Complete())
	// Output: false
}
func ExampleCheckCaptureJoin() {
	var a, b cudaprof.CaptureProviderResult
	fmt.Println(cudaprof.CheckCaptureJoin(a, b, "") != nil)
	// Output: true
}

func ExampleCapturePlan_Validate() {
	var plan cudaprof.CapturePlan
	fmt.Println(plan.Validate() != nil)
	// Output: true
}
func ExampleReadCapturePlan() {
	_, err := cudaprof.ReadCapturePlan(strings.NewReader(`{"version":0}`), 1024)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleWriteCaptureManifest() {
	var output bytes.Buffer
	err := cudaprof.WriteCaptureManifest(&output, cudaprof.CaptureManifest{})
	fmt.Println(err != nil, output.Len())
	// Output: true 0
}
func ExampleReadCaptureManifest() {
	_, err := cudaprof.ReadCaptureManifest(strings.NewReader(`{}`), 1024)
	fmt.Println(err != nil)
	// Output: true
}
