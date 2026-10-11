package cudaflightrecorder_test

import (
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func ExampleNewCUPTIActivityCapture() {
	// Set ExclusiveOwnership only when the application controls CUPTI activity
	// registration and enable/disable APIs, including other profiling libraries.
	provider, err := cudaflightrecorder.NewCUPTIActivityCapture(cudaflightrecorder.CUPTIActivityCaptureConfig{
		Activities: cuptiactivity.Config{
			Kinds:       []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel},
			BufferBytes: 64 << 10, MaxBuffers: 4, MaxRecords: 1024,
		},
		CollectionWindow:   100 * time.Millisecond,
		ExclusiveOwnership: true,
	})
	// Construction validates configuration; Trigger performs native admission.
	fmt.Println(provider != nil, err)
	// Output: true <nil>
}
