package cudacapture_test

import (
	"bytes"
	"fmt"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func Example() {
	// Construct a minimal in-memory capture with code roles.
	fn := &profile.Function{ID: 1, Name: "vectorAdd", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{1000}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("__global__ void vectorAdd() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("\x7fELF\x02\x01\x01\x00"), // Minimal bytes
			"kernels.sass":  []byte("// SASS\n"),
		},
	}

	// Write capture to a single pprof byte stream.
	var buf bytes.Buffer
	err := cudacapture.Write(&buf, cap)
	fmt.Printf("write: %v\n", err == nil)

	// Read capture back from the pprof stream.
	readCap, err := cudacapture.Read(&buf)
	fmt.Printf("read: %v, files: %d\n", err == nil, len(readCap.Files))

	// Output:
	// write: true
	// read: true, files: 4
}

func ExampleAttachKernelLabels() {
	labels := make(map[string][]string)
	rec := cudacapture.ActivityRecord{
		Subtype:             "kernel",
		Name:                "vectorAdd",
		GridDim:             [3]uint32{256, 1, 1},
		BlockDim:            [3]uint32{64, 1, 1},
		RegistersPerThread:  32,
		DynamicSharedMemory: 1024,
	}
	cudacapture.AttachKernelLabels(labels, rec)

	fmt.Println("grid:", labels["grid"][0])
	fmt.Println("block:", labels["block"][0])
	fmt.Println("registers:", labels["registers"][0])
	fmt.Println("smem:", labels["smem"][0])

	// Output:
	// grid: 256,1,1
	// block: 64,1,1
	// registers: 32
	// smem: 1024
}

func ExampleBuildActivityProfile() {
	activities := []cudacapture.ActivityRecord{
		{
			Subtype:             "kernel",
			Name:                "vectorAdd",
			Valid:               true,
			Start:               1000,
			End:                 2000,
			GridDim:             [3]uint32{128, 1, 1},
			BlockDim:            [3]uint32{32, 1, 1},
			RegistersPerThread:  16,
			DynamicSharedMemory: 256,
		},
	}
	p := cudacapture.BuildActivityProfile("session-1", 1234, activities)
	fmt.Println("samples:", len(p.Sample))
	fmt.Println("grid:", p.Sample[0].Label["grid"][0])

	// Output:
	// samples: 1
	// grid: 128,1,1
}
