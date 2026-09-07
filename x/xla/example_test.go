package xla_test

import (
	"bytes"
	"context"
	"fmt"
	"log"

	"github.com/tmc/lib/x/xla"
)

func Example() {
	// Initialize a new engine by loading the PJRT plugin (e.g., CPU, GPU, or TPU)
	engine, err := xla.NewEngine("/path/to/pjrt_c_api_cpu_plugin.so")
	if err != nil {
		fmt.Printf("Engine initialization failed: %v\n", err)
		return
	}
	defer engine.Close()

	// Get locally addressable devices
	devices, err := engine.Devices()
	if err != nil || len(devices) == 0 {
		log.Fatalf("No devices found: %v", err)
	}

	// Compile an XLA program (e.g., from MLIR or HLO bytecode)
	programCode := []byte("... MLIR module bytecode ...")
	exec, err := engine.Compile(programCode, "MLIR", xla.DefaultCompileOptions())
	if err != nil {
		log.Fatalf("Compilation failed: %v", err)
	}
	defer exec.Close()

	// Transfer input data from host memory to device memory
	inputData := []byte{1, 2, 3, 4} // Example raw float32 bytes
	dims := []int64{1}              // 1D array of 1 element

	arg, err := engine.NewBufferFromHost(inputData, xla.F32, dims, devices[0])
	if err != nil {
		log.Fatalf("Failed to create input buffer: %v", err)
	}
	defer arg.Close()

	// Execute the compiled program on the device
	ctx := context.Background()
	outputs, err := exec.Execute(ctx, []*xla.Buffer{arg})
	if err != nil {
		log.Fatalf("Execution failed: %v", err)
	}
	defer outputs[0].Close()

	// Read the output buffer back to host memory using io.WriterTo
	var buf bytes.Buffer
	if _, err := outputs[0].WriteTo(&buf); err != nil {
		log.Fatalf("Failed to read output: %v", err)
	}

	fmt.Printf("Output size: %d bytes\n", buf.Len())
}
