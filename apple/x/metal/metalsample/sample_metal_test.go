//go:build darwin

package metalsample

import (
	"context"
	"math"
	"testing"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
)

func TestArgmaxSamplingMatchesCPU(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	lib, err := dev.CompileLibrary(ArgmaxSamplingMSL)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(lib.GetID()).Release()
	pipeline, err := dev.NewComputePipeline(lib, "ArgmaxSampling")
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(pipeline.GetID()).Release()

	long := make([]float32, 1301)
	for i := range long {
		long[i] = -float32(i + 1)
	}
	long[5], long[1200] = 10, 10
	tests := []struct {
		name    string
		logits  []float32
		threads uint
	}{
		{name: "empty", threads: 1},
		{name: "serial", logits: []float32{-1, 3, 2}, threads: 1},
		{name: "odd width", logits: []float32{-4, 7, 2, 7, 1}, threads: 3},
		{name: "nan and infinity", logits: []float32{float32(math.NaN()), float32(math.Inf(-1)), 4, float32(math.Inf(1)), float32(math.Inf(1))}, threads: 7},
		{name: "all nan", logits: []float32{float32(math.NaN()), float32(math.NaN())}, threads: 32},
		{name: "all negative infinity", logits: []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}, threads: 32},
		{name: "multiple strides", logits: long, threads: 256},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := runMetalArgmax(t, dev, pipeline, test.logits, test.threads)
			if want := SampleArgmaxCPU(test.logits); got != want {
				t.Fatalf("GPU argmax = %d, CPU argmax = %d", got, want)
			}
		})
	}
}

func runMetalArgmax(t *testing.T, dev *metal.Device, pipeline apple.MTLComputePipelineState, logits []float32, threads uint) int32 {
	t.Helper()
	vocabSize := int32(len(logits))
	if len(logits) == 0 {
		logits = []float32{0}
	}
	output := int32(-1)
	logitsBuffer, err := dev.AllocBuffer(len(logits) * 4)
	if err != nil {
		t.Fatal(err)
	}
	copy(unsafe.Slice((*float32)(logitsBuffer.Contents()), len(logits)), logits)
	outputBuffer, err := dev.AllocBuffer(4)
	if err != nil {
		t.Fatal(err)
	}
	*(*int32)(outputBuffer.Contents()) = output
	vocabBuffer, err := dev.AllocBuffer(4)
	if err != nil {
		t.Fatal(err)
	}
	*(*int32)(vocabBuffer.Contents()) = vocabSize
	for _, buffer := range []apple.MTLBuffer{logitsBuffer, outputBuffer, vocabBuffer} {
		defer objectivec.ObjectFromID(buffer.GetID()).Release()
	}

	commandBuffer := dev.Queue().CommandBuffer()
	encoder := commandBuffer.ComputeCommandEncoder()
	encoder.SetComputePipelineState(pipeline)
	encoder.SetBufferWithOffsetAtIndex(logitsBuffer, 0, 0)
	encoder.SetBufferWithOffsetAtIndex(outputBuffer, 0, 1)
	encoder.SetBufferWithOffsetAtIndex(vocabBuffer, 0, 2)
	encoder.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: 1, Height: 1, Depth: 1},
		apple.MTLSize{Width: threads, Height: 1, Depth: 1},
	)
	encoder.EndEncoding()
	commandBuffer.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), commandBuffer); err != nil {
		t.Fatal(err)
	}
	return *(*int32)(outputBuffer.Contents())
}
