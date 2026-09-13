//go:build darwin

package metalbench

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
)

func TestDynamicEngineOutputs(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	for _, dim := range []int{256, 257} {
		for _, batch := range []int{1, 7, 8, 17} {
			t.Run(fmt.Sprintf("dim%d/batch%d", dim, batch), func(t *testing.T) {
				const outDim = 19
				static, err := NewStaticBaselineEngine(dev, dim, outDim)
				if err != nil {
					t.Fatal(err)
				}
				defer static.Free()
				adaptive, err := NewAdaptivePGOEngine(dev, dim, outDim)
				if err != nil {
					t.Fatal(err)
				}
				defer adaptive.Free()
				weights := static.weightBuf.HostSlice()
				input := static.inBuf.HostSlice()
				for i := range weights {
					weights[i] = float32(i%7 - 3)
				}
				for i := range input {
					input[i] = float32(i%5 - 2)
				}
				copy(adaptive.weightBuf.HostSlice(), weights)
				copy(adaptive.inBuf.HostSlice(), input)
				for i := range static.outBuf.HostSlice() {
					static.outBuf.HostSlice()[i] = -9999
					adaptive.outBuf.HostSlice()[i] = -9999
				}
				phase := WorkloadPhase{BatchSize: batch, Steps: 1}
				if _, err := static.ExecutePhase(phase); err != nil {
					t.Fatal(err)
				}
				if _, err := adaptive.ExecutePhase(phase); err != nil {
					t.Fatal(err)
				}
				for b := 0; b < batch; b++ {
					for row := 0; row < outDim; row++ {
						var want float32
						for k := 0; k < dim; k++ {
							want += input[b*dim+k] * weights[row*dim+k]
						}
						i := b*outDim + row
						for _, result := range []struct {
							name string
							got  float32
						}{{"static", static.outBuf.HostSlice()[i]}, {"adaptive", adaptive.outBuf.HostSlice()[i]}} {
							if result.got != want {
								t.Fatalf("%s output[%d] = %g, want %g", result.name, i, result.got, want)
							}
						}
					}
				}
			})
		}
	}
}
