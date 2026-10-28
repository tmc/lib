package cudainspect_test

import (
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudainspect"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func ExampleCalibrateWorkload() {
	raw, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex")
	if err != nil {
		log.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil {
		log.Fatal(err)
	}
	sass, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.sass")
	if err != nil {
		log.Fatal(err)
	}
	ins, err := cudainspect.NewInspector(os.TempDir())
	if err != nil {
		log.Fatal(err)
	}
	rep, err := ins.Report(context.Background(), cubin, cudainspect.Request{
		Symbol:   "addk",
		Required: []string{"precision_attribution", "target"},
		SASS:     string(sass),
	})
	if err != nil {
		log.Fatal(err)
	}
	model, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		log.Fatal(err)
	}

	res, err := cudainspect.CalibrateWorkload(context.Background(), cudainspect.CalibrationRequest{
		Workload:  "gemm-10240-fp32",
		Report:    rep,
		Model:     model,
		Shape:     cudainspect.GEMMShape{M: 10240, N: 10240, K: 10240, Repetitions: 1},
		Separates: "fp32 from fp16_tensor",
		FailsWhen: "a kernel interleaves HMMA and FFMA in comparable shares",
		Observe: func(context.Context) (cudainspect.WorkloadObservation, error) {
			return cudainspect.WorkloadObservation{
				ArtifactSHA256: rep.Artifact.SHA256,
				Entry:          "addk",
				Complete:       true,
				Identity: cudacensus.Identity{
					DeviceName:    "NVIDIA GB10",
					Architecture:  "sm_121",
					ComputeMajor:  12,
					ComputeMinor:  1,
					DriverVersion: "580.65.06",
				},
				Residency: cudaroofline.DeviceResident,
				Timing: cudainspect.DeviceTiming{
					Duration: 100 * time.Millisecond,
					Clock:    "cuda_event",
				},
				SampleInterval: 10 * time.Millisecond,
				Readings: []cudacensus.Reading{
					{Counter: cudacensus.FP32Active, Value: 0.5, Available: true},
					{Counter: cudacensus.DRAMBandwidthUtil, Value: 0.05, Available: true},
				},
				RanAt: time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC),
			}, nil
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Calibration.Report())
	fmt.Printf("receipt samples=%d error=%.4f\n", res.Receipt.Samples, res.Receipt.Error)
	// Output:
	// gemm-10240-fp32 on NVIDIA GB10 sm_121, driver 580.65.06: analytic 21.4748 TFLOPS (2*m*n*k, m=10240 n=10240 k=10240) over 100ms, estimated 20.0000 TFLOPS by instruction_mix, relative error -6.87% (static_mix)
	// receipt samples=1 error=0.0687
}
