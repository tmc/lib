// Package cudaoccupancy models GPU hardware resource limits and calculates
// theoretical warp occupancy directly from compiled CUDA binary (CUBIN) images.
//
// It inspects CUBIN ELF sections (.nv.info, .nv.shared, and .nv.constant0)
// using pure Go without requiring the CUDA driver, runtime, or external tools.
//
// # Basic Usage
//
// Inspect a CUBIN and calculate theoretical occupancy:
//
//	kernels, err := cudaoccupancy.InspectCubin(cubinBytes)
//	if err != nil {
//		log.Fatal(err)
//	}
//	for _, k := range kernels {
//		report, err := k.Calculate("sm_121", 256, 0)
//		if err != nil {
//			log.Fatal(err)
//		}
//		fmt.Printf("%s: %.1f%% (%s)\n", k.Name, report.Occupancy*100, report.LimitingFactor)
//	}
package cudaoccupancy
