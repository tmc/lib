// Package cudaoccupancy models GPU hardware resource limits and calculates
// theoretical warp occupancy directly from compiled CUDA binary (CUBIN) images.
//
// It inspects CUBIN ELF sections (.nv.info, .nv.shared, and .nv.constant0)
// using pure Go without requiring the CUDA driver, runtime, or external tools.
//
// # Section Structure & Attribute Locations
//
// In CUBIN ELF files, kernel attributes are partitioned across two sections:
//   - .nv.info (global): carries EIATTR_REGCOUNT (0x2f, the actual 32-bit register
//     allocation per thread) and EIATTR_FRAME_SIZE (0x11, local stack frame size),
//     keyed by function symbol index in .symtab.
//   - .nv.info.<kernel> (per-function): carries EIATTR_MAXREG_COUNT (0x1b, the
//     compiler limit rather than the allocation), EIATTR_PARAM_CBANK (0x0a),
//     and parameter descriptors.
//   - .nv.shared.<kernel>: the section size directly records the physical static
//     shared memory charged per CTA (including the hardware/driver reservation).
//
// Note that presence of an attribute in .nv.compat (such as EICOMPAT_ATTR_INST_TCGEN05_MMA)
// does not imply target architecture support (ptxas rejects tcgen05 on sm_121a).
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
