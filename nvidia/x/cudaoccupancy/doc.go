// Package cudaoccupancy models GPU hardware resource limits and calculates
// theoretical warp occupancy directly from compiled CUDA binary (CUBIN) images.
//
// It inspects CUBIN ELF sections (.nv.info, .nv.shared, and .nv.constant0)
// using pure Go without requiring the CUDA driver, runtime, or external tools.
//
// # Section Structure & Attribute Locations
//
// In CUBIN ELF files, kernel attributes are partitioned across sections:
//   - .nv.info (global): carries EIATTR_REGCOUNT (0x2f, 32-bit registers allocated
//     per thread) and EIATTR_FRAME_SIZE (0x11, local stack frame size), keyed by
//     function symbol index in .symtab.
//   - .nv.info.<kernel> (per-function): carries EIATTR_MAXREG_COUNT (0x1b, compiler
//     register limit), EIATTR_PARAM_CBANK (0x0a), and EIATTR_REQ_NTID (0x10).
//   - .nv.shared.<kernel>: section size records physical static shared memory
//     charged per CTA (including hardware/driver reservation), corresponding to
//     EIATTR_SHARED_SIZE_SYSMEM (0x30).
//
// # Basic Usage
//
// Inspect a CUBIN reader and calculate analytical occupancy:
//
//	infos, err := cudaoccupancy.ReadKernelInfo(cubinReader)
//	if err != nil {
//		log.Fatal(err)
//	}
//	for _, k := range infos {
//		res, err := cudaoccupancy.CalculateOccupancy("sm_90", k.Registers, k.SharedMemBytes, 256)
//		if err != nil {
//			log.Fatal(err)
//		}
//		fmt.Printf("%s: %d/%d warps (%.1f%%) limited by %s\n",
//			k.Name, res.TheoreticalWarps, res.MaxWarps, res.Occupancy*100, res.LimitingFactor)
//	}
package cudaoccupancy
