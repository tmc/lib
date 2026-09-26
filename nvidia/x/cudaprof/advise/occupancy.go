package advise

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// auditCUBINResourceOccupancy inspects CUBIN ELF .nv.info sections (EIATTR_REG_COUNT,
// EIATTR_SHARED_SIZE_SYSMEM, EIATTR_FRAME_SIZE, EIATTR_MAXREG_COUNT) using x/cudaoccupancy
// to calculate target-specific SM warp occupancy, occupancy cliffs, and hardware limiting factors.
func auditCUBINResourceOccupancy(cap *cudacapture.Capture, report *Report, infos []cudaoccupancy.KernelResourceInfo, launchesByKernel map[string]launchRecord, math map[string]floatMath) {
	if cap == nil || len(infos) == 0 {
		return
	}

	arch := parseArchFromCapture(cap)
	dev, deviceErr := cudaoccupancy.LookupDevice(arch)

	for _, k := range infos {
		threadsPerBlock := 128
		if l, ok := launchesByKernel[k.Name]; ok && l.BlockThreads > 0 {
			threadsPerBlock = l.BlockThreads
		} else if k.ReqNTID[0] > 0 {
			threadsPerBlock = k.ReqNTID[0] * max(1, k.ReqNTID[1]) * max(1, k.ReqNTID[2])
		}

		// 1. Stack frame size (EIATTR_FRAME_SIZE)
		if k.FrameSizeBytes > 0 {
			rec := "Reduce local variable lifetimes, eliminate large stack arrays, or apply __launch_bounds__ to eliminate local memory stack frame spills."
			if math[k.Name].costs(k.FrameSizeBytes) {
				rec += fastMathLead
			}
			report.Items = append(report.Items, Item{
				rule:          "frame-size",
				Severity:      SeverityReadFirst,
				Category:      CategoryRegisterPressure,
				EvidenceLevel: EvidenceObserved,
				Location:      Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q allocates %d bytes of local stack frame per thread in CUBIN ELF (EIATTR_FRAME_SIZE), causing spills to high-latency local memory",
					k.Name, k.FrameSizeBytes),
				ActionableRecommendation: rec,
				NextMeasurement:          "verify zero local memory traffic with l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
			})
		}

		// 2. Compiler register limit (EIATTR_MAXREG_COUNT)
		if k.MaxRegCount > 0 && k.Registers >= k.MaxRegCount {
			report.Items = append(report.Items, Item{
				rule:          "maxreg",
				Severity:      SeverityCost,
				Category:      CategoryRegisterPressure,
				EvidenceLevel: EvidenceObserved,
				Location:      Location{Kernel: k.Name},
				Finding: fmt.Sprintf("Kernel %q reached the compiler register limit of %d GPRs (EIATTR_MAXREG_COUNT), which may force register spilling or sub-optimal instruction scheduling",
					k.Name, k.MaxRegCount),
				ActionableRecommendation: "Raise compiler register budget (-maxrregcount) or restructure compute loops to reduce register demand.",
				NextMeasurement:          "inspect SASS for spills or verify -maxrregcount with cuobjdump -res-usage",
			})
		}

		// 3. Analytical occupancy & hardware limiting factor
		// Unknown hardware cannot borrow another architecture's resource limits.
		if deviceErr != nil {
			continue
		}
		kRes := cudaoccupancy.KernelResource{
			Registers:       k.Registers,
			StaticSharedMem: k.SharedMemBytes,
		}
		rep := kRes.CalculateDevice(dev, threadsPerBlock, 0)
		if rep.Occupancy < 1.0 {
			warpsPerBlock := (threadsPerBlock + dev.WarpSize - 1) / dev.WarpSize
			if warpsPerBlock <= 0 {
				warpsPerBlock = 1
			}

			severity := SeverityCost

			finding := fmt.Sprintf("Microarchitecture theoretical occupancy on %s for kernel %q is %.1f%% (%d/%d active warps, %d blocks/SM) constrained by %s (limits: %d blocks from registers, %d from smem, %d from block count, %d from block size)",
				dev.Architecture, k.Name, rep.Occupancy*100, rep.TheoreticalWarps, rep.MaxWarps, rep.ActiveBlocks, rep.LimitingFactor, rep.RegLimit, rep.SmemLimit, rep.BlockLimit, rep.WarpLimit)

			var rec string
			switch rep.LimitingFactor {
			case cudaoccupancy.LimitRegisters:
				targetBlocks := rep.ActiveBlocks + 1
				regsPerBlockTarget := dev.MaxRegistersPerSM / targetBlocks
				regsPerWarpTarget := regsPerBlockTarget / warpsPerBlock
				maxRegsTarget := regsPerWarpTarget / dev.WarpSize
				targetWarps := targetBlocks * warpsPerBlock
				targetOcc := (float64(targetWarps) / float64(dev.MaxWarpsPerSM)) * 100
				rec = fmt.Sprintf("Register file capacity (%d regs/SM) limits residency to %d blocks/SM. Reducing registers from %d to <=%d GPRs would allow %d blocks/SM (%d warps, %.1f%% occupancy). Test with __launch_bounds__ or -maxrregcount=%d.",
					dev.MaxRegistersPerSM, rep.ActiveBlocks, k.Registers, maxRegsTarget, targetBlocks, targetWarps, targetOcc, maxRegsTarget) + occupancyCaveat
			case cudaoccupancy.LimitSharedMemory:
				rec = fmt.Sprintf("Static shared memory (%d bytes/block) limits residency to %d blocks/SM on %s (%d bytes/SM). Reduce shared memory or dynamically allocate to allow higher CTA residency.",
					k.SharedMemBytes, rep.ActiveBlocks, dev.Architecture, dev.MaxSharedMemoryPerSM) + occupancyCaveat
			case cudaoccupancy.LimitBlocks:
				rec = fmt.Sprintf("Maximum thread blocks per SM (%d on %s) limits occupancy because block size (%d threads) is small. Restructure thread block geometry to >=128 or 256 threads to reach higher occupancy.",
					dev.MaxBlocksPerSM, dev.Architecture, threadsPerBlock) + occupancyCaveat
			case cudaoccupancy.LimitBlockSize:
				rec = "Restructure thread block geometry to fit within hardware block size limits (<= 1024 threads per block)."
			default:
				rec = "Tune thread block dimensions and resource consumption to maximize concurrent warp execution." + occupancyCaveat
			}

			report.Items = append(report.Items, Item{
				rule:                     "occupancy",
				Severity:                 severity,
				Category:                 CategoryRegisterPressure,
				EvidenceLevel:            EvidenceSuspected,
				Location:                 Location{Kernel: k.Name},
				Finding:                  finding,
				ActionableRecommendation: rec,
				NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
			})
		}
	}
}
