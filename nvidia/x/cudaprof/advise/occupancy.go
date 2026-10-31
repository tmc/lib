package advise

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// auditCUBINResourceOccupancy inspects CUBIN ELF .nv.info sections (EIATTR_REG_COUNT,
// EIATTR_SHARED_SIZE_SYSMEM, EIATTR_FRAME_SIZE, EIATTR_MAXREG_COUNT) using x/cudaoccupancy
// to calculate target-specific SM warp occupancy, occupancy cliffs, and hardware limiting factors.
func auditCUBINResourceOccupancy(cap *cudacapture.Capture, report *Report, infos []kernelResource, launchesByKernel map[string]launchRecord, math map[string]floatMath, tmem map[string]tensorMemory) {
	if cap == nil || len(infos) == 0 {
		return
	}

	arch := parseArchFromCapture(cap)
	dev, deviceErr := cudaoccupancy.LookupDevice(arch)

	for _, k := range infos {
		threadsPerBlock := 128
		assumed := ""
		blockBasis := "assumed default; actual block geometry unknown"
		if l, ok := launchesByKernel[k.Name]; ok && l.BlockThreads > 0 {
			threadsPerBlock = l.BlockThreads
			blockBasis = "recorded launch"
		} else if k.ReqNTID[0] > 0 {
			threadsPerBlock = k.ReqNTID[0] * max(1, k.ReqNTID[1]) * max(1, k.ReqNTID[2])
			blockBasis = "CUBIN required dimensions"
		} else {
			assumed = "; no launch or .reqntid records the block size, so 128 threads assumed"
		}

		// 1. Stack frame size (EIATTR_FRAME_SIZE)
		if k.FrameSizeBytes > 0 {
			report.Items = append(report.Items, Item{
				rule:                 "frame-size",
				Severity:             SeverityContext,
				Category:             CategoryRegisterPressure,
				EvidenceLevel:        EvidenceObserved,
				NoRuntimeAttribution: true,
				Location:             Location{Kernel: k.Name, Module: k.module},
				Finding: fmt.Sprintf("Kernel %q allocates %d bytes of local stack frame per thread in CUBIN ELF (EIATTR_FRAME_SIZE). Static frame allocation alone does not establish register spilling, executed local-memory traffic or latency cost.",
					k.Name, k.FrameSizeBytes),
				ActionableRecommendation: "Inspect compiler spill diagnostics and source-correlated local-memory instructions. Measure dynamic local reads and writes for the same kernel and case before attributing cost or choosing a register, stack-array or math-mode change.",
				NextMeasurement:          "inspect ptxas spill-load/store diagnostics; collect source-bound dynamic local-load and local-store traffic, then compare matched-case timing",
			})
		}

		// 2. Compiler register limit (EIATTR_MAXREG_COUNT)
		if k.MaxRegCount > 0 && k.Registers >= k.MaxRegCount {
			report.Items = append(report.Items, Item{
				rule:          "maxreg",
				Severity:      SeverityCost,
				Category:      CategoryRegisterPressure,
				EvidenceLevel: EvidenceObserved,
				Location:      Location{Kernel: k.Name, Module: k.module},
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
		tm := tmem[k.Name]
		kRes := cudaoccupancy.KernelResource{
			Registers:       k.Registers,
			StaticSharedMem: k.SharedMemBytes,
			TMEMColumns:     tm.columns,
			ClusterSize:     tm.cluster,
			// A pair is checked against the cluster only when the PTX
			// records one; a cluster set at launch is not in the code.
			CTAPair: tm.pair && tm.cluster > 0,
		}
		// This is a conditional static-resource model, not a complete launch:
		// the capture does not provide dynamic shared bytes here.
		rep := kRes.CalculateDevice(dev, threadsPerBlock, 0)
		if rep.Occupancy < 1.0 {
			warpsPerBlock := (threadsPerBlock + dev.WarpSize - 1) / dev.WarpSize
			if warpsPerBlock <= 0 {
				warpsPerBlock = 1
			}

			severity := SeverityCost

			finding := fmt.Sprintf("Microarchitecture theoretical occupancy on %s for kernel %q is %.1f%% (%d/%d active warps, %d blocks/SM) constrained by %s (limits: %d blocks from registers, %d from smem, %d from block count, %d from block size)",
				dev.Architecture, k.Name, rep.Occupancy*100, rep.TheoreticalWarps, rep.MaxWarps, rep.ActiveBlocks, rep.LimitingFactor, rep.RegLimit, rep.SmemLimit, rep.BlockLimit, rep.WarpLimit)
			if tm.columns > 0 {
				finding += fmt.Sprintf(", %d from %d TMEM columns", rep.TMEMLimit, tm.columns)
			}
			finding += assumed
			finding += fmt.Sprintf(" Conditional static-resource model: %d threads/block (%s); dynamic shared memory is unknown and excluded. This is not achieved or complete launch residency.", threadsPerBlock, blockBasis)

			var rec string
			switch rep.LimitingFactor {
			case cudaoccupancy.LimitRegisters:
				targetBlocks := rep.ActiveBlocks + 1
				regsPerBlockTarget := dev.MaxRegistersPerSM / targetBlocks
				regsPerWarpTarget := regsPerBlockTarget / warpsPerBlock
				regsPerWarpTarget -= regsPerWarpTarget % dev.RegAllocationUnit
				maxRegsTarget := regsPerWarpTarget / dev.WarpSize
				candidate := kRes
				candidate.Registers = maxRegsTarget
				next := candidate.CalculateDevice(dev, threadsPerBlock, 0)
				rec = fmt.Sprintf("Register file capacity (%d regs/SM) limits residency to %d blocks/SM in this static-resource model. ", dev.MaxRegistersPerSM, rep.ActiveBlocks)
				if next.ActiveBlocks < targetBlocks {
					rec += fmt.Sprintf("Reducing registers from %d to <=%d GPRs still models %d blocks/SM: other known resource limits prevent %d blocks/SM (registers %d, smem %d, block count %d, block size %d). Register reduction alone is insufficient in this model.",
						k.Registers, maxRegsTarget, next.ActiveBlocks, targetBlocks, next.RegLimit, next.SmemLimit, next.BlockLimit, next.WarpLimit)
				} else {
					rec += fmt.Sprintf("Reducing registers from %d to <=%d GPRs could model %d blocks/SM (%d warps, %.1f%% occupancy) with the same known static inputs. Test with __launch_bounds__ or -maxrregcount=%d.",
						k.Registers, maxRegsTarget, next.ActiveBlocks, next.TheoreticalWarps, next.Occupancy*100, maxRegsTarget)
				}
				rec += " Verify actual block geometry, dynamic shared bytes and sufficient grid work before using this residency target." + occupancyCaveat
			case cudaoccupancy.LimitSharedMemory:
				rec = fmt.Sprintf("Static shared memory (%d bytes/block) limits residency to %d blocks/SM on %s (%d bytes/SM). A higher-residency target also depends on registers, block geometry and unknown dynamic shared memory; moving static allocation to dynamic does not itself free SM capacity.",
					k.SharedMemBytes, rep.ActiveBlocks, dev.Architecture, dev.MaxSharedMemoryPerSM) + occupancyCaveat
			case cudaoccupancy.LimitBlocks:
				rec = fmt.Sprintf("Maximum thread blocks per SM (%d on %s) limits occupancy because block size (%d threads) is small. Restructure thread block geometry to >=128 or 256 threads to reach higher occupancy.",
					dev.MaxBlocksPerSM, dev.Architecture, threadsPerBlock) + occupancyCaveat
			case cudaoccupancy.LimitTensorMemory:
				rec = fmt.Sprintf("Tensor Memory (%d columns/SM on %s) limits residency: each CTA allocates %d columns with tcgen05.alloc, so %d CTAs fit. Allocate fewer columns per CTA (a smaller accumulator tile, or one allocation shared by a CTA pair) to allow more.",
					dev.TMEMColumnsPerSM, dev.Architecture, tm.columns, rep.ActiveBlocks) + occupancyCaveat
			case cudaoccupancy.LimitCluster:
				rec = fmt.Sprintf("The kernel's cluster (%d CTAs, cta_group::2 %v) does not fit %s, whose largest cluster is %d CTAs and whose CTA pairs need an even cluster: it cannot launch as compiled.", tm.cluster, tm.pair, dev.Architecture, dev.MaxClusterSize)
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
				Location:                 Location{Kernel: k.Name, Module: k.module},
				Finding:                  finding,
				ActionableRecommendation: rec,
				NextMeasurement:          "record actual block geometry and dynamic shared bytes, then measure sm__warps_active.avg.pct_of_peak_sustained_active and A/B wall-clock timing",
			})
		}
	}
}
