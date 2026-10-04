package cuptirange

import "fmt"

// ReplayModeSupport reports whether the selected replay mode has a qualified
// native contract for version. UserReplay retains the existing capture contract.
//
// Kernel replay is documented by the CUDA 13.2 Range Profiling API, section
// 2.9: https://docs.nvidia.com/cupti/13.2.0/main/main.html#range-profiling-api
// Qualifying another host struct layout does not qualify its replay contract.
//
// 130400 is admitted on its own evidence. Its section 2.9 text is 13.2's plus a
// pointer to the new cache-state reset, and its cupti_range_profiler.h adds
// only Enable_Params.cacheControlMode, which a struct sized to
// pRangeProfilerObject leaves at CUPTI_CACHE_CONTROL_MODE_NONE. On GB20B with
// driver 580.173.02 the kernel-replay controls in x/cuptibench (DS4 bounds,
// graph allocation identity, forward and reversed graph kernel association,
// host pass budget) passed against CUPTI 130400 as they do against 130202.
//
// A later release in the same major takes the replay contract of the newest
// layout at or below it, as [layoutVersion] does for struct sizes.
func ReplayModeSupport(version uint32, mode ReplayMode) error {
	switch mode {
	case UserReplay:
		return nil
	case KernelReplay:
		if l, _ := layoutVersion(version); l == 130200 || l == 130202 || l == 130400 {
			return nil
		}
		return fmt.Errorf("cuptirange: kernel replay is not qualified for CUPTI version %d", version)
	default:
		return fmt.Errorf("cuptirange: invalid replay mode %d", mode)
	}
}
