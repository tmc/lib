package cuptirange

import "fmt"

// ReplayModeSupport reports whether the selected replay mode has a qualified
// native contract for version. UserReplay retains the existing capture contract.
//
// Kernel replay is documented by the CUDA 13.2 Range Profiling API, section
// 2.9: https://docs.nvidia.com/cupti/13.2.0/main/main.html#range-profiling-api
// Qualifying another host struct layout does not qualify its replay contract.
func ReplayModeSupport(version uint32, mode ReplayMode) error {
	switch mode {
	case UserReplay:
		return nil
	case KernelReplay:
		if version == 130200 || version == 130202 {
			return nil
		}
		return fmt.Errorf("cuptirange: kernel replay is not qualified for CUPTI version %d", version)
	default:
		return fmt.Errorf("cuptirange: invalid replay mode %d", mode)
	}
}
