package cuptipcsampling

import (
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// CUPTI reads the caller-declared Size to decide how much of a parameter struct
// it may fill, then writes the fields in place through the caller's pointer. A
// size that does not match the layout the loaded library expects is a native
// write at the wrong offset, not an error return, so the declared size must be
// derived from the same Go type CUPTI is handed rather than written out by hand.
//
// Each function below ends the prefix at the struct's last field. The values
// they produce are pinned by TestParamSizes against sizes measured with
// sizeof(3) on every CUDA toolkit installed on the GB10 host, 12.6 through 13.4:
// 24 for the four context-only structs, 32 for GetData and 40 for
// ConfigurationInfo. Regenerating nvidia/cupti against a header that moves a
// field changes these functions and fails that test, which is the point; a
// literal would have stayed silently wrong.

func enableParamsSize() uint {
	var p cupti.PCSamplingEnableParams
	return uint(unsafe.Offsetof(p.Ctx) + unsafe.Sizeof(p.Ctx))
}

func disableParamsSize() uint {
	var p cupti.PCSamplingDisableParams
	return uint(unsafe.Offsetof(p.Ctx) + unsafe.Sizeof(p.Ctx))
}

func startParamsSize() uint {
	var p cupti.PCSamplingStartParams
	return uint(unsafe.Offsetof(p.Ctx) + unsafe.Sizeof(p.Ctx))
}

func stopParamsSize() uint {
	var p cupti.PCSamplingStopParams
	return uint(unsafe.Offsetof(p.Ctx) + unsafe.Sizeof(p.Ctx))
}

func getDataParamsSize() uint {
	var p cupti.PCSamplingGetDataParams
	return uint(unsafe.Offsetof(p.PcSamplingData) + unsafe.Sizeof(p.PcSamplingData))
}

func configurationInfoParamsSize() uint {
	var p cupti.PCSamplingConfigurationInfoParams
	return uint(unsafe.Offsetof(p.PPCSamplingConfigurationInfo) + unsafe.Sizeof(p.PPCSamplingConfigurationInfo))
}
