package cupti

import "unsafe"

// PCSamplingReasonLayout describes the parameter sizes used by the native
// PC-sampling reason inventory calls on a 64-bit host.
//
// The layout is unchanged in the validated CUPTI releases listed by
// [PCSamplingReasonLayoutFor]. A release not listed there has not had its
// ABI checked and must be rejected by callers that pass these parameters.
type PCSamplingReasonLayout struct {
	CountParamsSize uint
	NamesParamsSize uint
}

// PCSamplingReasonLayoutFor reports the validated parameter layout for a CUPTI
// API version. CUPTI 27 is the legacy CUDA 12 encoding; 130001, 130200 and
// 130400 are CUDA 13 releases. The parameter structs contain pointers and are
// admitted only on 64-bit hosts.
func PCSamplingReasonLayoutFor(version uint32) (PCSamplingReasonLayout, bool) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		return PCSamplingReasonLayout{}, false
	}
	switch version {
	case 27, 130001, 130200, 130400:
		return PCSamplingReasonLayout{
			CountParamsSize: uint(unsafe.Sizeof(PCSamplingGetNumStallReasonsParams{})),
			NamesParamsSize: uint(unsafe.Sizeof(PCSamplingGetStallReasonsParams{})),
		}, true
	default:
		return PCSamplingReasonLayout{}, false
	}
}
