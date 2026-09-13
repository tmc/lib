package cuptipcsampling

import (
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// CaptureResolver maps PC sampling coordinates to source and SASS locations
// using decoded code and assembly from a cudacapture.Code bundle.
type CaptureResolver struct {
	Code       *cudacapture.Code
	CUBINCRC   uint64
	Generation uint64
}

// NewCaptureResolver creates a CodeResolver from decoded capture code and expected CUBIN CRC.
func NewCaptureResolver(code *cudacapture.Code, cubinCRC uint64) *CaptureResolver {
	return &CaptureResolver{
		Code:     code,
		CUBINCRC: cubinCRC,
	}
}

// NewCaptureResolverFromCubin attempts to compute the CUBIN CRC using IdentifyCubin
// and constructs a CaptureResolver. If IdentifyCubin fails (e.g. absent CUPTI library),
// CUBINCRC remains 0 and matches any CRC.
func NewCaptureResolverFromCubin(code *cudacapture.Code, cubin []byte) *CaptureResolver {
	r := &CaptureResolver{Code: code}
	if len(cubin) > 0 {
		if id, err := IdentifyCubin(cubin, len(cubin)); err == nil {
			r.CUBINCRC = id.CRC
		}
	}
	return r
}

// NewCaptureResolverFromCapture creates a CaptureResolver from a Capture bundle and decoded Code.
// It sets Generation if modules are recorded in the capture, and computes CUBINCRC via
// IdentifyCubin if the CUBIN file is present.
func NewCaptureResolverFromCapture(code *cudacapture.Code, cap *cudacapture.Capture) *CaptureResolver {
	r := &CaptureResolver{Code: code}
	if cap == nil {
		return r
	}
	if cubinBytes, ok := cap.Files[cap.Code.CUBIN]; ok && len(cubinBytes) > 0 {
		if id, err := IdentifyCubin(cubinBytes, len(cubinBytes)); err == nil {
			r.CUBINCRC = id.CRC
		}
	}
	if len(cap.Modules) > 0 {
		r.Generation = cap.Modules[0].Generation
	}
	return r
}

// ResolvePC implements CodeResolver.
// It maps the function-relative pcOffset to the CUBIN-absolute offset (k.Offset + pcOffset)
// and looks up source, PTX, and SASS lines.
func (r *CaptureResolver) ResolvePC(mod ModuleID, kernelName string, pcOffset uint64) CodeLocation {
	if r == nil || r.Code == nil {
		return CodeLocation{KernelName: kernelName}
	}
	if mod.CRC != 0 && r.CUBINCRC != 0 && mod.CRC != r.CUBINCRC {
		return CodeLocation{KernelName: kernelName}
	}
	if mod.Generation != 0 && r.Generation != 0 && mod.Generation != r.Generation {
		return CodeLocation{KernelName: kernelName}
	}
	k := r.Code.Kernel(kernelName)
	if k == nil {
		return CodeLocation{KernelName: kernelName}
	}

	loc := CodeLocation{
		KernelName:  k.Name,
		SourceFile:  r.Code.SourceFile,
		SourceLine:  int64(k.SourceLine),
		PTXFile:     r.Code.PTXFile,
		PTXLine:     int64(k.PTXLine),
		SASSFile:    r.Code.SASSFile,
		CUBINOffset: k.Offset + pcOffset,
		Resolved:    true,
	}

	if inst := k.Instruction(pcOffset); inst != nil {
		if inst.SourceLine > 0 {
			loc.SourceLine = int64(inst.SourceLine)
		}
		if inst.SASSLine > 0 {
			loc.SASSLine = int64(inst.SASSLine)
		}
	}
	return loc
}
