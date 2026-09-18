package cuptipcsampling

import (
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// CaptureResolver maps PC sampling coordinates to source and SASS locations
// using decoded code and assembly from a cudacapture.Code bundle.
type CaptureResolver struct {
	Code     *cudacapture.Code
	CUBINCRC uint64
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
// It computes CUBINCRC via IdentifyCubin if the CUBIN file is present.
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
	k := r.Code.Kernel(kernelName)
	if k == nil {
		return CodeLocation{KernelName: kernelName}
	}

	// A source file and a source line are one fact and are taken from one
	// tier together. Taking the line from the instruction while keeping the
	// unit's file names a line that was measured in a different file, which is
	// a location no tier ever reported: it is wrong in the direction that
	// looks right, because the line is plausible and the reader has no way to
	// see that the two halves came from different places.
	//
	// The unit's file is the last resort, used only for a kernel whose
	// defining file is unknown. A single-file module makes all three the same
	// file, which is why this went unnoticed until a module had fourteen.
	sourceFile, sourceLine := r.Code.SourceFile, int64(k.SourceLine)
	if k.SourceFile != "" {
		sourceFile = k.SourceFile
	}

	loc := CodeLocation{
		KernelName:  k.Name,
		PTXFile:     r.Code.PTXFile,
		PTXLine:     int64(k.PTXLine),
		SASSFile:    r.Code.SASSFile,
		CUBINOffset: k.Offset + pcOffset,
		Resolved:    true,
	}

	if inst := k.Instruction(pcOffset); inst != nil {
		// cudacapture leaves Instruction.SourceFile empty exactly when
		// SourceLine is zero, so this moves the pair or neither half. An
		// instruction the compiler never located falls back to the kernel's
		// definition, file and line alike.
		if inst.SourceLine > 0 {
			sourceFile, sourceLine = inst.SourceFile, int64(inst.SourceLine)
		}
		if inst.SASSLine > 0 {
			loc.SASSLine = int64(inst.SASSLine)
		}
		if inst.PTXLine > 0 {
			loc.PTXLine = int64(inst.PTXLine)
		}
		loc.InlinedAt = inst.InlinedAt
	}
	loc.SourceFile, loc.SourceLine = sourceFile, sourceLine
	return loc
}
