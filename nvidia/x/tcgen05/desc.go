package tcgen05

import "fmt"

// ScaleVec encodes the block-scale vector multiplier for tcgen05.mma.
type ScaleVec uint8

const (
	ScaleVecNone ScaleVec = iota
	ScaleVec1X
	ScaleVec2X
	ScaleVec4X
)

func (s ScaleVec) String() string {
	switch s {
	case ScaleVecNone:
		return "none"
	case ScaleVec1X:
		return "1X"
	case ScaleVec2X:
		return "2X"
	case ScaleVec4X:
		return "4X"
	}
	return fmt.Sprintf("ScaleVec(%d)", uint8(s))
}

// InstDesc models the 32-bit instruction descriptor (idesc) consumed by
// tcgen05.mma (PTX ISA 9.4 section 9.7.18.4 and 9.7.18.10.9.3).
type InstDesc struct {
	Sparsity        uint8
	SparsityVersion uint8 // Bit 12: 0 (v0) for sm_100a/sm_103a, 1 (v1) for sm_107a/sm_107f
	Saturate        bool
	DType           DType
	AType           DType
	BType           DType
	NegateA         bool
	NegateB         bool
	TransposeA      bool
	TransposeB      bool
	M               uint16
	N               uint16
	ScaleVec        ScaleVec
}

// Validate checks target-independent descriptor field ranges.
func (d InstDesc) Validate() error {
	if d.Sparsity > 3 {
		return fmt.Errorf("tcgen05 instdesc: invalid Sparsity %d (want 0..3)", d.Sparsity)
	}
	if d.SparsityVersion > 1 {
		return fmt.Errorf("tcgen05 instdesc: invalid SparsityVersion %d (want 0 or 1)", d.SparsityVersion)
	}
	if d.ScaleVec > ScaleVec4X {
		return fmt.Errorf("tcgen05 instdesc: invalid ScaleVec %d", uint8(d.ScaleVec))
	}
	if d.M != 0 && (d.M%8 != 0 || d.M > 256) {
		return fmt.Errorf("tcgen05 instdesc: invalid M %d (want multiple of 8 in 8..256)", d.M)
	}
	if d.N != 0 && (d.N%8 != 0 || d.N > 512) {
		return fmt.Errorf("tcgen05 instdesc: invalid N %d (want multiple of 8 in 8..512)", d.N)
	}
	return nil
}

// ValidateForTarget validates d against PTX ISA 9.4 §9.7.18.10.9.3 target rules:
// bit 12 (SparsityVersion) must be v0 (0) for sm_100a/sm_103a/sm_100f/sm_103f
// and v1 (1) for sm_107a/sm_107f when Sparsity != 0.
func (d InstDesc) ValidateForTarget(target string) error {
	if err := d.Validate(); err != nil {
		return err
	}
	switch target {
	case "sm_100a", "sm_103a", "sm_100f", "sm_103f":
		if d.Sparsity != 0 && d.SparsityVersion != 0 {
			return fmt.Errorf("tcgen05 instdesc: target %s requires SparsityVersion v0 (0) in bit 12 when Sparsity != 0, got %d", target, d.SparsityVersion)
		}
		if d.SparsityVersion != 0 {
			return fmt.Errorf("tcgen05 instdesc: target %s requires SparsityVersion v0 (0) in bit 12, got %d", target, d.SparsityVersion)
		}
	case "sm_107a", "sm_107f":
		if d.Sparsity != 0 && d.SparsityVersion != 1 {
			return fmt.Errorf("tcgen05 instdesc: target %s requires SparsityVersion v1 (1) in bit 12 when Sparsity != 0, got %d", target, d.SparsityVersion)
		}
	default:
		return fmt.Errorf("tcgen05 instdesc: unsupported target %q (want sm_100a, sm_103a, sm_107a, sm_100f, sm_103f, or sm_107f)", target)
	}
	return nil
}

// Encode packs d into a 32-bit tcgen05.mma instruction descriptor word.
func (d InstDesc) Encode() uint32 {
	var w uint32
	w |= uint32(d.Sparsity & 0x3)
	if d.Saturate {
		w |= 1 << 2
	}
	w |= uint32(d.DType&0x7) << 3
	w |= uint32(d.AType&0x7) << 6
	w |= uint32(d.BType&0x7) << 9
	w |= uint32(d.SparsityVersion&0x1) << 12
	if d.NegateB {
		w |= 1 << 13
	}
	if d.TransposeA {
		w |= 1 << 14
	}
	if d.TransposeB {
		w |= 1 << 15
	}
	w |= uint32((d.N/8)&0x7f) << 16
	w |= uint32(d.ScaleVec&0x3) << 23
	if d.NegateA {
		w |= 1 << 25
	}
	w |= uint32((d.M/8)&0x3f) << 26
	return w
}

// DecodeInstDesc unpacks a 32-bit tcgen05.mma instruction descriptor word.
func DecodeInstDesc(w uint32) InstDesc {
	return InstDesc{
		Sparsity:        uint8(w & 0x3),
		SparsityVersion: uint8((w >> 12) & 0x1),
		Saturate:        (w>>2)&1 != 0,
		DType:           DType((w >> 3) & 0x7),
		AType:           DType((w >> 6) & 0x7),
		BType:           DType((w >> 9) & 0x7),
		NegateA:         (w>>25)&1 != 0,
		NegateB:         (w>>13)&1 != 0,
		TransposeA:      (w>>14)&1 != 0,
		TransposeB:      (w>>15)&1 != 0,
		N:               uint16(((w >> 16) & 0x7f) * 8),
		ScaleVec:        ScaleVec((w >> 23) & 0x3),
		M:               uint16(((w >> 26) & 0x3f) * 8),
	}
}

// Swizzle specifies the shared-memory swizzle pattern for a tcgen05 matrix descriptor.
type Swizzle uint8

const (
	SwizzleNone Swizzle = 0
	Swizzle32B  Swizzle = 32
	Swizzle64B  Swizzle = 64
	Swizzle128B Swizzle = 128
)

func (s Swizzle) String() string {
	switch s {
	case SwizzleNone:
		return "none"
	case Swizzle32B:
		return "32B"
	case Swizzle64B:
		return "64B"
	case Swizzle128B:
		return "128B"
	}
	return fmt.Sprintf("Swizzle(%d)", uint8(s))
}

// SharedDesc models the 64-bit shared-memory matrix descriptor consumed by
// tcgen05.mma and tcgen05.cp (PTX ISA 9.4 section 9.7.18.3).
// BaseAddr, LeadingByteOffset, and StrideByteOffset are 16-byte aligned byte
// offsets within the 256 KB per-CTA shared memory window (18-bit byte range).
type SharedDesc struct {
	BaseAddr          uint32
	LeadingByteOffset uint32
	StrideByteOffset  uint32
	Swizzle           Swizzle
}

// Encode packs d into a 64-bit tcgen05 shared-memory descriptor word.
func (d SharedDesc) Encode() uint64 {
	var swCode uint64
	switch d.Swizzle {
	case Swizzle128B:
		swCode = 1
	case Swizzle64B:
		swCode = 2
	case Swizzle32B:
		swCode = 3
	default:
		swCode = 0
	}
	base := uint64((d.BaseAddr & 0x3ffff) >> 4)
	lbo := uint64((d.LeadingByteOffset & 0x3ffff) >> 4)
	sbo := uint64((d.StrideByteOffset & 0x3ffff) >> 4)
	return base | (lbo << 16) | (sbo << 32) | (swCode << 62)
}

// DecodeSharedDesc unpacks a 64-bit tcgen05 shared-memory descriptor word.
func DecodeSharedDesc(w uint64) SharedDesc {
	var sw Swizzle
	switch (w >> 62) & 0x3 {
	case 1:
		sw = Swizzle128B
	case 2:
		sw = Swizzle64B
	case 3:
		sw = Swizzle32B
	default:
		sw = SwizzleNone
	}
	return SharedDesc{
		BaseAddr:          uint32((w & 0x3fff) << 4),
		LeadingByteOffset: uint32(((w >> 16) & 0x3fff) << 4),
		StrideByteOffset:  uint32(((w >> 32) & 0x3fff) << 4),
		Swizzle:           sw,
	}
}

// TMEMAddr represents a 32-bit Tensor Memory address composed of a 16-bit lane
// row index in the upper half and a 16-bit column offset in the lower half.
type TMEMAddr struct {
	Row uint16
	Col uint16
}

// Encode returns the 32-bit TMEM address word (uint32(Row)<<16) | uint32(Col).
func (a TMEMAddr) Encode() uint32 {
	return (uint32(a.Row) << 16) | uint32(a.Col)
}

// DecodeTMEMAddr unpacks a 32-bit TMEM address word into its (Row, Col) pair.
func DecodeTMEMAddr(w uint32) TMEMAddr {
	return TMEMAddr{
		Row: uint16(w >> 16),
		Col: uint16(w & 0xffff),
	}
}
