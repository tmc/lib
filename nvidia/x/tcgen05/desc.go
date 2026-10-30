package tcgen05

import "fmt"

// A Kind is the .kind qualifier of a tcgen05.mma instruction.
// It selects the layout of the instruction descriptor.
type Kind uint8

const (
	KindTF32     Kind = iota + 1 // .kind::tf32
	KindF16                      // .kind::f16
	KindF8F6F4                   // .kind::f8f6f4
	KindI8                       // .kind::i8
	KindTI16                     // .kind::ti16
	KindMXF8F6F4                 // .kind::mxf8f6f4
	KindMXF4                     // .kind::mxf4
	KindMXF4NVF4                 // .kind::mxf4nvf4
)

var kindNames = [...]string{
	KindTF32:     "tf32",
	KindF16:      "f16",
	KindF8F6F4:   "f8f6f4",
	KindI8:       "i8",
	KindTI16:     "ti16",
	KindMXF8F6F4: "mxf8f6f4",
	KindMXF4:     "mxf4",
	KindMXF4NVF4: "mxf4nvf4",
}

// String returns the PTX spelling of k without the .kind:: prefix.
func (k Kind) String() string {
	if int(k) < len(kindNames) && kindNames[k] != "" {
		return kindNames[k]
	}
	return fmt.Sprintf("Kind(%d)", uint8(k))
}

// parseKind returns the Kind spelled s, as printed by [Kind.String].
func parseKind(s string) (Kind, bool) {
	for k, name := range kindNames {
		if name != "" && name == s {
			return Kind(k), true
		}
	}
	return 0, false
}

// blockScaled reports whether k uses the block-scaled descriptor layout
// (PTX ISA 9.4 Tables 52 and 53) rather than Table 51.
func (k Kind) blockScaled() bool {
	return k == KindMXF8F6F4 || k == KindMXF4 || k == KindMXF4NVF4
}

// An ElemType is a matrix element type as named by the tcgen05.mma
// instruction descriptor tables.
type ElemType uint8

const (
	ElemF16 ElemType = iota + 1
	ElemBF16
	ElemTF32
	ElemF32
	ElemS32
	ElemE4M3
	ElemE5M2
	ElemE2M3
	ElemE3M2
	ElemE2M1
	ElemU8
	ElemS8
	ElemS1Z4M11
)

var elemNames = [...]string{
	ElemF16:     "f16",
	ElemBF16:    "bf16",
	ElemTF32:    "tf32",
	ElemF32:     "f32",
	ElemS32:     "s32",
	ElemE4M3:    "e4m3",
	ElemE5M2:    "e5m2",
	ElemE2M3:    "e2m3",
	ElemE3M2:    "e3m2",
	ElemE2M1:    "e2m1",
	ElemU8:      "u8",
	ElemS8:      "s8",
	ElemS1Z4M11: "s1z4m11",
}

func (t ElemType) String() string {
	if int(t) < len(elemNames) && elemNames[t] != "" {
		return elemNames[t]
	}
	return fmt.Sprintf("ElemType(%d)", uint8(t))
}

// A ScaleType is the element type of the block scale factors of a
// block-scaled tcgen05.mma.
type ScaleType uint8

const (
	ScaleUE8M0 ScaleType = iota + 1
	ScaleUE4M3
	ScaleUE5M3
)

var scaleNames = [...]string{ScaleUE8M0: "ue8m0", ScaleUE4M3: "ue4m3", ScaleUE5M3: "ue5m3"}

func (s ScaleType) String() string {
	if int(s) < len(scaleNames) && scaleNames[s] != "" {
		return scaleNames[s]
	}
	return fmt.Sprintf("ScaleType(%d)", uint8(s))
}

// InstDesc is the 32-bit instruction descriptor (idesc) operand of
// tcgen05.mma, laid out per PTX ISA 9.4 section 9.7.18.4.2.
//
// The layout depends on Kind. Fields the layout for Kind does not contain
// must be zero; [InstDesc.Encode] rejects them otherwise, so that
// DecodeInstDesc(d.Kind, w) == d for every encodable d.
type InstDesc struct {
	Kind Kind

	Sparse           bool  // tcgen05.mma.sp
	SparsitySelector uint8 // Table 51 kinds, sparse only: 0..3
	SparsityVersion  uint8 // mxf4 and mxf4nvf4, sparse only: 0 (v0) or 1 (v1)
	Saturate         bool  // i8 only

	DType ElemType // Table 51 kinds only: f16 or f32 (f16, f8f6f4), f32 (tf32), s32 (i8, ti16)
	AType ElemType
	BType ElemType

	NegateA, NegateB       bool
	TransposeA, TransposeB bool

	// M and N are the MMA shape. K is the MMA reduction extent of a single
	// instruction, counting the logical (pre-compression) elements when
	// Sparse is set; it must be zero for i8 and ti16, whose layout has no
	// K field.
	M, N, K int

	// MaxShift is the maximum shift while attempting B-matrix reuse in
	// tcgen05.mma.ws: 0, 8, 16, or 32. Table 51 kinds only.
	MaxShift int

	// Block-scaled kinds only.
	ScaleType     ScaleType
	ScaleAID      uint8 // matrix A scale factor data ID: 0..3 (mxf8f6f4), 0 or 2 (mxf4, mxf4nvf4)
	ScaleBID      uint8 // matrix B scale factor data ID: same range as ScaleAID
	ScaleA128Lane bool  // matrix A scale factor layout: 128 lanes instead of 32
}

var f8f6f4Codes = map[ElemType]uint32{ElemE4M3: 0, ElemE5M2: 1, ElemE2M3: 3, ElemE3M2: 4, ElemE2M1: 5}

// abCodes holds the atype and btype codes of Tables 51 to 53.
var abCodes = map[Kind]map[ElemType]uint32{
	KindTF32:     {ElemTF32: 2},
	KindF16:      {ElemF16: 0, ElemBF16: 1},
	KindF8F6F4:   f8f6f4Codes,
	KindI8:       {ElemU8: 0, ElemS8: 1},
	KindTI16:     {ElemS1Z4M11: 3},
	KindMXF8F6F4: f8f6f4Codes,
	KindMXF4:     {ElemE2M1: 1},
	KindMXF4NVF4: {ElemE2M1: 1},
}

// dtypeCodes holds the dtype codes of Table 51.
var dtypeCodes = map[Kind]map[ElemType]uint32{
	KindTF32:   {ElemF32: 1},
	KindF16:    {ElemF16: 0, ElemF32: 1},
	KindF8F6F4: {ElemF16: 0, ElemF32: 1},
	KindI8:     {ElemS32: 2},
	KindTI16:   {ElemS32: 2},
}

// scaleCodes holds the scale matrix type codes of Tables 52 and 53.
var scaleCodes = map[Kind]map[ScaleType]uint32{
	KindMXF8F6F4: {ScaleUE8M0: 1},
	KindMXF4:     {ScaleUE8M0: 1},
	KindMXF4NVF4: {ScaleUE4M3: 0, ScaleUE8M0: 1, ScaleUE5M3: 2},
}

// kSizes maps each K dimension code to its {dense, sparse} K. A zero entry
// is not encodable. Kinds without a K field have no entry.
var kSizes = map[Kind][][2]int{
	KindTF32:     {{8, 16}},
	KindF16:      {{16, 32}},
	KindF8F6F4:   {{32, 64}, {64, 0}},
	KindMXF8F6F4: {{32, 64}, {64, 128}},
	KindMXF4:     {{64, 128}, {96, 192}, {128, 0}},
	KindMXF4NVF4: {{64, 128}, {96, 192}, {128, 0}},
}

var maxShiftCodes = map[int]uint32{0: 0, 8: 1, 16: 2, 32: 3}

func lookup[T comparable](m map[T]uint32, code uint32) T {
	for v, c := range m {
		if c == code {
			return v
		}
	}
	var zero T
	return zero
}

func b2u(b bool) uint32 {
	if b {
		return 1
	}
	return 0
}

// kCode returns the K dimension code for d.
func (d InstDesc) kCode() (uint32, error) {
	sizes, ok := kSizes[d.Kind]
	if !ok {
		if d.K != 0 {
			return 0, fmt.Errorf("K %d set but .kind::%v has no K field", d.K, d.Kind)
		}
		return 0, nil
	}
	for code, s := range sizes {
		k := s[0]
		if d.Sparse {
			k = s[1]
		}
		if k != 0 && k == d.K {
			return uint32(code), nil
		}
	}
	form := "dense"
	if d.Sparse {
		form = "sparse"
	}
	return 0, fmt.Errorf("K %d not encodable for %s .kind::%v", d.K, form, d.Kind)
}

// Encode packs d into the 32-bit instruction descriptor word. It returns an
// error if d has a field the layout for d.Kind cannot represent.
func (d InstDesc) Encode() (uint32, error) {
	w, err := d.encode()
	if err != nil {
		return 0, fmt.Errorf("tcgen05 instdesc: %w", err)
	}
	return w, nil
}

// Validate reports whether d is encodable.
func (d InstDesc) Validate() error {
	_, err := d.Encode()
	return err
}

func (d InstDesc) encode() (uint32, error) {
	k := d.Kind
	ab, ok := abCodes[k]
	if !ok {
		return 0, fmt.Errorf("invalid kind %v", k)
	}
	a, ok := ab[d.AType]
	if !ok {
		return 0, fmt.Errorf("atype %v not valid for .kind::%v", d.AType, k)
	}
	b, ok := ab[d.BType]
	if !ok {
		return 0, fmt.Errorf("btype %v not valid for .kind::%v", d.BType, k)
	}
	if d.N < 8 || d.N%8 != 0 || d.N>>3 > 0x3f {
		return 0, fmt.Errorf("N %d not encodable (want multiple of 8 in 8..504)", d.N)
	}
	kc, err := d.kCode()
	if err != nil {
		return 0, err
	}
	if !d.Sparse && (d.SparsitySelector != 0 || d.SparsityVersion != 0) {
		return 0, fmt.Errorf("sparsity selector or version set on a dense descriptor")
	}

	w := b2u(d.Sparse)<<2 | a<<7 | b<<10 |
		b2u(d.NegateA)<<13 | b2u(d.NegateB)<<14 |
		b2u(d.TransposeA)<<15 | b2u(d.TransposeB)<<16 |
		uint32(d.N>>3)<<17

	if !k.blockScaled() {
		// Table 51.
		dt, ok := dtypeCodes[k][d.DType]
		if !ok {
			return 0, fmt.Errorf("dtype %v not valid for .kind::%v", d.DType, k)
		}
		if d.M < 16 || d.M%16 != 0 || d.M>>4 > 0x1f {
			return 0, fmt.Errorf("M %d not encodable for .kind::%v (want multiple of 16 in 16..496)", d.M, k)
		}
		if d.SparsitySelector > 3 {
			return 0, fmt.Errorf("sparsity selector %d out of range 0..3", d.SparsitySelector)
		}
		if d.SparsityVersion != 0 {
			return 0, fmt.Errorf("sparsity version set on .kind::%v", k)
		}
		if d.Saturate && k != KindI8 {
			return 0, fmt.Errorf("saturate set on .kind::%v", k)
		}
		if k == KindI8 && (d.NegateA || d.NegateB) {
			return 0, fmt.Errorf("negate set on .kind::i8")
		}
		if k == KindTI16 && (d.TransposeA || d.TransposeB) {
			return 0, fmt.Errorf("transpose set on .kind::ti16")
		}
		shift, ok := maxShiftCodes[d.MaxShift]
		if !ok {
			return 0, fmt.Errorf("max shift %d not encodable (want 0, 8, 16, or 32)", d.MaxShift)
		}
		if d.ScaleType != 0 || d.ScaleAID != 0 || d.ScaleBID != 0 || d.ScaleA128Lane {
			return 0, fmt.Errorf("scale factor fields set on .kind::%v", k)
		}
		w |= uint32(d.SparsitySelector) | b2u(d.Saturate)<<3 | dt<<4 |
			uint32(d.M>>4)<<24 | kc<<29 | shift<<30
		return w, nil
	}

	// Tables 52 and 53.
	if d.DType != 0 || d.Saturate || d.SparsitySelector != 0 || d.MaxShift != 0 {
		return 0, fmt.Errorf("dtype, saturate, sparsity selector, or max shift set on .kind::%v", k)
	}
	if d.M < 128 || d.M%128 != 0 || d.M>>7 > 3 {
		return 0, fmt.Errorf("M %d not encodable for .kind::%v (want 128, 256, or 384)", d.M, k)
	}
	st, ok := scaleCodes[k][d.ScaleType]
	if !ok {
		return 0, fmt.Errorf("scale type %v not valid for .kind::%v", d.ScaleType, k)
	}
	w |= uint32(d.ScaleBID)<<4 | b2u(d.ScaleA128Lane)<<26 | uint32(d.M>>7)<<27 | uint32(d.ScaleAID)<<29
	if k == KindMXF8F6F4 {
		if d.ScaleAID > 3 || d.ScaleBID > 3 {
			return 0, fmt.Errorf("scale factor data ID out of range 0..3")
		}
		if d.SparsityVersion != 0 {
			return 0, fmt.Errorf("sparsity version set on .kind::%v", k)
		}
		return w | st<<23 | kc<<31, nil
	}
	if d.ScaleAID != 0 && d.ScaleAID != 2 || d.ScaleBID != 0 && d.ScaleBID != 2 {
		return 0, fmt.Errorf("scale factor data ID must be 0 or 2 for .kind::%v", k)
	}
	if d.TransposeA {
		return 0, fmt.Errorf("transpose A set on .kind::%v", k)
	}
	if d.SparsityVersion > 1 {
		return 0, fmt.Errorf("sparsity version %d out of range 0..1", d.SparsityVersion)
	}
	return w | (kc>>1)<<3 | uint32(d.SparsityVersion)<<12 | st<<23 | (kc&1)<<31, nil
}

// DecodeInstDesc unpacks a 32-bit tcgen05.mma instruction descriptor word
// laid out for kind. It returns an error if w sets a reserved bit or holds a
// code the layout does not define.
func DecodeInstDesc(kind Kind, w uint32) (InstDesc, error) {
	ab, ok := abCodes[kind]
	if !ok {
		return InstDesc{}, fmt.Errorf("tcgen05 instdesc: invalid kind %v", kind)
	}
	bit := func(n uint) bool { return w>>n&1 != 0 }
	d := InstDesc{
		Kind:       kind,
		Sparse:     bit(2),
		AType:      lookup(ab, w>>7&7),
		NegateA:    bit(13),
		NegateB:    bit(14),
		TransposeA: bit(15),
		TransposeB: bit(16),
		N:          int(w>>17&0x3f) << 3,
	}
	var kc uint32
	if !kind.blockScaled() {
		d.SparsitySelector = uint8(w & 3)
		d.Saturate = bit(3)
		d.DType = lookup(dtypeCodes[kind], w>>4&3)
		d.BType = lookup(ab, w>>10&7)
		d.M = int(w>>24&0x1f) << 4
		kc = w >> 29 & 1
		d.MaxShift = lookup(maxShiftCodes, w>>30&3)
	} else {
		d.ScaleBID = uint8(w >> 4 & 3)
		d.ScaleA128Lane = bit(26)
		d.M = int(w>>27&3) << 7
		d.ScaleAID = uint8(w >> 29 & 3)
		if kind == KindMXF8F6F4 {
			d.BType = lookup(ab, w>>10&7)
			d.ScaleType = lookup(scaleCodes[kind], w>>23&1)
			kc = w >> 31
		} else {
			d.BType = lookup(ab, w>>10&3)
			d.SparsityVersion = uint8(w >> 12 & 1)
			d.ScaleType = lookup(scaleCodes[kind], w>>23&3)
			kc = (w>>3&1)<<1 | w>>31
		}
	}
	if sizes := kSizes[kind]; int(kc) < len(sizes) {
		d.K = sizes[kc][0]
		if d.Sparse {
			d.K = sizes[kc][1]
		}
	}
	got, err := d.Encode()
	if err != nil {
		return InstDesc{}, fmt.Errorf("tcgen05 instdesc: decode 0x%08x as .kind::%v: %w", w, kind, err)
	}
	if got != w {
		return InstDesc{}, fmt.Errorf("tcgen05 instdesc: decode 0x%08x as .kind::%v: reserved bits 0x%08x set", w, kind, w^got)
	}
	return d, nil
}

// ValidateForTarget reports whether d is encodable and valid on target,
// applying the target notes of PTX ISA 9.4 section 9.7.18.4.2.1: the K
// dimension select bits (29 for f8f6f4, 31 for mxf8f6f4, 3 for mxf4 and
// mxf4nvf4), the matrix A scale factor layout bit, and the UE5M3 scale type
// require sm_107a or sm_107f, and the mxf4/mxf4nvf4 sparsity version must
// be v1 on sm_107a/sm_107f and v0 elsewhere.
func (d InstDesc) ValidateForTarget(target string) error {
	w, err := d.Encode()
	if err != nil {
		return err
	}
	var sm107 bool
	switch target {
	case "sm_100a", "sm_100f", "sm_103a", "sm_103f", "sm_110a", "sm_110f":
	case "sm_107a", "sm_107f":
		sm107 = true
	default:
		return fmt.Errorf("tcgen05 instdesc: unsupported target %q", target)
	}
	var sm107Bits uint32
	switch d.Kind {
	case KindF8F6F4:
		sm107Bits = 1 << 29
	case KindMXF8F6F4:
		sm107Bits = 1<<26 | 1<<31
	case KindMXF4, KindMXF4NVF4:
		sm107Bits = 1<<3 | 1<<26
	}
	if !sm107 && w&sm107Bits != 0 {
		return fmt.Errorf("tcgen05 instdesc: .kind::%v K %d or 128-lane scale layout requires sm_107a or sm_107f, not %s", d.Kind, d.K, target)
	}
	if !sm107 && d.ScaleType == ScaleUE5M3 {
		return fmt.Errorf("tcgen05 instdesc: scale type ue5m3 requires sm_107a or sm_107f, not %s", target)
	}
	if d.Sparse && (d.Kind == KindMXF4 || d.Kind == KindMXF4NVF4) {
		want := uint8(0)
		if sm107 {
			want = 1
		}
		if d.SparsityVersion != want {
			return fmt.Errorf("tcgen05 instdesc: target %s requires sparsity version v%d, got v%d", target, want, d.SparsityVersion)
		}
	}
	return nil
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
