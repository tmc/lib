package qualify

import (
	"errors"
	"fmt"
	"iter"
	"math"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// An ArchScheduleSpec describes how a block-scaled FP4 GEMM
// D[M×N] = Σ_k (A[m,k]·sA[m,k/G])·(B[k,n]·sB[n,k/G]) is lowered for one
// target:
//
//   - sm_90a: wgmma.mma_async m64nNk16 on bf16, with A decoded from packed
//     E2M1 by the [bitopt] NVFP4 program into a swizzled shared-memory
//     tile and the block scales applied in software; TileM a multiple of
//     64, TileN a multiple of 8 up to 256.
//   - sm_100a, sm_103a, sm_107a: tcgen05.mma kind::mxf4nvf4 with
//     block_scale.scale_vec::4X (G = 16) or ::2X (G = 32), accumulating in
//     Tensor Memory; CTAGroupSize 2 pairs two CTAs over 256 rows; TileM =
//     128·CTAGroupSize, TileN a multiple of 16 up to 256.
//   - sm_120a, sm_121a: warp-level mma.sync.m16n8k64 kind::mxf4nvf4 with
//     block scales, no Tensor Memory; 4×2 warps per CTA tile.
//
// TileK must be a multiple of 64 on every target.
type ArchScheduleSpec struct {
	Arch                string
	ProblemM            int
	ProblemN            int
	ProblemK            int
	TileM, TileN, TileK int
	CTAGroupSize        int // tcgen05 only: 1 or 2
	ScaleBlockSize      int // 16 (NVFP4) or 32 (MXFP4)
	SwizzleBytes        int // 128, 64 or 32
}

// An EquivalenceCertificate records the result of comparing two lowered
// schedules MAC by MAC.
type EquivalenceCertificate struct {
	RefArch          string
	CandArch         string
	CoordinatesTotal int // M·N·K multiply-accumulates checked in each schedule
	BijectiveMatch   bool
	// MaxAccumULPBound bounds |D_ref − D_cand| relative to Σ_k|A_mk·B_kn|
	// for IEEE FP32 accumulation in each schedule's chunked order:
	// γ_{c+K/c} per schedule, c its MMA K, summed over the two.
	MaxAccumULPBound float64
	// ProbabilisticULPBound is the corresponding √(c+K/c)·u estimate.
	ProbabilisticULPBound float64
	MismatchDetail        string // the first mismatch, empty when BijectiveMatch
}

// A mac is one multiply-accumulate as a lowered kernel performs it: the
// product A[am,ak]·sA[sam,sak] · B[bk,bn]·sB[sbn,sbk] is added into the
// value the kernel stores at D[m,n].
type mac struct {
	m, n     int
	am, ak   int
	bk, bn   int
	sam, sak int
	sbn, sbk int
}

// A lowering enumerates the multiply-accumulates of one schedule from the
// hardware layouts of its instructions and the kernel's own index maps.
type lowering interface {
	arch() string
	mmaK() int    // K per MMA instruction: the accumulation chunk
	check() error // instruction-level layout checks: swizzle, decode, TMEM
	macs() iter.Seq[mac]
}

func (s ArchScheduleSpec) checkTiles(mMul, nMul, nMax int) error {
	switch {
	case s.ProblemM <= 0 || s.ProblemN <= 0 || s.ProblemK <= 0:
		return fmt.Errorf("problem %dx%dx%d", s.ProblemM, s.ProblemN, s.ProblemK)
	case s.TileM <= 0 || s.TileM%mMul != 0 || s.ProblemM%s.TileM != 0:
		return fmt.Errorf("TileM %d must be a multiple of %d dividing M=%d", s.TileM, mMul, s.ProblemM)
	case s.TileN <= 0 || s.TileN%nMul != 0 || s.TileN > nMax || s.ProblemN%s.TileN != 0:
		return fmt.Errorf("TileN %d must be a multiple of %d up to %d dividing N=%d", s.TileN, nMul, nMax, s.ProblemN)
	case s.TileK <= 0 || s.TileK%64 != 0 || s.ProblemK%s.TileK != 0:
		return fmt.Errorf("TileK %d must be a multiple of 64 dividing K=%d", s.TileK, s.ProblemK)
	case s.ScaleBlockSize != 16 && s.ScaleBlockSize != 32:
		return fmt.Errorf("ScaleBlockSize %d", s.ScaleBlockSize)
	case s.SwizzleBytes != 32 && s.SwizzleBytes != 64 && s.SwizzleBytes != 128:
		return fmt.Errorf("SwizzleBytes %d", s.SwizzleBytes)
	}
	return nil
}

// lowerSchedule returns the lowering of s as this package models it.
func lowerSchedule(s ArchScheduleSpec) (lowering, error) {
	switch s.Arch {
	case "sm_90a":
		if err := s.checkTiles(64, 8, 256); err != nil {
			return nil, err
		}
		dec := bitopt.NVFP4Program1Word()
		return &wgmmaLowering{
			s:       s,
			cStore:  cudahostemu.MMAM16N8C,
			decode:  &dec,
			pack:    dec.PackWords,
			readerS: swizzleMode(s.SwizzleBytes),
		}, nil
	case "sm_100a", "sm_103a", "sm_107a":
		if s.CTAGroupSize != 1 && s.CTAGroupSize != 2 {
			return nil, fmt.Errorf("CTAGroupSize %d", s.CTAGroupSize)
		}
		if s.TileM != 128*s.CTAGroupSize {
			return nil, fmt.Errorf("TileM %d, want %d for cta_group::%d", s.TileM, 128*s.CTAGroupSize, s.CTAGroupSize)
		}
		if err := s.checkTiles(128, 16, 256); err != nil {
			return nil, err
		}
		return &tcgen05Lowering{s: s, peerRowOffset: 128, readerS: tcgenSwizzle(s.SwizzleBytes)}, nil
	case "sm_120a", "sm_121a":
		if err := s.checkTiles(16*4, 8*2, 256); err != nil {
			return nil, err
		}
		return &mmaSyncLowering{s: s, aLoad: cudahostemu.MMAM16N8K64NVFP4A}, nil
	}
	return nil, fmt.Errorf("unsupported arch %q", s.Arch)
}

func swizzleMode(bytes int) cudahostemu.SwizzleMode {
	switch bytes {
	case 128:
		return cudahostemu.Swizzle128B
	case 64:
		return cudahostemu.Swizzle64B
	case 32:
		return cudahostemu.Swizzle32B
	}
	return cudahostemu.SwizzleNone
}

func tcgenSwizzle(bytes int) tcgen05.Swizzle {
	switch bytes {
	case 128:
		return tcgen05.Swizzle128B
	case 64:
		return tcgen05.Swizzle64B
	case 32:
		return tcgen05.Swizzle32B
	}
	return tcgen05.SwizzleNone
}

// checkSwizzle checks that a rows×rowBytes shared-memory tile written by TMA
// with writer swizzle and read through an MMA descriptor with reader
// swizzle maps every 16-byte chunk to a distinct address inside the tile and
// that the reader finds each chunk where the writer put it. Rows are laid
// out in atoms of swizzle-width bytes, as a TMA box with that swizzle does.
func checkSwizzle(rows, rowBytes, width int, writer, reader cudahostemu.SwizzleMode) error {
	if rowBytes%width != 0 {
		return fmt.Errorf("row of %d bytes is not a whole number of %d-byte swizzle atoms", rowBytes, width)
	}
	size := rows * rowBytes
	seen := make([]bool, size/16)
	for r := range rows {
		for c := 0; c < rowBytes; c += 16 {
			atom, within := c/width, r*width+c%width
			base := atom * rows * width
			w := base + cudahostemu.SwizzleOffset(within, writer)
			rd := base + cudahostemu.SwizzleOffset(within, reader)
			if w != rd {
				return fmt.Errorf("row %d byte %d: TMA writes offset %d, MMA descriptor reads %d", r, c, w, rd)
			}
			if w < 0 || w >= size || w%16 != 0 || seen[w/16] {
				return fmt.Errorf("row %d byte %d: offset %d is outside the tile or written twice", r, c, w)
			}
			seen[w/16] = true
		}
	}
	return nil
}

// wgmmaLowering is the sm_90a schedule.
type wgmmaLowering struct {
	s       ArchScheduleSpec
	cStore  func(lane, reg int) (row, col int) // the epilogue's register-to-tile map
	decode  *bitopt.Program                    // device E2M1-to-bf16 program
	pack    func(codes []uint8) []uint32       // offline weight packing
	readerS cudahostemu.SwizzleMode            // swizzle in the wgmma descriptor
	perm    [8]int                             // decoded slot -> code index, set by check
}

func (l *wgmmaLowering) arch() string { return l.s.Arch }
func (l *wgmmaLowering) mmaK() int    { return 16 }

func (l *wgmmaLowering) check() error {
	if err := l.decode.Check(); err != nil {
		return fmt.Errorf("NVFP4 decode program: %w", err)
	}
	codes := []uint8{0, 1, 2, 3, 4, 5, 6, 7}
	got := l.decode.UnpackWords(l.pack(codes))
	used := 0
	for p, c := range got[:8] {
		if c > 7 || used&(1<<c) != 0 {
			return fmt.Errorf("NVFP4 decode: slot %d reads code %d, not a permutation of the packed codes", p, c)
		}
		used |= 1 << c
		l.perm[p] = int(c)
	}
	// The wgmma descriptor's swizzle field (bits 62-63) as the hardware
	// decodes it.
	desc := cudahostemu.WGMMADesc(0, 16, 8*l.s.SwizzleBytes, l.readerS)
	reader := cudahostemu.SwizzleMode(desc >> 62)
	return checkSwizzle(l.s.TileM, 2*l.s.TileK, l.s.SwizzleBytes, swizzleMode(l.s.SwizzleBytes), reader)
}

func (l *wgmmaLowering) macs() iter.Seq[mac] {
	s := l.s
	return func(yield func(mac) bool) {
		for tm := 0; tm < s.ProblemM; tm += s.TileM {
			for tn := 0; tn < s.ProblemN; tn += s.TileN {
				for wg := 0; wg < s.TileM/64; wg++ {
					for w := range 4 {
						for lane := range 32 {
							for i := 0; i < s.TileN/2; i++ {
								hr, hc := cudahostemu.MMAM16N8C(lane, i%4)
								sr, sc := l.cStore(lane, i%4)
								hm := tm + 64*wg + 16*w + hr
								hn := tn + 8*(i/4) + hc
								m := tm + 64*wg + 16*w + sr
								n := tn + 8*(i/4) + sc
								for k0 := 0; k0 < s.ProblemK; k0 += 16 {
									for slot := range 16 {
										ak := k0 + slot&^7 + l.perm[slot&7]
										bk := k0 + slot
										x := mac{m, n, hm, ak, bk, hn, hm, ak / s.ScaleBlockSize, hn, bk / s.ScaleBlockSize}
										if !yield(x) {
											return
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// tcgen05Lowering is the sm_100a, sm_103a and sm_107a schedule.
type tcgen05Lowering struct {
	s             ArchScheduleSpec
	peerRowOffset int // rows between the paired CTAs' halves in the epilogue
	readerS       tcgen05.Swizzle
}

func (l *tcgen05Lowering) arch() string { return l.s.Arch }
func (l *tcgen05Lowering) mmaK() int    { return 64 }

func (l *tcgen05Lowering) check() error {
	// FP32 accumulators use one TMEM column per output column.
	if l.s.TileN > 512 {
		return fmt.Errorf("TileN %d exceeds 512 TMEM columns", l.s.TileN)
	}
	desc := tcgen05.SharedDesc{LeadingByteOffset: 16, StrideByteOffset: 8 * uint32(l.s.SwizzleBytes), Swizzle: l.readerS}
	reader := swizzleMode(int(tcgen05.DecodeSharedDesc(desc.Encode()).Swizzle))
	// E2M1 rows hold two codes per byte.
	return checkSwizzle(l.s.TileM, l.s.TileK/2, l.s.SwizzleBytes, swizzleMode(l.s.SwizzleBytes), reader)
}

func (l *tcgen05Lowering) macs() iter.Seq[mac] {
	s := l.s
	vecK := 64 / (64 / s.ScaleBlockSize) // K per scale in scale_vec::4X or ::2X
	return func(yield func(mac) bool) {
		for tm := 0; tm < s.ProblemM; tm += s.TileM {
			for tn := 0; tn < s.ProblemN; tn += s.TileN {
				for peer := range s.CTAGroupSize {
					for w := range 4 {
						for lane := range 32 {
							for col := 0; col < s.TileN; col++ {
								a := tcgen05.DecodeTMEMAddr(tcgen05.TMEMAddr{Row: uint16(32*w + lane), Col: uint16(col)}.Encode())
								hm := tm + 128*peer + int(a.Row)
								m := tm + l.peerRowOffset*peer + int(a.Row)
								n := tn + int(a.Col)
								for k0 := 0; k0 < s.ProblemK; k0 += 64 {
									for k := range 64 {
										kb := (k0 + k/vecK*vecK) / s.ScaleBlockSize
										x := mac{m, n, hm, k0 + k, k0 + k, n, hm, kb, n, kb}
										if !yield(x) {
											return
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// mmaSyncLowering is the sm_120a and sm_121a schedule.
type mmaSyncLowering struct {
	s     ArchScheduleSpec
	aLoad func(lane, reg int) (row, colOct int) // the kernel's A fragment loads
	// inv[row][k] is the (lane, reg, nibble) where the hardware takes
	// A[row, k] from, set by check.
	inv [16][64][3]int
}

func (l *mmaSyncLowering) arch() string { return l.s.Arch }
func (l *mmaSyncLowering) mmaK() int    { return 64 }

func (l *mmaSyncLowering) check() error {
	var seen [16][64]bool
	for lane := range 32 {
		for reg := range 4 {
			row, oct := cudahostemu.MMAM16N8K64NVFP4A(lane, reg)
			for nib := range 8 {
				if seen[row][oct+nib] {
					return fmt.Errorf("mma.sync A fragment: A[%d,%d] held twice", row, oct+nib)
				}
				seen[row][oct+nib] = true
				l.inv[row][oct+nib] = [3]int{lane, reg, nib}
			}
		}
	}
	return checkSwizzle(l.s.TileM, l.s.TileK/2, l.s.SwizzleBytes, swizzleMode(l.s.SwizzleBytes), swizzleMode(l.s.SwizzleBytes))
}

func (l *mmaSyncLowering) macs() iter.Seq[mac] {
	s := l.s
	const wm, wn = 4, 2
	return func(yield func(mac) bool) {
		for tm := 0; tm < s.ProblemM; tm += s.TileM {
			for tn := 0; tn < s.ProblemN; tn += s.TileN {
				for wy := range wm {
					for wx := range wn {
						for i := 0; i < s.TileM/wm/16; i++ {
							for j := 0; j < s.TileN/wn/8; j++ {
								m0 := tm + wy*s.TileM/wm + 16*i
								n0 := tn + wx*s.TileN/wn + 8*j
								for lane := range 32 {
									for creg := range 4 {
										r, c := cudahostemu.MMAM16N8C(lane, creg)
										m, n := m0+r, n0+c
										for k0 := 0; k0 < s.ProblemK; k0 += 64 {
											for k := range 64 {
												p := l.inv[r][k]
												ar, aoct := l.aLoad(p[0], p[1])
												ak := k0 + aoct + p[2]
												kb := (k0 + k/s.ScaleBlockSize*s.ScaleBlockSize) / s.ScaleBlockSize
												x := mac{m, n, m0 + ar, ak, k0 + k, n, m0 + r, kb, n, kb}
												if !yield(x) {
													return
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// verifyLowering checks that l performs exactly the multiply-accumulates of
// the GEMM ref describes: every D[m,n] accumulates A[m,k]·B[k,n] for each
// k exactly once with the scales sA[m,k/G] and sB[n,k/G], G the reference
// block size. It returns "" or the first mismatch.
func verifyLowering(l lowering, ref ArchScheduleSpec) string {
	if err := l.check(); err != nil {
		return fmt.Sprintf("%s: %v", l.arch(), err)
	}
	M, N, K, G := ref.ProblemM, ref.ProblemN, ref.ProblemK, ref.ScaleBlockSize
	count := make([]uint8, M*N*K)
	for x := range l.macs() {
		if x.m < 0 || x.m >= M || x.n < 0 || x.n >= N || x.ak < 0 || x.ak >= K {
			return fmt.Sprintf("%s: stores (m,n,k)=(%d,%d,%d) outside the %dx%dx%d problem", l.arch(), x.m, x.n, x.ak, M, N, K)
		}
		if x.am != x.m || x.bn != x.n || x.ak != x.bk {
			return fmt.Sprintf("%s: (m,n,k)=(%d,%d,%d) accumulates A[%d,%d]·B[%d,%d]", l.arch(), x.m, x.n, x.bk, x.am, x.ak, x.bk, x.bn)
		}
		if x.sam != x.m || x.sbn != x.n || x.sak != x.ak/G || x.sbk != x.bk/G {
			return fmt.Sprintf("%s: (m,n,k)=(%d,%d,%d) applies scales sA[%d,%d]·sB[%d,%d], want sA[%d,%d]·sB[%d,%d]",
				l.arch(), x.m, x.n, x.ak, x.sam, x.sak, x.sbn, x.sbk, x.m, x.ak/G, x.n, x.bk/G)
		}
		i := (x.m*N+x.n)*K + x.ak
		if count[i]++; count[i] > 1 {
			return fmt.Sprintf("%s: (m,n,k)=(%d,%d,%d) accumulated twice", l.arch(), x.m, x.n, x.ak)
		}
	}
	for i, c := range count {
		if c == 0 {
			return fmt.Sprintf("%s: (m,n,k)=(%d,%d,%d) never accumulated", l.arch(), i/K/N, i/K%N, i%K)
		}
	}
	return ""
}

func accumBounds(l lowering, k int) (worst, prob float64) {
	c := l.mmaK()
	n := c + (k+c-1)/c
	return gammaF32(n), math.Sqrt(float64(n)) / (1 << 24)
}

func certify(ref ArchScheduleSpec, lr, lc lowering, refBad, candBad string) EquivalenceCertificate {
	wr, pr := accumBounds(lr, ref.ProblemK)
	wc, pc := accumBounds(lc, ref.ProblemK)
	cert := EquivalenceCertificate{
		RefArch:               lr.arch(),
		CandArch:              lc.arch(),
		CoordinatesTotal:      ref.ProblemM * ref.ProblemN * ref.ProblemK,
		MaxAccumULPBound:      wr + wc,
		ProbabilisticULPBound: pr + pc,
	}
	switch {
	case refBad != "":
		cert.MismatchDetail = "reference " + refBad
	case candBad != "":
		cert.MismatchDetail = "candidate " + candBad
	default:
		cert.BijectiveMatch = true
	}
	return cert
}

// ProveCrossISAEquivalence enumerates every multiply-accumulate of the two
// lowered schedules, from the instruction layouts the hardware uses
// (accumulator fragments, Tensor Memory addresses, A fragments, block-scale
// lanes) and the kernel's own index maps (epilogue stores, fragment loads,
// FP4 packing and decode, TMA and descriptor swizzles), and checks that
// each computes, for every (m, n), the sum over k ∈ [0, K) of
// A[m,k]·sA[m,k/G]·B[k,n]·sB[n,k/G] with each term exactly once, G being
// ref's ScaleBlockSize. Two schedules that pass compute the same terms, so
// they differ only by FP32 accumulation order, which the certificate
// bounds.
//
// The proof is by exhaustive enumeration over the given problem size, not
// symbolic; it covers index maps and data placement, not the arithmetic of
// the hardware's internal adders, which are not IEEE FP32 adders on every
// target. It returns an error for a spec it cannot lower.
func ProveCrossISAEquivalence(ref, cand ArchScheduleSpec) (EquivalenceCertificate, error) {
	lr, err := lowerSchedule(ref)
	if err != nil {
		return EquivalenceCertificate{}, fmt.Errorf("prove equivalence: %s: %w", ref.Arch, err)
	}
	lc, err := lowerSchedule(cand)
	if err != nil {
		return EquivalenceCertificate{}, fmt.Errorf("prove equivalence: %s: %w", cand.Arch, err)
	}
	return proveLowerings(ref, cand, lr, lc), nil
}

func proveLowerings(ref, cand ArchScheduleSpec, lr, lc lowering) EquivalenceCertificate {
	if ref.ProblemM != cand.ProblemM || ref.ProblemN != cand.ProblemN || ref.ProblemK != cand.ProblemK {
		c := certify(ref, lr, lc, "", "")
		c.BijectiveMatch = false
		c.MismatchDetail = fmt.Sprintf("problem %dx%dx%d vs %dx%dx%d", ref.ProblemM, ref.ProblemN, ref.ProblemK, cand.ProblemM, cand.ProblemN, cand.ProblemK)
		return c
	}
	return certify(ref, lr, lc, verifyLowering(lr, ref), verifyLowering(lc, ref))
}

// TargetISAs lists the targets [VerifyAllTargetISAs] covers, sm_90a first.
var TargetISAs = []string{"sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a", "sm_121a"}

// DefaultSchedule returns the schedule this package lowers an m×n×k
// NVFP4 GEMM to on arch: the largest tiles of the target's shape that
// divide the problem, cta_group::2 on tcgen05 targets when m allows,
// TileK 64, and the widest swizzle that fits a tile row in shared memory
// (128 bytes for sm_90a's decoded bf16, 32 bytes for packed E2M1).
func DefaultSchedule(arch string, m, n, k int) (ArchScheduleSpec, error) {
	s := ArchScheduleSpec{Arch: arch, ProblemM: m, ProblemN: n, ProblemK: k, TileK: 64, ScaleBlockSize: 16, SwizzleBytes: 128}
	largest := func(dim int, opts ...int) int {
		for _, o := range opts {
			if dim%o == 0 {
				return o
			}
		}
		return 0
	}
	switch arch {
	case "sm_90a":
		s.TileM = largest(m, 128, 64)
		s.TileN = largest(n, 256, 128, 64, 32, 16, 8)
	case "sm_100a", "sm_103a", "sm_107a":
		s.SwizzleBytes = min(128, s.TileK/2)
		s.CTAGroupSize = 1
		if m%256 == 0 {
			s.CTAGroupSize = 2
		}
		s.TileM = 128 * s.CTAGroupSize
		s.TileN = largest(n, 256, 128, 64, 32, 16)
	case "sm_120a", "sm_121a":
		s.SwizzleBytes = min(128, s.TileK/2)
		s.TileM = largest(m, 128, 64)
		s.TileN = largest(n, 128, 64, 32, 16)
	default:
		return s, fmt.Errorf("unsupported arch %q", arch)
	}
	if _, err := lowerSchedule(s); err != nil {
		return s, fmt.Errorf("default schedule %s for %dx%dx%d: %w", arch, m, n, k, err)
	}
	return s, nil
}

// VerifyAllTargetISAs lowers an m×n×k NVFP4 GEMM with [DefaultSchedule] for
// every target in [TargetISAs] and returns the certificates of each target
// against sm_90a. It returns an error if any schedule fails to lower or any
// certificate is not a match.
func VerifyAllTargetISAs(m, n, k int) ([]EquivalenceCertificate, error) {
	specs := make([]ArchScheduleSpec, len(TargetISAs))
	lows := make([]lowering, len(TargetISAs))
	bad := make([]string, len(TargetISAs))
	for i, arch := range TargetISAs {
		s, err := DefaultSchedule(arch, m, n, k)
		if err != nil {
			return nil, fmt.Errorf("verify all target isas: %w", err)
		}
		l, err := lowerSchedule(s)
		if err != nil {
			return nil, fmt.Errorf("verify all target isas: %w", err)
		}
		specs[i], lows[i] = s, l
	}
	var wg sync.WaitGroup
	for i, l := range lows {
		wg.Go(func() { bad[i] = verifyLowering(l, specs[0]) })
	}
	wg.Wait()
	var certs []EquivalenceCertificate
	var errs []error
	for i := 1; i < len(lows); i++ {
		c := certify(specs[0], lows[0], lows[i], bad[0], bad[i])
		certs = append(certs, c)
		if !c.BijectiveMatch {
			errs = append(errs, fmt.Errorf("%s vs %s: %s", c.RefArch, c.CandArch, c.MismatchDetail))
		}
	}
	return certs, errors.Join(errs...)
}
