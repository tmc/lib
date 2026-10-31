package qualify

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
)

// A FaultKind classifies an anomaly caught by the ABFT check or the barrier
// watchdog.
type FaultKind uint8

const (
	FaultNone            FaultKind = iota
	FaultTensorCoreSDC             // a row or column checksum disagrees beyond tolerance
	FaultNaNInfGenerated           // a non-finite output from finite inputs
	FaultMBarrierTimeout           // an mbarrier wait exceeded the watchdog timeout
)

func (k FaultKind) String() string {
	switch k {
	case FaultNone:
		return "none"
	case FaultTensorCoreSDC:
		return "tensor-core-sdc"
	case FaultNaNInfGenerated:
		return "nan-inf-generated"
	case FaultMBarrierTimeout:
		return "mbarrier-timeout"
	}
	return fmt.Sprintf("FaultKind(%d)", uint8(k))
}

// FaultRecordSize is the size in bytes of an encoded [FaultRecord].
const FaultRecordSize = 32

// A FaultRecord is one entry of the device-to-host fault ring. Its 32-byte
// little-endian encoding, which the emitted PTX and CUDA code write, is:
//
//	offset  size  field
//	0       1     Kind
//	1       1     flags: bit 0 = Corrected
//	2       2     Rank
//	4       2     PeerRank
//	6       2     LayerID
//	8       4     TileMIdx
//	12      4     TileNIdx
//	16      2     CorruptRow (int16, -1 if unknown)
//	18      2     CorruptCol (int16, -1 if unknown)
//	20      4     ObservedDiff (float32 bits)
//	24      8     ElapsedNs
type FaultRecord struct {
	Kind         FaultKind
	Corrected    bool // a transient SDC that passed the check on replay
	Rank         uint16
	PeerRank     uint16
	LayerID      uint16
	TileMIdx     uint32
	TileNIdx     uint32
	CorruptRow   int32 // row of the tile whose checksum diverged, or -1
	CorruptCol   int32 // column of the tile whose checksum diverged, or -1
	ObservedDiff float32
	ElapsedNs    uint64
}

// MarshalBinary encodes r in the 32-byte ring layout.
func (r FaultRecord) MarshalBinary() ([]byte, error) {
	if r.CorruptRow < math.MinInt16 || r.CorruptRow > math.MaxInt16 || r.CorruptCol < math.MinInt16 || r.CorruptCol > math.MaxInt16 {
		return nil, fmt.Errorf("fault record: row %d or column %d does not fit in int16", r.CorruptRow, r.CorruptCol)
	}
	b := make([]byte, FaultRecordSize)
	b[0] = byte(r.Kind)
	if r.Corrected {
		b[1] = 1
	}
	le := binary.LittleEndian
	le.PutUint16(b[2:], r.Rank)
	le.PutUint16(b[4:], r.PeerRank)
	le.PutUint16(b[6:], r.LayerID)
	le.PutUint32(b[8:], r.TileMIdx)
	le.PutUint32(b[12:], r.TileNIdx)
	le.PutUint16(b[16:], uint16(int16(r.CorruptRow)))
	le.PutUint16(b[18:], uint16(int16(r.CorruptCol)))
	le.PutUint32(b[20:], math.Float32bits(r.ObservedDiff))
	le.PutUint64(b[24:], r.ElapsedNs)
	return b, nil
}

// UnmarshalBinary decodes a 32-byte ring entry.
func (r *FaultRecord) UnmarshalBinary(b []byte) error {
	if len(b) != FaultRecordSize {
		return fmt.Errorf("fault record: %d bytes, want %d", len(b), FaultRecordSize)
	}
	le := binary.LittleEndian
	*r = FaultRecord{
		Kind:         FaultKind(b[0]),
		Corrected:    b[1]&1 != 0,
		Rank:         le.Uint16(b[2:]),
		PeerRank:     le.Uint16(b[4:]),
		LayerID:      le.Uint16(b[6:]),
		TileMIdx:     le.Uint32(b[8:]),
		TileNIdx:     le.Uint32(b[12:]),
		CorruptRow:   int32(int16(le.Uint16(b[16:]))),
		CorruptCol:   int32(int16(le.Uint16(b[18:]))),
		ObservedDiff: math.Float32frombits(le.Uint32(b[20:])),
		ElapsedNs:    le.Uint64(b[24:]),
	}
	return nil
}

// ABFTConfig configures tile checksum verification and the barrier
// watchdog.
type ABFTConfig struct {
	TileM, TileN, TileK int
	// RelTolerance is the relative rounding error of the stored output on
	// top of FP32 accumulation: 0 for f32 output, 2^-8 for bf16, 2^-4 for
	// e4m3. The checks allow RelTolerance·Σ_j|D_ij| per row.
	RelTolerance      float32
	WatchdogTimeoutNs uint64 // mbarrier wait bound; 0 disables the watchdog
	// AutoRecomputeOnSDC makes [RunTileWithABFT] recompute a tile once
	// when its check fails, and report a corrected fault if the replay
	// passes.
	AutoRecomputeOnSDC bool
}

func (c ABFTConfig) check() error {
	if c.TileM <= 0 || c.TileN <= 0 || c.TileK <= 0 {
		return fmt.Errorf("tile %dx%dx%d", c.TileM, c.TileN, c.TileK)
	}
	if c.TileM > math.MaxInt16 || c.TileN > math.MaxInt16 {
		return fmt.Errorf("tile %dx%d exceeds the int16 fault coordinates", c.TileM, c.TileN)
	}
	if !(c.RelTolerance >= 0) {
		return fmt.Errorf("RelTolerance %v", c.RelTolerance)
	}
	return nil
}

// gammaF32 is Higham's γ_n = n·u/(1-n·u) for FP32 (u = 2^-24): the
// relative bound on the error of any n-term FP32 summation order.
func gammaF32(n int) float64 {
	const u = 1.0 / (1 << 24)
	nu := float64(n) * u
	if nu >= 1 {
		return math.Inf(1)
	}
	return nu / (1 - nu)
}

// abftBounds returns, for D = A·B, the observed row and column sums of d,
// the checksum references A·(B·1) and (1ᵀ·A)·B, and the per-row and
// per-column tolerances
//
//	τ_i = γ_K·Σ_k|A_ik|·Σ_j|B_kj| + RelTolerance·Σ_j|D_ij| + ε_i
//
// (columns alike), where γ_K bounds every FP32 accumulation order of a
// K-term dot product (Wilkinson) and ε_i bounds the float64 error of the
// host sums. A clean tile computed in FP32 with any K order, then stored
// with relative rounding at most RelTolerance, is within τ in every row
// and column.
func abftBounds(a, b, d []float32, cfg ABFTConfig) (rowObs, rowRef, rowTol, colObs, colRef, colTol []float64) {
	m, n, k := cfg.TileM, cfg.TileN, cfg.TileK
	bsum := make([]float64, k)
	babs := make([]float64, k)
	for kk := range k {
		for j := range n {
			v := float64(b[kk*n+j])
			bsum[kk] += v
			babs[kk] += math.Abs(v)
		}
	}
	asum := make([]float64, k)
	aabs := make([]float64, k)
	for i := range m {
		for kk := range k {
			v := float64(a[i*k+kk])
			asum[kk] += v
			aabs[kk] += math.Abs(v)
		}
	}
	gk := gammaF32(k)
	// Float64 sums of at most m+n+k+2 terms: γ for u = 2^-53.
	host := float64(m+n+k+2) * 0x1p-53 * 1.01
	rel := float64(cfg.RelTolerance)
	rowObs, rowRef, rowTol = make([]float64, m), make([]float64, m), make([]float64, m)
	for i := range m {
		var mag, dabs float64
		for kk := range k {
			v := float64(a[i*k+kk])
			rowRef[i] += v * bsum[kk]
			mag += math.Abs(v) * babs[kk]
		}
		for j := range n {
			v := float64(d[i*n+j])
			rowObs[i] += v
			dabs += math.Abs(v)
		}
		rowTol[i] = gk*mag + rel*dabs + host*(mag+dabs)
	}
	colObs, colRef, colTol = make([]float64, n), make([]float64, n), make([]float64, n)
	for j := range n {
		var mag, dabs float64
		for kk := range k {
			v := float64(b[kk*n+j])
			colRef[j] += asum[kk] * v
			mag += aabs[kk] * math.Abs(v)
		}
		for i := range m {
			v := float64(d[i*n+j])
			colObs[j] += v
			dabs += math.Abs(v)
		}
		colTol[j] = gk*mag + rel*dabs + host*(mag+dabs)
	}
	return
}

// VerifyTileABFT checks a computed tile d = a·b (a is TileM×TileK, b
// TileK×TileN, d TileM×TileN, all row-major) with row checksums
// D·1 = A·(B·1) and column checksums 1ᵀ·D = (1ᵀ·A)·B.
//
// A row or column whose checksums differ by more than its tolerance τ (see
// [ABFTConfig.RelTolerance]; τ bounds the rounding error of every FP32
// accumulation order) is a fault. The record names the worst row and the
// worst column, measured as |difference|/τ, so a single corrupted element
// whose change exceeds 2τ in both its row and column is located exactly. A
// change smaller than τ is indistinguishable from rounding and is not
// reported. A non-finite element of d with finite a and b is reported as
// [FaultNaNInfGenerated] at its coordinates. A clean tile returns a record
// with Kind [FaultNone] and coordinates -1.
func VerifyTileABFT(a, b, d []float32, cfg ABFTConfig) (FaultRecord, error) {
	rec := FaultRecord{CorruptRow: -1, CorruptCol: -1}
	if err := cfg.check(); err != nil {
		return rec, fmt.Errorf("verify tile abft: %w", err)
	}
	m, n, k := cfg.TileM, cfg.TileN, cfg.TileK
	if len(a) != m*k || len(b) != k*n || len(d) != m*n {
		return rec, fmt.Errorf("verify tile abft: lengths %d, %d, %d, want %d, %d, %d", len(a), len(b), len(d), m*k, k*n, m*n)
	}
	for _, s := range [][]float32{a, b} {
		for _, v := range s {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return rec, errors.New("verify tile abft: non-finite input")
			}
		}
	}
	for i, v := range d {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			rec.Kind = FaultNaNInfGenerated
			rec.CorruptRow, rec.CorruptCol = int32(i/n), int32(i%n)
			rec.ObservedDiff = v
			return rec, nil
		}
	}
	rowObs, rowRef, rowTol, colObs, colRef, colTol := abftBounds(a, b, d, cfg)
	worst := func(obs, ref, tol []float64) (int, float64) {
		at, ratio := -1, 1.0
		for i := range obs {
			r := math.Abs(obs[i]-ref[i]) / tol[i]
			if tol[i] == 0 && obs[i] != ref[i] {
				r = math.Inf(1)
			}
			if r > ratio {
				at, ratio = i, r
			}
		}
		return at, ratio
	}
	r, _ := worst(rowObs, rowRef, rowTol)
	c, _ := worst(colObs, colRef, colTol)
	if r < 0 && c < 0 {
		return rec, nil
	}
	rec.Kind = FaultTensorCoreSDC
	rec.CorruptRow, rec.CorruptCol = int32(r), int32(c)
	if r >= 0 {
		rec.ObservedDiff = float32(rowObs[r] - rowRef[r])
	} else {
		rec.ObservedDiff = float32(colObs[c] - colRef[c])
	}
	return rec, nil
}

// RunTileWithABFT computes a tile with gemm, which must fill d = a·b, and
// checks it with [VerifyTileABFT]. When the check reports a fault and
// cfg.AutoRecomputeOnSDC is set, it recomputes the tile once: if the replay
// passes, it returns the replayed tile and the first fault with Corrected
// set; otherwise the replay's fault.
func RunTileWithABFT(a, b []float32, cfg ABFTConfig, gemm func(a, b, d []float32)) ([]float32, FaultRecord, error) {
	if err := cfg.check(); err != nil {
		return nil, FaultRecord{}, fmt.Errorf("run tile with abft: %w", err)
	}
	d := make([]float32, cfg.TileM*cfg.TileN)
	gemm(a, b, d)
	rec, err := VerifyTileABFT(a, b, d, cfg)
	if err != nil || rec.Kind == FaultNone || !cfg.AutoRecomputeOnSDC {
		return d, rec, err
	}
	clear(d)
	gemm(a, b, d)
	again, err := VerifyTileABFT(a, b, d, cfg)
	if err != nil {
		return d, again, err
	}
	if again.Kind == FaultNone {
		rec.Corrected = true
		return d, rec, nil
	}
	return d, again, nil
}

// abftTau is the relative tolerance the device epilogue applies to
// Σ_j|D_ij|: 2·γ_{K+N} covers the FP32 accumulation of D, of the checksum
// reference and of the row sum.
func abftTau(cfg ABFTConfig) float32 {
	return float32(2*gammaF32(cfg.TileK+cfg.TileN)) + cfg.RelTolerance
}

// EmitABFTAndWatchdogPTX emits a PTX 8.6 module for sm_100a with three
// device functions:
//
//   - prefix_abft_row(taddr, mma_ok) -> code checks one accumulator row of
//     a TileM×TileN FP32 tile in Tensor Memory, read with
//     tcgen05.ld.sync.aligned.32x32b.x16.b32: it sums the row and its
//     absolute values in registers and compares the sum with the checksum
//     the MMA wrote in column TileN (from a B operand extended with its row
//     sums; the MMA N grows by its 16-column granularity, so the extra MMA
//     work is 16/TileN). It returns 0 when |Σ_j D_ij − ref| ≤ τ·Σ_j|D_ij|,
//     with τ = 2·γ_{K+N} + RelTolerance, 1 for a checksum fault and 2 for a
//     non-finite sum.
//   - prefix_wait(bar, parity, timeout_ns) -> elapsed polls
//     mbarrier.try_wait.parity and gives up once %globaltimer has advanced
//     by more than timeout_ns, returning the elapsed time (0 on success).
//   - prefix_report(ring, ...) appends a 32-byte [FaultRecord] to a ring in
//     global memory: a u32 slot counter at offset 0, a u32 capacity at
//     offset 4, and records from offset 32.
//
// PTX ISA 9.4 offers tcgen05.ld.red only with .min and .max, so the row sum
// is not folded into the load; tcgen05.ld.red.max.abs.NaN (sm_103f, sm_110f)
// could replace the separate finiteness test on those targets. The device
// tolerance scales with Σ_j|D_ij| rather than Σ_k|A_ik||B_kj|, which the
// device does not have, so rows with heavy cancellation can be flagged;
// [VerifyTileABFT] uses the rigorous bound.
//
// The emitted text is checked in this package by a PTX interpreter on
// clean and corrupted rows. It has not been assembled or run here.
func EmitABFTAndWatchdogPTX(cfg ABFTConfig, prefix string) (string, error) {
	if err := cfg.check(); err != nil {
		return "", fmt.Errorf("emit abft ptx: %w", err)
	}
	if prefix == "" {
		return "", errors.New("emit abft ptx: empty prefix")
	}
	if cfg.TileN%16 != 0 || cfg.TileN+16 > 512 {
		return "", fmt.Errorf("emit abft ptx: TileN %d must be a multiple of 16 with TileN+16 <= 512 TMEM columns", cfg.TileN)
	}
	var b strings.Builder
	b.WriteString(".version 8.6\n.target sm_100a\n.address_size 64\n\n")
	tau := abftTau(cfg)
	fmt.Fprintf(&b, `// %[1]s_abft_row checks the TMEM accumulator row at taddr against the
// checksum in column %[2]d. tau = 2*gamma_%[3]d + RelTolerance = %[4]g.
.visible .func (.param .b32 r_out) %[1]s_abft_row(.param .b32 p_taddr)
{
	.reg .pred %%p;
	.reg .b32 %%t, %%tc, %%x<16>, %%code;
	.reg .f32 %%sum, %%abs, %%a, %%ref, %%diff, %%tol;
	ld.param.b32 %%t, [p_taddr];
	mov.f32 %%sum, 0f00000000;
	mov.f32 %%abs, 0f00000000;
`, prefix, cfg.TileN, cfg.TileK+cfg.TileN, tau)
	regs := make([]string, 16)
	for i := range regs {
		regs[i] = fmt.Sprintf("%%x%d", i)
	}
	vec := "{" + strings.Join(regs, ", ") + "}"
	for c := 0; c < cfg.TileN; c += 16 {
		fmt.Fprintf(&b, "\tadd.u32 %%tc, %%t, %d;\n\ttcgen05.ld.sync.aligned.32x32b.x16.b32 %s, [%%tc];\n\ttcgen05.wait::ld.sync.aligned;\n", c, vec)
		for _, r := range regs {
			fmt.Fprintf(&b, "\tmov.b32 %%a, %s;\n\tadd.rn.f32 %%sum, %%sum, %%a;\n\tabs.f32 %%a, %%a;\n\tadd.rn.f32 %%abs, %%abs, %%a;\n", r)
		}
	}
	fmt.Fprintf(&b, `	add.u32 %%tc, %%t, %d;
	tcgen05.ld.sync.aligned.32x32b.x1.b32 {%%x0}, [%%tc];
	tcgen05.wait::ld.sync.aligned;
	mov.b32 %%ref, %%x0;
	mov.u32 %%code, 2;
	testp.finite.f32 %%p, %%sum;
	@!%%p bra ABFT_DONE;
	mov.u32 %%code, 0;
	sub.rn.f32 %%diff, %%sum, %%ref;
	abs.f32 %%diff, %%diff;
	mul.rn.f32 %%tol, %%abs, 0f%08X;
	setp.gt.f32 %%p, %%diff, %%tol;
	@%%p mov.u32 %%code, 1;
ABFT_DONE:
	st.param.b32 [r_out], %%code;
	ret;
}

`, cfg.TileN, math.Float32bits(tau))
	writeWatchdogPTX(&b, cfg, prefix)
	return b.String(), nil
}

// EmitWatchdogPTX emits the prefix_wait and prefix_report functions of
// [EmitABFTAndWatchdogPTX] alone, as a PTX 8.0 module for sm_90, which
// sm_120a and sm_121a also accept.
func EmitWatchdogPTX(cfg ABFTConfig, prefix string) (string, error) {
	if prefix == "" {
		return "", errors.New("emit watchdog ptx: empty prefix")
	}
	var b strings.Builder
	b.WriteString(".version 8.0\n.target sm_90\n.address_size 64\n\n")
	writeWatchdogPTX(&b, cfg, prefix)
	return b.String(), nil
}

func writeWatchdogPTX(b *strings.Builder, cfg ABFTConfig, prefix string) {
	fmt.Fprintf(b, `// %[1]s_wait waits for phase parity of the shared-memory mbarrier at bar,
// giving up after timeout_ns (default %[2]d) of %%globaltimer. It returns 0
// on completion and the elapsed nanoseconds on timeout.
.visible .func (.param .b64 r_out) %[1]s_wait(.param .b32 p_bar, .param .b32 p_parity, .param .b64 p_timeout)
{
	.reg .pred %%done, %%late;
	.reg .b32 %%bar, %%par;
	.reg .b64 %%limit, %%t0, %%now, %%el;
	ld.param.b32 %%bar, [p_bar];
	ld.param.b32 %%par, [p_parity];
	ld.param.b64 %%limit, [p_timeout];
	mov.u64 %%t0, %%globaltimer;
	mov.u64 %%el, 0;
WAIT_LOOP:
	mbarrier.try_wait.parity.shared::cta.b64 %%done, [%%bar], %%par;
	@%%done bra WAIT_DONE;
	mov.u64 %%now, %%globaltimer;
	sub.u64 %%el, %%now, %%t0;
	setp.le.u64 %%late, %%el, %%limit;
	@%%late bra WAIT_LOOP;
	bra WAIT_OUT;
WAIT_DONE:
	mov.u64 %%el, 0;
WAIT_OUT:
	st.param.b64 [r_out], %%el;
	ret;
}

// %[1]s_report appends a 32-byte fault record to the ring at p_ring.
.visible .func %[1]s_report(.param .b64 p_ring, .param .b32 p_kind_flags_rank, .param .b32 p_peer_layer,
	.param .b32 p_tile_m, .param .b32 p_tile_n, .param .b32 p_row_col, .param .b32 p_diff, .param .b64 p_elapsed)
{
	.reg .pred %%full;
	.reg .b32 %%slot, %%cap, %%w<6>;
	.reg .b64 %%ring, %%rec, %%off, %%el;
	ld.param.b64 %%ring, [p_ring];
	atom.global.add.u32 %%slot, [%%ring], 1;
	ld.global.u32 %%cap, [%%ring+4];
	setp.ge.u32 %%full, %%slot, %%cap;
	@%%full bra REPORT_DONE;
	mul.wide.u32 %%off, %%slot, 32;
	add.u64 %%rec, %%ring, %%off;
	ld.param.b32 %%w0, [p_kind_flags_rank];
	ld.param.b32 %%w1, [p_peer_layer];
	ld.param.b32 %%w2, [p_tile_m];
	ld.param.b32 %%w3, [p_tile_n];
	ld.param.b32 %%w4, [p_row_col];
	ld.param.b32 %%w5, [p_diff];
	ld.param.b64 %%el, [p_elapsed];
	st.global.v4.b32 [%%rec+32], {%%w0, %%w1, %%w2, %%w3};
	st.global.v2.b32 [%%rec+48], {%%w4, %%w5};
	st.global.b64 [%%rec+56], %%el;
REPORT_DONE:
	ret;
}
`, prefix, cfg.WatchdogTimeoutNs)
}

// ABFTCUDA returns CUDA C++ source, for cudahostemu and nvcc, with
//
//   - qs_abft_gemm: one thread per row of an m×n tile computes
//     D = A·B in FP32 (__fmul_rn, __fadd_rn), flips bits xor_mask of
//     D[inj_row][inj_col] when inj_row >= 0 to inject a fault, and checks
//     its row against the checksum A·(B·1) with the device tolerance of
//     [EmitABFTAndWatchdogPTX]; a failing row appends a [FaultRecord] (column
//     -1) to the ring with the qs_report_fault helper;
//   - qs_report_fault: the CUDA twin of prefix_report.
//
// The ring layout is that of [EmitABFTAndWatchdogPTX].
func ABFTCUDA(cfg ABFTConfig) (string, error) {
	if err := cfg.check(); err != nil {
		return "", fmt.Errorf("abft cuda: %w", err)
	}
	return strings.ReplaceAll(abftCUDASrc, "$TAU", fmt.Sprintf("__uint_as_float(%#xu)", math.Float32bits(abftTau(cfg)))), nil
}

const abftCUDASrc = `
#include <cstdint>

__device__ void qs_report_fault(uint32_t* ring, uint32_t kind_flags_rank, uint32_t peer_layer,
                                uint32_t tile_m, uint32_t tile_n, uint32_t row_col, float diff, uint64_t elapsed) {
    uint32_t slot = atomicAdd(&ring[0], 1u);
    if (slot >= ring[1]) return;
    uint32_t* rec = ring + 8 + slot * 8;
    rec[0] = kind_flags_rank; rec[1] = peer_layer; rec[2] = tile_m; rec[3] = tile_n;
    rec[4] = row_col; rec[5] = __float_as_uint(diff);
    rec[6] = (uint32_t)elapsed; rec[7] = (uint32_t)(elapsed >> 32);
}

__global__ void qs_abft_gemm(const float* a, const float* b, float* d, uint32_t* ring,
                             int m, int n, int k, int inj_row, int inj_col, uint32_t xor_mask, uint32_t tile_m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) acc = __fadd_rn(acc, __fmul_rn(a[i * k + kk], b[kk * n + j]));
        if (i == inj_row && j == inj_col) acc = __uint_as_float(__float_as_uint(acc) ^ xor_mask);
        d[i * n + j] = acc;
    }
    float ref = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        float bs = 0.0f;
        for (int j = 0; j < n; ++j) bs = __fadd_rn(bs, b[kk * n + j]);
        ref = __fadd_rn(ref, __fmul_rn(a[i * k + kk], bs));
    }
    float sum = 0.0f, abs_sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        sum = __fadd_rn(sum, d[i * n + j]);
        abs_sum = __fadd_rn(abs_sum, fabsf(d[i * n + j]));
    }
    uint32_t kind = 0;
    if (!isfinite(sum)) kind = 2;
    else if (fabsf(__fsub_rn(sum, ref)) > __fmul_rn(abs_sum, $TAU)) kind = 1;
    if (kind) qs_report_fault(ring, kind, 0u, tile_m, 0u, ((uint32_t)i & 0xffffu) | 0xffff0000u, __fsub_rn(sum, ref), 0ull);
}
`
