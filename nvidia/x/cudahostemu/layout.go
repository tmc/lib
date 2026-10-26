package cudahostemu

// SwizzleMode is the shared-memory swizzle mode of a Hopper WGMMA matrix
// descriptor (bits 62..63 of the 64-bit descriptor).
type SwizzleMode uint8

const (
	SwizzleNone SwizzleMode = 0 // no swizzle (16-byte core matrix)
	Swizzle128B SwizzleMode = 1 // 128-byte swizzle
	Swizzle64B  SwizzleMode = 2 // 64-byte swizzle
	Swizzle32B  SwizzleMode = 3 // 32-byte swizzle
)

// MMAM16N8K16A returns the matrix A row (0..15) and starting K-column pair
// (0, 2, ..., 14) held by 32-bit register reg (0..3) of warp lane (0..31) in
// mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.f32.
//
// The low 16 bits of reg hold A[row, colPair] and the high 16 bits hold
// A[row, colPair+1].
func MMAM16N8K16A(lane, reg int) (row, colPair int) {
	l := lane & 31
	g, t := l>>2, l&3
	return g + 8*(reg&1), 2*t + 8*((reg>>1)&1)
}

// MMAM16N8K16B returns the starting K-row pair (0, 2, ..., 14) and N-column
// (0..7) held by 32-bit register reg (0..1) of warp lane (0..31) in
// mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.f32.
//
// The low 16 bits of reg hold B[rowPair, col] and the high 16 bits hold
// B[rowPair+1, col].
func MMAM16N8K16B(lane, reg int) (rowPair, col int) {
	l := lane & 31
	g, t := l>>2, l&3
	return 2*t + 8*(reg&1), g
}

// MMAM16N8C returns the accumulator row (0..15) and column (0..7) held by
// FP32 register reg (0..3) of warp lane (0..31) in mma.sync.aligned.m16n8k16
// and mma.sync.aligned.m16n8k32.
func MMAM16N8C(lane, reg int) (row, col int) {
	l := lane & 31
	g, t := l>>2, l&3
	return g + 8*((reg>>1)&1), 2*t + (reg & 1)
}

// MMAM16N8K32A returns the matrix A row (0..15) and starting K-column quad
// (0, 4, ..., 28) held by 32-bit register reg (0..3) of warp lane (0..31) in
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.
//
// Byte b (0..3, from low byte to high byte) of reg holds A[row, colQuad+b].
func MMAM16N8K32A(lane, reg int) (row, colQuad int) {
	l := lane & 31
	g, t := l>>2, l&3
	return g + 8*(reg&1), 4*t + 16*((reg>>1)&1)
}

// MMAM16N8K32B returns the starting K-row quad (0, 4, ..., 28) and N-column
// (0..7) held by 32-bit register reg (0..1) of warp lane (0..31) in
// mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.
//
// Byte b (0..3, from low byte to high byte) of reg holds B[rowQuad+b, col].
func MMAM16N8K32B(lane, reg int) (rowQuad, col int) {
	l := lane & 31
	g, t := l>>2, l&3
	return 4*t + 16*(reg&1), g
}

// LDMatrixM8N8X4 returns the 8x8 matrix index (0..3) and row (0..7) whose
// shared-memory row address lane (0..31) supplies to
// ldmatrix.sync.aligned.m8n8.x4.shared.b16, together with the byte offset
// colBytes (0, 4, 8, or 12) within row (lane&31)/4 of matrix j that lane
// receives in destination register j (0..3).
func LDMatrixM8N8X4(lane int) (matrix, row, colBytes int) {
	l := lane & 31
	return l >> 3, l & 7, (l & 3) * 4
}

// WGMMADesc encodes a 64-bit Hopper WGMMA shared-memory matrix descriptor from
// byte offsets (each shifted right by 4 into a 14-bit field) and a swizzle mode.
func WGMMADesc(smemOffsetBytes, leadingBytes, strideBytes int, swizzle SwizzleMode) uint64 {
	start := (uint64(smemOffsetBytes) >> 4) & 0x3fff
	lbo := (uint64(leadingBytes) >> 4) & 0x3fff
	sbo := (uint64(strideBytes) >> 4) & 0x3fff
	return start | (lbo << 16) | (sbo << 32) | ((uint64(swizzle) & 3) << 62)
}

// SwizzleOffset returns the shared-memory byte offset after applying swizzle.
func SwizzleOffset(offsetBytes int, swizzle SwizzleMode) int {
	switch swizzle & 3 {
	case Swizzle128B:
		return offsetBytes ^ (((offsetBytes >> 7) & 7) << 4)
	case Swizzle64B:
		return offsetBytes ^ (((offsetBytes >> 7) & 3) << 4)
	case Swizzle32B:
		return offsetBytes ^ (((offsetBytes >> 7) & 1) << 4)
	default:
		return offsetBytes
	}
}
