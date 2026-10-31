package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/egraph/tilebridge"
	"github.com/tmc/lib/nvidia/x/safetensors"
	"github.com/tmc/lib/nvidia/x/tcgen05"
)

// A WeightTargetFormat is the device layout a [StreamLoader] produces.
type WeightTargetFormat uint8

const (
	// FormatSwizzledBF16 keeps BF16 bits unchanged in swizzled tiles.
	FormatSwizzledBF16 WeightTargetFormat = iota

	// FormatSwizzledFP8E4M3 holds E4M3 codes in swizzled tiles with one
	// FP32 scale per row (amax/448).
	FormatSwizzledFP8E4M3

	// FormatNVFP4ScaleVec4X holds packed E2M1 pairs in swizzled tiles and
	// UE4M3 block scales (one per 16 elements) packed four to a uint32 word,
	// the grouping a scale_vec::4X block-scaled MMA reads per 64-element K step.
	FormatNVFP4ScaleVec4X

	// FormatSparse2To4NVFP4 is FormatNVFP4ScaleVec4X pruned to 2:4 along K:
	// one byte of two kept E2M1 codes per 4-element group, and 4-bit
	// selectors packed eight to a uint32 metadata word.
	FormatSparse2To4NVFP4

	// FormatRubin3BitLUTBlock32 holds 3-bit indices into an 8-entry E4M3
	// codebook per 8 rows, with one UE8M0 scale per 32-element block, the
	// operand of Rubin's tcgen05.mma ... decompress::lut::b.block32.
	FormatRubin3BitLUTBlock32
)

var formatNames = [...]string{
	FormatSwizzledBF16:        "swizzled-bf16",
	FormatSwizzledFP8E4M3:     "swizzled-fp8-e4m3",
	FormatNVFP4ScaleVec4X:     "nvfp4-scale-vec-4x",
	FormatSparse2To4NVFP4:     "sparse-2to4-nvfp4",
	FormatRubin3BitLUTBlock32: "rubin-3bit-lut-block32",
}

func (f WeightTargetFormat) String() string {
	if int(f) < len(formatNames) {
		return formatNames[f]
	}
	return fmt.Sprintf("WeightTargetFormat(%d)", uint8(f))
}

// SwizzleOffset returns the physical shared-memory byte offset of logical
// byte offset off under an n-byte TMA swizzle (CU_TENSOR_MAP_SWIZZLE_32B,
// _64B or _128B): the 16-byte chunk bits [4, 4+log2(n/16)) are XORed with
// the bits [7, 7+log2(n/16)) above them. For the 128-byte swizzle and
// 128-byte rows, chunk c of row r lands at chunk c ^ (r mod 8). An n of 0 is
// the identity. The function is an involution, so it is its own inverse.
// The tile base must be aligned to 8*n bytes for hardware to agree.
func SwizzleOffset(n, off int) int {
	if n == 0 {
		return off
	}
	mask := n/16 - 1
	return off ^ (((off >> 7) & mask) << 4)
}

// BankConflictWays returns how many shared-memory wavefronts a set of
// simultaneous width-byte accesses at the given byte offsets needs per
// bank: the largest number of distinct 4-byte words that map to any one of
// the 32 four-byte banks. A warp-wide access of at most 128 bytes is
// conflict-free when it returns 1.
func BankConflictWays(offsets []int, width int) int {
	words := make(map[int]map[int]bool)
	for _, off := range offsets {
		for w := off / 4; w < (off+width+3)/4; w++ {
			b := w % 32
			if words[b] == nil {
				words[b] = make(map[int]bool)
			}
			words[b][w] = true
		}
	}
	ways := 0
	for _, s := range words {
		ways = max(ways, len(s))
	}
	return ways
}

// RepackSwizzled copies a row-major [rows, rowBytes] byte matrix into
// tile-major order: tiles of tileRows rows by span bytes, ordered by row
// tile then column panel, each tile holding the exact shared-memory image a
// TMA box load with an n-byte swizzle (n = span, or 0) would produce. A tile
// can therefore be moved to shared memory with a linear bulk copy.
func RepackSwizzled(src []byte, rows, rowBytes, span, swizzle, tileRows int) ([]byte, error) {
	if err := checkTileGeometry(rows, rowBytes, span, swizzle, tileRows); err != nil {
		return nil, fmt.Errorf("repack swizzled: %w", err)
	}
	if len(src) != rows*rowBytes {
		return nil, fmt.Errorf("repack swizzled: len(src)=%d, want %d", len(src), rows*rowBytes)
	}
	dst := make([]byte, len(src))
	panels := rowBytes / span
	tileBytes := tileRows * span
	for r := range rows {
		t, rr := r/tileRows, r%tileRows
		for p := range panels {
			base := (t*panels + p) * tileBytes
			for c := 0; c < span; c += 16 {
				off := SwizzleOffset(swizzle, rr*span+c)
				copy(dst[base+off:base+off+16], src[r*rowBytes+p*span+c:])
			}
		}
	}
	return dst, nil
}

// UnpackSwizzled inverts [RepackSwizzled].
func UnpackSwizzled(tiles []byte, rows, rowBytes, span, swizzle, tileRows int) ([]byte, error) {
	if err := checkTileGeometry(rows, rowBytes, span, swizzle, tileRows); err != nil {
		return nil, fmt.Errorf("unpack swizzled: %w", err)
	}
	if len(tiles) != rows*rowBytes {
		return nil, fmt.Errorf("unpack swizzled: len(tiles)=%d, want %d", len(tiles), rows*rowBytes)
	}
	dst := make([]byte, len(tiles))
	panels := rowBytes / span
	tileBytes := tileRows * span
	for r := range rows {
		t, rr := r/tileRows, r%tileRows
		for p := range panels {
			base := (t*panels + p) * tileBytes
			for c := 0; c < span; c += 16 {
				off := SwizzleOffset(swizzle, rr*span+c)
				copy(dst[r*rowBytes+p*span+c:r*rowBytes+p*span+c+16], tiles[base+off:])
			}
		}
	}
	return dst, nil
}

func checkTileGeometry(rows, rowBytes, span, swizzle, tileRows int) error {
	switch swizzle {
	case 0:
	case 32, 64, 128:
		if span != swizzle {
			return fmt.Errorf("span %d must equal swizzle %d", span, swizzle)
		}
	default:
		return fmt.Errorf("unsupported swizzle %d", swizzle)
	}
	if span < 16 || span > 128 || span%16 != 0 {
		return fmt.Errorf("span %d is not a 16-byte multiple in 16..128", span)
	}
	if rowBytes <= 0 || rowBytes%span != 0 {
		return fmt.Errorf("row bytes %d is not a positive multiple of span %d", rowBytes, span)
	}
	if tileRows <= 0 || tileRows%8 != 0 || tileRows > 256 || rows <= 0 || rows%tileRows != 0 {
		return fmt.Errorf("tile rows %d must be a multiple of 8 up to 256 dividing rows %d", tileRows, rows)
	}
	return nil
}

// A PackedTensorBundle is one weight matrix in a device-ready layout.
//
// WeightBytes is tile-major (see [RepackSwizzled]): WeightTMADesc describes
// it as a rank-3 tensor {Span/2 lanes, TileRows, tiles} of 16-bit lanes with
// no swizzle, so loading box (0, 0, i) copies tile i's pre-swizzled image to
// shared memory unchanged. SourceTMADesc describes the original row-major
// weight bytes (for example a host-registered checkpoint mapping) with the
// swizzle applied in flight; box (p, t) of SourceTMADesc and tile
// t*panels+p of WeightTMADesc produce the same shared-memory image. Both
// descriptors carry Base 0; the caller patches Base before encoding. A
// descriptor with Rank 0 means that payload is not TMA-addressable and
// should be moved with a bulk copy.
type PackedTensorBundle struct {
	Name       string
	Format     WeightTargetFormat
	Rows, Cols int

	// RowBytes is the width in bytes of one row of the weight payload,
	// Span the tile width in bytes, Swizzle the swizzle in bytes (0, 32,
	// 64 or 128), and TileRows the rows per tile.
	RowBytes, Span, Swizzle, TileRows int

	WeightBytes []byte
	// ScaleBytes holds per-row FP32 scales (FP8), little-endian
	// scale_vec::4X words (NVFP4 and 2:4 NVFP4), or UE8M0 block scales
	// [Rows, Cols/32] (3-bit LUT).
	ScaleBytes []byte
	// MetaBytes holds the 2:4 selector words [Rows, Cols/32] little-endian.
	MetaBytes []byte
	// LUTBytes holds the 8-entry E4M3 codebooks, one per 8 rows.
	LUTBytes []byte

	WeightTMADesc cudatma.Descriptor
	SourceTMADesc cudatma.Descriptor
	ScaleTMADesc  cudatma.Descriptor
}

// A StreamLoader packs checkpoint tensors into a [WeightTargetFormat].
type StreamLoader struct {
	// Target is the SM target the descriptors are validated for: one of
	// the TMA targets cudatma admits (sm_100a, sm_103a, sm_107a, sm_100f,
	// sm_103f, sm_107f, sm_121a). FormatRubin3BitLUTBlock32 requires
	// sm_107a or sm_107f.
	Target string
	Format WeightTargetFormat
	// Swizzle is the swizzle in bytes (32, 64 or 128); 0 selects the
	// widest of those dividing the payload row. The 3-bit LUT format
	// ignores it: its rows (48-byte multiples) are not swizzled.
	Swizzle int
	// TileRows is the rows per tile; 0 selects the largest of 64, 32, 16
	// and 8 that divides the row count.
	TileRows int
}

func (l *StreamLoader) tileRows(rows int) (int, error) {
	if l.TileRows != 0 {
		return l.TileRows, nil
	}
	for _, t := range []int{64, 32, 16, 8} {
		if rows%t == 0 {
			return t, nil
		}
	}
	return 0, fmt.Errorf("rows %d is not a multiple of 8", rows)
}

// PackTensor packs a row-major [rows, cols] BF16 tensor.
func (l *StreamLoader) PackTensor(name string, rows, cols int, rawBF16 []uint16) (PackedTensorBundle, error) {
	raw := make([]byte, 2*len(rawBF16))
	for i, v := range rawBF16 {
		binary.LittleEndian.PutUint16(raw[2*i:], v)
	}
	return l.PackTensorBytes(name, rows, cols, safetensors.BF16, raw)
}

// PackTensorBytes packs a row-major [rows, cols] tensor stored as
// little-endian dtype (BF16, F16 or F32) bytes, as found in a safetensors
// data section. raw is only read, so it may alias a read-only mapping.
func (l *StreamLoader) PackTensorBytes(name string, rows, cols int, dtype safetensors.Dtype, raw []byte) (PackedTensorBundle, error) {
	b, err := l.pack(name, rows, cols, dtype, raw)
	if err != nil {
		return PackedTensorBundle{}, fmt.Errorf("pack %s: %w", name, err)
	}
	return b, nil
}

func (l *StreamLoader) pack(name string, rows, cols int, dtype safetensors.Dtype, raw []byte) (PackedTensorBundle, error) {
	if rows <= 0 || cols <= 0 {
		return PackedTensorBundle{}, fmt.Errorf("invalid shape [%d, %d]", rows, cols)
	}
	if es := dtype.Size(); es == 0 || len(raw) != rows*cols*es {
		return PackedTensorBundle{}, fmt.Errorf("%d bytes do not hold [%d, %d] %s", len(raw), rows, cols, dtype)
	}
	tr, err := l.tileRows(rows)
	if err != nil {
		return PackedTensorBundle{}, err
	}
	swz := l.Swizzle
	b := PackedTensorBundle{Name: name, Format: l.Format, Rows: rows, Cols: cols, TileRows: tr}
	var payload []byte
	switch l.Format {
	case FormatSwizzledBF16:
		if dtype == safetensors.BF16 {
			payload = raw
		} else {
			vals, err := decodeFloats(dtype, raw)
			if err != nil {
				return b, err
			}
			payload = make([]byte, 2*len(vals))
			for i, v := range vals {
				binary.LittleEndian.PutUint16(payload[2*i:], safetensors.Float32ToBFloat16(v))
			}
		}
		b.RowBytes = 2 * cols
	case FormatSwizzledFP8E4M3:
		vals, err := decodeFloats(dtype, raw)
		if err != nil {
			return b, err
		}
		payload = make([]byte, rows*cols)
		b.ScaleBytes = make([]byte, 4*rows)
		for r := range rows {
			row := vals[r*cols : (r+1)*cols]
			var amax float32
			for _, v := range row {
				amax = max(amax, float32(math.Abs(float64(v))))
			}
			s := amax / 448
			if s == 0 {
				s = 1
			}
			binary.LittleEndian.PutUint32(b.ScaleBytes[4*r:], math.Float32bits(s))
			for c, v := range row {
				payload[r*cols+c] = tilebridge.E4M3Bits(v / s)
			}
		}
		b.RowBytes = cols
	case FormatNVFP4ScaleVec4X, FormatSparse2To4NVFP4:
		if cols%64 != 0 {
			return b, fmt.Errorf("%s needs cols %d to be a multiple of 64", l.Format, cols)
		}
		vals, err := decodeFloats(dtype, raw)
		if err != nil {
			return b, err
		}
		packed, scales, err := QuantizeTensorNVFP4(vals, rows, cols)
		if err != nil {
			return b, err
		}
		payload = packed
		b.RowBytes = cols / 2
		if l.Format == FormatSparse2To4NVFP4 {
			comp, compScales, meta, err := tcgen05.Prune2To4NVFP4(packed, scales, rows, cols)
			if err != nil {
				return b, err
			}
			payload, scales = comp, compScales
			b.RowBytes = cols / 4
			b.MetaBytes = make([]byte, 4*len(meta))
			for i, w := range meta {
				binary.LittleEndian.PutUint32(b.MetaBytes[4*i:], w)
			}
		}
		// scales is [rows, cols/16]; four consecutive bytes are one word.
		b.ScaleBytes = scales
	case FormatRubin3BitLUTBlock32:
		if l.Target != "sm_107a" && l.Target != "sm_107f" {
			return b, fmt.Errorf("%s requires sm_107a or sm_107f, not %q", l.Format, l.Target)
		}
		if rows%8 != 0 || cols%128 != 0 {
			return b, fmt.Errorf("%s needs rows %% 8 == 0 and cols %% 128 == 0, have [%d, %d]", l.Format, rows, cols)
		}
		vals, err := decodeFloats(dtype, raw)
		if err != nil {
			return b, err
		}
		// Quantize3BitLUTB takes B as [K, N]; the weight is [N, K].
		bt := make([]float32, len(vals))
		for n := range rows {
			for k := range cols {
				bt[k*rows+n] = vals[n*cols+k]
			}
		}
		idx, lut, scales, _, err := tcgen05.Quantize3BitLUTB(bt, cols, rows)
		if err != nil {
			return b, err
		}
		b.RowBytes = cols * 3 / 8
		payload = make([]byte, rows*b.RowBytes)
		for n := range rows {
			row := payload[n*b.RowBytes : (n+1)*b.RowBytes]
			for k := range cols {
				put3(row, k, idx[k*rows+n])
			}
		}
		b.ScaleBytes, b.LUTBytes = scales, lut
		swz = 0
	default:
		return b, fmt.Errorf("unknown format %d", uint8(l.Format))
	}

	if swz == 0 && l.Format != FormatRubin3BitLUTBlock32 {
		for _, n := range []int{128, 64, 32} {
			if b.RowBytes%n == 0 {
				swz = n
				break
			}
		}
	}
	b.Swizzle = swz
	b.Span = swz
	if swz == 0 {
		b.Span = noSwizzleSpan(b.RowBytes)
		if b.Span == 0 {
			return b, fmt.Errorf("row bytes %d has no 16-byte tile width", b.RowBytes)
		}
	}
	if b.WeightBytes, err = RepackSwizzled(payload, rows, b.RowBytes, b.Span, b.Swizzle, tr); err != nil {
		return b, err
	}
	if b.WeightTMADesc, err = l.tileDescriptor(b); err != nil {
		return b, fmt.Errorf("weight descriptor: %w", err)
	}
	if b.SourceTMADesc, err = l.sourceDescriptor(b); err != nil {
		return b, fmt.Errorf("source descriptor: %w", err)
	}
	if b.ScaleTMADesc, err = l.scaleDescriptor(b); err != nil {
		return b, fmt.Errorf("scale descriptor: %w", err)
	}
	return b, nil
}

// noSwizzleSpan is the widest 16-byte multiple up to 128 dividing rowBytes.
func noSwizzleSpan(rowBytes int) int {
	for w := 128; w >= 16; w -= 16 {
		if rowBytes%w == 0 {
			return w
		}
	}
	return 0
}

func put3(row []byte, k int, v uint8) {
	bit := 3 * k
	for i := range 3 {
		if v>>i&1 != 0 {
			row[(bit+i)/8] |= 1 << ((bit + i) % 8)
		}
	}
}

func get3(row []byte, k int) uint8 {
	bit := 3 * k
	var v uint8
	for i := range 3 {
		v |= (row[(bit+i)/8] >> ((bit + i) % 8) & 1) << i
	}
	return v
}

func decodeFloats(dtype safetensors.Dtype, raw []byte) ([]float32, error) {
	switch dtype {
	case safetensors.BF16, safetensors.F16, safetensors.F32:
		return safetensors.DecodeFloat32(dtype, raw)
	}
	return nil, fmt.Errorf("unsupported source dtype %s", dtype)
}

func (l *StreamLoader) baseDescriptor(elementBytes uint32) cudatma.Descriptor {
	return cudatma.Descriptor{
		Target:       l.Target,
		ElementBytes: elementBytes,
		L2Promotion:  driver.TensorMapL2PromotionL2256B,
	}
}

func (l *StreamLoader) tileDescriptor(b PackedTensorBundle) (cudatma.Descriptor, error) {
	d := l.baseDescriptor(2)
	if b.Format == FormatSwizzledBF16 {
		d.ElementType = cudatma.ElementTypeBfloat16
	}
	tiles := len(b.WeightBytes) / (b.Span * b.TileRows)
	d.Rank = 3
	d.Dims = [cudatma.MaxRank]uint64{uint64(b.Span / 2), uint64(b.TileRows), uint64(tiles)}
	d.Strides = [cudatma.MaxRank]uint64{uint64(b.Span), uint64(b.Span * b.TileRows)}
	d.Box = [cudatma.MaxRank]uint32{uint32(b.Span / 2), uint32(b.TileRows), 1}
	return d, d.Validate()
}

func (l *StreamLoader) sourceDescriptor(b PackedTensorBundle) (cudatma.Descriptor, error) {
	d := l.baseDescriptor(2)
	if b.Format == FormatSwizzledBF16 {
		d.ElementType = cudatma.ElementTypeBfloat16
	}
	d.Rank = 2
	d.Dims = [cudatma.MaxRank]uint64{uint64(b.RowBytes / 2), uint64(b.Rows)}
	d.Strides = [cudatma.MaxRank]uint64{uint64(b.RowBytes)}
	d.Box = [cudatma.MaxRank]uint32{uint32(b.Span / 2), uint32(b.TileRows)}
	d.Swizzle = uint32(b.Swizzle)
	return d, d.Validate()
}

// scaleDescriptor describes ScaleBytes as a rank-2 tensor of 32-bit
// words, one tile row of words per box. Payloads whose row is not a
// 16-byte multiple get the zero Descriptor.
func (l *StreamLoader) scaleDescriptor(b PackedTensorBundle) (cudatma.Descriptor, error) {
	rowBytes := len(b.ScaleBytes) / b.Rows
	if b.Format == FormatSwizzledFP8E4M3 || rowBytes == 0 || rowBytes%16 != 0 {
		return cudatma.Descriptor{}, nil
	}
	d := l.baseDescriptor(4)
	d.Rank = 2
	d.Dims = [cudatma.MaxRank]uint64{uint64(rowBytes / 4), uint64(b.Rows)}
	d.Strides = [cudatma.MaxRank]uint64{uint64(rowBytes)}
	d.Box = [cudatma.MaxRank]uint32{uint32(noSwizzleSpan(rowBytes) / 4), uint32(b.TileRows)}
	return d, d.Validate()
}

// ScaleVec4XWord returns the scale_vec::4X word for row r and 64-element K
// step j of an NVFP4 bundle: byte i (little-endian) is the UE4M3 scale of
// elements [64j+16i, 64j+16i+16).
func (b PackedTensorBundle) ScaleVec4XWord(r, j int) uint32 {
	words := b.Cols / 64
	return binary.LittleEndian.Uint32(b.ScaleBytes[4*(r*words+j):])
}

// UnpackTensorToFP32 inverts the layout of a [PackedTensorBundle] and
// dequantizes it to row-major [Rows, Cols] float32. For FormatSwizzledBF16
// the result is the packed BF16 values exactly; the other formats return
// the values their codes and scales represent.
func UnpackTensorToFP32(b PackedTensorBundle) ([]float32, error) {
	out, err := unpack(b)
	if err != nil {
		return nil, fmt.Errorf("unpack %s: %w", b.Name, err)
	}
	return out, nil
}

func unpack(b PackedTensorBundle) ([]float32, error) {
	payload, err := UnpackSwizzled(b.WeightBytes, b.Rows, b.RowBytes, b.Span, b.Swizzle, b.TileRows)
	if err != nil {
		return nil, err
	}
	rows, cols := b.Rows, b.Cols
	out := make([]float32, rows*cols)
	switch b.Format {
	case FormatSwizzledBF16:
		for i := range out {
			out[i] = safetensors.BFloat16ToFloat32(binary.LittleEndian.Uint16(payload[2*i:]))
		}
	case FormatSwizzledFP8E4M3:
		if len(b.ScaleBytes) != 4*rows {
			return nil, fmt.Errorf("have %d scale bytes, want %d", len(b.ScaleBytes), 4*rows)
		}
		for r := range rows {
			s := math.Float32frombits(binary.LittleEndian.Uint32(b.ScaleBytes[4*r:]))
			for c := range cols {
				out[r*cols+c] = tilebridge.FromE4M3Bits(payload[r*cols+c]) * s
			}
		}
	case FormatNVFP4ScaleVec4X, FormatSparse2To4NVFP4:
		if len(b.ScaleBytes) != rows*cols/16 {
			return nil, fmt.Errorf("have %d scale bytes, want %d", len(b.ScaleBytes), rows*cols/16)
		}
		// Read the block scales back through the scale_vec::4X words so a
		// misplaced word is visible in the values.
		scales := make([]byte, len(b.ScaleBytes))
		for r := range rows {
			for j := range cols / 64 {
				binary.LittleEndian.PutUint32(scales[r*cols/16+4*j:], b.ScaleVec4XWord(r, j))
			}
		}
		if b.Format == FormatSparse2To4NVFP4 {
			if payload, err = expand2To4NVFP4(payload, b.MetaBytes, rows, cols); err != nil {
				return nil, err
			}
		}
		return DequantizeTensorNVFP4(payload, scales, rows, cols, 1)
	case FormatRubin3BitLUTBlock32:
		kb := cols / 32
		if len(b.ScaleBytes) != rows*kb || len(b.LUTBytes) != rows {
			return nil, fmt.Errorf("have %d scale and %d codebook bytes, want %d and %d", len(b.ScaleBytes), len(b.LUTBytes), rows*kb, rows)
		}
		for n := range rows {
			row := payload[n*b.RowBytes : (n+1)*b.RowBytes]
			for k := range cols {
				code := b.LUTBytes[(n/8)*8+int(get3(row, k))]
				out[n*cols+k] = tcgen05.FromE4M3Bits(code) * tcgen05.FromUE8M0Bits(b.ScaleBytes[n*kb+k/32])
			}
		}
	default:
		return nil, fmt.Errorf("unknown format %d", uint8(b.Format))
	}
	return out, nil
}

// expand2To4NVFP4 rebuilds dense packed E2M1 bytes [rows, cols/2] from 2:4
// compressed bytes [rows, cols/4] and selector words [rows, cols/32].
func expand2To4NVFP4(comp, meta []byte, rows, cols int) ([]byte, error) {
	if len(meta) != 4*rows*cols/32 {
		return nil, fmt.Errorf("have %d metadata bytes, want %d", len(meta), 4*rows*cols/32)
	}
	dense := make([]byte, rows*cols/2)
	for r := range rows {
		for g := range cols / 4 {
			w := binary.LittleEndian.Uint32(meta[4*(r*cols/32+g/8):])
			nib := w >> (4 * (g % 8)) & 0xf
			i0, i1 := int(nib&3), int(nib>>2&3)
			if i0 >= i1 {
				return nil, fmt.Errorf("row %d group %d: invalid selectors (%d, %d)", r, g, i0, i1)
			}
			c := comp[r*cols/4+g]
			for _, kv := range [2]struct {
				i    int
				code byte
			}{{i0, c & 0xf}, {i1, c >> 4}} {
				pos := r*cols + 4*g + kv.i
				dense[pos/2] |= kv.code << (4 * (pos & 1))
			}
		}
	}
	return dense, nil
}

// SQNR returns the signal-to-quantization-noise ratio of approx against
// ref in decibels: 10 log10(Σref² / Σ(ref-approx)²). It is +Inf when the
// two agree exactly.
func SQNR(ref, approx []float32) float64 {
	var sig, noise float64
	for i := range ref {
		d := float64(ref[i]) - float64(approx[i])
		sig += float64(ref[i]) * float64(ref[i])
		noise += d * d
	}
	if noise == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(sig/noise)
}

// SwizzleRepackCUDA returns CUDA C++ for a kernel name(src, dst, rows,
// row_bytes) that performs [RepackSwizzled] on the device: each thread
// moves one 16-byte chunk from row-major src to its tile-major swizzled
// position in dst. row_bytes must be a multiple of span and rows of
// tileRows; both pointers must be 16-byte aligned. It is plain CUDA and
// runs on any sm_70 or later device.
func SwizzleRepackCUDA(name string, span, swizzle, tileRows int) (string, error) {
	if !isCIdent(name) {
		return "", fmt.Errorf("swizzle repack: bad kernel name %q", name)
	}
	if err := checkTileGeometry(tileRows, span, span, swizzle, tileRows); err != nil {
		return "", fmt.Errorf("swizzle repack: %w", err)
	}
	mask := 0
	if swizzle != 0 {
		mask = swizzle/16 - 1
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, `// %s: row-major to tile-major %d-byte-swizzled repack (span %d, %d rows per tile).
extern "C" __global__ void %s(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int rows, int row_bytes) {
	const int span = %d, tile_rows = %d, mask = %d;
	const long long chunks = (long long)rows * row_bytes / 16;
	const int panels = row_bytes / span;
	for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += (long long)gridDim.x * blockDim.x) {
		const long long byte = i * 16;
		const int r = (int)(byte / row_bytes), cb = (int)(byte %% row_bytes);
		const int t = r / tile_rows, rr = r %% tile_rows, p = cb / span, c = cb %% span;
		int off = rr * span + c;
		off ^= ((off >> 7) & mask) << 4;
		const long long tile = (long long)t * panels + p;
		*(uint4*)(dst + tile * tile_rows * span + off) = *(const uint4*)(src + byte);
	}
}
`, name, swizzle, span, tileRows, name, span, tileRows, mask)
	return sb.String(), nil
}

func isCIdent(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r != '_' && (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (i == 0 || r < '0' || r > '9') {
			return false
		}
	}
	return true
}
