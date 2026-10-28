package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cudatma"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

func TestSwizzleOffsetBijection(t *testing.T) {
	for _, n := range []int{0, 32, 64, 128} {
		seen := make(map[int]bool)
		for off := 0; off < 1024; off++ {
			p := SwizzleOffset(n, off)
			if p < 0 || p >= 1024 || seen[p] {
				t.Fatalf("swizzle %d: offset %d maps to %d, not a permutation of the 1024-byte atom", n, off, p)
			}
			seen[p] = true
			if SwizzleOffset(n, p) != off {
				t.Fatalf("swizzle %d: not an involution at %d", n, off)
			}
			if p%16 != off%16 {
				t.Fatalf("swizzle %d: moved byte %d within its 16-byte chunk", n, off)
			}
		}
	}
	// The 128-byte form is c_phys = c ^ (r mod 8) for 128-byte rows.
	for r := range 16 {
		for c := range 8 {
			if got, want := SwizzleOffset(128, r*128+c*16), r*128+(c^(r%8))*16; got != want {
				t.Fatalf("SwizzleOffset(128, row %d chunk %d) = %d, want %d", r, c, got, want)
			}
		}
	}
}

// TestSwizzleBankConflicts reads one logical 16-byte column from eight
// consecutive rows, the ldmatrix access pattern, and checks the swizzled
// addresses cover all 32 banks once.
func TestSwizzleBankConflicts(t *testing.T) {
	tests := []struct {
		swizzle, rowBytes, want int
	}{
		{0, 128, 8},
		{128, 128, 1},
		{64, 64, 1},
		{32, 32, 1},
	}
	for _, tt := range tests {
		for c := 0; c < tt.rowBytes; c += 16 {
			var offs []int
			for r := range 8 {
				offs = append(offs, SwizzleOffset(tt.swizzle, r*tt.rowBytes+c))
			}
			if tt.swizzle == 32 {
				// Eight 32-byte rows fill two 128-byte lines; read four
				// rows per wavefront.
				offs = offs[:4]
			}
			if got := BankConflictWays(offs, 16); got != tt.want {
				t.Errorf("swizzle %d column %d: %d-way, want %d (offsets %v)", tt.swizzle, c, got, tt.want, offs)
			}
		}
	}
}

func randBF16(r *rand.Rand, n int, dist string) ([]uint16, []float32) {
	bits := make([]uint16, n)
	vals := make([]float32, n)
	for i := range bits {
		var v float64
		switch dist {
		case "laplace":
			u := r.Float64() - 0.5
			v = -math.Copysign(math.Log(1-2*math.Abs(u)), u) / math.Sqrt2
		default:
			v = r.NormFloat64()
		}
		bits[i] = safetensors.Float32ToBFloat16(float32(v * 0.05))
		vals[i] = safetensors.BFloat16ToFloat32(bits[i])
	}
	return bits, vals
}

func TestPackBF16RoundTripBitExact(t *testing.T) {
	tests := []struct {
		rows, cols, swizzle, tileRows int
	}{
		{8, 64, 128, 0},
		{64, 128, 128, 0},
		{24, 256, 128, 0},
		{128, 320, 128, 32},
		{16, 32, 64, 0},
		{16, 16, 32, 8},
	}
	r := rand.New(rand.NewPCG(1, 2))
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%dx%d/sw%d", tt.rows, tt.cols, tt.swizzle), func(t *testing.T) {
			raw, _ := randBF16(r, tt.rows*tt.cols, "gauss")
			// Include the special values a checkpoint may carry.
			raw[0], raw[1], raw[2] = 0x7fc1, 0xff80, 0x8000
			l := StreamLoader{Target: "sm_121a", Format: FormatSwizzledBF16, Swizzle: tt.swizzle, TileRows: tt.tileRows}
			b, err := l.PackTensor("w", tt.rows, tt.cols, raw)
			if err != nil {
				t.Fatal(err)
			}
			got, err := UnpackTensorToFP32(b)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if math.Float32bits(v) != uint32(raw[i])<<16 {
					t.Fatalf("element %d: %#08x, want %#08x", i, math.Float32bits(v), uint32(raw[i])<<16)
				}
			}
			// The swizzle is load-bearing: unpacking as unswizzled must
			// differ whenever a tile has more than one row of chunks.
			if tt.swizzle != 0 {
				mut := b
				mut.Swizzle = 0
				bad, err := UnpackTensorToFP32(mut)
				if err != nil {
					t.Fatal(err)
				}
				if slicesEqualBits(bad, got) {
					t.Fatal("unpacking without the swizzle gave the same values")
				}
			}
		})
	}
}

func slicesEqualBits(a, b []float32) bool {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// tmaLoad models a tiled TMA load of the box at coords from global into a
// fresh shared-memory image, reading only the descriptor's fields: element
// (i0, i1, ...) of the box comes from global offset Σ (coords+i)·stride
// (stride 0 being the element size) and lands at its row-major box offset,
// swizzled per 16-byte chunk.
func tmaLoad(t *testing.T, d cudatma.Descriptor, global []byte, coords []int) []byte {
	t.Helper()
	if err := d.Validate(); err != nil {
		t.Fatalf("descriptor: %v", err)
	}
	eb := int(d.ElementBytes)
	strides := []int{eb}
	for i := 0; i < d.Rank-1; i++ {
		strides = append(strides, int(d.Strides[i]))
	}
	boxElems := 1
	for i := range d.Rank {
		boxElems *= int(d.Box[i])
	}
	smem := make([]byte, boxElems*eb)
	idx := make([]int, d.Rank)
	for e := range boxElems {
		rem := e
		for i := range d.Rank {
			idx[i] = rem % int(d.Box[i])
			rem /= int(d.Box[i])
		}
		g := 0
		for i := range d.Rank {
			g += (coords[i] + idx[i]) * strides[i]
		}
		s := SwizzleOffset(int(d.Swizzle), e*eb)
		copy(smem[s:s+eb], global[g:g+eb])
	}
	return smem
}

// TestDescriptorsAgree checks the packed tiles, the tile descriptor and the
// source descriptor against each other: loading tile i through
// WeightTMADesc and loading the same box from the row-major source through
// SourceTMADesc must give the same shared-memory image.
func TestDescriptorsAgree(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	tests := []struct {
		format     WeightTargetFormat
		target     string
		rows, cols int
	}{
		{FormatSwizzledBF16, "sm_121a", 64, 256},
		{FormatSwizzledFP8E4M3, "sm_100a", 32, 256},
		{FormatNVFP4ScaleVec4X, "sm_103a", 32, 512},
		{FormatSparse2To4NVFP4, "sm_107a", 16, 512},
		{FormatRubin3BitLUTBlock32, "sm_107f", 16, 256},
	}
	for _, tt := range tests {
		t.Run(tt.format.String(), func(t *testing.T) {
			raw, _ := randBF16(r, tt.rows*tt.cols, "gauss")
			l := StreamLoader{Target: tt.target, Format: tt.format}
			b, err := l.PackTensor("w", tt.rows, tt.cols, raw)
			if err != nil {
				t.Fatal(err)
			}
			source, err := UnpackSwizzled(b.WeightBytes, b.Rows, b.RowBytes, b.Span, b.Swizzle, b.TileRows)
			if err != nil {
				t.Fatal(err)
			}
			panels := b.RowBytes / b.Span
			for tile := range b.Rows / b.TileRows {
				for p := range panels {
					i := tile*panels + p
					a := tmaLoad(t, b.WeightTMADesc, b.WeightBytes, []int{0, 0, i})
					want := b.WeightBytes[i*b.Span*b.TileRows : (i+1)*b.Span*b.TileRows]
					if !bytes.Equal(a, want) {
						t.Fatalf("tile %d: tile descriptor load differs from the packed image", i)
					}
					s := tmaLoad(t, b.SourceTMADesc, source, []int{p * b.Span / 2, tile * b.TileRows})
					if !bytes.Equal(s, want) {
						t.Fatalf("tile %d: source descriptor load differs from the packed image", i)
					}
				}
			}
			if b.ScaleTMADesc.Rank != 0 {
				if err := b.ScaleTMADesc.Validate(); err != nil {
					t.Fatalf("scale descriptor: %v", err)
				}
			}
			for _, d := range []cudatma.Descriptor{b.WeightTMADesc, b.SourceTMADesc} {
				d.Base = 0x7f00_0000_0080
				if err := d.Validate(); err != nil {
					t.Fatalf("patched base: %v", err)
				}
			}
		})
	}
}

func TestQuantizedSQNR(t *testing.T) {
	const rows, cols = 64, 512
	tests := []struct {
		format WeightTargetFormat
		target string
		// minDB is the floor this layout reaches on both distributions.
		minDB float64
	}{
		{FormatSwizzledFP8E4M3, "sm_121a", 28},
		{FormatNVFP4ScaleVec4X, "sm_121a", 18},
		{FormatRubin3BitLUTBlock32, "sm_107a", 12},
		// Pruning half of each group of four discards real signal; the
		// floor here is for the dense input, not the pruned one.
		{FormatSparse2To4NVFP4, "sm_107a", 7},
	}
	for _, dist := range []string{"gauss", "laplace"} {
		raw, ref := randBF16(rand.New(rand.NewPCG(5, 6)), rows*cols, dist)
		for _, tt := range tests {
			l := StreamLoader{Target: tt.target, Format: tt.format}
			b, err := l.PackTensor("w", rows, cols, raw)
			if err != nil {
				t.Fatalf("%v: %v", tt.format, err)
			}
			got, err := UnpackTensorToFP32(b)
			if err != nil {
				t.Fatalf("%v: %v", tt.format, err)
			}
			db := SQNR(ref, got)
			t.Logf("%s %v: SQNR %.2f dB", dist, tt.format, db)
			if db < tt.minDB {
				t.Errorf("%s %v: SQNR %.2f dB < %.0f dB", dist, tt.format, db, tt.minDB)
			}
			if tt.format == FormatSparse2To4NVFP4 {
				for g := 0; g < len(got); g += 4 {
					nz := 0
					for _, v := range got[g : g+4] {
						if v != 0 {
							nz++
						}
					}
					if nz > 2 {
						t.Fatalf("group %d has %d nonzeros", g/4, nz)
					}
				}
			}
		}
	}
}

// TestScaleVec4XLayout checks each scale_vec::4X word holds the four block
// scales of one 64-element K step in byte order, and that moving words is
// visible after unpacking.
func TestScaleVec4XLayout(t *testing.T) {
	const rows, cols = 8, 256
	raw, _ := randBF16(rand.New(rand.NewPCG(7, 8)), rows*cols, "gauss")
	// Give every 16-element block a distinct magnitude so scales differ.
	for i := range raw {
		v := safetensors.BFloat16ToFloat32(raw[i]) * float32(1+i/16%13)
		raw[i] = safetensors.Float32ToBFloat16(v)
	}
	l := StreamLoader{Target: "sm_121a", Format: FormatNVFP4ScaleVec4X}
	b, err := l.PackTensor("w", rows, cols, raw)
	if err != nil {
		t.Fatal(err)
	}
	vals := make([]float32, len(raw))
	for i, h := range raw {
		vals[i] = safetensors.BFloat16ToFloat32(h)
	}
	_, scales, err := QuantizeTensorNVFP4(vals, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	for r := range rows {
		for j := range cols / 64 {
			w := b.ScaleVec4XWord(r, j)
			for i := range 4 {
				if got, want := byte(w>>(8*i)), scales[r*cols/16+4*j+i]; got != want {
					t.Fatalf("row %d step %d byte %d: %#x, want %#x", r, j, i, got, want)
				}
			}
		}
	}
	good, err := UnpackTensorToFP32(b)
	if err != nil {
		t.Fatal(err)
	}
	mut := b
	mut.ScaleBytes = bytes.Clone(b.ScaleBytes)
	copy(mut.ScaleBytes[0:4], b.ScaleBytes[4:8])
	copy(mut.ScaleBytes[4:8], b.ScaleBytes[0:4])
	bad, err := UnpackTensorToFP32(mut)
	if err != nil {
		t.Fatal(err)
	}
	if slicesEqualBits(good, bad) {
		t.Fatal("swapping two scale_vec::4X words did not change the values")
	}
}

func TestSparseMetadataCorruption(t *testing.T) {
	raw, _ := randBF16(rand.New(rand.NewPCG(9, 10)), 8*64, "gauss")
	l := StreamLoader{Target: "sm_100a", Format: FormatSparse2To4NVFP4}
	b, err := l.PackTensor("w", 8, 64, raw)
	if err != nil {
		t.Fatal(err)
	}
	b.MetaBytes = bytes.Clone(b.MetaBytes)
	b.MetaBytes[0] = 0x00 // selectors (0, 0) are not a valid pair
	if _, err := UnpackTensorToFP32(b); err == nil || !strings.Contains(err.Error(), "invalid selectors") {
		t.Fatalf("UnpackTensorToFP32 with corrupt metadata: %v", err)
	}
}

func TestPackRejects(t *testing.T) {
	raw := make([]uint16, 8*64)
	tests := []struct {
		name string
		l    StreamLoader
		rows int
		cols int
		want string
	}{
		{"hopper target", StreamLoader{Target: "sm_90a"}, 8, 64, "unsupported target"},
		{"lut off rubin", StreamLoader{Target: "sm_121a", Format: FormatRubin3BitLUTBlock32}, 8, 64, "requires sm_107a"},
		{"rows", StreamLoader{Target: "sm_121a"}, 4, 128, "multiple of 8"},
		{"swizzle", StreamLoader{Target: "sm_121a", Swizzle: 48}, 8, 64, "unsupported swizzle"},
		{"nvfp4 cols", StreamLoader{Target: "sm_121a", Format: FormatNVFP4ScaleVec4X}, 16, 32, "multiple of 64"},
		{"length", StreamLoader{Target: "sm_121a"}, 8, 32, "do not hold"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.l.PackTensor("w", tt.rows, tt.cols, raw[:min(len(raw), tt.rows*tt.cols)])
			if tt.name == "length" {
				_, err = tt.l.PackTensor("w", tt.rows, tt.cols, raw)
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err = %v, want %q", err, tt.want)
			}
		})
	}
}

func writeSafetensors(t *testing.T, tensors []safetensors.Tensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := safetensors.Write(&buf, tensors, nil); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestPackMappedZeroCopy(t *testing.T) {
	raw, _ := randBF16(rand.New(rand.NewPCG(11, 12)), 32*128, "gauss")
	data := make([]byte, 2*len(raw))
	for i, v := range raw {
		binary.LittleEndian.PutUint16(data[2*i:], v)
	}
	file := writeSafetensors(t, []safetensors.Tensor{
		{Name: "norm", Dtype: safetensors.BF16, Shape: []int{128}, Data: make([]byte, 256)},
		{Name: "w", Dtype: safetensors.BF16, Shape: []int{32, 128}, Data: data},
	})
	path := filepath.Join(t.TempDir(), "m.safetensors")
	if err := safetensors.WriteFile(path, []safetensors.Tensor{
		{Name: "norm", Dtype: safetensors.BF16, Shape: []int{128}, Data: make([]byte, 256)},
		{Name: "w", Dtype: safetensors.BF16, Shape: []int{32, 128}, Data: data},
	}, nil); err != nil {
		t.Fatal(err)
	}
	mapped, err := MapSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()
	view, err := ViewSafetensors(file)
	if err != nil {
		t.Fatal(err)
	}
	tb, err := view.TensorBytes("w")
	if err != nil {
		t.Fatal(err)
	}
	start := uintptr(unsafe.Pointer(unsafe.SliceData(file)))
	if p := uintptr(unsafe.Pointer(unsafe.SliceData(tb))); p < start || p >= start+uintptr(len(file)) {
		t.Fatal("TensorBytes copied instead of aliasing the file bytes")
	}
	l := StreamLoader{Target: "sm_121a"}
	want, err := l.PackTensor("w", 32, 128, raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []*MappedSafetensors{view, mapped} {
		got, err := l.PackMapped(m, "w")
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got.WeightBytes, want.WeightBytes) {
			t.Fatal("PackMapped differs from PackTensor")
		}
	}
	if _, err := l.PackMapped(view, "norm"); err == nil {
		t.Fatal("PackMapped accepted a rank-1 tensor")
	}
}

func TestPlanStream(t *testing.T) {
	infos := []safetensors.TensorInfo{
		{Name: "a", Shape: []int{256, 1024}, Size: 256 * 2048},
		{Name: "norm", Shape: []int{1024}, Size: 2048},
		{Name: "b", Shape: []int{64, 512}, Size: 64 * 1024},
	}
	p, err := PlanStream(infos, 128<<10, 2)
	if err != nil {
		t.Fatal(err)
	}
	var copied int64
	for _, s := range p.Steps {
		if s.Stage == StageCopy {
			copied += s.Bytes
		}
	}
	if copied != 256*2048+2048+64*1024 {
		t.Fatalf("plan copies %d bytes", copied)
	}
	serial := float64(copied)/(400*1e9) + float64(copied)/(400*1e9)
	if got := p.Makespan(400, 400); got >= serial*0.75 {
		t.Errorf("makespan %.3g s does not overlap (serial %.3g s)", got, serial)
	}

	// Dropping the wait that guards slot reuse must be caught.
	for i, s := range p.Steps {
		if s.Stage == StageCopy && len(s.Wait) > 0 {
			mut := p
			mut.Steps = append([]StreamStep(nil), p.Steps...)
			mut.Steps[i].Wait = nil
			if err := mut.Validate(); err == nil || !strings.Contains(err.Error(), "overwrites slot") {
				t.Fatalf("Validate without slot-reuse wait at step %d: %v", i, err)
			}
			break
		}
	}
	// So must a repack that does not wait for its copy.
	mut := p
	mut.Steps = append([]StreamStep(nil), p.Steps...)
	mut.Steps[1].Wait = nil
	if err := mut.Validate(); err == nil {
		t.Fatal("Validate accepted a repack that does not wait for its copy")
	}
	if _, err := PlanStream(infos, 1024, 2); err == nil {
		t.Fatal("PlanStream accepted slots smaller than 8 rows")
	}
}

func TestSwizzleRepackCUDA(t *testing.T) {
	src, err := SwizzleRepackCUDA("repack_sw128", 128, 128, 64)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`extern "C" __global__ void repack_sw128(`,
		"span = 128, tile_rows = 64, mask = 7",
		"off ^= ((off >> 7) & mask) << 4;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("source missing %q", want)
		}
	}
	if _, err := SwizzleRepackCUDA("bad name", 128, 128, 64); err == nil {
		t.Error("accepted a bad kernel name")
	}
	if _, err := SwizzleRepackCUDA("k", 128, 64, 64); err == nil {
		t.Error("accepted span != swizzle")
	}
}

func ExampleSwizzleOffset() {
	// Chunk 2 of rows 0..3 under the 128-byte swizzle.
	for r := range 4 {
		fmt.Print(SwizzleOffset(128, r*128+2*16)%128/16, " ")
	}
	fmt.Println()
	// Output: 2 3 0 1
}

func ExampleStreamLoader_PackTensor() {
	raw := make([]uint16, 64*128)
	for i := range raw {
		raw[i] = safetensors.Float32ToBFloat16(float32(i%7) - 3)
	}
	l := StreamLoader{Target: "sm_121a", Format: FormatNVFP4ScaleVec4X}
	b, err := l.PackTensor("model.layers.0.mlp.up_proj.weight", 64, 128, raw)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(b.Format, b.RowBytes, b.Span, b.TileRows, len(b.WeightBytes), len(b.ScaleBytes))
	fmt.Println(b.WeightTMADesc.Box[:3], b.SourceTMADesc.Swizzle)
	// Output:
	// nvfp4-scale-vec-4x 64 64 64 4096 512
	// [32 64 1] 64
}
