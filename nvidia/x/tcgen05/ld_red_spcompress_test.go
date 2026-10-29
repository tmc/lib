package tcgen05

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func bytesU32(b []byte) []uint32 {
	v := make([]uint32, len(b)/4)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return v
}

func TestTCGen05LdRed(t *testing.T) {
	// 1. Target and shape validation: must reject sm_100a and invalid dimensions/ops.
	for _, bad := range []TCGen05LdRed{
		{Target: "sm_100a"},
		{Target: "sm_100f"},
		{Target: "sm_90a"},
		{M: 16, N: 128},
		{M: 128, N: 24},
		{RedOp: "sum"},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("expected error for invalid TCGen05LdRed config %+v, got nil", bad)
		}
	}

	for _, validTarget := range []string{"sm_103a", "sm_103f", "sm_107a", "sm_107f"} {
		cfg := TCGen05LdRed{Target: validTarget}
		if err := cfg.Validate(); err != nil {
			t.Fatalf("Validate(%s) failed: %v", validTarget, err)
		}
	}

	// 2. Verify PTX emission for max + .abs + .NaN and min.
	kMaxAbsNaN := TCGen05LdRed{
		M:      128,
		N:      128,
		RedOp:  LdRedMax,
		Abs:    true,
		NaN:    true,
		Target: "sm_107a",
	}
	srcMax, err := kMaxAbsNaN.CUDA()
	if err != nil {
		t.Fatalf("kMaxAbsNaN.CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.NaN.f32",
		"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
		"tcgen05.wait::ld.sync.aligned;",
		"eg_tcgen05_ld_red_f32",
	} {
		if !strings.Contains(srcMax, want) {
			t.Fatalf("kMaxAbsNaN.CUDA() missing %q", want)
		}
	}
	if fullTU, err := kMaxAbsNaN.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
		t.Fatalf("kMaxAbsNaN.CUDAFile() failed or missing Prelude: %v", err)
	}

	kMin := TCGen05LdRed{
		M:      128,
		N:      128,
		RedOp:  LdRedMin,
		Abs:    false,
		NaN:    false,
		Target: "sm_103a",
	}
	srcMin, err := kMin.CUDA()
	if err != nil {
		t.Fatalf("kMin.CUDA() failed: %v", err)
	}
	if !strings.Contains(srcMin, "tcgen05.ld.red.sync.aligned.32x32b.x4.min.f32") {
		t.Fatalf("kMin.CUDA() missing tcgen05.ld.red.sync.aligned.32x32b.x4.min.f32")
	}

	// 3. Bit-exact host emulation vs Reference for both max (.abs) and min.
	const (
		M = 128
		N = 128
	)
	rng := rand.New(rand.NewPCG(1301, 1302))
	in := make([]float32, M*N)
	for i := range in {
		in[i] = float32(rng.IntN(41)-20) * 0.25
	}
	// Inject a large negative value in row 0 so max(.abs) picks |-19.5| = 19.5
	// while min picks -19.5.
	in[0*N+7] = -19.5
	// Inject a NaN in row 3 to test .NaN propagation when NaN=true.
	inWithNaN := make([]float32, len(in))
	copy(inWithNaN, in)
	inWithNaN[3*N+11] = float32(math.NaN())

	cases := []struct {
		name string
		cfg  TCGen05LdRed
		in   []float32
	}{
		{
			name: "max_abs_nan_sm107a",
			cfg:  kMaxAbsNaN,
			in:   inWithNaN,
		},
		{
			name: "min_signed_sm103a",
			cfg:  kMin,
			in:   in,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src, err := tc.cfg.CUDA()
			if err != nil {
				t.Fatalf("CUDA() failed: %v", err)
			}
			wantData, wantRed, err := tc.cfg.Reference(tc.in)
			if err != nil {
				t.Fatalf("Reference() failed: %v", err)
			}

			grid, block := tc.cfg.Grid()
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, eg_tcgen05_ld_red_f32, in_tmem, out_data, out_row_red, %d, %d)",
				grid[0], grid[1], grid[2], block[0], block[1], block[2], M, N)
			res := runEmu(t, src, launch,
				emuBuf{"in_tmem", "float", f32Bytes(tc.in)},
				emuBuf{"out_data", "float", poison(4 * M * N)},
				emuBuf{"out_row_red", "float", poison(4 * M)},
			)
			gotData := bytesF32(res["out_data"])
			gotRed := bytesF32(res["out_row_red"])

			for i := range wantData {
				if math.IsNaN(float64(wantData[i])) {
					if !math.IsNaN(float64(gotData[i])) {
						t.Fatalf("out_data[%d] = %v, want NaN", i, gotData[i])
					}
					continue
				}
				if math.Float32bits(gotData[i]) != math.Float32bits(wantData[i]) {
					t.Fatalf("out_data[%d] = %v (0x%08x), want %v (0x%08x)",
						i, gotData[i], math.Float32bits(gotData[i]),
						wantData[i], math.Float32bits(wantData[i]))
				}
			}
			for r := range wantRed {
				if math.IsNaN(float64(wantRed[r])) {
					if !math.IsNaN(float64(gotRed[r])) {
						t.Fatalf("out_row_red[%d] = %v, want NaN", r, gotRed[r])
					}
					continue
				}
				if math.Float32bits(gotRed[r]) != math.Float32bits(wantRed[r]) {
					t.Fatalf("out_row_red[%d] = %v (0x%08x), want %v (0x%08x)",
						r, gotRed[r], math.Float32bits(gotRed[r]),
						wantRed[r], math.Float32bits(wantRed[r]))
				}
			}
		})
	}
}

func TestTCGen05LdSpCompress(t *testing.T) {
	// 1. Target validation: must reject sm_100a and sm_103a, and accept sm_107a/sm_107f.
	for _, bad := range []TCGen05LdSpCompress{
		{Target: "sm_100a"},
		{Target: "sm_100f"},
		{Target: "sm_103a"},
		{Target: "sm_103f"},
		{M: 128, N: 6},
		{RedOp: "avg"},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("expected error for invalid TCGen05LdSpCompress config %+v, got nil", bad)
		}
	}

	for _, validTarget := range []string{"sm_107a", "sm_107f"} {
		cfg := TCGen05LdSpCompress{Target: validTarget}
		if err := cfg.Validate(); err != nil {
			t.Fatalf("Validate(%s) failed: %v", validTarget, err)
		}
	}

	// 2. Verify PTX emission with and without .red.
	kRed := TCGen05LdSpCompress{
		M:             128,
		N:             128,
		RedOp:         LdRedMax,
		Abs:           true,
		WithReduction: true,
		Target:        "sm_107a",
	}
	srcRed, err := kRed.CUDA()
	if err != nil {
		t.Fatalf("kRed.CUDA() failed: %v", err)
	}
	for _, want := range []string{
		"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
		"eg_tcgen05_ld_spcompress_f32",
		"#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070",
	} {
		if !strings.Contains(srcRed, want) {
			t.Fatalf("kRed.CUDA() missing %q", want)
		}
	}
	if fullTU, err := kRed.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
		t.Fatalf("kRed.CUDAFile() failed or missing Prelude: %v", err)
	}

	kNoRed := TCGen05LdSpCompress{
		M:             128,
		N:             128,
		RedOp:         LdRedMin,
		Abs:           false,
		WithReduction: false,
		Target:        "sm_107f",
	}
	srcNoRed, err := kNoRed.CUDA()
	if err != nil {
		t.Fatalf("kNoRed.CUDA() failed: %v", err)
	}
	if !strings.Contains(srcNoRed, "tcgen05.ld.spcompress.sync.aligned.32x32b.x4.min.sp::2:4.f32.b2") {
		t.Fatalf("kNoRed.CUDA() missing tcgen05.ld.spcompress.sync.aligned.32x32b.x4.min.sp::2:4.f32.b2")
	}

	// 3. Bit-exact cudahostemu.Run vs Reference on out_cdata, out_mdata, and out_row_red.
	const (
		M = 128
		N = 128
	)
	rng := rand.New(rand.NewPCG(2027, 1070))
	in := make([]float32, M*N)
	for i := range in {
		in[i] = float32(rng.IntN(33)-16) * 0.25
	}
	// Inject explicit magnitude ties in row 0 group 0 to test tie-breaking by smaller column index (0 < 1 < 2 < 3).
	in[0] = -3.5
	in[1] = 3.5
	in[2] = -3.5
	in[3] = 3.5
	// Inject distinct top-2 magnitudes at indices (1, 3) in row 0 group 1.
	in[4] = 0.5
	in[5] = -7.25
	in[6] = -1.0
	in[7] = 5.5

	wantCData, wantMData, wantRowRed, err := kRed.Reference(in)
	if err != nil {
		t.Fatalf("kRed.Reference() failed: %v", err)
	}

	// Verify tie-breaking in group 0 (indices 0 and 1 selected -> nibble 0 | (1<<2) = 0x4)
	// and group 1 (indices 1 and 3 selected -> nibble 1 | (3<<2) = 0xd).
	if nib0 := wantMData[0] & 0xf; nib0 != 0x4 {
		t.Fatalf("group 0 metadata nibble = 0x%x, want 0x4 (idx0=0, idx1=1)", nib0)
	}
	if nib1 := (wantMData[0] >> 4) & 0xf; nib1 != 0xd {
		t.Fatalf("group 1 metadata nibble = 0x%x, want 0xd (idx0=1, idx1=3)", nib1)
	}
	if wantCData[2] != -7.25 || wantCData[3] != 5.5 {
		t.Fatalf("group 1 compressed values = (%v, %v), want (-7.25, 5.5)", wantCData[2], wantCData[3])
	}

	mdataWords := kRed.MetadataWordsPerRow()
	grid, block := kRed.Grid()
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, eg_tcgen05_ld_spcompress_f32, in_tmem, out_cdata, out_mdata, out_row_red, %d, %d)",
		grid[0], grid[1], grid[2], block[0], block[1], block[2], M, N)
	res := runEmu(t, srcRed, launch,
		emuBuf{"in_tmem", "float", f32Bytes(in)},
		emuBuf{"out_cdata", "float", poison(4 * M * (N / 2))},
		emuBuf{"out_mdata", "uint32_t", poison(4 * M * mdataWords)},
		emuBuf{"out_row_red", "float", poison(4 * M)},
	)
	gotCData := bytesF32(res["out_cdata"])
	gotMData := bytesU32(res["out_mdata"])
	gotRowRed := bytesF32(res["out_row_red"])

	for i := range wantCData {
		if math.Float32bits(gotCData[i]) != math.Float32bits(wantCData[i]) {
			t.Fatalf("out_cdata[%d] = %v (0x%08x), want %v (0x%08x)",
				i, gotCData[i], math.Float32bits(gotCData[i]),
				wantCData[i], math.Float32bits(wantCData[i]))
		}
	}
	for i := range wantMData {
		if gotMData[i] != wantMData[i] {
			t.Fatalf("out_mdata[%d] = 0x%08x, want 0x%08x", i, gotMData[i], wantMData[i])
		}
	}
	for r := range wantRowRed {
		if math.Float32bits(gotRowRed[r]) != math.Float32bits(wantRowRed[r]) {
			t.Fatalf("out_row_red[%d] = %v (0x%08x), want %v (0x%08x)",
				r, gotRowRed[r], math.Float32bits(gotRowRed[r]),
				wantRowRed[r], math.Float32bits(wantRowRed[r]))
		}
	}

	// 4. Verify DecompressReference preserves the top-2 magnitudes per 4-element group.
	decompressed, err := kRed.DecompressReference(gotCData, gotMData, M, N)
	if err != nil {
		t.Fatalf("DecompressReference() failed: %v", err)
	}
	if len(decompressed) != M*N {
		t.Fatalf("len(decompressed) = %d, want %d", len(decompressed), M*N)
	}
	for r := range M {
		for g := range N / 4 {
			base := r*N + 4*g
			nib := (gotMData[r*mdataWords+(g>>3)] >> (4 * (g & 7))) & 0xf
			idx0 := int(nib & 0x3)
			idx1 := int((nib >> 2) & 0x3)
			if idx0 >= idx1 {
				t.Fatalf("row %d group %d invalid metadata indices (%d, %d)", r, g, idx0, idx1)
			}
			for j := range 4 {
				gotV := decompressed[base+j]
				if j == idx0 || j == idx1 {
					wantV := in[base+j]
					if math.Float32bits(gotV) != math.Float32bits(wantV) {
						t.Fatalf("row %d group %d selected col %d = %v, want %v", r, g, j, gotV, wantV)
					}
				} else {
					if gotV != 0 {
						t.Fatalf("row %d group %d pruned col %d = %v, want 0", r, g, j, gotV)
					}
					prunedMag := math.Abs(float64(in[base+j]))
					keptMag0 := math.Abs(float64(in[base+idx0]))
					keptMag1 := math.Abs(float64(in[base+idx1]))
					if prunedMag > keptMag0 || prunedMag > keptMag1 {
						t.Fatalf("row %d group %d pruned magnitude %v exceeds kept magnitudes (%v, %v)",
							r, g, prunedMag, keptMag0, keptMag1)
					}
				}
			}
		}
	}
}

func ExampleTCGen05LdSpCompress() {
	sp := TCGen05LdSpCompress{
		M:             1,
		N:             8,
		RedOp:         LdRedMax,
		Abs:           true,
		WithReduction: true,
		Target:        "sm_107a",
	}
	in := []float32{
		1.0, -4.0, 2.5, -0.5, // group 0: top-2 |x| at indices 1 (-4.0) and 2 (2.5)
		-6.0, 1.5, -1.0, 3.0, // group 1: top-2 |x| at indices 0 (-6.0) and 3 (3.0)
	}
	cdata, mdata, rowRed, err := sp.Reference(in)
	dense, _ := DecompressReference(cdata, mdata, 1, 8)
	src, _ := sp.CUDA()
	fmt.Printf("err=%v cdata=%v mdata=0x%02x rowRed=%.1f dense=%v has_ptx=%v\n",
		err,
		cdata,
		mdata[0],
		rowRed[0],
		dense,
		strings.Contains(src, "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2"),
	)
	// Output:
	// err=<nil> cdata=[-4 2.5 -6 3] mdata=0xc9 rowRed=6.0 dense=[0 -4 2.5 0 -6 0 0 3] has_ptx=true
}
