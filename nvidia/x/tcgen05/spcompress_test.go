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
	out := make([]uint32, len(b)/4)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return out
}

func TestSpCompressDesc(t *testing.T) {
	ops := []struct {
		op   SpCompressOp
		name string
		code uint32
	}{
		{SpOpMax, "MAX", 0},
		{SpOpMaxAbs, "MAXABS", 1},
		{SpOpMin, "MIN", 2},
		{SpOpMinAbs, "MINABS", 3},
	}
	types := []struct {
		elem SpElemType
		name string
		bits int
		code uint32
	}{
		{SpTypeF16, ".f16", 16, 0},
		{SpTypeBF16, ".bf16", 16, 1},
		{SpTypeE5M2, ".e5m2", 8, 2},
		{SpTypeE4M3, ".e4m3", 8, 3},
		{SpTypeE3M2, ".e3m2", 8, 4},
		{SpTypeE2M3, ".e2m3", 8, 5},
		{SpTypeU8, ".u8", 8, 6},
		{SpTypeS8, ".s8", 8, 7},
	}

	for _, o := range ops {
		if o.op.String() != o.name {
			t.Fatalf("SpCompressOp(%d).String() = %q, want %q", o.op, o.op.String(), o.name)
		}
		for _, et := range types {
			if et.elem.String() != et.name {
				t.Fatalf("SpElemType(%d).String() = %q, want %q", et.elem, et.elem.String(), et.name)
			}
			if et.elem.BitWidth() != et.bits {
				t.Fatalf("SpElemType(%s).BitWidth() = %d, want %d", et.name, et.elem.BitWidth(), et.bits)
			}
			d := SpCompressDesc{Op: o.op, ElemType: et.elem}
			if err := d.Validate(); err != nil {
				t.Fatalf("Validate(%+v) failed: %v", d, err)
			}
			enc := d.Encode()
			wantEnc := o.code | (et.code << 2)
			if enc != wantEnc {
				t.Fatalf("Encode(%+v) = 0x%08x, want 0x%08x", d, enc, wantEnc)
			}
			if (enc >> 5) != 0 {
				t.Fatalf("Encode(%+v) set reserved bits 5..31: 0x%08x", d, enc)
			}
			dec, err := DecodeSpCompressDesc(enc)
			if err != nil {
				t.Fatalf("DecodeSpCompressDesc(0x%08x) failed: %v", enc, err)
			}
			if dec != d {
				t.Fatalf("DecodeSpCompressDesc(0x%08x) = %+v, want %+v", enc, dec, d)
			}
		}
	}

	// Non-zero reserved bits in bits 5..31 must be rejected.
	for bit := 5; bit < 32; bit++ {
		bad := (uint32(1) << uint(bit)) | SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpTypeE4M3}.Encode()
		if _, err := DecodeSpCompressDesc(bad); err == nil {
			t.Fatalf("DecodeSpCompressDesc(0x%08x) with reserved bit %d set succeeded, want error", bad, bit)
		}
	}

	// Out-of-range enum values must fail Validate().
	if err := (SpCompressDesc{Op: SpCompressOp(4), ElemType: SpTypeE4M3}).Validate(); err == nil {
		t.Fatalf("expected error for invalid SpCompressOp(4)")
	}
	if err := (SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpElemType(8)}).Validate(); err == nil {
		t.Fatalf("expected error for invalid SpElemType(8)")
	}
}

func TestRubinSpCompressDecompressKernel(t *testing.T) {
	// Target validation: reject sm_100a and sm_103a; accept sm_107a and sm_107f.
	for _, badTarget := range []string{"sm_100a", "sm_103a", "sm_90a"} {
		badK := RubinSpCompressDecompressKernel{
			NumElements: 256,
			Desc:        SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpTypeE4M3},
			Ratio:       SpRatio2to4,
			Target:      badTarget,
		}
		if err := badK.Validate(); err == nil {
			t.Fatalf("expected error for non-Rubin target %q", badTarget)
		}
	}
	if err := (RubinSpCompressDecompressKernel{NumElements: 48}).Validate(); err == nil {
		t.Fatalf("expected error for non-multiple-of-32 NumElements")
	}
	if err := (RubinSpCompressDecompressKernel{Ratio: "3:5"}).Validate(); err == nil {
		t.Fatalf("expected error for invalid Ratio")
	}

	// Verify all 6 PTX ISA 9.4 §9.7.10.31 sparsity ratios pass validation.
	for _, r := range []string{SpRatio1to2, SpRatio2to4, SpRatio4to8, SpRatio1to4, SpRatio2to8, SpRatio4to16} {
		if err := (RubinSpCompressDecompressKernel{Ratio: r, Target: "sm_107a"}).Validate(); err != nil {
			t.Fatalf("Validate(Ratio=%q) failed: %v", r, err)
		}
	}

	// Deterministic input with positive, negative, zero, and tied values across 256 elements.
	const n = 256
	rng := rand.New(rand.NewPCG(1070, 2026))
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(rng.IntN(33)-16) * 0.25
	}
	// Group 0: distinct signed and absolute values: [-4.0, 1.5, -0.5, 3.0]
	//   MAX:    selects idx 1 (1.5) and idx 3 (3.0)   -> idx0=1, idx1=3
	//   MAXABS: selects idx 0 (-4.0) and idx 3 (3.0)  -> idx0=0, idx1=3
	//   MIN:    selects idx 0 (-4.0) and idx 2 (-0.5) -> idx0=0, idx1=2
	//   MINABS: selects idx 1 (1.5) and idx 2 (-0.5)  -> idx0=1, idx1=2
	in[0], in[1], in[2], in[3] = -4.0, 1.5, -0.5, 3.0
	// Group 1: exact ties to verify deterministic tie-breaking by lower index (0 < 1 < 2 < 3).
	in[4], in[5], in[6], in[7] = -2.0, 2.0, -2.0, 2.0

	wantGroup0 := map[SpCompressOp][2]int{
		SpOpMax:    {1, 3},
		SpOpMaxAbs: {0, 3},
		SpOpMin:    {0, 2},
		SpOpMinAbs: {1, 2},
	}

	for _, op := range []SpCompressOp{SpOpMaxAbs, SpOpMax, SpOpMinAbs, SpOpMin} {
		t.Run(op.String(), func(t *testing.T) {
			k := RubinSpCompressDecompressKernel{
				NumElements: n,
				Desc:        SpCompressDesc{Op: op, ElemType: SpTypeE4M3},
				Ratio:       SpRatio2to4,
				Target:      "sm_107a",
			}
			if err := k.Validate(); err != nil {
				t.Fatalf("Validate() failed: %v", err)
			}
			src, err := k.CUDA()
			if err != nil {
				t.Fatalf("CUDA() failed: %v", err)
			}
			for _, wantPTX := range []string{
				"spcompress.b8.b2.sp::2:4.x4 {%0}, {%1, %2}, {%3, %4, %5, %6}, %7;",
				"spdecompress.b8.b2.sp::2:4.x32",
				"__CUDA_ARCH__ == 1070",
				"eg_rubin_spcompress_roundtrip",
			} {
				if !strings.Contains(src, wantPTX) {
					t.Fatalf("CUDA() missing %q", wantPTX)
				}
			}
			if fullTU, err := k.CUDAFile(); err != nil || !strings.Contains(fullTU, "EG_PRELUDE") {
				t.Fatalf("CUDAFile() failed or missing Prelude: %v", err)
			}

			wantCData, wantMData, wantRecon, err := k.Reference(in)
			if err != nil {
				t.Fatalf("Reference() failed: %v", err)
			}

			// Verify group 0 expected indices for each op.
			g0Indices := wantGroup0[op]
			nib0 := wantMData[0] & 0xf
			wantNib0 := uint32(g0Indices[0]) | (uint32(g0Indices[1]) << 2)
			if nib0 != wantNib0 {
				t.Fatalf("op=%s group 0 nibble = 0x%x, want 0x%x (indices %v)", op, nib0, wantNib0, g0Indices)
			}
			if wantCData[0] != in[g0Indices[0]] || wantCData[1] != in[g0Indices[1]] {
				t.Fatalf("op=%s group 0 cdata = (%v, %v), want (%v, %v)",
					op, wantCData[0], wantCData[1], in[g0Indices[0]], in[g0Indices[1]])
			}

			// Execute via cudahostemu.Run (runEmu) and verify bit-exact agreement with Reference.
			spdesc := k.Desc.Encode()
			grid, block := k.Grid()
			launch := fmt.Sprintf(
				"eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, eg_rubin_spcompress_roundtrip, in_dense, out_cdata, out_mdata, out_reconstructed, %d, 0x%08xu)",
				grid[0], grid[1], grid[2], block[0], block[1], block[2], n, spdesc,
			)
			res := runEmu(t, src, launch,
				emuBuf{"in_dense", "float", f32Bytes(in)},
				emuBuf{"out_cdata", "float", poison(4 * (n / 2))},
				emuBuf{"out_mdata", "unsigned int", poison(4 * (n / 32))},
				emuBuf{"out_reconstructed", "float", poison(4 * n)},
			)

			gotCData := bytesF32(res["out_cdata"])
			gotMData := bytesU32(res["out_mdata"])
			gotRecon := bytesF32(res["out_reconstructed"])

			for i := range wantCData {
				if math.Float32bits(gotCData[i]) != math.Float32bits(wantCData[i]) {
					t.Fatalf("op=%s out_cdata[%d] = %v (0x%08x), want %v (0x%08x)",
						op, i, gotCData[i], math.Float32bits(gotCData[i]), wantCData[i], math.Float32bits(wantCData[i]))
				}
			}
			for i := range wantMData {
				if gotMData[i] != wantMData[i] {
					t.Fatalf("op=%s out_mdata[%d] = 0x%08x, want 0x%08x", op, i, gotMData[i], wantMData[i])
				}
			}
			for i := range wantRecon {
				if math.Float32bits(gotRecon[i]) != math.Float32bits(wantRecon[i]) {
					t.Fatalf("op=%s out_reconstructed[%d] = %v (0x%08x), want %v (0x%08x)",
						op, i, gotRecon[i], math.Float32bits(gotRecon[i]), wantRecon[i], math.Float32bits(wantRecon[i]))
				}
			}
		})
	}
}

func ExampleRubinSpCompressDecompressKernel() {
	desc := SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpTypeE4M3}
	k := RubinSpCompressDecompressKernel{
		NumElements: 32,
		Desc:        desc,
		Ratio:       SpRatio2to4,
		Target:      "sm_107a",
	}
	in := make([]float32, 32)
	in[0], in[1], in[2], in[3] = -4.0, 1.5, -0.5, 3.0
	cdata, mdata, recon, err := k.Reference(in)
	src, _ := k.CUDA()
	fmt.Printf("err=%v spdesc=0x%02x cdata0=(%.1f, %.1f) nib0=0x%x recon0=(%.1f, %.1f, %.1f, %.1f) ptx=%v\n",
		err, desc.Encode(), cdata[0], cdata[1], mdata[0]&0xf,
		recon[0], recon[1], recon[2], recon[3],
		strings.Contains(src, "spcompress.b8.b2.sp::2:4.x4") && strings.Contains(src, "spdecompress.b8.b2.sp::2:4.x32"))
	// Output:
	// err=<nil> spdesc=0x0d cdata0=(-4.0, 3.0) nib0=0xc recon0=(-4.0, 0.0, 0.0, 3.0) ptx=true
}
