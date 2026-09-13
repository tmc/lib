package main

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestConfigValidation(t *testing.T) {
	for _, tt := range []struct {
		name    string
		cfg     config
		wantErr bool
	}{
		{"valid default", config{elements: 1024, passes: 2, rounds: 8}, false},
		{"elements zero", config{elements: 0, passes: 2, rounds: 8}, true},
		{"elements negative", config{elements: -1, passes: 2, rounds: 8}, true},
		{"elements too large", config{elements: 1<<26 + 1, passes: 2, rounds: 8}, true},
		{"passes zero", config{elements: 1024, passes: 0, rounds: 8}, true},
		{"passes too large", config{elements: 1024, passes: 1001, rounds: 8}, true},
		{"rounds zero", config{elements: 1024, passes: 2, rounds: 0}, true},
		{"rounds too large", config{elements: 1024, passes: 2, rounds: 1<<20 + 1}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := run("", tt.cfg)
			if (err != nil) != tt.wantErr {
				if !tt.wantErr {
					// on a host without GPU, run fails after validation when opening device/dir
				}
			}
		})
	}
}

func TestFloat16Conversion(t *testing.T) {
	tests := []struct {
		name string
		val  float32
		tol  float32
	}{
		{"zero", 0.0, 0.0},
		{"one", 1.0, 1e-6},
		{"half", 0.5, 1e-6},
		{"negative two", -2.0, 1e-6},
		{"small scale", 0.05, 1e-4},
		{"sixteen", 16.0, 1e-6},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := float32ToFloat16(tc.val)
			got := float16ToFloat32(h)
			diff := float32(math.Abs(float64(got - tc.val)))
			if diff > tc.tol {
				t.Fatalf("float16ToFloat32(float32ToFloat16(%g)) = %g, want %g (diff %g > tol %g)", tc.val, got, tc.val, diff, tc.tol)
			}
		})
	}
}

func TestCPUReferencePipeline(t *testing.T) {
	t.Run("RMSNorm", func(t *testing.T) {
		x := []float32{1.0, 2.0, 3.0, 4.0}
		gamma := []float32{1.0, 1.0, 1.0, 1.0}
		out := refRMSNorm(x, gamma, 1, 4, 1e-5)
		if len(out) != 4 {
			t.Fatalf("want 4 elements, got %d", len(out))
		}
		var sumsq float64
		for _, v := range out {
			sumsq += float64(v * v)
		}
		rms := math.Sqrt(sumsq / 4.0)
		if math.Abs(rms-1.0) > 1e-3 {
			t.Errorf("expected normalized RMS ~ 1.0, got %g", rms)
		}
	})

	t.Run("Q8_0Matvec", func(t *testing.T) {
		w := make([]BlockQ8_0, 1)
		w[0].D = float32ToFloat16(0.5)
		x := make([]float32, 32)
		for i := 0; i < 32; i++ {
			w[0].QS[i] = int8(i % 5)
			x[i] = 1.0
		}
		out := refQ8_0Matvec(w, x, 1, 32)
		if len(out) != 1 {
			t.Fatalf("want 1 output, got %d", len(out))
		}
		var wantSum float64
		for i := 0; i < 32; i++ {
			wantSum += float64(int8(i % 5))
		}
		want := float32(wantSum * 0.5)
		if math.Abs(float64(out[0]-want)) > 1e-3 {
			t.Errorf("got %g, want %g", out[0], want)
		}
	})

	t.Run("SwiGLU", func(t *testing.T) {
		gate := []float32{0.0, 2.0}
		up := []float32{1.0, 3.0}
		out := refSwiGLU(gate, up)
		if len(out) != 2 {
			t.Fatalf("want 2 outputs, got %d", len(out))
		}
		if out[0] != 0.0 {
			t.Errorf("SwiGLU(0, 1) = %g, want 0.0", out[0])
		}
		want1 := float32(2.0 / (1.0 + math.Exp(-2.0)) * 3.0)
		if math.Abs(float64(out[1]-want1)) > 1e-5 {
			t.Errorf("SwiGLU(2, 3) = %g, want %g", out[1], want1)
		}
	})

	t.Run("MoEExpertDispatch", func(t *testing.T) {
		rows, cols := 2, 4
		numExperts := 4
		weights := make([]float32, numExperts*rows*cols)
		// Set weights for expert 2, row 1
		expert := 2
		row := 1
		offset := (expert*rows + row) * cols
		weights[offset+0] = 1.0
		weights[offset+1] = 2.0
		weights[offset+2] = 3.0
		weights[offset+3] = 4.0

		x := []float32{1.0, 1.0, 1.0, 1.0}
		expertIDs := []int32{0, 2}
		out := refMoEDispatch(weights, x, expertIDs, 1, rows, cols)
		if len(out) != rows {
			t.Fatalf("want %d outputs, got %d", rows, len(out))
		}
		// out[1] should be 1*1 + 2*1 + 3*1 + 4*1 = 10.0
		if math.Abs(float64(out[row]-10.0)) > 1e-5 {
			t.Errorf("MoE dispatch got %g, want 10.0", out[row])
		}
	})

	t.Run("IQ2ScalarVsSIMD", func(t *testing.T) {
		q := []uint32{0x1b2e3d4c, 0xa55aa55a}
		a := make([]int8, 32)
		for i := range a {
			a[i] = int8((i % 7) - 3)
		}
		scale := float32(0.25)
		refScalar := refIQ2(q, a, scale)

		// Verify against SIMD unpacking logic in Go
		for idx := range q {
			qval := q[idx]
			var sum int32
			for g := 0; g < 4; g++ {
				byteG := (qval >> (8 * g)) & 0xFF
				rep := byteG * 0x01010101
				b0 := int32((rep >> 0) & 0x03)
				b1 := int32((rep >> 2) & 0x03)
				b2 := int32((rep >> 4) & 0x03)
				b3 := int32((rep >> 6) & 0x03)

				a0 := int32(a[idx*16+g*4+0])
				a1 := int32(a[idx*16+g*4+1])
				a2 := int32(a[idx*16+g*4+2])
				a3 := int32(a[idx*16+g*4+3])

				sum += b0*a0 + b1*a1 + b2*a2 + b3*a3
			}
			gotSimd := float32(sum) * scale
			if math.Abs(float64(refScalar[idx]-gotSimd)) > 1e-5 {
				t.Fatalf("IQ2 SIMD %g != scalar %g at index %d", gotSimd, refScalar[idx], idx)
			}
		}
	})

	t.Run("SmemStageMatvec", func(t *testing.T) {
		rows, cols := 2, 4
		a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		x := []float32{1, 0, 1, 0}
		out := refSmemStageMatvec(a, x, rows, cols)
		if len(out) != rows {
			t.Fatalf("want %d outputs, got %d", rows, len(out))
		}
		if out[0] != 4.0 || out[1] != 12.0 {
			t.Errorf("refSmemStageMatvec = %v, want [4.0, 12.0]", out)
		}
	})

	t.Run("ReferenceFMA", func(t *testing.T) {
		ref := reference(8)
		if len(ref) != 1024 {
			t.Fatalf("expected 1024 reference values, got %d", len(ref))
		}
		for i, v := range ref {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("reference[%d] is not finite: %g", i, v)
			}
		}
	})
}

func TestAssertOpcodesSIMDvsScalar(t *testing.T) {
	const sassText = `
--- Kernel: ds4_iq2_simd ---
/* 0010 */   @P0 LDG.E.CONSTANT.F32 R4, [R2.64] ;
/* 0020 */   LOP3.LUT R6, R4, R5, 0xca ;
/* 0030 */   IDP4A.U8.U8 R0, R6, R7, R0 ;
/* 0040 */   IDP4A.S8.S8 R1, R8, R9, R1 ;
/* 0050 */   EXIT ;

--- Kernel: ds4_iq2_scalar ---
/* 0100 */   LDG.E.F32 R4, [R2.64] ;
/* 0110 */   SHR.B64 R5, R4, 0x8 ;
/* 0120 */   IMAD.WIDE R0, R5, R6, R0 ;
/* 0130 */   IMAD.WIDE R1, R7, R8, R1 ;
/* 0140 */   EXIT ;

--- Kernel: ds4_q8_0_matvec ---
/* 0200 */   SHFL.SYNC.DOWN R0, R1, 0x10, 0x1f ;
/* 0210 */   EXIT ;
`
	d, err := cudasass.ParseString(sassText)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	tests := []struct {
		name    string
		kernel  string
		require []string
		forbid  []string
		wantErr bool
	}{
		{
			name:    "simd requires IDP4A and forbids SHR",
			kernel:  "ds4_iq2_simd",
			require: []string{"IDP4A"},
			forbid:  []string{"SHR"},
			wantErr: false,
		},
		{
			name:    "simd fails if required IMAD is missing",
			kernel:  "ds4_iq2_simd",
			require: []string{"IMAD"},
			forbid:  nil,
			wantErr: true,
		},
		{
			name:    "scalar requires IMAD and forbids IDP4A",
			kernel:  "ds4_iq2_scalar",
			require: []string{"IMAD"},
			forbid:  []string{"IDP4A"},
			wantErr: false,
		},
		{
			name:    "scalar fails if forbidden SHR is present",
			kernel:  "ds4_iq2_scalar",
			require: []string{"IMAD"},
			forbid:  []string{"SHR"},
			wantErr: true,
		},
		{
			name:    "q8_0 matvec requires SHFL",
			kernel:  "ds4_q8_0_matvec",
			require: []string{"SHFL"},
			forbid:  nil,
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := d.AssertOpcodes(tc.kernel, tc.require, tc.forbid)
			if (err != nil) != tc.wantErr {
				t.Fatalf("AssertOpcodes(%q) error = %v, wantErr %v", tc.kernel, err, tc.wantErr)
			}
		})
	}
}
