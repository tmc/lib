package advise

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The fixtures below are trimmed from nvdisasm and nvcc -ptx output for one
// kernel compiled for sm_121 by CUDA 13.4, with and without --use_fast_math:
//
//	extern "C" __global__ void rope(float *x, const float *t, int n) {
//		int i = blockIdx.x * blockDim.x + threadIdx.x;
//		if (i >= n) return;
//		float a = t[i] / (float)(i + 1);
//		float s, c;
//		sincosf(a, &s, &c);
//		x[i] = x[i] * c + s * expf(-a) + sqrtf(a);
//	}
//
// cuobjdump -res-usage reports STACK:32 for the precise build and STACK:0 for
// the fast one. preciseSASS drops named parts of the precise listing to build
// the cases that must not fire.

const (
	sassHeader = `	.section	.text.rope,"ax",@progbits
        .type           rope,@function
rope:
.text.rope:
`
	// The division: MUFU.RCP, Newton steps, FCHK and the slow-path call.
	sassDivision = `        /*0100*/                   MUFU.RCP R4, R3 ;
        /*0110*/                   FFMA R5, -R3, R4, 1 ;
        /*0120*/                   FFMA R5, R4, R5, R4 ;
        /*0130*/                   FCHK P0, R0, R3 ;
        /*0140*/                   FFMA R4, R0, R5, RZ ;
        /*0170*/              @!P0 BRA ` + "`(.L_x_16)" + ` ;
        /*0180*/                   MOV R2, 0x1a0 ;
        /*0190*/                   CALL.REL.NOINC ` + "`($__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath)" + ` ;
.L_x_16:
        /*01a0*/                   NOP ;
`
	// The sincosf argument reduction's guard.
	sassGuard = `        /*01f0*/                   FMUL R0, R15.reuse, 0.63661974668502807617 ;
        /*0210*/                   FSETP.GE.AND P0, PT, |R15|.reuse, 105615, PT ;
        /*02f0*/              @!P0 BRA ` + "`(.L_x_18)" + ` ;
`
	// The Payne-Hanek slow path's local array.
	sassLocal = `        /*0460*/                   STL.64 [R1], R2 ;
        /*04d0*/                   STL.64 [R1+0x8], R8 ;
        /*0550*/                   LDL R0, [R19+0x18] ;
        /*0560*/                   LDL R7, [R19+0x14] ;
`
	sassTail = `        /*0680*/                   DMUL R2, R6, UR6 ;
        /*0690*/                   F2F.F32.F64 R2, R2 ;
.L_x_18:
        /*07e0*/                   MUFU.RSQ R4, R15 ;
        /*0870*/              @!P0 BRA ` + "`(.L_x_20)" + ` ;
        /*08a0*/                   CALL.REL.NOINC ` + "`($__internal_1_$__cuda_sm20_sqrt_rn_f32_slowpath)" + ` ;
.L_x_20:
        /*08c0*/                   FMUL.FTZ R0, R4.reuse, R15 ;
        /*0920*/                   FADD R3, R9, R0 ;
        /*0930*/                   STG.E desc[UR4][R12.64], R3 ;
        /*0940*/                   EXIT ;
        .type           $__internal_1_$__cuda_sm20_sqrt_rn_f32_slowpath,@function
$__internal_1_$__cuda_sm20_sqrt_rn_f32_slowpath:
        /*09c0*/               @P0 FADD.FTZ R2, R0, 1 ;
        /*0a10*/               @P0 FMUL.FTZ R4, R3, R2 ;
        /*0ab0*/                   RET.REL.NODEC R2 ` + "`(rope)" + ` ;
        .type           $__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath,@function
$__internal_2_$__cuda_sm3x_div_rn_noftz_f32_slowpath:
        /*0b90*/                   FSETP.GTU.FTZ.AND P0, PT, |R0|, +INF , PT ;
        /*0cd0*/              @!P0 FFMA R6, R0, 1.84467440737095516160e+19, RZ ;
        /*0d00*/                   RET.REL.NODEC R2 ` + "`(rope)" + ` ;
`

	// sassFast is the complete --use_fast_math listing.
	sassFast = sassHeader + `        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*00f0*/                   IADD R7, R7, 0x1 ;
        /*0100*/                   I2FP.F32.S32 R7, R7 ;
        /*0110*/                   MUFU.RCP R7, R7 ;
        /*0120*/                   FMUL.FTZ R0, R2, R7 ;
        /*0130*/                   FMUL.RZ R11, R0.reuse, 0.15915493667125701904 ;
        /*0140*/                   FMUL.FTZ R10, R0, -1.4426950216293334961 ;
        /*0150*/                   MUFU.SQRT R3, R0 ;
        /*0160*/                   MUFU.COS R8, R11 ;
        /*0170*/                   MUFU.SIN R6, R11 ;
        /*0180*/                   MUFU.EX2 R10, R10 ;
        /*0190*/                   FMUL.FTZ R9, R8, R9 ;
        /*01a0*/                   FFMA.FTZ R6, R6, R10, R9 ;
        /*01b0*/                   FADD.FTZ R3, R6, R3 ;
        /*01c0*/                   STG.E desc[UR4][R4.64], R3 ;
        /*01d0*/                   EXIT ;
`

	// sassInteger is an integer-only kernel whose division uses MUFU.RCP,
	// with local memory added so that only the absence of float math can
	// keep it from firing.
	sassInteger = sassHeader + `        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0100*/                   I2F.U32.RP R6, R5 ;
        /*0110*/                   MUFU.RCP R6, R6 ;
        /*0120*/                   IADD R4, R6, 0xffffffe ;
        /*0130*/                   F2I.FTZ.U32.TRUNC.NTZ R5, R4 ;
        /*0140*/                   STL [R1], R5 ;
        /*0150*/                   LDL R5, [R1] ;
        /*01c0*/                   STG.E desc[UR4][R4.64], R3 ;
        /*01d0*/                   EXIT ;
`

	// ptxPrecise and ptxFast are the kernel's PTX, trimmed.
	ptxPrecise = `.version 9.0
.target sm_121
.address_size 64

.visible .entry rope(
	.param .u64 .ptr .align 1 rope_param_0,
	.param .u64 .ptr .align 1 rope_param_1,
	.param .u32 rope_param_2
)
{
	.local .align 4 .b8 	__local_depot0[28];
	.reg .pred 	%p<9>;
	.reg .b32 	%r<81>;
	.reg .b64 	%rd<26>;

	cvt.rn.f32.s32 	%r11, %r10;
	div.rn.f32 	%r2, %r9, %r11;
	mul.f32 	%r12, %r2, 0f3F22F983;
	abs.f32 	%r3, %r2;
	setp.ltu.f32 	%p2, %r3, 0f47CE4780;
	st.local.b32 	[%rd7], %rd9;
	ld.local.b32 	%r78, [%rd1+24];
	mul.f32 	%r42, %r81, %r81;
	add.f32 	%r67, %r66, 0fCB40007F;
	ex2.approx.ftz.f32 	%r72, %r70;
	sqrt.rn.f32 	%r76, %r2;
	add.f32 	%r77, %r76, %r75;
	ret;
}
`
	ptxFast = `.version 9.0
.target sm_121
.address_size 64

.visible .entry rope(
	.param .u64 .ptr .align 1 rope_param_0,
	.param .u64 .ptr .align 1 rope_param_1,
	.param .u32 rope_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<8>;

	cvt.rn.f32.s32 	%r8, %r7;
	div.approx.ftz.f32 	%r9, %r6, %r8;
	sin.approx.ftz.f32 	%r10, %r9;
	cos.approx.ftz.f32 	%r11, %r9;
	mul.ftz.f32 	%r13, %r11, %r12;
	ex2.approx.ftz.f32 	%r15, %r14;
	fma.rn.ftz.f32 	%r16, %r10, %r15, %r13;
	sqrt.approx.ftz.f32 	%r17, %r9;
	add.ftz.f32 	%r18, %r17, %r16;
	ret;
}
`
)

// preciseSASS assembles the precise listing, leaving out the named parts.
func preciseSASS(without ...string) string {
	parts := []struct{ name, text string }{
		{"division", sassDivision},
		{"guard", sassGuard},
		{"local", sassLocal},
	}
	s := sassHeader
	for _, p := range parts {
		keep := true
		for _, w := range without {
			if w == p.name {
				keep = false
			}
		}
		if keep {
			s += p.text
		}
	}
	return s + sassTail
}

func TestFloatMathMarkers(t *testing.T) {
	tests := []struct {
		name string
		sass string
		ptx  string
		want floatMath // at is not compared
	}{
		{
			name: "precise SASS",
			sass: preciseSASS(),
			want: floatMath{tier: "SASS", reductions: 1, divisions: 1, roots: 1, fp32: 6, ftz: 1, local: 4},
		},
		{
			name: "fast SASS",
			sass: sassFast,
			want: floatMath{tier: "SASS", fp32: 6, ftz: 5},
		},
		{
			name: "integer SASS",
			sass: sassInteger,
			want: floatMath{tier: "SASS", local: 2},
		},
		{
			name: "precise PTX",
			ptx:  ptxPrecise,
			want: floatMath{tier: "PTX", reductions: 1, divisions: 1, roots: 1, fp32: 4},
		},
		{
			name: "fast PTX",
			ptx:  ptxFast,
			want: floatMath{tier: "PTX", fp32: 3, ftz: 3},
		},
		{
			name: "SASS reciprocal",
			sass: sassHeader + "        /*0100*/                   CALL.REL.NOINC `($__internal_2_$__cuda_sm20_rcp_rn_f32_slowpath)` ;\n" +
				"        /*0110*/                   EXIT ;\n",
			want: floatMath{tier: "SASS", divisions: 1},
		},
		{
			name: "PTX reciprocal",
			ptx:  strings.Replace(ptxFast, "div.approx.ftz.f32 	%r9, %r6, %r8;", "rcp.rn.f32 	%r9, %r8;", 1),
			want: floatMath{tier: "PTX", divisions: 1, fp32: 3, ftz: 3},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got floatMath
			if tt.sass != "" {
				d, err := cudasass.Parse(strings.NewReader(tt.sass))
				if err != nil {
					t.Fatal(err)
				}
				got = sassFloatMath(d.Function("rope"))
			} else {
				mod, err := cudaptx.ParseBytes([]byte(tt.ptx))
				if err != nil {
					t.Fatal(err)
				}
				got = ptxFloatMath(mod.Kernels[0], "kernels.ptx")
			}
			got.at = Location{}
			if got != tt.want {
				t.Errorf("markers = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestFloatMathCosts(t *testing.T) {
	tests := []struct {
		name  string
		m     floatMath
		frame int
		want  bool
	}{
		{"reduction and local memory", floatMath{reductions: 2, local: 10}, 0, true},
		{"reduction and stack frame", floatMath{reductions: 2}, 40, true},
		{"reduction kept in registers", floatMath{reductions: 4, divisions: 4}, 0, false},
		{"division and unrelated local memory", floatMath{divisions: 17, local: 39}, 40, false},
		{"no float math", floatMath{local: 28}, 32, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.costs(tt.frame); got != tt.want {
				t.Errorf("costs(%d) = %v, want %v", tt.frame, got, tt.want)
			}
		})
	}
}

// frameCUBIN returns a CUBIN whose only kernel, rope, has a stack frame of
// frame bytes.
func frameCUBIN(frame uint32) []byte {
	var nv bytes.Buffer
	nv.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
	binary.Write(&nv, binary.LittleEndian, uint32(26))
	nv.Write([]byte{0x04, cudaoccupancy.EIATTR_FRAME_SIZE, 0x04, 0x00})
	binary.Write(&nv, binary.LittleEndian, frame)
	return synthesizeTestELFForAdvise(map[string][]byte{
		".text.rope":    make([]byte, 32),
		".nv.info.rope": nv.Bytes(),
	})
}

func TestAdvice_PreciseFloatMath(t *testing.T) {
	const finding = "precise float math that costs it local memory"
	tests := []struct {
		name     string
		sass     string
		ptx      string
		frame    uint32 // with a CUBIN when nonzero
		opts     string // recorded compile options
		want     bool
		wantText []string // in the finding when want
		evidence EvidenceLevel
	}{
		{
			name:     "precise SASS with local memory",
			sass:     preciseSASS(),
			want:     true,
			wantText: []string{"(read from SASS)", "1 sinf/cosf argument-reduction guards", "1 IEEE divisions", "1 IEEE square roots", "1 of 6 FP32", "4 local-memory instructions"},
			evidence: EvidenceSuspected,
		},
		{
			name:     "precise SASS with a stack frame",
			sass:     preciseSASS("local"),
			frame:    32,
			want:     true,
			wantText: []string{"a 32-byte stack frame"},
			evidence: EvidenceSuspected,
		},
		{
			name:     "precise PTX with SASS local memory",
			sass:     preciseSASS(),
			ptx:      ptxPrecise,
			want:     true,
			wantText: []string{"(read from PTX)", "0 of 4 FP32", "4 local-memory instructions"},
			evidence: EvidenceSuspected,
		},
		{
			name:     "recorded options without fast math",
			sass:     preciseSASS(),
			opts:     "-arch=sm_121 -lineinfo",
			want:     true,
			wantText: []string{`records compile options "-arch=sm_121 -lineinfo"`},
			evidence: EvidenceObserved,
		},
		{
			// The options cannot be this module's, so they are not evidence.
			name:     "recorded options with fast math",
			sass:     preciseSASS(),
			opts:     "-arch=sm_121 --use_fast_math",
			want:     true,
			wantText: []string{"(read from SASS)"},
			evidence: EvidenceSuspected,
		},
		{name: "fast math", sass: sassFast, frame: 8},
		{name: "fast PTX over precise SASS", sass: preciseSASS(), ptx: ptxFast},
		{name: "integer only", sass: sassInteger, frame: 16},
		{name: "precise math kept in registers", sass: preciseSASS("local")},
		{name: "division without reduction", sass: preciseSASS("guard"), frame: 40},
		{name: "precise PTX without machine code", ptx: ptxPrecise},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files := make(map[string][]byte)
			var code cudacapture.CodeFiles
			if tt.sass != "" {
				code.SASS = "kernels.sass"
				files[code.SASS] = []byte(tt.sass)
			}
			if tt.ptx != "" {
				code.PTX = "kernels.ptx"
				files[code.PTX] = []byte(tt.ptx)
			}
			if tt.frame > 0 {
				code.CUBIN = "kernels.cubin"
				files[code.CUBIN] = frameCUBIN(tt.frame)
			}
			cap := &cudacapture.Capture{Code: code, Files: files}
			if tt.opts != "" {
				cap.Identity = &cudacapture.ComparisonIdentity{CompileOpts: tt.opts}
			}
			report, err := Analyze(cap)
			if err != nil {
				t.Fatal(err)
			}
			var got []Item
			for _, item := range findings(report) {
				if strings.Contains(item.Finding, finding) {
					got = append(got, item)
				}
			}
			if !tt.want {
				if len(got) != 0 {
					t.Fatalf("got a precise float math finding, want none: %+v", got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("got %d precise float math findings, want 1; items: %+v", len(got), report.Items)
			}
			item := got[0]
			for _, s := range tt.wantText {
				if !strings.Contains(item.Finding, s) {
					t.Errorf("finding lacks %q: %s", s, item.Finding)
				}
			}
			if item.EvidenceLevel != tt.evidence {
				t.Errorf("evidence = %s, want %s", item.EvidenceLevel, tt.evidence)
			}
			if item.Location.Kernel != "rope" {
				t.Errorf("location kernel = %q, want rope", item.Location.Kernel)
			}
			for _, s := range []string{"--use_fast_math", "__sinf", "Under nvcc", "changes numerics", "output-agreement gate"} {
				if !strings.Contains(item.ActionableRecommendation, s) {
					t.Errorf("recommendation lacks %q: %s", s, item.ActionableRecommendation)
				}
			}
			if !strings.Contains(item.NextMeasurement, "cuobjdump -res-usage") {
				t.Errorf("next measurement = %q, want cuobjdump -res-usage", item.NextMeasurement)
			}
		})
	}
}

// TestAdvice_LocalMemoryNamesFastMath checks that the stack-frame and spill
// findings point at fast math only for a kernel whose precise math costs it
// that memory.
func TestAdvice_LocalMemoryNamesFastMath(t *testing.T) {
	tests := []struct {
		name string
		sass string
		want bool
	}{
		{"precise", preciseSASS(), true},
		{"division without reduction", preciseSASS("guard"), false},
		{"fast", sassFast + "        /*01e0*/                   STL [R1], R5 ;\n", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass", CUBIN: "kernels.cubin"},
				Files: map[string][]byte{
					"kernels.sass":  []byte(tt.sass),
					"kernels.cubin": frameCUBIN(32),
				},
			}
			report, err := Analyze(cap)
			if err != nil {
				t.Fatal(err)
			}
			var n int
			for _, item := range findings(report) {
				if !strings.Contains(item.Finding, "stack frame per thread") && !strings.Contains(item.Finding, "Register spill point") {
					continue
				}
				n++
				if got := strings.Contains(item.ActionableRecommendation, fastMathLead); got != tt.want {
					t.Errorf("%s: names fast math = %v, want %v", item.Finding, got, tt.want)
				}
			}
			if n != 2 {
				t.Errorf("got %d stack-frame and spill findings, want 2", n)
			}
		})
	}
}
