package cublaslog

import (
	"strings"
	"testing"
)

// Lines logged by libcublasLt 13 on GB10 for torch's a@w.t() (through
// cuBLAS) and F.linear with a bias (cublasLtMatmul directly).
const (
	tstLine  = "A=0X32F600000 Adesc=[type=R_16BF rows=1024 cols=768 ld=1024] B=0X32EE00000 Bdesc=[type=R_16BF rows=1024 cols=512 ld=1024] C=0X32EF00600 Cdesc=[type=R_16BF rows=768 cols=512 ld=768] D=0X32EF00600 Ddesc=[type=R_16BF rows=768 cols=512 ld=768] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T smCountTarget=48] algo=[algoId=67 tile=MATMUL_TILE_64x184 stages=MATMUL_STAGES_64xAUTO customOption=29] workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=0 stream=0X0"
	biasLine = "A=0X32F600000 Adesc=[type=R_16BF rows=1024 cols=768 ld=1024] B=0X32EE00000 Bdesc=[type=R_16BF rows=1024 cols=512 ld=1024] C=0X333400000 Cdesc=[type=R_16BF rows=768 cols=512 ld=768] D=0X333400000 Ddesc=[type=R_16BF rows=768 cols=512 ld=768] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T epilogue=EPILOGUE_BIAS biasPointer=0x32ef00000] algo=[algoId=21 tile=MATMUL_TILE_16x16 stages=MATMUL_STAGES_32x1] workSpace=0X331400000 workSpaceSizeInBytes=1048576 beta=0 outOfPlace=0 stream=0X0"
)

func TestParse(t *testing.T) {
	tests := []struct {
		name  string
		line  string
		shape [4]int64 // M, N, K, batch
		op    string
		types string
		epi   string
		beta  string
		algo  string
	}{
		{"cublas internal", tstLine, [4]int64{768, 512, 1024, 1}, "TN", "R_16BF,R_16BF->R_16BF", "DEFAULT", "0", "67 64x184/64xAUTO customOption=29"},
		{"bias epilogue", biasLine, [4]int64{768, 512, 1024, 1}, "TN", "R_16BF,R_16BF->R_16BF", "BIAS", "0", "21 16x16/32x1"},
		{
			"batched split-K accumulate",
			"Adesc=[type=R_32F rows=64 cols=32 ld=64 batchCount=8 strideBatch=2048] Bdesc=[type=R_32F rows=16 cols=32 ld=16] Cdesc=[type=R_16F rows=64 cols=16 ld=64] Ddesc=[type=R_32F rows=64 cols=16 ld=64] computeDesc=[computeType=COMPUTE_32F_FAST_TF32 scaleType=R_32F transb=OP_T] algo=[algoId=6 tile=MATMUL_TILE_64x64 stages=MATMUL_STAGES_16x5 splitK=4 reductionScheme=CUBLASLT_REDUCTION_SCHEME_INPLACE] beta=1 stream=0X1",
			[4]int64{64, 16, 32, 8}, "NT", "R_32F,R_32F,R_16F->R_32F", "DEFAULT", "1",
			"6 64x64/16x5 splitK=4 reductionScheme=CUBLASLT_REDUCTION_SCHEME_INPLACE",
		},
		{
			"no Cdesc",
			"Adesc=[type=R_16BF rows=8 cols=4 ld=8] Bdesc=[type=R_16BF rows=4 cols=2 ld=4] Ddesc=[type=R_16BF rows=8 cols=2 ld=8] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F] algo=[algoId=1]",
			[4]int64{8, 2, 4, 1}, "NN", "R_16BF,R_16BF->R_16BF", "DEFAULT", "", "1",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := Parse(tt.line)
			if err != nil {
				t.Fatal(err)
			}
			if got := [4]int64{m.M, m.N, m.K, m.Batch}; got != tt.shape {
				t.Errorf("M, N, K, Batch = %v, want %v", got, tt.shape)
			}
			if got := m.TransA + m.TransB; got != tt.op {
				t.Errorf("op = %s, want %s", got, tt.op)
			}
			if got := m.Types(); got != tt.types {
				t.Errorf("Types() = %s, want %s", got, tt.types)
			}
			if m.Epilogue != tt.epi || m.Beta != tt.beta {
				t.Errorf("epilogue, beta = %s, %q; want %s, %q", m.Epilogue, m.Beta, tt.epi, tt.beta)
			}
			if got := m.Algo.String(); got != tt.algo {
				t.Errorf("Algo = %q, want %q", got, tt.algo)
			}
		})
	}
}

func TestParseFields(t *testing.T) {
	m, err := Parse(biasLine)
	if err != nil {
		t.Fatal(err)
	}
	if m.Compute != "COMPUTE_32F" || m.Scale != "R_32F" || m.Stream != "0X0" {
		t.Errorf("compute, scale, stream = %s, %s, %s", m.Compute, m.Scale, m.Stream)
	}
	if m.A.LD != 1024 || m.B.LD != 1024 || m.C.LD != 768 || m.D.Order != "COL" {
		t.Errorf("layouts = %+v %+v %+v %+v", m.A, m.B, m.C, m.D)
	}
	if got := m.Fields["workSpaceSizeInBytes"]; got != "1048576" {
		t.Errorf("Fields[workSpaceSizeInBytes] = %q", got)
	}
	if !strings.Contains(m.Fields["computeDesc"], "biasPointer=0x32ef00000") {
		t.Errorf("Fields[computeDesc] = %q", m.Fields["computeDesc"])
	}
}

func TestParseRefuses(t *testing.T) {
	tests := []struct {
		name, line, want string
	}{
		{"empty", "", "no Adesc"},
		{"unclosed group", "Adesc=[type=R_16BF rows=1", "unclosed group"},
		{"no key", "=3", "malformed field"},
		{"no rows", "Adesc=[type=R_16BF cols=1 ld=1] Bdesc=[type=R_16BF rows=1 cols=1 ld=1]", "Adesc: no rows"},
		{"no output", "Adesc=[type=R_16BF rows=1 cols=1 ld=1] Bdesc=[type=R_16BF rows=1 cols=1 ld=1]", "no Cdesc or Ddesc"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.line)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("Parse error = %v, want %q", err, tt.want)
			}
		})
	}
}
