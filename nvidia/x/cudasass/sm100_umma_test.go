package cudasass

import (
	"os"
	"testing"
)

// TestSM100UMMAFixtures classifies real ptxas 13.4 output: an NVFP4
// tcgen05 GEMM (egraph/hopper UMMAGEMM, decode form) compiled for sm_100a
// and sm_103a and dumped with cuobjdump -sass, encodings removed.
func TestSM100UMMAFixtures(t *testing.T) {
	tests := []struct {
		file   string
		mma    string // the block-scaled MMA spelling
		counts map[SM100Kind]int
	}{
		{"testdata/sm100a-umma-13.4.sass", "UTCOMMA.4X", map[SM100Kind]int{SM100UTMMA: 4, SM100UTMALDG: 8, SM100UTMACP: 8, SM100UTMAFlush: 2, SM100LDTM: 1}},
		{"testdata/sm103a-umma-13.4.sass", "UTCOMMA.BLOCK16", map[SM100Kind]int{SM100UTMMA: 4, SM100UTMALDG: 8, SM100UTMACP: 8, SM100UTMAFlush: 2, SM100LDTM: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			b, err := os.ReadFile(tt.file)
			if err != nil {
				t.Fatal(err)
			}
			d, err := ParseString(string(b))
			if err != nil {
				t.Fatal(err)
			}
			if len(d.Functions) != 1 {
				t.Fatalf("%d functions, want 1", len(d.Functions))
			}
			fn := *d.Functions[0]
			s := SummarizeSM100Function(fn)
			for kind, want := range tt.counts {
				if got := s.Count(kind); got != want {
					t.Errorf("%s: %d, want %d", kind, got, want)
				}
			}
			if s.BlockScaled != s.UTMMA || s.ScaleVec4X != s.UTMMA {
				t.Errorf("%d MMAs, %d block-scaled, %d scale_vec::4X; want all", s.UTMMA, s.BlockScaled, s.ScaleVec4X)
			}
			found := false
			for _, in := range fn.Instructions {
				found = found || in.Opcode == tt.mma
			}
			if !found {
				t.Errorf("no %s instruction", tt.mma)
			}
			if f := AuditSM100Function(fn); len(f) != 0 {
				t.Errorf("audit findings: %v", f)
			}
		})
	}
}
