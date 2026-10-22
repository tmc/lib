package cudamega

import (
	"strings"
	"testing"
)

func TestMoEMixtureSource(t *testing.T) {
	for _, workers := range []uint32{1, 2, 4, 8} {
		p := mixtureProgram(t, workers)
		s, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"moe_mixture_worker(", "if(n!=8u||!epoch||(layer!=0&&layer!=2))", "unsigned long long row=", "weight_decode(weights[row*17u+h])", "routing[0]", "routing[1]", "__fadd_rn(result,__fmul_rn", "producers_0=", "producers_1=", "atomicAdd(completed + 2, 1u)", "start_order["} {
			if !strings.Contains(s, want) {
				t.Fatalf("workers=%d source lacks %q", workers, want)
			}
		}
		ordinary, err := p.MoEOrdinarySource()
		if err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"moe_weight_slab(", "moe_mixture_ordinary(float *scratch", "routing[0],scratch[0+e]", "routing[1],scratch[8+e]", "scratch[16+e]=sum"} {
			if !strings.Contains(ordinary, want) {
				t.Fatalf("ordinary source lacks %q", want)
			}
		}
		for _, bad := range []string{"moe_mixture_worker", "completed", "start_order", "__syncthreads", "atomicAdd"} {
			if strings.Contains(ordinary, bad) {
				t.Fatalf("ordinary source contains scheduler %q", bad)
			}
		}
	}
}
