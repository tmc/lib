package hopper

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestFuseTails checks that a single-CTA RMSNorm after a skinny gemm runs
// as the gemm's tail and gives the unfused plan's output bit for bit, for
// both row-kernel geometries (32 threads per row up to 1024 columns, 256
// beyond) and for several rows in one tail block.
func TestFuseTails(t *testing.T) {
	tests := []struct {
		m, d  int
		fused int
	}{
		{1, 512, 1},
		{3, 512, 1},
		{1, 2048, 1},
		{2, 2048, 0}, // two rows of 2048 launch two CTAs
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("m%d_d%d", tt.m, tt.d), func(t *testing.T) {
			const k, n = 256, 256
			build := func() *Plan {
				b := NewBuilder()
				attn := b.Input("attn", F32, tt.m, k)
				x := b.Input("x", F32, tt.m, tt.d)
				h := b.Map(MustLambda("(add $0 $1)"), b.MatMul(b.Cast(BF16, attn), b.Input("w_o", BF16, tt.d, k)), x)
				norm := b.Cast(BF16, b.RMSNorm(h, b.Input("norm_w", F32, tt.d), tt.d, 1e-5))
				root := b.MatMul(norm, b.Input("w2", BF16, n, tt.d))
				e, err := Optimize(b.G, root)
				if err != nil {
					t.Fatal(err)
				}
				p, err := Compile(e, "tail", F32)
				if err != nil {
					t.Fatal(err)
				}
				return p
			}
			plain, fused := build(), build()
			if got := fused.FuseTails(); got != tt.fused {
				t.Fatalf("FuseTails = %d, want %d", got, tt.fused)
			}
			if got, want := len(fused.Kernels), len(plain.Kernels)-tt.fused; got != want {
				t.Fatalf("%d kernels after FuseTails, want %d", got, want)
			}
			if tt.fused == 0 {
				return
			}
			g := fused.Kernels[0].GEMM
			if g == nil || g.Tail == nil || !slices.Contains(fused.Kernels[0].Args, g.Name+"_count") {
				t.Fatalf("first kernel %+v is not a gemm with a tail and counter", fused.Kernels[0])
			}
			r := rand.New(rand.NewPCG(7, uint64(tt.m*tt.d)))
			env := randEnv(r, plain.Inputs...)
			want := runPlan(t, plain, env)
			got := runPlan(t, fused, env)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("out[%d] = %g fused, %g unfused", i, got[i], want[i])
				}
			}
		})
	}
}
