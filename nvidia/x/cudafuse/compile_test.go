package cudafuse_test

import (
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestFusedLoopHasNoWideDivision compiles fused kernels and checks the PTX
// for a 64-bit division. The grid-stride loop's index is 64-bit, and an
// unrolled loop is sized by dividing by its stride, which ptxas lowers to a
// __cuda_sm20_div_u64 call executed by every thread of every fused kernel.
// NVRTC 13.4 unrolls the loop for sm_100 and later and not for sm_90, so the
// targets are Blackwell ones.
func TestFusedLoopHasNoWideDivision(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 requires NVRTC")
	}
	shape := []int{1 << 20}
	tests := []struct {
		name  string
		build func(g *cudair.Graph) *cudair.Node
	}{
		{"unary", func(g *cudair.Graph) *cudair.Node {
			return g.AddUnary("relu", g.AddInput("x", shape, cudair.Float32))
		}},
		{"gate", func(g *cudair.Graph) *cudair.Node {
			x := g.AddInput("x", shape, cudair.Float32)
			w := g.AddInput("w", shape, cudair.Float32)
			return g.AddBinary("mul", g.AddUnary("silu", x), w)
		}},
	}
	for _, arch := range []string{"sm_100", "sm_120"} {
		for _, tt := range tests {
			t.Run(arch+"/"+tt.name, func(t *testing.T) {
				g := cudair.NewGraph()
				if err := g.SetOutputs(tt.build(g)); err != nil {
					t.Fatal(err)
				}
				src, err := cudafuse.Fuse(g, "fused_"+tt.name)
				if err != nil {
					t.Fatal(err)
				}
				mod, err := nvrtcx.Compile(src, nvrtcx.WithName("fused_"+tt.name+".cu"), nvrtcx.WithPhysicalArchTarget(arch))
				if err != nil {
					t.Skipf("NVRTC cannot compile for %s: %v", arch, err)
				}
				for _, op := range []string{"div.u64", "rem.u64", "div.s64", "rem.s64"} {
					if strings.Contains(mod.PTX(), op) {
						t.Errorf("PTX has %s:\n%s", op, mod.PTX())
					}
				}
			})
		}
	}
}
