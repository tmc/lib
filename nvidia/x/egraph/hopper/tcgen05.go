package hopper

import "github.com/tmc/lib/nvidia/x/tcgen05"

// TileEpilogue selects the elementwise epilogue applied to TCGen05GEMM accumulators.
type TileEpilogue = tcgen05.TileEpilogue

const (
	TileLinear = tcgen05.TileLinear // identity: y = acc
	TileSiLU   = tcgen05.TileSiLU   // SiLU: y = acc / (1 + exp(-acc))
	TileGLU    = tcgen05.TileGLU    // SwiGLU over paired (gate, up) columns: y = silu(gate) * up
)

// TCGen05GEMM configures a Blackwell sm_100a/sm_103a Tensor Memory (TMEM) GEMM
// kernel using tcgen05.alloc, tcgen05.mma.cta_group::1, tcgen05.ld, and tcgen05.dealloc.
type TCGen05GEMM struct {
	Name     string
	N, K     int
	BM, BN   int          // e.g., BM=128, BN=128 or 256 (accumulators live in TMEM, not registers!)
	BK       int          // reduction tile step (default 64)
	Weight   WeightFormat // WeightBF16, WeightE4M3, WeightNVFP4
	Out      DType
	Epilogue TileEpilogue // TileLinear, TileSiLU, TileGLU
}

func (g TCGen05GEMM) toTCGen05() tcgen05.TCGen05GEMM {
	return tcgen05.TCGen05GEMM{
		Name:     g.Name,
		N:        g.N,
		K:        g.K,
		BM:       g.BM,
		BN:       g.BN,
		BK:       g.BK,
		Weight:   tcgen05.WeightFormat(g.Weight),
		Out:      tcgen05.DType(g.Out),
		Epilogue: g.Epilogue,
	}
}

// SharedBytes returns the dynamic shared memory bytes per CTA needed by g.
func (g TCGen05GEMM) SharedBytes() int {
	return g.toTCGen05().SharedBytes()
}

// Grid returns the launch grid and block dimensions for m activation rows.
func (g TCGen05GEMM) Grid(m int) (grid, block [3]int) {
	return g.toTCGen05().Grid(m)
}

// CUDA returns the kernel source without [Prelude].
func (g TCGen05GEMM) CUDA() (string, error) {
	return g.toTCGen05().CUDA()
}

// Launcher returns the host launch wrapper for g.
func (g TCGen05GEMM) Launcher() (string, error) {
	return g.toTCGen05().Launcher()
}

// Source returns [Prelude], the Blackwell tcgen05 kernel, and its launcher.
func (g TCGen05GEMM) Source() (string, error) {
	return g.toTCGen05().Source()
}

// Reference computes the float64 reference output for WeightBF16 or WeightE4M3
// with FP32 epilogue and Out rounding.
func (g TCGen05GEMM) Reference(m int, x []uint16, w GEMMWeights) ([]float32, error) {
	return g.toTCGen05().Reference(m, x, tcgen05.GEMMWeights{
		BF16:       w.BF16,
		E4M3:       w.E4M3,
		Packed:     w.Packed,
		Scales:     w.Scales,
		GlobalScal: w.GlobalScal,
	})
}

// ReferenceNVFP4 computes the float64 reference output for WeightNVFP4 with
// block-scaled E2M1 operands, FP32 epilogue, and Out rounding.
func (g TCGen05GEMM) ReferenceNVFP4(m int, aFP4, aScale, bFP4, bScale []uint8) ([]float32, error) {
	return g.toTCGen05().ReferenceNVFP4(m, aFP4, aScale, bFP4, bScale)
}
