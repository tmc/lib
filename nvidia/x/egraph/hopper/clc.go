package hopper

import "github.com/tmc/lib/nvidia/x/tcgen05"

// CLCHelpers returns the device-side Blackwell sm_100a/sm_103a Cluster Launch
// Control (CLC) inline-PTX and host-emulation wrappers.
func CLCHelpers() string {
	return tcgen05.CLCHelpers()
}

// CLCPersistentTileGEMM is a Blackwell sm_100a/sm_103a persistent TileGEMM kernel
// that launches a fixed grid of ActiveCTAs resident blocks and dynamically steals
// output tiles from the hardware launch grid via clusterlaunchcontrol.try_cancel
// and clusterlaunchcontrol.query_cancel.
type CLCPersistentTileGEMM struct {
	Name       string
	N, K       int
	BM, BN, BK int
	ActiveCTAs int
	Weight     WeightFormat
	Out        DType
	Epilogue   Lambda
}

func (g CLCPersistentTileGEMM) toTCGen05() (tcgen05.CLCPersistentTileGEMM, error) {
	var ep tcgen05.Lambda
	if g.Epilogue.root != nil {
		var err error
		ep, err = tcgen05.ParseLambda(g.Epilogue.src)
		if err != nil {
			return tcgen05.CLCPersistentTileGEMM{}, err
		}
	}
	return tcgen05.CLCPersistentTileGEMM{
		Name:       g.Name,
		N:          g.N,
		K:          g.K,
		BM:         g.BM,
		BN:         g.BN,
		BK:         g.BK,
		ActiveCTAs: g.ActiveCTAs,
		Weight:     tcgen05.WeightFormat(g.Weight),
		Out:        tcgen05.DType(g.Out),
		Epilogue:   ep,
	}, nil
}

// SharedBytes returns the dynamic shared-memory allocation per CTA in bytes.
func (g CLCPersistentTileGEMM) SharedBytes() int {
	tg, err := g.toTCGen05()
	if err != nil {
		return 0
	}
	return tg.SharedBytes()
}

// TotalTiles returns the total number of (BM x BN) output tiles covering an
// (m x N) output matrix.
func (g CLCPersistentTileGEMM) TotalTiles(m int) int {
	tg, err := g.toTCGen05()
	if err != nil {
		return 0
	}
	return tg.TotalTiles(m)
}

// CUDA returns the __global__ kernel definition (including CLCHelpers) without
// [Prelude].
func (g CLCPersistentTileGEMM) CUDA() (string, error) {
	tg, err := g.toTCGen05()
	if err != nil {
		return "", err
	}
	return tg.CUDA()
}

// Launcher returns the inline C++ host launch helper for g.
func (g CLCPersistentTileGEMM) Launcher() (string, error) {
	tg, err := g.toTCGen05()
	if err != nil {
		return "", err
	}
	return tg.Launcher()
}

// Source returns the self-contained CUDA C++ translation unit ([Prelude] +
// [CLCHelpers] + kernel + launcher) for g.
func (g CLCPersistentTileGEMM) Source() (string, error) {
	tg, err := g.toTCGen05()
	if err != nil {
		return "", err
	}
	return tg.Source()
}

// Reference computes the float64 reference output of g on activation matrix x
// (m x K) and weights w (N x K), rounded to g.Out.
func (g CLCPersistentTileGEMM) Reference(m int, x []uint16, w GEMMWeights) ([]float32, error) {
	tg, err := g.toTCGen05()
	if err != nil {
		return nil, err
	}
	return tg.Reference(m, x, tcgen05.GEMMWeights{
		BF16:       w.BF16,
		E4M3:       w.E4M3,
		Packed:     w.Packed,
		Scales:     w.Scales,
		GlobalScal: w.GlobalScal,
	})
}
