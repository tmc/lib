//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaExpertGatherSource holds the expert-major MoE gather kernels. They
// recompute the same sums as the per-token kernels in cuda_experts_q81.cu,
// in the same order, with the expert weight row read once per tile of pairs
// instead of once per pair.
//
//go:embed cuda_experts_gather.cu
var cudaExpertGatherSource string

// cudaMoEGatherTile mirrors DS4_MOE_GATHER_TILE in cuda_experts_gather.cu.
// TestCUDAMoEGatherConstants fails if the two drift apart.
const cudaMoEGatherTile = 4

// cudaMoEMaxExperts mirrors DS4_MOE_MAX_EXPERTS, the counting sort's bin bound.
const cudaMoEMaxExperts = 256

// cudaExpertGatherWarps is the number of expert rows one block owns.
const cudaExpertGatherWarps = 2

// cudaMoEGrouping is the device-side CSR listing of one token group's expert
// selections, produced by ds4_moe_group_csr and consumed by both gather
// kernels. One grouping serves every projection of a layer, so the caller
// allocates it once per group width and refills it per layer.
//
// The zero value is not usable; call newCUDAMoEGrouping.
type cudaMoEGrouping struct {
	offsets cudamem.Slice[int32] // experts+2, bin b spans [offsets[b], offsets[b+1])
	pairs   cudamem.Slice[int32] // pairs, sorted by bin then by pair id
	active  cudamem.Slice[int32] // experts+1, the non-empty bins, ascending
	count   cudamem.Slice[int32] // 1, the number of non-empty bins
	npairs  int
	experts int
}

// newCUDAMoEGrouping allocates a grouping for pairs selections over experts
// experts. Bin experts collects out-of-range ids, so there are experts+1 bins.
func newCUDAMoEGrouping(dev *cuda.Device, experts, pairs int) (*cudaMoEGrouping, error) {
	if experts <= 0 || experts > cudaMoEMaxExperts || pairs <= 0 {
		return nil, fmt.Errorf("cuda: MoE grouping: invalid shape")
	}
	g := &cudaMoEGrouping{npairs: pairs, experts: experts}
	for _, alloc := range []struct {
		dst    *cudamem.Slice[int32]
		length int
	}{
		{&g.offsets, experts + 2},
		{&g.pairs, pairs},
		{&g.active, experts + 1},
		{&g.count, 1},
	} {
		s, err := cudamem.Alloc[int32](dev, alloc.length)
		if err != nil {
			_ = g.Free()
			return nil, fmt.Errorf("cuda: MoE grouping: %w", err)
		}
		*alloc.dst = s
	}
	return g, nil
}

// Free releases every buffer, reporting the first failure.
func (g *cudaMoEGrouping) Free() error {
	var first error
	for _, s := range []cudamem.Slice[int32]{g.offsets, g.pairs, g.active, g.count} {
		if s.Ptr() == 0 {
			continue
		}
		if err := s.Free(); err != nil && first == nil {
			first = err
		}
	}
	*g = cudaMoEGrouping{}
	return first
}

// profile reports whether the grouping bins the tensor's experts.
func (g *cudaMoEGrouping) profile(experts uint64) error {
	if g == nil || g.npairs <= 0 {
		return fmt.Errorf("cuda: MoE gather: no grouping")
	}
	if experts > cudaMoEMaxExperts || uint64(g.experts) != experts {
		return fmt.Errorf("cuda: MoE gather: grouping covers %d experts, tensor has %d", g.experts, experts)
	}
	if g.offsets.Len() != g.experts+2 || g.active.Len() != g.experts+1 || g.pairs.Len() != g.npairs || g.count.Len() != 1 {
		return fmt.Errorf("cuda: MoE gather: grouping buffers do not match its shape")
	}
	return nil
}

// encodeMoEGroupCSR appends the counting sort that groups ids by expert. ids
// is token-major and its length is the number of (token, slot) pairs.
func (r *cudaFoundation) encodeMoEGroupCSR(ids cudamem.Slice[int32], g *cudaMoEGrouping) error {
	if g == nil || ids.Len() != g.npairs {
		return fmt.Errorf("cuda: MoE group CSR: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(r.moeGroupCSRFn).Grid(1).Block(256).On(r.stream()).Launch(
			ids, g.offsets, g.pairs, g.active, g.count, uint32(g.npairs), uint32(g.experts)); err != nil {
			return fmt.Errorf("cuda: launch MoE group CSR: %w", err)
		}
		return nil
	})
}

// gatherGrid reports the block grid for a gather launch: one block per
// warpsPerBlock rows in x, and one per possibly active bin in y. At most
// min(pairs, experts+1) bins can be non-empty, and blocks past the count the
// grouping kernel recorded retire before reading a weight.
func (g *cudaMoEGrouping) gatherGrid(rows uint64) (uint32, uint32) {
	blocks := (rows + cudaExpertGatherWarps - 1) / cudaExpertGatherWarps
	bins := g.experts + 1
	if g.npairs < bins {
		bins = g.npairs
	}
	return uint32(blocks), uint32(bins)
}

// encodeMatvecIQ2SelectedQ81Gather appends the grouped paired IQ2_XXS
// projections for a token group. The activation is one Q8_1 row per token,
// shared by that token's slots; outputs are pair-major, pair p being token
// p/slots slot p%slots, which is what the per-token kernel writes when its
// outputs are concatenated in token order.
func (r *cudaFoundation) encodeMatvecIQ2SelectedQ81Gather(gate, up cudaModelTensor, g *cudaMoEGrouping, slots int, in cudamem.Slice[byte], gateOut, upOut cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(gate, 16, 66); err != nil {
		return err
	}
	if err := cudaExpertProfile(up, 16, 66); err != nil {
		return err
	}
	for i := range gate.dimensions {
		if gate.dimensions[i] != up.dimensions[i] {
			return fmt.Errorf("cuda: IQ2 Q8_1 gather: mismatched dimensions")
		}
	}
	cols, rows, experts := gate.dimensions[0], gate.dimensions[1], gate.dimensions[2]
	if err := g.profile(experts); err != nil {
		return err
	}
	pairs := uint64(g.npairs)
	if slots <= 0 || pairs%uint64(slots) != 0 {
		return fmt.Errorf("cuda: IQ2 Q8_1 gather: pairs are not a whole number of tokens")
	}
	tokens := pairs / uint64(slots)
	if pairs > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*tokens || uint64(gateOut.Len()) != pairs*rows || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("cuda: IQ2 Q8_1 gather: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		x, y := g.gatherGrid(rows)
		if err := cudalaunch.Kernel(r.iq2GatherQ81Fn).Grid(x, y).Block(32*cudaExpertGatherWarps).On(r.stream()).Launch(
			gate.device, up.device, g.offsets, g.pairs, g.active, g.count, in, gateOut, upOut,
			uint32(cols), uint32(rows), uint32(slots), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch IQ2 Q8_1 gather: %w", err)
		}
		return nil
	})
}

// encodeMatvecQ2KSelectedQ81Gather appends the grouped Q2_K projection for a
// token group. Q2 reads one Q8_1 activation row per pair, so both the
// activation and the output are pair-major.
func (r *cudaFoundation) encodeMatvecQ2KSelectedQ81Gather(t cudaModelTensor, g *cudaMoEGrouping, in cudamem.Slice[byte], out cudamem.Slice[float32]) error {
	if err := cudaExpertProfile(t, 10, 84); err != nil {
		return err
	}
	cols, rows, experts := t.dimensions[0], t.dimensions[1], t.dimensions[2]
	if err := g.profile(experts); err != nil {
		return err
	}
	pairs := uint64(g.npairs)
	if pairs > math.MaxInt32/rows || uint64(in.Len()) != cols/32*36*pairs || uint64(out.Len()) != pairs*rows {
		return fmt.Errorf("cuda: Q2_K Q8_1 gather: invalid shape")
	}
	return r.device.WithContext(func() error {
		if err := r.loadMathKernels(); err != nil {
			return err
		}
		x, y := g.gatherGrid(rows)
		if err := cudalaunch.Kernel(r.q2KGatherQ81Fn).Grid(x, y).Block(32*cudaExpertGatherWarps).On(r.stream()).Launch(
			t.device, g.offsets, g.pairs, g.active, g.count, in, out,
			uint32(cols), uint32(rows), uint32(experts)); err != nil {
			return fmt.Errorf("cuda: launch Q2_K Q8_1 gather: %w", err)
		}
		return nil
	})
}
