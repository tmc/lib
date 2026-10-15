package cudasearch_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
	"github.com/tmc/lib/nvidia/x/cudasearch"
)

func reductionSearch(width, registers int) *cudasearch.LinearPrunedSearch {
	levels := []cudalayout.StorageLevel{0, 1, 2, 3}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	bypass := cudalayout.NewDatatypeBypass()
	for _, l := range levels {
		factors[l] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
		if l != 3 {
			for r := cudalayout.TensorA; r <= cudalayout.TensorC; r++ {
				bypass.SetBypass(l, r, true)
			}
		}
	}
	factors[2]["K"] = 2
	s := cudasearch.NewLinearPrunedSearch([]cudalayout.IndexFactorization{cudalayout.NewIndexFactorization(levels, factors)}, nil, []cudalayout.DatatypeBypass{bypass}, cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 128, MaxRegistersPerBlock: registers, MaxThreadsPerBlock: 64, TMAAlignmentBytes: 128}, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: width}, map[string]int{"B": 1, "M": 1, "N": 1, "K": 2})
	s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"K": 2}, nil), cudalayout.NewSpatialTiling(nil, nil)}
	s.SpatialProfile = &cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"K": 1}}
	s.SplitK = true
	return s
}

func TestMapspaceReductionWholeRecipeScore(t *testing.T) {
	s := reductionSearch(1, 64)
	stats, err := s.Search(context.Background())
	// Two scalar MACs, all buffers bypassed. Identity: A2+B2+C(read1+
	// write2)=7. Split: A2+B2+partial writes2=6, plus tree reads2/write1=3.
	// Leaf-only scoring chooses split; full recipe scoring must choose identity.
	if err != nil || stats.EvaluatedCandidates != 2 || stats.Best == nil || stats.Best.SplitK != nil || stats.BestIntensity != 4.0/7 || stats.Best.TotalDRAM().TotalBytes != 7 {
		t.Fatalf("whole recipe score: %+v %v", stats, err)
	}
	s.SpatialTilings = s.SpatialTilings[:1]
	stats, err = s.Search(context.Background())
	if err != nil || stats.Best == nil || stats.Best.SplitK == nil || stats.Best.Movement.DRAM.TotalBytes != 6 || stats.Best.TotalDRAM().TotalBytes != 9 || stats.BestIntensity != 4.0/9 || stats.Best.SplitK.ExecutedFLOPs != 5 {
		t.Fatalf("partial/whole distinction: %+v %v", stats, err)
	}
	if stats.Best.SplitK.WorkspaceBytes != 2 || stats.Best.SplitK.OutputBytes != 1 || stats.Best.SplitK.ReductionAdditions != 1 {
		t.Fatal(stats.Best.SplitK)
	}
	s.SplitK = false
	stats, err = s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 0 || stats.UnsupportedCandidates != 1 || stats.DownstreamSkipped != 1 || stats.Best != nil {
		t.Fatal("zero opt-in changed", stats, err)
	}
	s.SplitK = true
	s.SpatialProfile = nil
	stats, err = s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 0 || stats.UnsupportedCandidates != 1 || stats.Best != nil {
		t.Fatal("missing spatial profile admitted", stats, err)
	}
}

func TestMapspaceReductionCapacityAuthority(t *testing.T) {
	for _, limit := range []int{191, 192} {
		s := reductionSearch(5, limit)
		stats, err := s.Search(context.Background())
		want := 1
		if limit == 192 {
			want = 2
		}
		if err != nil || stats.EvaluatedCandidates != want || stats.DownstreamSkipped != 2-want || stats.FactorizationsPruned != 0 || stats.UnsupportedCandidates != 0 {
			t.Fatalf("limit%d: %+v %v", limit, stats, err)
		}
		// Core registers/shared are entirely bypassed. The merge still needs
		// ceil(2*5/4)*64=192 registers under the canonical capacity authority.
		s.SpatialTilings = s.SpatialTilings[:1]
		stats, err = s.Search(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if limit == 191 {
			if stats.FactorizationsPruned != 1 || stats.EvaluatedCandidates != 0 || stats.DownstreamSkipped != 1 || stats.Best != nil {
				t.Fatal("merge capacity not pre-pruned", stats)
			}
		} else {
			if stats.EvaluatedCandidates != 1 || stats.Best == nil || stats.Best.SplitK.MergeRegisterBytes != 10 {
				t.Fatal("canonical threshold rejected", stats)
			}
		}
	}
	// Configured threads, not two partial owners or one active merge lane.
	s := reductionSearch(5, 96)
	s.Constraints.MaxThreadsPerBlock = 32
	s.SpatialTilings = s.SpatialTilings[:1]
	stats, err := s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 1 {
		t.Fatal("32-thread capacity", stats, err)
	}
	s.Constraints.MaxThreadsPerBlock = 64
	stats, err = s.Search(context.Background())
	if err != nil || stats.FactorizationsPruned != 1 || stats.EvaluatedCandidates != 0 {
		t.Fatal("configured threads ignored", stats, err)
	}
}

func TestMapspaceReductionSplitTieAndDetached(t *testing.T) {
	s := reductionSearch(1, 64)
	s.SpatialTilings = s.SpatialTilings[:1]
	second := cudalayout.NewSpatialTiling(s.SpatialTilings[0].CTAMesh, nil)
	second.CTAMesh["B"] = 1
	s.SpatialTilings = append(s.SpatialTilings, second)
	want, err := s.Search(context.Background())
	if err != nil || want.Best == nil || want.Best.SplitK == nil {
		t.Fatal(want, err)
	}
	if _, ok := want.Best.Mapping.Spatial.CTAMesh["B"]; ok {
		t.Fatal("equal later candidate replaced first")
	}
	parts, err := s.Split(1)
	if err != nil || !parts[0].SplitK {
		t.Fatal(parts, err)
	}
	got, err := parts[0].Search(context.Background())
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("split changed recipe", err)
	}
	got.Best.SplitK.Rounds[0].ReadBytes = 0
	got.Best.SplitK.DRAM.TensorTransfers["C"] = 0
	got.Best.SplitK.GEMM.Levels[0].Tensors[2] = cudaroofline.SpatialTensorMovement{}
	got.Best.Mapping.Factorization.Factors[2]["K"] = 99
	got.Best.Mapping.Permutation.Orders[0][0] = "Q"
	got.Best.Mapping.Bypass.SetBypass(0, 0, false)
	got.Best.Mapping.Spatial.CTAMesh["K"] = 99
	got.Best.SpatialProfile.LaneMesh["K"] = 99
	again, err := s.Search(context.Background())
	if err != nil || !reflect.DeepEqual(again, want) {
		t.Fatal("result aliases input/state", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stats, err := s.Search(ctx)
	if !errors.Is(err, context.Canceled) || stats.EvaluatedCandidates != 0 || stats.Best != nil {
		t.Fatal(stats, err)
	}
}

func TestMapspaceReductionRefusalsAndTotal(t *testing.T) {
	if got := (cudasearch.MapSpaceSearchResult{}).TotalDRAM(); got.TotalBytes != 0 || got.TensorTransfers != nil {
		t.Fatal(got)
	}
	s := reductionSearch(1, 64)
	s.SpatialTilings = s.SpatialTilings[:1]
	s.SpatialProfile.LaneMesh["K"] = 3
	stats, err := s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 0 || stats.Best != nil || stats.DownstreamSkipped != 1 {
		t.Fatal("invalid lane ratio scored", stats, err)
	}
	s = reductionSearch(1, 64)
	s.SpatialProfile.LaneMesh = map[string]int{"Q": 1}
	stats, err = s.Search(context.Background())
	if !errors.Is(err, cudalayout.ErrUnsupported) || stats.EvaluatedCandidates != 0 {
		t.Fatal(stats, err)
	}
}
