package cudasearch

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func spatialProfileSearch() *LinearPrunedSearch {
	levels := []cudalayout.StorageLevel{0, 1, 2, 3}
	factors := map[cudalayout.StorageLevel]map[string]int{}
	for _, l := range levels {
		factors[l] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
	}
	factors[2]["M"] = 2
	factors[3]["N"] = 2
	factors[3]["K"] = 2
	s := NewLinearPrunedSearch([]cudalayout.IndexFactorization{cudalayout.NewIndexFactorization(levels, factors)}, nil, nil, cudalayout.HardwareConstraints{MaxSharedMemoryPerBlock: 1024, MaxRegistersPerBlock: 1024, MaxThreadsPerBlock: 32}, map[cudalayout.TensorRole]int{0: 1, 1: 1, 2: 1}, map[string]int{"B": 1, "M": 2, "N": 2, "K": 2})
	s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"M": 2}, nil), cudalayout.NewSpatialTiling(map[string]int{"N": 2}, nil)}
	s.SpatialProfile = &cudaroofline.GEMMSpatialProfile{LaneMesh: map[string]int{"M": 1}}
	return s
}

func TestMapspaceSpatialProfileScoring(t *testing.T) {
	s := spatialProfileSearch()
	stats, err := s.Search(context.Background())
	if err != nil || stats.EvaluatedCandidates != 2 || stats.DownstreamSkipped != 0 || stats.UnsupportedCandidates != 0 || stats.Best == nil {
		t.Fatalf("stats=%+v error=%v", stats, err)
	}
	if stats.Best.Mapping.Spatial.CTAMesh["N"] != 2 || stats.Best.Movement.DRAM.TotalBytes != 16 || stats.BestIntensity != 1 || stats.Best.SpatialMovement == nil || stats.Best.SpatialProfile == nil {
		t.Fatalf("wrong spatial winner: %+v", stats.Best)
	}
	s.SpatialTilings = s.SpatialTilings[:1]
	other, err := s.Search(context.Background())
	if err != nil || other.Best == nil || other.Best.Movement.DRAM.TotalBytes != 36 || other.Best.Movement.DRAM.TensorTransfers["C"] != 12 {
		t.Fatalf("owner-order spill count missing: %+v %v", other.Best, err)
	}
	s.SpatialProfile = nil
	nilStats, err := s.Search(context.Background())
	if err != nil || nilStats.EvaluatedCandidates != 0 || nilStats.UnsupportedCandidates != 1 || nilStats.DownstreamSkipped != 1 || nilStats.Best != nil {
		t.Fatal(nilStats, err)
	}
}

func TestMapspaceSpatialProfileSplitAndDetached(t *testing.T) {
	s := spatialProfileSearch()
	want, err := s.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	parts, err := s.Split(1)
	if err != nil {
		t.Fatal(err)
	}
	got, err := parts[0].Search(context.Background())
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("split lost profile", got, err)
	}
	got.Best.SpatialProfile.LaneMesh["M"] = 3
	got.Best.SpatialMovement.Levels[0].Tensors[0] = cudaroofline.SpatialTensorMovement{}
	again, err := s.Search(context.Background())
	if err != nil || !reflect.DeepEqual(again, want) || s.SpatialProfile.LaneMesh["M"] != 1 {
		t.Fatal("returned profile/report aliases input or search state", err)
	}
}

func TestMapspaceSpatialProfileCapacity(t *testing.T) {
	for _, register := range []bool{false, true} {
		t.Run(map[bool]string{false: "shared", true: "register"}[register], func(t *testing.T) {
			s := spatialProfileSearch()
			f := s.Factorizations[0]
			f.Factors[1] = map[string]int{"B": 1, "M": 2, "N": 2, "K": 2}
			f.Factors[2]["M"] = 2
			f.Factors[3] = map[string]int{"B": 1, "M": 1, "N": 1, "K": 1}
			s.Problem = map[string]int{"B": 1, "M": 4, "N": 2, "K": 2}
			s.DTypes = map[cudalayout.TensorRole]int{0: 4, 1: 4, 2: 4}
			s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"M": 2}, map[string]int{"N": 2})}
			s.SpatialProfile.LaneMesh = map[string]int{"M": 2}
			s.Constraints.MaxThreadsPerBlock = 64
			bypass := cudalayout.NewDatatypeBypass()
			if register {
				s.Constraints.MaxRegistersPerBlock = 128
				bypass.SetBypass(0, 2, true)
			} else {
				s.Constraints.MaxSharedMemoryPerBlock = 32
				bypass.SetBypass(1, 2, true)
			}
			s.Bypasses = []cudalayout.DatatypeBypass{cudalayout.NewDatatypeBypass(), bypass}
			stats, err := s.Search(context.Background())
			if err != nil || stats.FactorizationsPruned != 0 || stats.EvaluatedCandidates != 1 || stats.DownstreamSkipped != 1 || stats.Best == nil {
				t.Fatal("per-instance allocation was bypassed by union/aggregate traffic", stats, err)
			}
			if !reflect.DeepEqual(stats.Best.Mapping.Bypass, bypass) {
				t.Fatal("over-capacity candidate became best")
			}
			s.Bypasses = s.Bypasses[:1]
			stats, err = s.Search(context.Background())
			if err != nil || stats.FactorizationsPruned != 1 || stats.EvaluatedCandidates != 0 || stats.DownstreamSkipped != 1 || stats.Best != nil {
				t.Fatal("no fitting allocation was not pruned", stats, err)
			}
		})
	}
}

func TestMapspaceSpatialProfileRefusals(t *testing.T) {
	for _, mesh := range []map[string]int{{"Q": 1}, {"M": 0}, {"M": -1}, {"M": 33}, {"M": 32, "N": 2}} {
		s := spatialProfileSearch()
		s.SpatialProfile.LaneMesh = mesh
		stats, err := s.Search(context.Background())
		if !errors.Is(err, cudalayout.ErrUnsupported) || stats.EvaluatedCandidates != 0 || stats.Best != nil {
			t.Fatal(mesh, stats, err)
		}
	}
	s := spatialProfileSearch()
	s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(map[string]int{"K": 2}, nil)}
	stats, err := s.Search(context.Background())
	if err != nil || stats.UnsupportedCandidates != 1 || stats.DownstreamSkipped != 1 || stats.EvaluatedCandidates != 0 || stats.Best != nil {
		t.Fatal("K reduction not explicitly unsupported", stats, err)
	}
	s = spatialProfileSearch()
	s.SpatialProfile.LaneMesh = map[string]int{"M": 3}
	stats, err = s.Search(context.Background())
	if err != nil || stats.UnsupportedCandidates != 0 || stats.DownstreamSkipped != 2 || stats.EvaluatedCandidates != 0 || stats.Best != nil {
		t.Fatal("invalid lane divisibility scored", stats, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stats, err = s.Search(ctx)
	if !errors.Is(err, context.Canceled) || stats.EvaluatedCandidates != 0 {
		t.Fatal(stats, err)
	}
}

func TestMapspaceSpatialProfileLaneThreadBound(t *testing.T) {
	for _, threads := range []int{0, 1, 2, 32} {
		s := spatialProfileSearch()
		s.SpatialTilings = nil
		s.Factorizations[0].Factors[1]["M"] = 2
		s.Factorizations[0].Factors[2]["M"] = 1
		s.SpatialProfile.LaneMesh = map[string]int{"M": 2}
		s.Constraints.MaxRegistersPerBlock = 0
		s.Constraints.MaxThreadsPerBlock = threads
		stats, err := s.Search(context.Background())
		want := 0
		if threads >= 2 {
			want = 1
		}
		if err != nil || stats.EvaluatedCandidates != want {
			t.Fatalf("threads%d: %+v %v", threads, stats, err)
		}
	}
}
