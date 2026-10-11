package cudasearch

import (
	"context"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

func spatialAxisSearch() *LinearPrunedSearch {
	fact := cudalayout.NewIndexFactorization(
		[]cudalayout.StorageLevel{cudalayout.LevelRegister, cudalayout.LevelShared, cudalayout.LevelDRAM},
		map[cudalayout.StorageLevel]map[string]int{
			cudalayout.LevelRegister: {"B": 1, "M": 1, "N": 1, "K": 1},
			cudalayout.LevelShared:   {"B": 1, "M": 4, "N": 4, "K": 4},
			cudalayout.LevelDRAM:     {"B": 1, "M": 2, "N": 2, "K": 2},
		})
	return NewLinearPrunedSearch([]cudalayout.IndexFactorization{fact},
		[]cudalayout.LoopPermutation{{}, {}},
		[]cudalayout.DatatypeBypass{{}, {}},
		cudalayout.HardwareConstraints{MaxThreadsPerBlock: 128, MaxRegistersPerBlock: 1024, MaxSharedMemoryPerBlock: 1024},
		map[cudalayout.TensorRole]int{cudalayout.TensorA: 4, cudalayout.TensorB: 4, cudalayout.TensorC: 4},
		map[string]int{"B": 1, "M": 8, "N": 8, "K": 8})
}

func TestMapspaceSpatialAxis(t *testing.T) {
	s := spatialAxisSearch()
	s.SpatialTilings = []cudalayout.SpatialTiling{
		cudalayout.NewSpatialTiling(map[string]int{"M": 2}, map[string]int{"M": 2}),
		cudalayout.NewSpatialTiling(map[string]int{"N": 2}, map[string]int{"N": 4}),
		cudalayout.NewSpatialTiling(map[string]int{"M": 3}, nil),
		cudalayout.NewSpatialTiling(nil, map[string]int{"M": 4, "N": 4}),
	}
	stats, err := s.Search(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if stats.FactorizationsPruned != 0 || stats.EvaluatedCandidates != 8 || stats.DownstreamSkipped != 8 {
		t.Fatalf("stats=%+v, want 8 valid candidates and 8 skipped", stats)
	}
	parts, err := s.Split(1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parts[0].SpatialTilings, s.SpatialTilings) {
		t.Fatal("Split lost spatial candidates")
	}
	pstats, err := parts[0].Search(context.Background())
	if err != nil || pstats != stats {
		t.Fatalf("split stats=%+v, error=%v, want %+v", pstats, err, stats)
	}
}

func TestMapspaceSpatialAllInvalidPrunes(t *testing.T) {
	s := spatialAxisSearch()
	s.SpatialTilings = []cudalayout.SpatialTiling{
		cudalayout.NewSpatialTiling(map[string]int{"M": 3}, nil),
		cudalayout.NewSpatialTiling(nil, map[string]int{"N": 0}),
	}
	stats, err := s.Search(context.Background())
	if err != nil || stats.FactorizationsPruned != 1 || stats.DownstreamSkipped != 8 || stats.EvaluatedCandidates != 0 {
		t.Fatalf("stats=%+v error=%v, want all eight downstream candidates pruned", stats, err)
	}
}

func TestMapspaceSpatialNeedsThreadBound(t *testing.T) {
	for _, threads := range []int{0, 63, 64, 127, 128} {
		s := spatialAxisSearch()
		s.Constraints.MaxThreadsPerBlock = threads
		s.Constraints.MaxRegistersPerBlock = 0
		s.SpatialTilings = []cudalayout.SpatialTiling{cudalayout.NewSpatialTiling(nil, map[string]int{"M": 2})}
		stats, err := s.Search(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		want := 0
		if threads >= 64 {
			want = 4
		}
		if stats.EvaluatedCandidates != want {
			t.Fatalf("threads=%d stats=%+v, want %d evaluations", threads, stats, want)
		}
	}
}
