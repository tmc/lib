package cudashape_test

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func mustShape(t *testing.T, dims ...int) cudashape.Shape {
	t.Helper()
	s, err := cudashape.New(dims...)
	if err != nil {
		t.Fatalf("cudashape.New(%v): unexpected error %v", dims, err)
	}
	return s
}

func TestShape_InferGrid_Ranks1To3(t *testing.T) {
	tests := []struct {
		name      string
		tensor    cudashape.Shape
		partition cudashape.Shape
		want      cudashape.Grid3D
	}{
		{
			name:      "rank 1 exact division",
			tensor:    mustShape(t, 1024),
			partition: mustShape(t, 128),
			want:      cudashape.Grid3D{X: 8, Y: 1, Z: 1},
		},
		{
			name:      "rank 1 ceiling division",
			tensor:    mustShape(t, 1000),
			partition: mustShape(t, 128),
			want:      cudashape.Grid3D{X: 8, Y: 1, Z: 1},
		},
		{
			name:      "rank 1 partition larger than tensor",
			tensor:    mustShape(t, 50),
			partition: mustShape(t, 128),
			want:      cudashape.Grid3D{X: 1, Y: 1, Z: 1},
		},
		{
			name:      "rank 2 exact division",
			tensor:    mustShape(t, 1024, 2048),
			partition: mustShape(t, 64, 128),
			want:      cudashape.Grid3D{X: 16, Y: 16, Z: 1},
		},
		{
			name:      "rank 2 asymmetric ceiling division",
			tensor:    mustShape(t, 15, 1000),
			partition: mustShape(t, 2, 64),
			want:      cudashape.Grid3D{X: 8, Y: 16, Z: 1},
		},
		{
			name:      "rank 3 exact division",
			tensor:    mustShape(t, 16, 1024, 2048),
			partition: mustShape(t, 2, 64, 128),
			want:      cudashape.Grid3D{X: 8, Y: 16, Z: 16},
		},
		{
			name:      "rank 3 asymmetric ceiling division",
			tensor:    mustShape(t, 15, 1000, 3000),
			partition: mustShape(t, 2, 64, 128),
			want:      cudashape.Grid3D{X: 8, Y: 16, Z: 24},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudashape.InferGrid(tt.tensor, tt.partition)
			if err != nil {
				t.Fatalf("InferGrid(%v, %v): unexpected error %v", tt.tensor, tt.partition, err)
			}
			if !got.Equal(tt.want) {
				t.Errorf("InferGrid(%v, %v) = %v, want %v", tt.tensor, tt.partition, got, tt.want)
			}
			if got.Dims() != tt.want.Dims() {
				t.Errorf("Dims() = %v, want %v", got.Dims(), tt.want.Dims())
			}
		})
	}
}

func TestShape_InferGrid_NonPositiveOrOverflow(t *testing.T) {
	t.Run("rank mismatch", func(t *testing.T) {
		t1 := mustShape(t, 100, 200)
		p1 := mustShape(t, 10)
		_, err := cudashape.InferGrid(t1, p1)
		if !errors.Is(err, cudashape.ErrRankMismatch) {
			t.Errorf("expected ErrRankMismatch, got %v", err)
		}
	})

	t.Run("rank 0 unsupported", func(t *testing.T) {
		t0 := mustShape(t)
		p0 := mustShape(t)
		_, err := cudashape.InferGrid(t0, p0)
		if !errors.Is(err, cudashape.ErrRankMismatch) {
			t.Errorf("expected ErrRankMismatch for rank 0, got %v", err)
		}
	})

	t.Run("rank 4 unsupported", func(t *testing.T) {
		t4 := mustShape(t, 2, 4, 8, 16)
		p4 := mustShape(t, 1, 2, 4, 8)
		_, err := cudashape.InferGrid(t4, p4)
		if !errors.Is(err, cudashape.ErrRankMismatch) {
			t.Errorf("expected ErrRankMismatch for rank 4, got %v", err)
		}
	})

	t.Run("tensor zero dimension", func(t *testing.T) {
		t0 := mustShape(t, 0, 100)
		p0 := mustShape(t, 10, 10)
		_, err := cudashape.InferGrid(t0, p0)
		if !errors.Is(err, cudashape.ErrNegativeDimension) {
			t.Errorf("expected ErrNegativeDimension for zero tensor dim, got %v", err)
		}
	})

	t.Run("partition zero dimension", func(t *testing.T) {
		t0 := mustShape(t, 100, 100)
		p0 := mustShape(t, 0, 10)
		_, err := cudashape.InferGrid(t0, p0)
		if !errors.Is(err, cudashape.ErrNegativeDimension) {
			t.Errorf("expected ErrNegativeDimension for zero partition dim, got %v", err)
		}
	})

	t.Run("overflow uint32", func(t *testing.T) {
		// math.MaxUint32 is 4294967295. If tensor extent is (MaxUint32 + 10) and partition is 1,
		// ceil division exceeds uint32 max.
		largeDim := int(math.MaxUint32) + 100
		tLarge, err := cudashape.New(largeDim)
		if err != nil {
			// On some platforms or limits, New might reject if > MaxByteLimit.
			// MaxByteLimit is 1<<48, and math.MaxUint32 is 1<<32, so New succeeds.
			t.Fatalf("cudashape.New: %v", err)
		}
		p1 := mustShape(t, 1)
		_, err = cudashape.InferGrid(tLarge, p1)
		if !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Errorf("expected ErrShapeOverflow, got %v", err)
		}
	})
	t.Run("CUDA Y limit", func(t *testing.T) {
		_, err := cudashape.InferGrid(mustShape(t, 1, 100000), mustShape(t, 1, 1))
		if !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("expected ErrShapeOverflow, got %v", err)
		}
	})
	t.Run("CUDA Z limit", func(t *testing.T) {
		_, err := cudashape.InferGrid(mustShape(t, 1, 1, 100000), mustShape(t, 1, 1, 1))
		if !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("expected ErrShapeOverflow, got %v", err)
		}
	})
	t.Run("CUDA Y limit boundary", func(t *testing.T) {
		got, err := cudashape.InferGrid(mustShape(t, 1, 65535), mustShape(t, 1, 1))
		if err != nil {
			t.Fatalf("InferGrid at Y limit: %v", err)
		}
		if got.Y != 65535 {
			t.Fatalf("InferGrid at Y limit = %v, want Y=65535", got)
		}
		if _, err := cudashape.InferGrid(mustShape(t, 1, 65536), mustShape(t, 1, 1)); !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("InferGrid above Y limit error = %v, want ErrShapeOverflow", err)
		}
	})
	t.Run("CUDA X upper boundary", func(t *testing.T) {
		got, err := cudashape.InferGrid(mustShape(t, 1<<31-1), mustShape(t, 1))
		if err != nil {
			t.Fatalf("InferGrid at X upper boundary: %v", err)
		}
		if got.X != 1<<31-1 {
			t.Fatalf("InferGrid at X upper boundary = %v, want X=%d", got, 1<<31-1)
		}
	})
	t.Run("CUDA X limit", func(t *testing.T) {
		_, err := cudashape.InferGrid(mustShape(t, 1<<31), mustShape(t, 1))
		if !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("expected ErrShapeOverflow, got %v", err)
		}
	})

	t.Run("Z grid upper boundary", func(t *testing.T) {
		atLimit, err := cudashape.InferGrid(mustShape(t, 1, 1, 65535), mustShape(t, 1, 1, 1))
		if err != nil {
			t.Fatalf("InferGrid at Z limit: %v", err)
		}
		if atLimit.Z != 65535 {
			t.Fatalf("InferGrid at Z limit = %v, want Z=65535", atLimit)
		}
		if _, err := cudashape.InferGrid(mustShape(t, 1, 1, 65536), mustShape(t, 1, 1, 1)); !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("InferGrid above Z limit error = %v, want ErrShapeOverflow", err)
		}
	})
}
