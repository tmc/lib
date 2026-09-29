package cudalayout_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalayout"
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

func mustPartition(t *testing.T, ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...cudalayout.PartitionOption) cudalayout.Partition {
	t.Helper()
	p, err := cudalayout.NewPartition(ptr, tensorShape, partitionShape, elemBytes, opts...)
	if err != nil {
		t.Fatalf("NewPartition: unexpected error %v", err)
	}
	return p
}

func mustPartitionMut(t *testing.T, ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...cudalayout.PartitionOption) cudalayout.PartitionMut {
	t.Helper()
	p, err := cudalayout.NewPartitionMut(ptr, tensorShape, partitionShape, elemBytes, opts...)
	if err != nil {
		t.Fatalf("NewPartitionMut: unexpected error %v", err)
	}
	return p
}

func TestPartition_ValidateDisjoint(t *testing.T) {
	t.Run("overflowing stride extent is rejected", func(t *testing.T) {
		_, err := cudalayout.NewPartition(0, mustShape(t, 4, 4), mustShape(t, 4, 4), 4,
			cudalayout.WithPartitionStrides([]int{1 << 62, 1 << 62}))
		if !errors.Is(err, cudalayout.ErrInvalidPartition) || !errors.Is(err, cudashape.ErrShapeOverflow) {
			t.Fatalf("want invalid partition and shape overflow, got %v", err)
		}
	})
	t.Run("masked outer tails are valid", func(t *testing.T) {
		p1 := mustPartition(t, 1, mustShape(t, 10), mustShape(t, 4), 4)
		if err := p1.ValidateDisjoint(); err != nil {
			t.Fatalf("1D masked tail: %v", err)
		}
		p2 := mustPartition(t, 1, mustShape(t, 10, 8), mustShape(t, 4, 8), 4)
		if err := p2.ValidateDisjoint(); err != nil {
			t.Fatalf("2D outer masked tail: %v", err)
		}
	})
	t.Run("3D column-major order", func(t *testing.T) {
		p := mustPartition(t, 1, mustShape(t, 16, 16, 16), mustShape(t, 4, 4, 4), 4,
			cudalayout.WithPartitionStrides([]int{1, 16, 256}))
		if err := p.ValidateDisjoint(); err != nil {
			t.Fatalf("column-major partition: %v", err)
		}
	})
	t.Run("packed 2D column-major partial inner axis wraps", func(t *testing.T) {
		p := mustPartition(t, 1, mustShape(t, 10, 8), mustShape(t, 4, 8), 4,
			cudalayout.WithPartitionStrides([]int{1, 10}))
		if err := p.ValidateDisjoint(); !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("want wrapping error, got %v", err)
		}
	})
	t.Run("3D column-major partial inner axis wraps", func(t *testing.T) {
		p := mustPartition(t, 1, mustShape(t, 10, 16, 16), mustShape(t, 4, 4, 4), 4,
			cudalayout.WithPartitionStrides([]int{1, 10, 160}))
		if err := p.ValidateDisjoint(); !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("want wrapping error, got %v", err)
		}
	})
	t.Run("3D axis 2 partial tail wraps", func(t *testing.T) {
		p := mustPartition(t, 1, mustShape(t, 16, 1024, 1000), mustShape(t, 2, 64, 64), 4,
			cudalayout.WithPartitionStrides([]int{1024 * 1000, 1000, 1}))
		if err := p.ValidateDisjoint(); !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("want ErrWrappingPartition, got %v", err)
		}
	})
	t.Run("3D overlapping middle stride", func(t *testing.T) {
		p := mustPartition(t, 1, mustShape(t, 16, 16, 16), mustShape(t, 2, 4, 4), 4,
			cudalayout.WithPartitionStrides([]int{256, 8, 1}))
		if err := p.ValidateDisjoint(); !errors.Is(err, cudalayout.ErrOverlappingPartition) {
			t.Fatalf("want overlap error, got %v", err)
		}
	})
	t.Run("NewPartitionFromTile refuses swizzle", func(t *testing.T) {
		tile, err := cudalayout.NewTile(8, 8, 4, cudalayout.WithSwizzle(cudalayout.Swizzle32B))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := cudalayout.NewPartitionFromTile(1, tile, 4, 4); !errors.Is(err, cudalayout.ErrInvalidPartition) {
			t.Fatalf("want invalid partition, got %v", err)
		}
	})
	t.Run("valid evenly divisible 2D partition", func(t *testing.T) {
		tShape := mustShape(t, 1024, 1024)
		pShape := mustShape(t, 64, 64)
		p := mustPartition(t, driver.DevicePtr(0x1000), tShape, pShape, 4)

		if err := p.ValidateDisjoint(); err != nil {
			t.Fatalf("ValidateDisjoint failed: %v", err)
		}
		if p.IsMutable() {
			t.Errorf("expected immutable Partition, got mutable")
		}
		if p.Ptr() != driver.DevicePtr(0x1000) {
			t.Errorf("Ptr() = %v, want 0x1000", p.Ptr())
		}
		if p.TileStride() != 1024 {
			t.Errorf("TileStride() = %d, want 1024", p.TileStride())
		}
		partStrides := p.PartitionStrides()
		if len(partStrides) != 2 || partStrides[0] != 64*1024 || partStrides[1] != 64*1 {
			t.Errorf("PartitionStrides() = %v, want [65536, 64]", partStrides)
		}
	})

	t.Run("valid ceiling covered 2D partition with stride padding", func(t *testing.T) {
		tShape := mustShape(t, 1000, 1000)
		pShape := mustShape(t, 64, 64)
		// Grid along axis 1 is ceil(1000/64) = 16.
		// Stride padding >= 16 * 64 = 1024 prevents row wrapping.
		p := mustPartition(t, driver.DevicePtr(0x2000), tShape, pShape, 4,
			cudalayout.WithPartitionStrides([]int{1024, 1}))

		if err := p.ValidateDisjoint(); err != nil {
			t.Fatalf("ValidateDisjoint with padding failed: %v", err)
		}
	})

	t.Run("invalid non-divisible wrapping partition on packed tensor", func(t *testing.T) {
		// Non-divisible column partition on packed tensor (stride 1000 < 16 * 64 = 1024).
		tShape := mustShape(t, 1000, 1000)
		pShape := mustShape(t, 64, 64)
		p := mustPartition(t, driver.DevicePtr(0x3000), tShape, pShape, 4,
			cudalayout.WithPartitionStrides([]int{1000, 1}))

		err := p.ValidateDisjoint()
		if !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("expected ErrWrappingPartition for non-divisible packed partition, got %v", err)
		}
	})
	t.Run("wrapping boundary on padded stride", func(t *testing.T) {
		tShape := mustShape(t, 1000, 1000)
		pShape := mustShape(t, 64, 64)
		for _, tt := range []struct {
			name      string
			outer     int
			wantError bool
		}{
			{name: "one below grid envelope", outer: 1023, wantError: true},
			{name: "at grid envelope", outer: 1024},
		} {
			t.Run(tt.name, func(t *testing.T) {
				p := mustPartition(t, 1, tShape, pShape, 4,
					cudalayout.WithPartitionStrides([]int{tt.outer, 1}))
				err := p.ValidateDisjoint()
				if tt.wantError && !errors.Is(err, cudalayout.ErrWrappingPartition) {
					t.Fatalf("ValidateDisjoint() = %v, want ErrWrappingPartition", err)
				}
				if !tt.wantError && err != nil {
					t.Fatalf("ValidateDisjoint() = %v, want nil", err)
				}
			})
		}
	})

	t.Run("invalid tile column span exceeding stride", func(t *testing.T) {
		tShape := mustShape(t, 100, 100)
		pShape := mustShape(t, 10, 120) // tile column span 120 > row stride 100
		p := mustPartition(t, driver.DevicePtr(0x4000), tShape, pShape, 4,
			cudalayout.WithPartitionStrides([]int{100, 1}))

		err := p.ValidateDisjoint()
		if !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("expected ErrWrappingPartition for tile span exceeding stride, got %v", err)
		}
	})

	t.Run("invalid stride smaller than tensor width", func(t *testing.T) {
		tShape := mustShape(t, 100, 100)
		pShape := mustShape(t, 10, 10)
		p := mustPartition(t, driver.DevicePtr(0x5000), tShape, pShape, 4,
			cudalayout.WithPartitionStrides([]int{50, 1})) // row stride 50 < row width 100

		err := p.ValidateDisjoint()
		if !errors.Is(err, cudalayout.ErrOverlappingPartition) {
			t.Fatalf("expected ErrOverlappingPartition for stride < width, got %v", err)
		}
	})
	t.Run("overlap boundary at tensor width", func(t *testing.T) {
		tShape := mustShape(t, 100, 100)
		pShape := mustShape(t, 10, 10)
		for _, tt := range []struct {
			name      string
			outer     int
			wantError bool
		}{
			{name: "one below width", outer: 99, wantError: true},
			{name: "at width", outer: 100},
		} {
			t.Run(tt.name, func(t *testing.T) {
				p := mustPartition(t, 1, tShape, pShape, 4,
					cudalayout.WithPartitionStrides([]int{tt.outer, 1}))
				err := p.ValidateDisjoint()
				if tt.wantError && !errors.Is(err, cudalayout.ErrOverlappingPartition) {
					t.Fatalf("ValidateDisjoint() = %v, want ErrOverlappingPartition", err)
				}
				if !tt.wantError && err != nil {
					t.Fatalf("ValidateDisjoint() = %v, want nil", err)
				}
			})
		}
	})

	t.Run("valid 1D partition", func(t *testing.T) {
		tShape := mustShape(t, 1000)
		pShape := mustShape(t, 128)
		p := mustPartition(t, driver.DevicePtr(0x6000), tShape, pShape, 4)

		if err := p.ValidateDisjoint(); err != nil {
			t.Fatalf("ValidateDisjoint 1D failed: %v", err)
		}
	})

	t.Run("valid 3D partition", func(t *testing.T) {
		tShape := mustShape(t, 16, 1024, 1024)
		pShape := mustShape(t, 2, 64, 64)
		p := mustPartition(t, driver.DevicePtr(0x7000), tShape, pShape, 4)

		if err := p.ValidateDisjoint(); err != nil {
			t.Fatalf("ValidateDisjoint 3D failed: %v", err)
		}
	})

	t.Run("invalid 3D non-divisible wrapping partition", func(t *testing.T) {
		tShape := mustShape(t, 16, 1000, 1000)
		pShape := mustShape(t, 2, 64, 64)
		// packed strides: [1000000, 1000, 1]
		p := mustPartition(t, driver.DevicePtr(0x8000), tShape, pShape, 4,
			cudalayout.WithPartitionStrides([]int{1000000, 1000, 1}))

		err := p.ValidateDisjoint()
		if !errors.Is(err, cudalayout.ErrWrappingPartition) {
			t.Fatalf("expected ErrWrappingPartition for 3D wrapping, got %v", err)
		}
	})

	t.Run("mutable partition view", func(t *testing.T) {
		tShape := mustShape(t, 64, 64)
		pShape := mustShape(t, 16, 16)
		pMut := mustPartitionMut(t, driver.DevicePtr(0x9000), tShape, pShape, 4)

		if !pMut.IsMutable() {
			t.Errorf("PartitionMut should report IsMutable() = true")
		}
		readOnly := pMut.AsReadOnly()
		if readOnly.IsMutable() {
			t.Errorf("AsReadOnly() should report IsMutable() = false")
		}
		if err := pMut.ValidateDisjoint(); err != nil {
			t.Fatalf("pMut.ValidateDisjoint failed: %v", err)
		}
	})
}

func TestPartition_TileStride32(t *testing.T) {
	for _, tt := range []struct {
		name      string
		stride    int
		want      int32
		wantError bool
	}{
		{name: "fits", stride: 64, want: 64},
		{name: "max int32", stride: (1 << 31) - 1, want: (1 << 31) - 1},
		{name: "overflows", stride: 1 << 32, wantError: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := mustPartition(t, 1, mustShape(t, 2), mustShape(t, 1), 4,
				cudalayout.WithPartitionStrides([]int{tt.stride}))
			got, err := p.TileStride32()
			if tt.wantError {
				if err == nil {
					t.Fatal("TileStride32() returned nil error for overflowing stride")
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("TileStride32() = (%d, %v), want (%d, nil)", got, err, tt.want)
			}
		})
	}
}
