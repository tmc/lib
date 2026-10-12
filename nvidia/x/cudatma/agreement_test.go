package cudatma

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// agreeingPair returns a descriptor and a tile that address the same bytes:
// a 16x64 bf16 tile under a 128-byte swizzle.
func agreeingPair(t *testing.T) (Descriptor, cudalayout.SwizzledTile) {
	t.Helper()
	tile, err := cudalayout.NewSwizzledTile(16, 64, 2, cudalayout.TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	d := Descriptor{
		Target:       TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [MaxRank]uint64{64, 16},
		Strides:      [MaxRank]uint64{128},
		Box:          [MaxRank]uint32{64, 16},
		ElementBytes: 2,
		Swizzle:      128,
	}
	return d, tile
}

func TestAgreesWithTile(t *testing.T) {
	d, tile := agreeingPair(t)
	if err := d.AgreesWithTile(tile); err != nil {
		t.Fatalf("AgreesWithTile() = %v, want nil", err)
	}
}

// TestAgreesWithTileRejects is the gate: each case is a pair that a mode-only
// comparison accepts or that nothing downstream would ever report. Every
// mutation must be caught.
func TestAgreesWithTileRejects(t *testing.T) {
	tests := []struct {
		name string
		// mutate perturbs one side of an agreeing pair.
		mutate func(*Descriptor, *cudalayout.SwizzledTile, *testing.T)
		want   string
	}{
		{
			// The case AgreesWith alone cannot see: modes match, shapes do not.
			name: "same swizzle, narrower box",
			mutate: func(d *Descriptor, _ *cudalayout.SwizzledTile, _ *testing.T) {
				d.Box[0] = 32
			},
			want: "columns",
		},
		{
			name: "same swizzle, fewer rows",
			mutate: func(d *Descriptor, _ *cudalayout.SwizzledTile, _ *testing.T) {
				d.Box[1] = 8
			},
			want: "rows",
		},
		{
			name: "different swizzle mode",
			mutate: func(d *Descriptor, _ *cudalayout.SwizzledTile, _ *testing.T) {
				d.Swizzle = 64
			},
			want: "swizzle",
		},
		{
			name: "unswizzled descriptor, swizzled tile",
			mutate: func(d *Descriptor, _ *cudalayout.SwizzledTile, _ *testing.T) {
				d.Swizzle = 0
			},
			want: "swizzle",
		},
		{
			name: "different element width",
			mutate: func(d *Descriptor, tile *cudalayout.SwizzledTile, tt *testing.T) {
				d.ElementBytes = 4
				d.Box[0] = 32
				d.Dims[0] = 32
			},
			want: "element size",
		},
		{
			name: "invalid descriptor",
			mutate: func(d *Descriptor, _ *cudalayout.SwizzledTile, _ *testing.T) {
				d.Rank = 1
			},
			want: "does not describe a tensor map",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, tile := agreeingPair(t)
			tt.mutate(&d, &tile, t)
			err := d.AgreesWithTile(tile)
			if err == nil {
				t.Fatalf("AgreesWithTile() = nil, want an error mentioning %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("AgreesWithTile() = %v, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestAgreesWithTileElementWidth checks that a 32-bit pair agrees, so the
// element comparison is not passing by only ever seeing one width.
func TestAgreesWithTileElementWidth(t *testing.T) {
	tile, err := cudalayout.NewSwizzledTile(16, 32, 4, cudalayout.TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	d := Descriptor{
		Target:       TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [MaxRank]uint64{32, 16},
		Strides:      [MaxRank]uint64{128},
		Box:          [MaxRank]uint32{32, 16},
		ElementBytes: 4,
		Swizzle:      128,
	}
	if err := d.AgreesWithTile(tile); err != nil {
		t.Fatalf("AgreesWithTile() = %v, want nil", err)
	}
}

func TestAgreesWithTileRankOne(t *testing.T) {
	d, tile := agreeingPair(t)
	d.Rank = 1
	d.Strides = [MaxRank]uint64{}
	d.Box[1] = 0
	d.Dims[1] = 0
	tile, err := cudalayout.NewSwizzledTile(1, 64, 2, cudalayout.TMASwizzle128B)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.AgreesWithTile(tile); err != nil {
		t.Fatal(err)
	}
}

func TestAgreesWithTileOuterPlanes(t *testing.T) {
	d, tile := agreeingPair(t)
	d.Rank = 3
	d.Dims[2] = 2
	d.Box[2] = 2
	d.Strides[1] = 2048
	if err := d.AgreesWithTile(tile); err == nil {
		t.Fatal("two axes admitted outer planes")
	}
	d.Interleave = 1
	if err := d.AgreesWithTile(tile); err == nil {
		t.Fatal("two axes admitted interleaved physical agreement")
	}
}
