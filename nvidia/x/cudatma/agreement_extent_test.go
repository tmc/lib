package cudatma_test

import (
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudatma"
)

func TestAgreesWithTileExtentWidth(t *testing.T) {
	d := cudatma.Descriptor{
		Target:       cudatma.TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [cudatma.MaxRank]uint64{64, 16},
		Strides:      [cudatma.MaxRank]uint64{128},
		Box:          [cudatma.MaxRank]uint32{64, 16},
		ElementBytes: 2,
		Swizzle:      128,
	}
	if err := d.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name       string
		rows, cols uint64
		want       string
	}{
		{"ordinary", 16, 64, ""},
		{"row wrap", 16 + 1<<32, 64, "rows"},
		{"column wrap", 16, 64 + 1<<32, "columns"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if strconv.IntSize < 64 && (tt.rows > 1<<31-1 || tt.cols > 1<<31-1) {
				t.Skip("tile extent needs a 64-bit int")
			}
			tile, err := cudalayout.NewSwizzledTile(int(tt.rows), int(tt.cols), 2, cudalayout.TMASwizzle128B)
			if err != nil {
				t.Fatalf("NewSwizzledTile: %v", err)
			}
			err = d.AgreesWithTile(tile)
			if tt.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("AgreesWithTile = %v, want %s mismatch", err, tt.want)
			}
		})
	}
}
