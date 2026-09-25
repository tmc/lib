package cudagroup

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func FuzzRankNeverPanics(f *testing.F) {
	f.Add(uint32(0), uint32(0), uint32(0))
	f.Add(uint32(31), uint32(0), uint32(0))
	f.Fuzz(func(t *testing.T, x, y, z uint32) {
		g, err := New(CTA, driver.Dim3{X: 256, Y: 2, Z: 1})
		if err != nil {
			t.Fatal(err)
		}
		_, _ = g.Rank(driver.Dim3{X: x, Y: y, Z: z})
	})
}
