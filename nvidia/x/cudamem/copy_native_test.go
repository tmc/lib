//go:build native

package cudamem_test

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestCopyWithLayoutsNative(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native copy qualification requires gpu-lock")
	}
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, tc := range []struct {
		name, source, destination string
		waw                       bool
	}{
		{"contiguous", "4:1", "4:1", false},
		{"residual", "(6,4):(1,6)", "(4,6):(1,4)", false},
		{"incompatible", "(5,7):(7,1)", "(7,5):(5,1)", false},
		{"zero stride", "4:0", "4:0", false},
		{"overlap", "(2,2):(1,1)", "(2,2):(1,1)", false},
		{"gaps", "(4,2):(2,16)", "(4,2):(3,20)", false},
		{"write after write", "4:1", "4:0", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srcLayout, err := cudalayout.ParseLayout(tc.source)
			if err != nil {
				t.Fatal(err)
			}
			dstLayout, err := cudalayout.ParseLayout(tc.destination)
			if err != nil {
				t.Fatal(err)
			}
			sourceSize, err := srcLayout.Cosize()
			if err != nil {
				t.Fatal(err)
			}
			destinationSize, err := dstLayout.Cosize()
			if err != nil {
				t.Fatal(err)
			}
			src, err := cudamem.Alloc[uint32](dev, sourceSize)
			t.Cleanup(func() {
				if err := src.Free(); err != nil {
					t.Error(err)
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			dst, err := cudamem.Alloc[uint32](dev, destinationSize)
			t.Cleanup(func() {
				if err := dst.Free(); err != nil {
					t.Error(err)
				}
			})
			if err != nil {
				t.Fatal(err)
			}
			source := make([]uint32, sourceSize)
			for i := range source {
				source[i] = 0x01020300 + uint32(i)
			}
			want := make([]uint32, destinationSize)
			for i := range want {
				want[i] = 0xa5a5a5a5
			}
			if err := src.CopyFromHost(source); err != nil {
				t.Fatal(err)
			}
			if err := dst.CopyFromHost(want); err != nil {
				t.Fatal(err)
			}
			plan, err := cudamem.CopyWithLayouts(dst, src, srcLayout, dstLayout)
			if tc.waw {
				if !errors.Is(err, cudamem.ErrWriteAfterWrite) {
					t.Fatalf("copy error = %v, want WAW", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				for i := 0; i < srcLayout.Size(); i++ {
					so, err := srcLayout.Index(i)
					if err != nil {
						t.Fatal(err)
					}
					do, err := dstLayout.Index(i)
					if err != nil {
						t.Fatal(err)
					}
					want[do] = source[so]
				}
			}
			got := make([]uint32, destinationSize)
			if err := dst.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, want) {
				t.Fatalf("device copy = %v, want %v", got, want)
			}
			t.Logf("actual device copy %s -> %s: width=%d collapse=%d; WAW=%t, %d elements checked", tc.source, tc.destination, plan.VectorWidth, plan.CollapseFactor, tc.waw, len(got))
		})
	}
}
