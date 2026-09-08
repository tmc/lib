package gomlx

import (
	"os"
	"slices"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func BenchmarkTransfer(b *testing.B) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		b.Skip("set TMC_COMPUTE_PLUGIN to benchmark native transfers")
	}
	backend, err := New(path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if err := backend.Close(); err != nil {
			b.Error(err)
		}
	}()
	for _, tc := range []struct {
		name string
		size int
	}{{"scalar", 1}, {"logits", 151936}} {
		b.Run(tc.name, func(b *testing.B) {
			data := make([]float32, tc.size)
			for i := range data {
				data[i] = float32(i%16+1) / 16
			}
			shape := shapes.Make(dtypes.Float32, tc.size)
			input, err := backend.BufferFromFlatData(0, data, shape)
			if err != nil {
				b.Fatal(err)
			}
			defer func() {
				if err := input.Finalize(); err != nil {
					b.Error(err)
				}
			}()
			dst := make([]float32, len(data))
			if err := input.ToFlatData(dst); err != nil {
				b.Fatal(err)
			}
			if !slices.Equal(data, dst) {
				b.Fatal("transfer correctness mismatch")
			}
			wrong := slices.Clone(dst)
			wrong[0]++
			if slices.Equal(data, wrong) {
				b.Fatal("wrong-result control accepted")
			}
			b.Run("upload", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(len(data) * 4))
				for b.Loop() {
					v, err := backend.BufferFromFlatData(0, data, shape)
					if err != nil {
						b.Fatal(err)
					}
					if err := v.Finalize(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("download", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(len(data) * 4))
				for b.Loop() {
					if err := input.ToFlatData(dst); err != nil {
						b.Fatal(err)
					}
				}
				if !slices.Equal(data, dst) {
					b.Fatal("download correctness mismatch")
				}
			})
		})
	}
}
