package cudaprepack_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
)

func tensorBox() cudaprepack.BoxDescriptor {
	return cudaprepack.BoxDescriptor{
		Rank: 2, DataType: cudaprepack.TypeFloat16, GlobalAddress: 4096,
		GlobalDims: [3]uint64{64, 8, 1}, GlobalStrides: [3]int64{128, 0, 0},
		BoxDims: [3]uint32{16, 4, 1}, ElementStrides: [3]uint32{1, 1, 1},
	}
}

func TestTensorMapScalarProfiles(t *testing.T) {
	for _, arch := range []string{cudaprepack.ArchSM100a, cudaprepack.ArchSM121a} {
		for _, op := range []cudaprepack.TensorMapOperation{cudaprepack.TensorMapTiledLoad, cudaprepack.TensorMapTiledStore} {
			for _, dt := range []cudaprepack.DataType{cudaprepack.TypeFloat16, cudaprepack.TypeFloat32} {
				for _, rank := range []int{2, 3} {
					for _, sw := range []cudaprepack.SwizzleMode{cudaprepack.SwizzleNone, cudaprepack.Swizzle32B, cudaprepack.Swizzle64B, cudaprepack.Swizzle128B} {
						t.Run(fmt.Sprintf("%s/%s/%s/%d/%s", arch, op, dt, rank, sw), func(t *testing.T) {
							b := tensorBox()
							b.DataType, b.Rank, b.Swizzle = dt, rank, sw
							b.BoxDims[0] = uint32(32 / (dt.BitsPerElement() / 8))
							b.GlobalDims[0], b.GlobalStrides[0] = 256, 1024
							if rank == 3 {
								b.GlobalDims[2], b.GlobalStrides[1] = 2, 8192
							}
							if err := b.ValidateTensorMap(arch, op); err != nil {
								t.Fatal(err)
							}
						})
					}
				}
			}
		}
	}
}

func TestTensorMapFieldBoundaries(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cudaprepack.BoxDescriptor)
	}{
		{"largest unswizzled scalar box", func(b *cudaprepack.BoxDescriptor) {
			b.DataType = cudaprepack.TypeFloat32
			b.GlobalDims[0] = 256
			b.GlobalStrides[0] = 1024
			b.BoxDims[0] = 256
		}},
		{"base need not align to box boundary", func(b *cudaprepack.BoxDescriptor) { b.GlobalAddress += 16; b.BoxDims[0] = 64 }},
		{"dimension boundary", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[0] = 1 << 32; b.GlobalStrides[0] = 1 << 33 }},
		{"stride boundary", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = 1<<40 - 16 }},
		{"sampling", func(b *cudaprepack.BoxDescriptor) { b.ElementStrides = [3]uint32{8, 8, 1} }},
		{"load tail", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[1] = 1 }},
		{"interleave16", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides[1] = 1024
			b.Interleave = cudaprepack.Interleave16B
			b.BoxDims[0] = 1
		}},
		{"interleave32", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides[1] = 1024
			b.Interleave = cudaprepack.Interleave32B
			b.Swizzle = cudaprepack.Swizzle32B
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := tensorBox()
			tt.change(&b)
			if err := b.ValidateTensorMap(cudaprepack.ArchSM121a, cudaprepack.TensorMapTiledLoad); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTensorMapRejectedFields(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cudaprepack.BoxDescriptor)
	}{
		{"dimension limit", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[0] = 1<<32 + 1; b.GlobalStrides[0] = 1 << 35 }},
		{"box limit", func(b *cudaprepack.BoxDescriptor) { b.BoxDims[1] = 257 }},
		{"sampling limit", func(b *cudaprepack.BoxDescriptor) { b.ElementStrides[1] = 9 }},
		{"stride limit", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = 1 << 40 }},
		{"inner width", func(b *cudaprepack.BoxDescriptor) { b.BoxDims[0] = 7 }},
		{"swizzle span", func(b *cudaprepack.BoxDescriptor) { b.BoxDims[0] = 32; b.Swizzle = cudaprepack.Swizzle32B }},
		{"atom swizzle subset", func(b *cudaprepack.BoxDescriptor) { b.Swizzle = cudaprepack.Swizzle128BAtom32B }},
		{"interleave rank", func(b *cudaprepack.BoxDescriptor) { b.Interleave = cudaprepack.Interleave16B }},
		{"interleave32 base", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides[1] = 1024
			b.Interleave = cudaprepack.Interleave32B
			b.Swizzle = cudaprepack.Swizzle32B
			b.GlobalAddress += 16
		}},
		{"interleave32 swizzle", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides[1] = 1024
			b.Interleave = cudaprepack.Interleave32B
		}},
		{"interleave32 stride", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides = [3]int64{144, 1152, 0}
			b.Interleave = cudaprepack.Interleave32B
			b.Swizzle = cudaprepack.Swizzle32B
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := tensorBox()
			tt.change(&b)
			if err := b.ValidateTensorMap(cudaprepack.ArchSM121a, cudaprepack.TensorMapTiledLoad); err == nil {
				t.Fatal("invalid tiled metadata accepted")
			}
		})
	}
	b := tensorBox()
	if err := b.ValidateTensorMap("sm_999a", cudaprepack.TensorMapTiledLoad); err == nil {
		t.Fatal("unknown target accepted")
	}
	for _, op := range []cudaprepack.TensorMapOperation{"", "im2col", "tiled-reduction", "global-to-tmem"} {
		if err := b.ValidateTensorMap(cudaprepack.ArchSM121a, op); err == nil {
			t.Fatalf("unknown operation %q accepted", op)
		}
	}
	b.GlobalDims[1] = 1
	if err := b.ValidateTensorMap(cudaprepack.ArchSM121a, cudaprepack.TensorMapTiledStore); err == nil {
		t.Fatal("store outside full-box profile accepted")
	}
}

func TestHostDescriptorMalformedStorage(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*cudaprepack.BoxDescriptor)
	}{
		{"rank", func(b *cudaprepack.BoxDescriptor) { b.Rank = 4 }},
		{"dtype", func(b *cudaprepack.BoxDescriptor) { b.DataType = "future" }},
		{"swizzle", func(b *cudaprepack.BoxDescriptor) { b.Swizzle = 99 }},
		{"interleave", func(b *cudaprepack.BoxDescriptor) { b.Interleave = 99 }},
		{"base", func(b *cudaprepack.BoxDescriptor) { b.GlobalAddress++ }},
		{"zero dimension", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[0] = 0 }},
		{"zero box", func(b *cudaprepack.BoxDescriptor) { b.BoxDims[1] = 0 }},
		{"zero element stride", func(b *cudaprepack.BoxDescriptor) { b.ElementStrides[1] = 0 }},
		{"negative stride", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = -128 }},
		{"zero stride", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = 0 }},
		{"unaligned stride", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = 129 }},
		{"overlapping rows", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[0] = 16 }},
		{"overlapping planes", func(b *cudaprepack.BoxDescriptor) { b.Rank = 3; b.GlobalDims[2] = 2; b.GlobalStrides[1] = 128 }},
		{"unused dimension", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[2] = 2 }},
		{"unused box", func(b *cudaprepack.BoxDescriptor) { b.BoxDims[2] = 2 }},
		{"unused element stride", func(b *cudaprepack.BoxDescriptor) { b.ElementStrides[2] = 2 }},
		{"unused global stride", func(b *cudaprepack.BoxDescriptor) { b.GlobalStrides[1] = 128 }},
		{"row overflow", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[0] = math.MaxUint64 }},
		{"storage overflow", func(b *cudaprepack.BoxDescriptor) { b.GlobalDims[1] = math.MaxUint64; b.GlobalAddress = 0 }},
		{"address overflow", func(b *cudaprepack.BoxDescriptor) { b.GlobalAddress = math.MaxUint64 - 15 }},
		{"box overflow", func(b *cudaprepack.BoxDescriptor) {
			b.Rank = 3
			b.GlobalDims[2] = 2
			b.GlobalStrides[1] = 1024
			b.BoxDims = [3]uint32{math.MaxUint32, math.MaxUint32, math.MaxUint32}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := tensorBox()
			tt.change(&b)
			if err := b.Validate(cudaprepack.ArchSM121a); err == nil {
				t.Fatal("invalid host metadata accepted")
			}
			if b.TotalSizeBytes() != 0 || b.BoxSizeBytes() != 0 {
				t.Fatal("malformed metadata returned usable byte extents")
			}
		})
	}
	var nilBox *cudaprepack.BoxDescriptor
	if nilBox.Validate(cudaprepack.ArchSM121a) == nil || nilBox.TotalSizeBytes() != 0 || nilBox.BoxSizeBytes() != 0 {
		t.Fatal("nil descriptor accepted")
	}
	b := tensorBox()
	if got := b.TotalSizeBytes(); got != 1024 {
		t.Fatal("host padded size", got)
	}
	if got := b.BoxSizeBytes(); got != 128 {
		t.Fatal("host box size", got)
	}
}

func TestHostFormatsAreNotTensorMapDtypes(t *testing.T) {
	for _, dt := range []cudaprepack.DataType{cudaprepack.TypeQ8_0, cudaprepack.TypeIQ2_XXS, cudaprepack.TypeMXFP4, cudaprepack.TypeFP8_E4M3, cudaprepack.TypeFP8_E5M2} {
		t.Run(string(dt), func(t *testing.T) {
			b := tensorBox()
			b.DataType = dt
			if err := b.Validate(cudaprepack.ArchSM121a); err != nil {
				t.Fatal("host storage refused", err)
			}
			for _, op := range []cudaprepack.TensorMapOperation{cudaprepack.TensorMapTiledLoad, cudaprepack.TensorMapTiledStore} {
				if err := b.ValidateTensorMap(cudaprepack.ArchSM121a, op); err == nil {
					t.Fatal("host format reinterpreted as a CUDA scalar dtype")
				}
			}
		})
	}
	for _, tt := range []struct {
		name string
		pack func() (*cudaprepack.PrepackedTensor, error)
	}{
		{"q8", func() (*cudaprepack.PrepackedTensor, error) { return cudaprepack.RepackQ8_0(make([]byte, 34), 1, 32) }},
		{"iq2", func() (*cudaprepack.PrepackedTensor, error) {
			return cudaprepack.RepackIQ2_XXS(make([]byte, 66), 1, 256)
		}},
		{"mxfp4", func() (*cudaprepack.PrepackedTensor, error) {
			return cudaprepack.PackMXFP4(make([]float32, 32), 1, 32, 32)
		}},
		{"fp8", func() (*cudaprepack.PrepackedTensor, error) {
			return cudaprepack.QuantizeFP8(make([]float32, 32), 1, 32, cudaprepack.TypeFP8_E4M3)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p, err := tt.pack()
			if err != nil {
				t.Fatal(err)
			}
			if err := p.BoxDesc.Validate(p.Arch); err != nil {
				t.Fatal("ordinary pack no longer describes valid host storage", err)
			}
		})
	}
}
