package dsv41

import (
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudakvrot"
	"github.com/tmc/lib/nvidia/x/cudamoe"
	"github.com/tmc/lib/nvidia/x/mxfp4"
)

// YoCoTopology builds the 40-layer You-Only-Cache-Once (YoCo) shared KV topology
// for V4.1 Flash from Schedule, mapping producer layers (0, 1, 2, 8, 14, 20)
// and consumer layers (3..7 -> 2, 9..13 -> 8, 15..19 -> 14, 21..39 -> 20).
func YoCoTopology() (*cudakv.SharedLayerTopology, error) {
	ownerLayer := [4]int{2, 8, 14, 20}
	specs := make([]cudakv.SharedLayerSpec, 40)
	for i := 0; i < 40; i++ {
		l, err := Schedule(i)
		if err != nil {
			return nil, err
		}
		switch {
		case l.Owner < 0:
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: i,
				Role:          cudakv.LayerKVProducer,
				IsSWA:         true,
				WindowSize:    512,
			}
		case l.PublishKV:
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: i,
				Role:          cudakv.LayerKVProducer,
				IsSWA:         false,
				WindowSize:    512,
			}
		default:
			src := ownerLayer[l.Owner]
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: src,
				Role:          cudakv.LayerKVSharedConsumer,
				IsSWA:         false,
				WindowSize:    512,
			}
		}
	}
	return cudakv.NewSharedLayerTopology(specs)
}

// SWAReplayWindow computes the bounded 512-token SWA replay window for a V4.1
// layer using github.com/tmc/lib/nvidia/x/cudakvrot.
func SWAReplayWindow(seqLenK, queryLen, speculativeOvershoot int) (cudakvrot.SWAReplayWindow, error) {
	return cudakvrot.ResolveSWABoundedReplay(seqLenK, queryLen, cudakvrot.SWAReplayConfig{
		WindowSize:           512,
		SpeculativeOvershoot: speculativeOvershoot,
		PageBlockSize:        64,
	})
}

// DefaultCandidateBlockConfig returns the V4.1 Flash DeepSelect block-level top-k
// configuration (block size 8, 2048 candidate blocks, 512 top tokens).
func DefaultCandidateBlockConfig() cudaattn.CandidateBlockConfig {
	return cudaattn.CandidateBlockConfig{
		BlockSize:        8,
		BlockTopK:        2048,
		TokenTopK:        512,
		RecentBlocks:     1,
		ChunkBlocks:      8,
		ComputeChunkSize: 64,
	}
}

// EngramLayerMultipliers computes the deterministic odd hash multipliers for a
// V4.1 Engram layer (layers 1 and 14) via github.com/tmc/lib/nvidia/x/cudamoe.
func EngramLayerMultipliers(layer, maxNgramSize int, seed int64) []int64 {
	return cudamoe.ComputeEngramMultipliers(maxNgramSize, layer, seed)
}

// QuantizeNVFP4 packs 16-element blocks of float32 weights into V4.1 NVFP4
// (packed E2M1 nibbles + E4M3 per-16-element scales) via github.com/tmc/lib/nvidia/x/mxfp4.
func QuantizeNVFP4(src []float32) (packed []byte, e4m3Scales []uint8, err error) {
	return mxfp4.PackE2M1WithE4M3Scale(src)
}

// DequantizeNVFP4 unpacks V4.1 NVFP4 blocks back to float32.
func DequantizeNVFP4(packed []byte, e4m3Scales []uint8, count int) ([]float32, error) {
	return mxfp4.UnpackE2M1WithE4M3Scale(packed, e4m3Scales, count)
}
