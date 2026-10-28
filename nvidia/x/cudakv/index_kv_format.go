package cudakv

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// KVPageTokenFormat identifies a sub-byte paged KV or indexer KV token storage layout.
type KVPageTokenFormat int

const (
	// FormatMLA_V32_FP8 is the DeepSeek-V3.2 FP8 paged KV token layout (656 B/token: 512B NoPE, 128B RoPE, 16B scales at 640).
	FormatMLA_V32_FP8 KVPageTokenFormat = iota + 1
	// FormatMLA_V41_FP8 is the DeepSeek-V4.1 FP8 paged KV token layout (528 B/token: 448B NoPE, 64B RoPE, 16B scales at 512).
	FormatMLA_V41_FP8
	// FormatMLA_V41_FP4 is the DeepSeek-V4.1 FP4 paged KV token layout (288 B/token: 224B NoPE, 32B RoPE, 28B scales at 256, 4B pad).
	FormatMLA_V41_FP4
	// FormatIndexKV_MXFP8 is the DeepSeek-V4.1 MXFP8 indexer KV token layout (132 B/token: 128B data, 4B UE8M0 scales at 128).
	FormatIndexKV_MXFP8
	// FormatIndexKV_FP4 is the DeepSeek-V4.1 FP4 indexer KV token layout (68 B/token: 32B data, 4B scales at 64).
	FormatIndexKV_FP4
)

// KVPageFormatDescriptor describes the byte layout of a single token in a paged KV block.
type KVPageFormatDescriptor struct {
	Format        KVPageTokenFormat
	BytesPerToken int
	NoPEBytes     int
	RoPEBytes     int
	DataBytes     int
	ScaleBytes    int
	ScaleOffset   int
	PaddingBytes  int
}

// Descriptor returns the byte layout descriptor for f.
func (f KVPageTokenFormat) Descriptor() (KVPageFormatDescriptor, error) {
	switch f {
	case FormatMLA_V32_FP8:
		return KVPageFormatDescriptor{
			Format:        FormatMLA_V32_FP8,
			BytesPerToken: 656,
			NoPEBytes:     512,
			RoPEBytes:     128,
			DataBytes:     640,
			ScaleBytes:    16,
			ScaleOffset:   640,
			PaddingBytes:  0,
		}, nil
	case FormatMLA_V41_FP8:
		return KVPageFormatDescriptor{
			Format:        FormatMLA_V41_FP8,
			BytesPerToken: 528,
			NoPEBytes:     448,
			RoPEBytes:     64,
			DataBytes:     512,
			ScaleBytes:    16,
			ScaleOffset:   512,
			PaddingBytes:  0,
		}, nil
	case FormatMLA_V41_FP4:
		return KVPageFormatDescriptor{
			Format:        FormatMLA_V41_FP4,
			BytesPerToken: 288,
			NoPEBytes:     224,
			RoPEBytes:     32,
			DataBytes:     256,
			ScaleBytes:    28,
			ScaleOffset:   256,
			PaddingBytes:  4,
		}, nil
	case FormatIndexKV_MXFP8:
		return KVPageFormatDescriptor{
			Format:        FormatIndexKV_MXFP8,
			BytesPerToken: 132,
			NoPEBytes:     128,
			RoPEBytes:     0,
			DataBytes:     128,
			ScaleBytes:    4,
			ScaleOffset:   128,
			PaddingBytes:  0,
		}, nil
	case FormatIndexKV_FP4:
		return KVPageFormatDescriptor{
			Format:        FormatIndexKV_FP4,
			BytesPerToken: 68,
			NoPEBytes:     32,
			RoPEBytes:     0,
			DataBytes:     32,
			ScaleBytes:    4,
			ScaleOffset:   64,
			PaddingBytes:  32,
		}, nil
	default:
		return KVPageFormatDescriptor{}, fmt.Errorf("cudakv: unknown kv page token format %d: %w", int(f), cudaerrors.ErrUnsupported)
	}
}

// String returns the canonical name of f.
func (f KVPageTokenFormat) String() string {
	switch f {
	case FormatMLA_V32_FP8:
		return "MLA_V32_FP8"
	case FormatMLA_V41_FP8:
		return "MLA_V41_FP8"
	case FormatMLA_V41_FP4:
		return "MLA_V41_FP4"
	case FormatIndexKV_MXFP8:
		return "IndexKV_MXFP8"
	case FormatIndexKV_FP4:
		return "IndexKV_FP4"
	default:
		return fmt.Sprintf("KVPageTokenFormat(%d)", int(f))
	}
}

// TokenSlotOffsets computes exact byte offsets within a paged KV block for a given token format.
func TokenSlotOffsets(
	format KVPageTokenFormat,
	pageBaseOffset int64,
	pageBlockSize int,
	tokenInPage int,
) (nopeOffset, ropeOffset, scaleOffset int64, err error) {
	desc, err := format.Descriptor()
	if err != nil {
		return 0, 0, 0, err
	}
	if pageBaseOffset < 0 {
		return 0, 0, 0, fmt.Errorf("cudakv: negative page base offset %d: %w", pageBaseOffset, cudaerrors.ErrUnsupported)
	}
	if pageBlockSize <= 0 {
		return 0, 0, 0, fmt.Errorf("cudakv: page block size %d must be positive: %w", pageBlockSize, cudaerrors.ErrUnsupported)
	}
	if tokenInPage < 0 || tokenInPage >= pageBlockSize {
		return 0, 0, 0, fmt.Errorf("cudakv: token index %d out of page range [0, %d): %w", tokenInPage, pageBlockSize, cudaerrors.ErrUnsupported)
	}
	tokenBase := pageBaseOffset + int64(tokenInPage)*int64(desc.BytesPerToken)
	nopeOffset = tokenBase
	ropeOffset = tokenBase + int64(desc.NoPEBytes)
	scaleOffset = tokenBase + int64(desc.ScaleOffset)
	return nopeOffset, ropeOffset, scaleOffset, nil
}
