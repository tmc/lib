package cudakv_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestKVPageTokenFormatAndOffsets(t *testing.T) {
	tests := []struct {
		name          string
		format        cudakv.KVPageTokenFormat
		wantBytes     int
		wantNoPE      int
		wantRoPE      int
		wantData      int
		wantScale     int
		wantScaleOff  int
		wantPad       int
		pageBase      int64
		pageBlockSize int
		tokenInPage   int
		wantNoPEOff   int64
		wantRoPEOff   int64
		wantScaleByte int64
	}{
		{
			name:          "MLA_V32_FP8",
			format:        cudakv.FormatMLA_V32_FP8,
			wantBytes:     656,
			wantNoPE:      512,
			wantRoPE:      128,
			wantData:      640,
			wantScale:     16,
			wantScaleOff:  640,
			wantPad:       0,
			pageBase:      4096,
			pageBlockSize: 64,
			tokenInPage:   3,
			wantNoPEOff:   4096 + 3*656,
			wantRoPEOff:   4096 + 3*656 + 512,
			wantScaleByte: 4096 + 3*656 + 640,
		},
		{
			name:          "MLA_V41_FP8",
			format:        cudakv.FormatMLA_V41_FP8,
			wantBytes:     528,
			wantNoPE:      448,
			wantRoPE:      64,
			wantData:      512,
			wantScale:     16,
			wantScaleOff:  512,
			wantPad:       0,
			pageBase:      1024,
			pageBlockSize: 64,
			tokenInPage:   5,
			wantNoPEOff:   1024 + 5*528,
			wantRoPEOff:   1024 + 5*528 + 448,
			wantScaleByte: 1024 + 5*528 + 512,
		},
		{
			name:          "MLA_V41_FP4",
			format:        cudakv.FormatMLA_V41_FP4,
			wantBytes:     288,
			wantNoPE:      224,
			wantRoPE:      32,
			wantData:      256,
			wantScale:     28,
			wantScaleOff:  256,
			wantPad:       4,
			pageBase:      0,
			pageBlockSize: 64,
			tokenInPage:   7,
			wantNoPEOff:   7 * 288,
			wantRoPEOff:   7*288 + 224,
			wantScaleByte: 7*288 + 256,
		},
		{
			name:          "IndexKV_MXFP8",
			format:        cudakv.FormatIndexKV_MXFP8,
			wantBytes:     132,
			wantNoPE:      128,
			wantRoPE:      0,
			wantData:      128,
			wantScale:     4,
			wantScaleOff:  128,
			wantPad:       0,
			pageBase:      8192,
			pageBlockSize: 64,
			tokenInPage:   2,
			wantNoPEOff:   8192 + 2*132,
			wantRoPEOff:   8192 + 2*132 + 128,
			wantScaleByte: 8192 + 2*132 + 128,
		},
		{
			name:          "IndexKV_FP4",
			format:        cudakv.FormatIndexKV_FP4,
			wantBytes:     68,
			wantNoPE:      32,
			wantRoPE:      0,
			wantData:      32,
			wantScale:     4,
			wantScaleOff:  64,
			wantPad:       32,
			pageBase:      512,
			pageBlockSize: 64,
			tokenInPage:   10,
			wantNoPEOff:   512 + 10*68,
			wantRoPEOff:   512 + 10*68 + 32,
			wantScaleByte: 512 + 10*68 + 64,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			desc, err := tc.format.Descriptor()
			if err != nil {
				t.Fatalf("Descriptor() unexpected error: %v", err)
			}
			if desc.BytesPerToken != tc.wantBytes || desc.NoPEBytes != tc.wantNoPE || desc.RoPEBytes != tc.wantRoPE ||
				desc.DataBytes != tc.wantData || desc.ScaleBytes != tc.wantScale || desc.ScaleOffset != tc.wantScaleOff ||
				desc.PaddingBytes != tc.wantPad {
				t.Fatalf("Descriptor() = %+v, want bytes=%d nope=%d rope=%d data=%d scale=%d scaleOff=%d pad=%d",
					desc, tc.wantBytes, tc.wantNoPE, tc.wantRoPE, tc.wantData, tc.wantScale, tc.wantScaleOff, tc.wantPad)
			}
			if tc.format.String() != tc.name {
				t.Fatalf("String() = %q, want %q", tc.format.String(), tc.name)
			}

			nopeOff, ropeOff, scaleOff, err := cudakv.TokenSlotOffsets(tc.format, tc.pageBase, tc.pageBlockSize, tc.tokenInPage)
			if err != nil {
				t.Fatalf("TokenSlotOffsets() unexpected error: %v", err)
			}
			if nopeOff != tc.wantNoPEOff || ropeOff != tc.wantRoPEOff || scaleOff != tc.wantScaleByte {
				t.Fatalf("TokenSlotOffsets() = (%d, %d, %d), want (%d, %d, %d)",
					nopeOff, ropeOff, scaleOff, tc.wantNoPEOff, tc.wantRoPEOff, tc.wantScaleByte)
			}
		})
	}

	// Negative controls: invalid format, negative base, non-positive block size, out-of-range token index.
	if _, err := cudakv.KVPageTokenFormat(99).Descriptor(); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("unknown format Descriptor error = %v, want ErrUnsupported", err)
	}
	if _, _, _, err := cudakv.TokenSlotOffsets(cudakv.FormatMLA_V41_FP8, -1, 64, 0); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("negative pageBaseOffset error = %v, want ErrUnsupported", err)
	}
	if _, _, _, err := cudakv.TokenSlotOffsets(cudakv.FormatMLA_V41_FP8, 0, 0, 0); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("zero pageBlockSize error = %v, want ErrUnsupported", err)
	}
	if _, _, _, err := cudakv.TokenSlotOffsets(cudakv.FormatMLA_V41_FP8, 0, 64, 64); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("out-of-range tokenInPage error = %v, want ErrUnsupported", err)
	}
}
