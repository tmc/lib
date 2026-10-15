package cudacensus_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacensus"
	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

func TestInstructionMixRejectsOverflow(t *testing.T) {
	for _, tt := range []struct {
		name   string
		counts map[cudaroofline.Precision]uint64
	}{
		{"wrap to one", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64, cudaroofline.FP16Tensor: 2}},
		{"wrap to zero", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64, cudaroofline.FP16Tensor: 1}},
		{"overflow tie", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64, cudaroofline.FP16Tensor: math.MaxUint64}},
		{"three counts", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64 - 2, cudaroofline.FP16Tensor: 2, cudaroofline.FP64: 1}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mix := cudacensus.InstructionMix{Counts: tt.counts}
			p, share, err := mix.Dominant()
			if err == nil || !strings.Contains(err.Error(), "overflows uint64") || p != "" || share != 0 {
				t.Errorf("Dominant = %q, %v, %v; want empty, zero, overflow error", p, share, err)
			}
			attr, err := cudacensus.AttributeFromMix(mix)
			if err == nil || !strings.Contains(err.Error(), "overflows uint64") {
				t.Errorf("AttributeFromMix error = %v, want overflow error", err)
			}
			if attr.Known || attr.Precision != "" || attr.Share != 0 || attr.Method != cudacensus.Unattributed || !strings.Contains(attr.Reason, "overflows uint64") {
				t.Errorf("AttributeFromMix = %+v, want explicitly unattributed overflow", attr)
			}
		})
	}
}

func TestInstructionMixValidBounds(t *testing.T) {
	for _, tt := range []struct {
		name   string
		counts map[cudaroofline.Precision]uint64
		total  uint64
		share  float64
	}{
		{"one", map[cudaroofline.Precision]uint64{cudaroofline.FP32: 1}, 1, 1},
		{"ordinary", map[cudaroofline.Precision]uint64{cudaroofline.FP32: 3, cudaroofline.FP16Tensor: 1, cudaroofline.FP64: 0}, 4, 0.75},
		{"maximum single", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64}, math.MaxUint64, 1},
		{"maximum total", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64 - 2, cudaroofline.FP16Tensor: 2}, math.MaxUint64, float64(uint64(math.MaxUint64-2)) / float64(uint64(math.MaxUint64))},
		{"near half", map[cudaroofline.Precision]uint64{cudaroofline.FP32: 1 << 63, cudaroofline.FP16Tensor: 1<<63 - 1}, math.MaxUint64, float64(uint64(1<<63)) / float64(uint64(math.MaxUint64))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mix := cudacensus.InstructionMix{Counts: tt.counts}
			if got := mix.Total(); got != tt.total {
				t.Errorf("Total = %d, want %d", got, tt.total)
			}
			p, share, err := mix.Dominant()
			if err != nil || p != cudaroofline.FP32 || share != tt.share {
				t.Errorf("Dominant = %q, %v, %v; want fp32, %v, nil", p, share, err, tt.share)
			}
			attr, err := cudacensus.AttributeFromMix(mix)
			if err != nil || !attr.Known || attr.Method != cudacensus.FromInstructionMix || attr.Precision != p || attr.Share != share || attr.Reason != "" {
				t.Errorf("AttributeFromMix = %+v, %v; want known matching dominant", attr, err)
			}
			if math.IsNaN(attr.Share) || math.IsInf(attr.Share, 0) || attr.Share <= 0 || attr.Share > 1 {
				t.Errorf("accepted share = %v, want finite in (0, 1]", attr.Share)
			}
		})
	}
}

func TestInstructionMixEmptyAndTieBounds(t *testing.T) {
	for _, tt := range []struct {
		name   string
		counts map[cudaroofline.Precision]uint64
		reason string
	}{
		{"nil", nil, "empty instruction mix"},
		{"zero", map[cudaroofline.Precision]uint64{cudaroofline.FP32: 0}, "empty instruction mix"},
		{"tie", map[cudaroofline.Precision]uint64{cudaroofline.FP32: 2, cudaroofline.FP16Tensor: 2, cudaroofline.FP64: 0}, "tie"},
		{"large tie", map[cudaroofline.Precision]uint64{cudaroofline.FP32: math.MaxUint64 / 2, cudaroofline.FP16Tensor: math.MaxUint64 / 2}, "tie"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			mix := cudacensus.InstructionMix{Counts: tt.counts}
			p, share, err := mix.Dominant()
			if err == nil || !strings.Contains(err.Error(), tt.reason) || p != "" || share != 0 {
				t.Errorf("Dominant = %q, %v, %v; want no decision with %q", p, share, err, tt.reason)
			}
			attr, err := cudacensus.AttributeFromMix(mix)
			if err == nil || attr.Known || attr.Precision != "" || attr.Share != 0 || attr.Method != cudacensus.Unattributed || attr.Reason != err.Error() {
				t.Errorf("AttributeFromMix = %+v, %v; want explicitly unattributed", attr, err)
			}
		})
	}
}

func TestInstructionMixTotalModulo(t *testing.T) {
	mix := cudacensus.InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP32: math.MaxUint64, cudaroofline.FP16Tensor: 2,
	}}
	if got := mix.Total(); got != 1 {
		t.Fatalf("Total = %d, want preserved modulo sum 1", got)
	}
}
