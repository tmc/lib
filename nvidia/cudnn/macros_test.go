package cudnn

import (
	"fmt"
	"math"
	"testing"
)

func TestStatusMacros(t *testing.T) {
	for _, r := range []Result{-2147483648, -2001, -1004, -1, 0, 999, 1004, 2001, 2147483647} {
		category, specific := r.Category(), r.SpecificError()
		if got := FullErrorCode(category, specific); got != r {
			t.Errorf("round trip %d: got %d", r, got)
		}
		if category%1000 != 0 || specific <= -1000 || specific >= 1000 {
			t.Errorf("invalid parts for %d: %d,%d", r, category, specific)
		}
		if r < 0 && specific > 0 || r > 0 && specific < 0 {
			t.Errorf("wrong remainder sign for %d: %d", r, specific)
		}
	}
	if got := FullErrorCode(Result(math.MaxInt32), 1); got != Result(math.MinInt32) {
		t.Fatalf("Go overflow contract = %d", got)
	}
}

func ExampleFullErrorCode() {
	fmt.Println(int32(FullErrorCode(2000, 1)))
	// Output: 2001
}

func ExampleResult_Category() {
	fmt.Println(int32(Result(2001).Category()))
	// Output: 2000
}

func ExampleResult_SpecificError() {
	fmt.Println(Result(2001).SpecificError())
	// Output: 1
}

// Expected bits are from the selected cuDNN9.20.0 macros evaluated as C doubles.
func TestMacroConstants(t *testing.T) {
	tests := []struct {
		name string
		got  float64
		want uint64
	}{
		{"CUDNN_MAX_SM_MAJOR_NUMBER", float64(MaxSmMajorNumber), 0x4028000000000000},
		{"CUDNN_MAX_SM_MINOR_NUMBER", float64(MaxSmMinorNumber), 0x3ff0000000000000},
		{"CUDNN_SEV_ERROR_EN", float64(SevErrorEn), 0x4000000000000000},
		{"CUDNN_SEV_WARNING_EN", float64(SevWarningEn), 0x4010000000000000},
		{"CUDNN_SEV_INFO_EN", float64(SevInfoEn), 0x4020000000000000},
		{"CUDNN_DIM_MAX", float64(DimMax), 0x4020000000000000},
		{"CUDNN_LRN_MIN_N", float64(LrnMinN), 0x3ff0000000000000},
		{"CUDNN_LRN_MAX_N", float64(LrnMaxN), 0x4030000000000000},
		{"CUDNN_LRN_MIN_K", float64(LrnMinK), 0x3ee4f8b588e368f1},
		{"CUDNN_LRN_MIN_BETA", float64(LrnMinBeta), 0x3f847ae147ae147b},
		{"CUDNN_BN_MIN_EPSILON", float64(BnMinEpsilon), 0x0000000000000000},
		{"CUDNN_RNN_PADDED_IO_DISABLED", float64(RnnPaddedIoDisabled), 0x0000000000000000},
		{"CUDNN_RNN_PADDED_IO_ENABLED", float64(RnnPaddedIoEnabled), 0x3ff0000000000000},
		{"CUDNN_SEQDATA_DIM_COUNT", float64(SeqdataDimCount), 0x4010000000000000},
		{"CUDNN_ATTN_QUERYMAP_ALL_TO_ONE", float64(AttnQuerymapAllToOne), 0x0000000000000000},
		{"CUDNN_ATTN_QUERYMAP_ONE_TO_ONE", float64(AttnQuerymapOneToOne), 0x3ff0000000000000},
		{"CUDNN_ATTN_DISABLE_PROJ_BIASES", float64(AttnDisableProjBiases), 0x0000000000000000},
		{"CUDNN_ATTN_ENABLE_PROJ_BIASES", float64(AttnEnableProjBiases), 0x4000000000000000},
		{"CUDNN_ATTN_WKIND_COUNT", float64(AttnWkindCount), 0x4020000000000000},
	}
	for _, tt := range tests {
		if got := math.Float64bits(tt.got); got != tt.want {
			t.Errorf("%s bits=%016x, want %016x", tt.name, got, tt.want)
		}
	}
}
