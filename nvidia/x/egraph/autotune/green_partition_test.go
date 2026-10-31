package autotune

import (
	"errors"
	"testing"
	"time"
)

func TestGreenContextSplitAlignmentAndBounds(t *testing.T) {
	cases := []struct {
		name      string
		totalSMs  int
		bgLoad    float64
		wantFGMin float64
	}{
		{name: "gb10_48sm_low_load", totalSMs: 48, bgLoad: 0.1, wantFGMin: 0.85},
		{name: "gb10_48sm_high_load", totalSMs: 48, bgLoad: 0.9, wantFGMin: 0.85},
		{name: "h100_132sm_med_load", totalSMs: 132, bgLoad: 0.5, wantFGMin: 0.85},
		{name: "b200_148sm_full_load", totalSMs: 148, bgLoad: 1.0, wantFGMin: 0.85},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec, err := BuildGreenContextPartitionSpec(tc.totalSMs, tc.bgLoad)
			if err != nil {
				t.Fatalf("BuildGreenContextPartitionSpec(%d, %g): %v", tc.totalSMs, tc.bgLoad, err)
			}
			if spec.ForegroundDecodeSMs%2 != 0 || spec.BackgroundAutotuneSMs%2 != 0 {
				t.Fatalf("expected 2-SM TPC alignment, got fg=%d bg=%d", spec.ForegroundDecodeSMs, spec.BackgroundAutotuneSMs)
			}
			if spec.BackgroundAutotuneSMs < 2 || spec.BackgroundAutotuneSMs > 8 {
				t.Fatalf("background SMs=%d outside [2, 8]", spec.BackgroundAutotuneSMs)
			}
			if spec.ForegroundFraction < tc.wantFGMin {
				t.Fatalf("foreground fraction=%.3f < %.3f (fg=%d bg=%d)",
					spec.ForegroundFraction, tc.wantFGMin, spec.ForegroundDecodeSMs, spec.BackgroundAutotuneSMs)
			}
			if spec.ForegroundDecodeSMs+spec.BackgroundAutotuneSMs != (tc.totalSMs/2)*2 {
				t.Fatalf("fg+bg=%d != aligned total %d",
					spec.ForegroundDecodeSMs+spec.BackgroundAutotuneSMs, (tc.totalSMs/2)*2)
			}
		})
	}

	if _, _, err := PlanGreenContextSplit(2, 0.5); err == nil {
		t.Fatalf("expected error for totalSMs < 4")
	}
	if _, _, err := PlanGreenContextSplit(48, 1.5); err == nil {
		t.Fatalf("expected error for bgCompileLoad > 1")
	}
}

func TestGreenContextValidateNoInterferenceReceipt(t *testing.T) {
	alone := 1000 * time.Microsecond
	withBGGood := 1012 * time.Microsecond // 1.2% jitter <= 2%
	if err := ValidateNoInterferenceReceipt(withBGGood, alone, 0.02); err != nil {
		t.Fatalf("expected valid no-interference receipt, got: %v", err)
	}

	withBGBad := 1045 * time.Microsecond // 4.5% jitter > 2%
	err := ValidateNoInterferenceReceipt(withBGBad, alone, 0.02)
	if !errors.Is(err, ErrGreenContextInterference) {
		t.Fatalf("expected ErrGreenContextInterference, got: %v", err)
	}
}
