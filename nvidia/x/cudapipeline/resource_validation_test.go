package cudapipeline_test

import (
	"math/big"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func TestValidatePipelineNilInput(t *testing.T) {
	var pipeline *cudapipeline.Pipeline
	for _, report := range []*cudapipeline.Report{
		cudapipeline.ValidatePipeline(nil),
		pipeline.Validate(),
	} {
		if report.Valid() || report.ActualStages != 0 || len(report.Issues) != 1 {
			t.Fatalf("nil pipeline report = %+v", report)
		}
		issue := report.Issues[0]
		if issue.StageIndex != -1 || issue.Severity != cudapipeline.SeverityError ||
			issue.Category != cudapipeline.CatHardwareConstraint || issue.Message != "pipeline is nil" {
			t.Errorf("nil pipeline issue = %+v", issue)
		}
	}
}

func TestSM121ResourceInputBounds(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name  string
		input [4]int // shared bytes, warps/block, blocks/SM, registers/thread
	}{
		{"existing_valid", [4]int{32768, 4, 2, 32}},
		{"all_zero", [4]int{}},
		{"zero_shared", [4]int{0, 4, 2, 32}},
		{"zero_warps", [4]int{1024, 0, 2, 32}},
		{"zero_blocks", [4]int{1024, 4, 0, 32}},
		{"zero_registers", [4]int{1024, 4, 2, 0}},
		{"shared_per_block_limit", [4]int{101376, 1, 1, 1}},
		{"shared_total_limit", [4]int{51200, 1, 2, 1}},
		{"warp_block_limits", [4]int{0, 2, 24, 1}},
		{"register_total_limit", [4]int{0, 1, 1, 2048}},
		{"quantum_one", [4]int{0, 1, 1, 1}},
		{"quantum_exact", [4]int{0, 1, 1, 8}},
		{"quantum_next", [4]int{0, 1, 1, 9}},
		{"shared_limit_exceeded", [4]int{101377, 1, 1, 1}},
		{"shared_total_exceeded", [4]int{51201, 1, 2, 1}},
		{"warps_exceeded", [4]int{0, 49, 1, 1}},
		{"blocks_exceeded", [4]int{0, 1, 25, 1}},
		{"register_total_exceeded", [4]int{0, 1, 1, 2049}},
		{"negative_shared", [4]int{-1, 4, 2, 32}},
		{"negative_warps", [4]int{1024, -1, 2, 32}},
		{"negative_blocks", [4]int{1024, 4, -1, 32}},
		{"negative_registers", [4]int{1024, 4, 2, -1}},
		{"all_negative", [4]int{-1, -1, -1, -1}},
		{"min_int", [4]int{-maxInt - 1, 0, 0, 0}},
		{"shared_product_overflow", [4]int{maxInt, 1, 2, 0}},
		{"warp_product_overflow", [4]int{0, maxInt, 2, 0}},
		{"register_raw_overflow", [4]int{0, 1, 1, maxInt}},
		{"register_rounding_overflow", [4]int{0, 0, 1, maxInt / 32}},
		{"register_total_overflow", [4]int{0, maxInt, 1, 1}},
		{"register_representable_quantum_boundary", [4]int{0, 0, 1, (maxInt / 256) * 8}},
		{"shared_product_boundary", [4]int{maxInt / 2, 0, 2, 0}},
		{"warp_product_boundary", [4]int{0, maxInt / 2, 2, 0}},
		{"zero_product_max_int", [4]int{maxInt, 0, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := tt.input
			summary, violations := cudapipeline.ValidateSM121Resources(a[0], a[1], a[2], a[3])
			if summary.SharedMemBytes != a[0] || summary.WarpsPerBlock != a[1] ||
				summary.BlocksPerSM != a[2] || summary.RegistersPerThread != a[3] {
				t.Fatalf("input quantities changed: %+v", summary)
			}
			wantValid, wantTotals, unavailable := resourceOracle(a, maxInt)
			if summary.WithinLimits != wantValid || (len(violations) == 0) != wantValid {
				t.Errorf("valid = %v, errors = %v; want %v", summary.WithinLimits, violations, wantValid)
			}
			gotTotals := [4]int{summary.TotalSharedMemPerSM, summary.TotalWarpsPerSM,
				summary.RegistersPerWarp, summary.TotalRegsPerSM}
			if gotTotals != wantTotals {
				t.Errorf("derived quantities = %v, want %v", gotTotals, wantTotals)
			}
			if unavailable {
				found := false
				for _, violation := range violations {
					found = found || strings.Contains(violation.Message, "unavailable")
				}
				if !found {
					t.Errorf("missing unavailable diagnostic: %v", violations)
				}
			}
		})
	}
}

// Use arbitrary-precision arithmetic as an independent oracle. No machine-int
// product or rounding expression from the implementation is reused here.
func resourceOracle(input [4]int, maxInt int) (bool, [4]int, bool) {
	var totals [4]int
	for _, value := range input {
		if value < 0 {
			return false, totals, true
		}
	}
	integer := func(n int) *big.Int { return big.NewInt(int64(n)) }
	product := func(a, b int) *big.Int { return new(big.Int).Mul(integer(a), integer(b)) }
	shared := product(input[0], input[2])
	warps := product(input[1], input[2])
	rawRegisters := product(input[3], 32)
	registers := new(big.Int).Add(rawRegisters, integer(255))
	registers.Quo(registers, integer(256))
	registers.Mul(registers, integer(256))
	allRegisters := new(big.Int).Mul(registers, warps)
	values := []*big.Int{shared, warps, registers, allRegisters}
	available := [4]bool{}
	for i, value := range values {
		available[i] = value.Cmp(integer(maxInt)) <= 0
		if available[i] {
			totals[i] = int(value.Int64())
		}
	}
	// Dependent totals are unavailable when a required intermediate is invalid.
	if !available[1] || !available[2] {
		available[3] = false
		totals[3] = 0
	}
	unavailable := false
	for _, ok := range available {
		unavailable = unavailable || !ok
	}
	valid := !unavailable && input[0] <= 101376 && input[2] <= 24 &&
		shared.Cmp(integer(102400)) <= 0 && warps.Cmp(integer(48)) <= 0 &&
		allRegisters.Cmp(integer(65536)) <= 0
	return valid, totals, unavailable
}

func TestSM121ResourceViolationOrder(t *testing.T) {
	tests := []struct {
		name  string
		input [4]int
		want  []string
	}{
		{"all_limits", [4]int{102000, 4, 25, 32}, []string{
			cudapipeline.ErrExceededSMEMLimit, cudapipeline.ErrExceededSMEMLimit,
			cudapipeline.ErrExceededBlocksPerSM, cudapipeline.ErrExceededWarpsPerSM,
			cudapipeline.ErrInvalidRegisterQuantum,
		}},
		{"all_negative", [4]int{-1, -1, -1, -1}, []string{
			cudapipeline.ErrExceededSMEMLimit, cudapipeline.ErrExceededWarpsPerSM,
			cudapipeline.ErrExceededBlocksPerSM, cudapipeline.ErrInvalidRegisterQuantum,
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := tt.input
			_, violations := cudapipeline.ValidateSM121Resources(a[0], a[1], a[2], a[3])
			if len(violations) != len(tt.want) {
				t.Fatalf("violations = %v, want categories %v", violations, tt.want)
			}
			for i, want := range tt.want {
				if violations[i].Violation != want {
					t.Errorf("violation %d = %q, want %q", i, violations[i].Violation, want)
				}
				if tt.name == "all_negative" && (violations[i].Expected != 0 ||
					violations[i].Actual != -1 || !strings.Contains(violations[i].Message, "nonnegative")) {
					t.Errorf("negative input diagnostic = %+v", violations[i])
				}
			}
		})
	}
}
