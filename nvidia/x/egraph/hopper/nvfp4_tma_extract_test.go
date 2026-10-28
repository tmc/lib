package hopper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func preparedTMAFixture(t *testing.T) (*Builder, egraph.Id, PreparedNVFP4TMA, []Tensor) {
	t.Helper()
	op := PreparedNVFP4TMA{N: 256, K: 384, BM: 64, Rows: 192, Experts: 3,
		Target: "sm_90a", Epilogue: NVFP4TMAFastSwiGLU, FastScales: true}
	inputs, err := preparedNVFP4TMASchema(op)
	if err != nil {
		t.Fatal(err)
	}
	for i, name := range []string{"x", "words", "scales", "gs256", "eids", "bvalid", "npost"} {
		inputs[i].Name = name
	}
	choices := []NVFP4TMASchedule{
		{Kernel: NVFP4TMAGEMM{Name: "choice_a", N: 256, K: 384, BM: 64, Stages: 3, GLU: true, FastScales: true, Producers: 1}, Grid: 7},
		{Kernel: NVFP4TMAGEMM{Name: "choice_b", N: 256, K: 384, BM: 64, Stages: 4, GLU: true, FastScales: true, Producers: 2}, Grid: 7},
	}
	b := NewBuilder()
	root, err := b.AddPreparedNVFP4TMAChoices(op, inputs, choices)
	if err != nil {
		t.Fatal(err)
	}
	return b, root, op, inputs
}

func TestPreparedNVFP4TMAExtraction(t *testing.T) {
	b, root, _, _ := preparedTMAFixture(t)
	choices, err := egraph.ExtractTopK(b.G, root, 4, egraph.AstSizeCost{})
	if err != nil {
		t.Fatal(err)
	}
	if len(choices) != 2 {
		t.Fatalf("extracted %d choices, want 2", len(choices))
	}
	var operation string
	seen := make(map[string]bool)
	for _, choice := range choices {
		plan, err := LowerPreparedNVFP4TMA(choice.Expr)
		if err != nil {
			t.Fatal(err)
		}
		if len(plan.Inputs) != 7 || plan.Schedule.Grid != 7 || !strings.Contains(plan.Source, "cp.async.bulk.tensor") {
			t.Fatal("incomplete TMA lowering")
		}
		if operation != "" && operation != plan.OperationSHA256 {
			t.Fatal("schedule changed operation identity")
		}
		operation = plan.OperationSHA256
		sum := sha256.Sum256([]byte(plan.Source))
		if plan.SourceSHA256 != hex.EncodeToString(sum[:]) {
			t.Fatal("source identity mismatch")
		}
		if seen[plan.SourceSHA256] {
			t.Fatal("different choices share source identity")
		}
		seen[plan.SourceSHA256] = true
		// Returning lowering metadata cannot alias or mutate the extracted graph.
		plan.Inputs[0].Shape[0] = -1
		if choice.Expr.Children[0].Data.(Tensor).Shape[0] != 192 {
			t.Fatal("lowering aliases extracted schema")
		}
		if _, err := Compile(choice.Expr, "default_path", BF16); err == nil {
			t.Fatal("generic compiler silently accepted prepared TMA ABI")
		}
	}
}

func TestPreparedNVFP4TMAMeasuredChoiceAdapter(t *testing.T) {
	b, root, _, _ := preparedTMAFixture(t)
	seen := make(map[string]int)
	// These are deliberately synthetic HOST fixture durations, not runtime costs.
	evaluator := PreparedNVFP4TMAEvaluator(func(ctx context.Context, plan PreparedNVFP4TMALowering) (time.Duration, error) {
		seen[plan.Schedule.Kernel.Name]++
		if plan.Schedule.Kernel.Name == "choice_b" {
			return 200 * time.Nanosecond, nil
		}
		return 500 * time.Nanosecond, nil
	})
	tuner := egraph.NewTuner(nil)
	ranked, err := tuner.SelectRankedWithExpressionEvaluator(context.Background(), b.G, root, egraph.AutotuneBudget(time.Second), "HOST-fixture-no-native-evidence", evaluator)
	if err != nil {
		t.Fatal(err)
	}
	if len(ranked) != 2 || ranked[0].Candidate.KernelName != "choice_b" || ranked[0].Latency != 200*time.Nanosecond {
		t.Fatalf("unexpected host ranking: %+v", ranked)
	}
	if seen["choice_a"] != 1 || seen["choice_b"] != 1 {
		t.Fatalf("callback counts %v; want exactly each supplied choice", seen)
	}
	for _, result := range ranked {
		plan, err := LowerPreparedNVFP4TMA(result.Candidate.Expr)
		if err != nil {
			t.Fatal(err)
		}
		if result.Candidate.CUDASource != plan.Source || result.Candidate.BlockDim != 384 || result.Candidate.GridDim != 7 || result.Candidate.SharedMem != plan.Schedule.Kernel.SharedBytes() {
			t.Fatal("tuner candidate metadata/source does not bind lowering")
		}
	}
}

func TestPreparedNVFP4TMATerminalCallbackStopsSearch(t *testing.T) {
	b, root, _, _ := preparedTMAFixture(t)
	calls := 0
	evaluator := PreparedNVFP4TMAEvaluator(func(context.Context, PreparedNVFP4TMALowering) (time.Duration, error) {
		calls++
		return 0, errors.Join(context.Canceled, errors.New("fixture terminal device health failure"))
	})
	_, err := egraph.NewTuner(nil).SelectRankedWithExpressionEvaluator(context.Background(), b.G, root, egraph.AutotuneBudget(time.Second), "HOST-terminal-control", evaluator)
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("terminal error %v after %d calls; want one rejected call and stop", err, calls)
	}
}

func TestPreparedNVFP4TMAUnsupportedContract(t *testing.T) {
	_, _, op, inputs := preparedTMAFixture(t)
	base := NVFP4TMASchedule{Kernel: NVFP4TMAGEMM{Name: "candidate", N: op.N, K: op.K, BM: op.BM, Stages: 3, GLU: true, FastScales: true}, Grid: 7}
	for _, tt := range []struct {
		name string
		edit func(*PreparedNVFP4TMA, []Tensor, *NVFP4TMASchedule)
	}{
		{"precise_silu", func(op *PreparedNVFP4TMA, _ []Tensor, _ *NVFP4TMASchedule) { op.Epilogue = "swiglu" }},
		{"other_target", func(op *PreparedNVFP4TMA, _ []Tensor, _ *NVFP4TMASchedule) { op.Target = "sm_121a" }},
		{"unpadded_rows", func(op *PreparedNVFP4TMA, _ []Tensor, _ *NVFP4TMASchedule) { op.Rows-- }},
		{"raw_words", func(_ *PreparedNVFP4TMA, x []Tensor, _ *NVFP4TMASchedule) { x[1].Shape = []int{3, 256, 192} }},
		{"non_bf16", func(_ *PreparedNVFP4TMA, x []Tensor, _ *NVFP4TMASchedule) { x[0].DType = F32 }},
		{"alias_names", func(_ *PreparedNVFP4TMA, x []Tensor, _ *NVFP4TMASchedule) { x[1].Name = x[0].Name }},
		{"different_bm", func(_ *PreparedNVFP4TMA, _ []Tensor, s *NVFP4TMASchedule) { s.Kernel.BM = 128 }},
		{"different_scale_policy", func(_ *PreparedNVFP4TMA, _ []Tensor, s *NVFP4TMASchedule) { s.Kernel.FastScales = false }},
		{"different_epilogue", func(_ *PreparedNVFP4TMA, _ []Tensor, s *NVFP4TMASchedule) { s.Kernel.GLU = false }},
		{"short_ring", func(_ *PreparedNVFP4TMA, _ []Tensor, s *NVFP4TMASchedule) { s.Kernel.Stages = 2 }},
		{"missing_grid", func(_ *PreparedNVFP4TMA, _ []Tensor, s *NVFP4TMASchedule) { s.Grid = 0 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			operation, tensors, schedule := op, append([]Tensor(nil), inputs...), base
			tt.edit(&operation, tensors, &schedule)
			b := NewBuilder()
			before := len(b.G.Classes())
			_, err := b.AddPreparedNVFP4TMAChoices(operation, tensors, []NVFP4TMASchedule{schedule})
			if !errors.Is(err, ErrPreparedNVFP4TMA) {
				t.Fatalf("error %v, want unsupported contract", err)
			}
			if len(b.G.Classes()) != before {
				t.Fatal("invalid choice partially changed graph")
			}
		})
	}
}

func TestPreparedNVFP4TMALoweringRejectsDrift(t *testing.T) {
	b, root, _, _ := preparedTMAFixture(t)
	expr, _, err := egraph.ExtractBest(b.G, root, egraph.AstSizeCost{})
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		edit func(*egraph.Expr)
	}{
		{"wrong_op", func(e *egraph.Expr) { e.Op = OpGroupedGEMMGLU }},
		{"missing_inputs", func(e *egraph.Expr) { e.Children = e.Children[:6] }},
		{"nonleaf", func(e *egraph.Expr) { e.Children[0].Children = []*egraph.Expr{{Op: "unexpected"}} }},
		{"schema", func(e *egraph.Expr) { v := e.Children[0].Data.(Tensor); v.DType = F32; e.Children[0].Data = v }},
		{"missing_payload", func(e *egraph.Expr) { e.Data = nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			copy := *expr
			copy.Children = append([]*egraph.Expr(nil), expr.Children...)
			for i, child := range copy.Children {
				node := *child
				copy.Children[i] = &node
			}
			tt.edit(&copy)
			if _, err := LowerPreparedNVFP4TMA(&copy); !errors.Is(err, ErrPreparedNVFP4TMA) {
				t.Fatalf("accepted drift: %v", err)
			}
		})
	}
	for _, run := range []func(context.Context, PreparedNVFP4TMALowering) (time.Duration, error){nil, func(context.Context, PreparedNVFP4TMALowering) (time.Duration, error) { return 0, nil }} {
		if _, _, err := PreparedNVFP4TMAEvaluator(run)(context.Background(), expr); err == nil {
			t.Fatal("accepted missing or nonpositive measured observation")
		}
	}
}

func TestPreparedNVFP4TMANativeWitnessSourceIdentity(t *testing.T) {
	// These hashes bind the two component sources retained with native job
	// 2f09d3ebcee14f38. They are not proof for other names/configs/inputs.
	for _, tt := range []struct {
		name                  string
		bm, stages, producers int
		sha                   string
	}{
		{"moe8_moe_w13_tma_port", 128, 10, 4, "49492b00706cecedc46c5fbd938a1423f4b7d11491011072d32653151d26c54a"},
		{"moe8_moe_w13_tma_port_bm256", 256, 6, 2, "4b69a81a26614f05400b3743f2489e55bd1ecb271c951f0e7b03b0259bffab43"},
	} {
		g := NVFP4TMAGEMM{Name: tt.name, N: 8192, K: 4096, BM: tt.bm, Stages: tt.stages, GLU: true, FastScales: true, GroupM: 1, Producers: tt.producers}
		source, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256([]byte(source))
		if hex.EncodeToString(sum[:]) != tt.sha {
			t.Fatalf("component source drift for %s; prior native witness cannot cover new bytes", tt.name)
		}
		g.Producers = 1
		source, err = g.Source()
		if err != nil {
			t.Fatal(err)
		}
		sum = sha256.Sum256([]byte(source))
		if hex.EncodeToString(sum[:]) == tt.sha {
			t.Fatal("novel schedule retained known-source identity")
		}
	}
}

func ExamplePreparedNVFP4TMA() {
	op := PreparedNVFP4TMA{N: 256, K: 384, BM: 64, Rows: 192, Experts: 3, Target: "sm_90a", Epilogue: NVFP4TMAFastSwiGLU, FastScales: true}
	fmt.Println(op.Target, op.Epilogue, op.BM)
	// Output: sm_90a swiglu-fast-exp 64
}

func ExampleBuilder_AddPreparedNVFP4TMAChoices() {
	op := PreparedNVFP4TMA{N: 128, K: 128, BM: 64, Rows: 64, Experts: 1, Target: "sm_90a", Epilogue: NVFP4TMAIdentity}
	inputs := []Tensor{
		{Name: "x", DType: BF16, Shape: []int{64, 128}},
		{Name: "words", DType: U8, Shape: []int{1, 1, 2, 4096}},
		{Name: "scales", DType: U8, Shape: []int{1, 1, 2, 512}},
		{Name: "gs256", DType: F32, Shape: []int{1}},
		{Name: "eids", DType: I32, Shape: []int{1}},
		{Name: "bvalid", DType: I32, Shape: []int{1}},
		{Name: "npost", DType: I32, Shape: []int{1}},
	}
	b := NewBuilder()
	root, err := b.AddPreparedNVFP4TMAChoices(op, inputs, []NVFP4TMASchedule{{Kernel: NVFP4TMAGEMM{Name: "expert", N: 128, K: 128, BM: 64, Stages: 3}, Grid: 1}})
	fmt.Println(err == nil, root != 0)
	// Output: true true
}

func ExampleNVFP4TMASchedule() {
	choice := NVFP4TMASchedule{Kernel: NVFP4TMAGEMM{Name: "expert", N: 128, K: 128, BM: 64}, Grid: 7}
	fmt.Println(choice.Kernel.BM, choice.Grid)
	// Output: 64 7
}

func ExampleLowerPreparedNVFP4TMA() {
	_, err := LowerPreparedNVFP4TMA(nil)
	fmt.Println(errors.Is(err, ErrPreparedNVFP4TMA))
	// Output: true
}

func ExamplePreparedNVFP4TMAEvaluator() {
	evaluate := PreparedNVFP4TMAEvaluator(nil)
	_, _, err := evaluate(context.Background(), nil)
	fmt.Println(err != nil)
	// Output: true
}
