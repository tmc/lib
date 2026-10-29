package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

func TestParityResultAdd(t *testing.T) {
	tol := parityTolerance{ATol: 0.1, RTol: 0.01}
	tests := []struct {
		name       string
		ours, ref  []float32
		hf         float64
		greedy     bool
		wantAbs    float64
		wantRatio  float64
		wantBF16   float64
		wantTop1   int
		wantGreedy int
		wantPass   bool
		wantGap    [2]float64 // ref, ours; for a disagreement
	}{
		{"equal", []float32{1, 5, 2}, []float32{1, 5, 2}, 0.5, true, 0, 0, 0, 1, 1, true, [2]float64{}},
		{"within", []float32{1.05, 5, 2}, []float32{1, 5, 2}, 0.1, true, 0.05, 0.05 / 0.11, 0.5, 1, 1, true, [2]float64{}},
		{"rtol scales", []float32{1, 5.1, 2}, []float32{1, 5, 2}, 0, false, 0.1, 0.1 / 0.15, 0, 1, 0, true, [2]float64{}},
		{"outside", []float32{1, 5, 2.3}, []float32{1, 5, 2}, 0, false, 0.3, 0.3 / 0.12, 0, 1, 0, false, [2]float64{}},
		{"flip", []float32{1, 4.9, 5.0}, []float32{1, 5, 4.95}, 0, true, 0.1, 0.1 / 0.15, 0, 0, 0, true, [2]float64{0.05, 0.1}},
		{"nan", []float32{1, 5, float32(math.NaN())}, []float32{1, 5, 2}, 0, false, math.Inf(1), math.Inf(1), 0, 1, 0, false, [2]float64{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var r parityResult
			r.add(7, tt.ours, tt.ref, tt.hf, tol, tt.greedy)
			near := func(a, b float64) bool { return a == b || math.Abs(a-b) < 1e-6 }
			if !near(r.MaxAbs, tt.wantAbs) || !near(r.MaxTolRatio, tt.wantRatio) || !near(r.MaxBF16Ratio, tt.wantBF16) {
				t.Errorf("max abs, tol ratio, bf16 ratio = %v, %v, %v; want %v, %v, %v",
					r.MaxAbs, r.MaxTolRatio, r.MaxBF16Ratio, tt.wantAbs, tt.wantRatio, tt.wantBF16)
			}
			if r.Top1 != tt.wantTop1 || r.Greedy != tt.wantGreedy || r.Passed != tt.wantPass {
				t.Errorf("top1, greedy, passed = %d, %d, %v; want %d, %d, %v", r.Top1, r.Greedy, r.Passed, tt.wantTop1, tt.wantGreedy, tt.wantPass)
			}
			if tt.wantTop1 == 1 {
				if len(r.Disagree) != 0 {
					t.Errorf("disagreements = %v, want none", r.Disagree)
				}
				return
			}
			if len(r.Disagree) != 1 {
				t.Fatalf("disagreements = %v, want one", r.Disagree)
			}
			d := r.Disagree[0]
			if d.Pos != 7 || !near(d.RefGap, tt.wantGap[0]) || !near(d.OursGap, tt.wantGap[1]) {
				t.Errorf("disagreement = %+v, want pos 7, ref gap %v, ours gap %v", d, tt.wantGap[0], tt.wantGap[1])
			}
		})
	}
}

func TestReadHFLogits(t *testing.T) {
	dir := t.TempDir()
	const vocab = 3
	vals := []float32{0, 1, 2, 3, 4, 5}
	data := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
	}
	if err := safetensors.WriteFile(filepath.Join(dir, "c.safetensors"), []safetensors.Tensor{
		{Name: "logits", Dtype: safetensors.F32, Shape: []int{2, vocab}, Data: data},
	}, nil); err != nil {
		t.Fatal(err)
	}
	tc := hfCase{Start: 3, Prompt: []int32{1, 2, 3, 4}, Generated: []int32{5}, LogitsFile: "c.safetensors"}
	rows, err := readHFLogits(dir, tc, vocab)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[1][2] != 5 {
		t.Fatalf("rows = %v, want [[0 1 2] [3 4 5]]", rows)
	}
	tc.Start = 2
	if _, err := readHFLogits(dir, tc, vocab); err == nil {
		t.Fatal("start 2: read 2 rows for 3 positions without error")
	}
}

// The audit must reject actual parity-constructor mutations, including a
// graph returned together with an error and compiler errors retaining owners.
func TestHFParityOwnership(t *testing.T) {
	const file = "hf_parity_cuda_test.go"
	for _, owner := range []string{"gm", "gpuW", "kv", "sp", "eg", "dg"} {
		_, bad := rewriteScan(t, releasesFirst, file, "if "+owner+" != nil {", "if "+owner+" == nil {")
		if !slices.ContainsFunc(bad, func(s string) bool { return strings.HasPrefix(s, file+":") }) {
			t.Errorf("missing release of %s was not rejected", owner)
		}
	}
	_, bad := rewriteScan(t, callErrorsReleased, file,
		"return nil, releaseErrorOwner(err)", "return nil, err", "gen.go")
	if !slices.ContainsFunc(bad, func(s string) bool { return strings.HasPrefix(s, file+":") }) {
		t.Error("retained compiler error owner was not rejected")
	}
}

func TestParityTolerance(t *testing.T) {
	for _, name := range []string{"LLAMA_HF_PARITY_ATOL", "LLAMA_HF_PARITY_RTOL"} {
		for _, value := range []string{"-1", "NaN", "Inf", "-Inf", "not a number"} {
			t.Run(name+"/"+value, func(t *testing.T) {
				t.Setenv("LLAMA_HF_PARITY_ATOL", "")
				t.Setenv("LLAMA_HF_PARITY_RTOL", "")
				t.Setenv(name, value)
				if _, err := (parityTolerance{0.005, 0.0005}).fromEnv(); err == nil {
					t.Fatal("accepted invalid tolerance")
				}
			})
		}
	}
	t.Setenv("LLAMA_HF_PARITY_ATOL", "0")
	t.Setenv("LLAMA_HF_PARITY_RTOL", "0")
	tol, err := defaultTolerance("egraph", &hfSuite{})
	if err != nil {
		t.Fatal(err)
	}
	tol, err = tol.fromEnv()
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		ours []float32
		pass bool
	}{{[]float32{1, 2}, true}, {[]float32{1, 2.001}, false}, {[]float32{1, float32(math.Inf(1))}, false}} {
		var r parityResult
		r.add(0, tt.ours, []float32{1, 2}, 0, tol, true)
		if r.Passed != tt.pass || math.IsNaN(r.MaxTolRatio) {
			t.Fatalf("ours=%v: passed=%v ratio=%v", tt.ours, r.Passed, r.MaxTolRatio)
		}
	}
}

func TestDefaultParityTolerance(t *testing.T) {
	t.Setenv("LLAMA_HF_PARITY_ATOL", "")
	t.Setenv("LLAMA_HF_PARITY_RTOL", "")
	if _, err := defaultTolerance("egraph", &hfSuite{}); err == nil {
		t.Fatal("accepted missing bfloat16 calibration without an override")
	}
	for _, tt := range []struct {
		kind string
		max  float64
		want parityTolerance
	}{
		{"f32", 0.25, parityTolerance{0.005, 0.0005}},
		{"bf16", 0.25, parityTolerance{0.25, 0}},
		{"egraph", 0, parityTolerance{0, 0}},
	} {
		var suite hfSuite
		if err := json.Unmarshal([]byte(`{"hf_bf16":{"max_abs":`+strconv.FormatFloat(tt.max, 'g', -1, 64)+`}}`), &suite); err != nil {
			t.Fatal(err)
		}
		got, err := defaultTolerance(tt.kind, &suite)
		if err != nil || got != tt.want {
			t.Errorf("defaultTolerance(%s, %v) = %v, %v; want %v", tt.kind, tt.max, got, err, tt.want)
		}
	}
}

func TestHFParityPrefillMetadataCapacity(t *testing.T) {
	const maxSeq, qtok = 2048, 15
	legacy, err := prefillSeqLenCapacity(maxSeq, maxSeq, qtok)
	if err != nil || legacy != 138 {
		t.Fatalf("default capacity = %d, %v; want unchanged 138", legacy, err)
	}
	capacity, err := prefillSeqLenCapacity(maxSeq, 1, qtok)
	if err != nil {
		t.Fatal(err)
	}
	for _, rows := range []int{1, 4, 8, 40, maxSeq} {
		for _, start := range []int{0, 1, 17, 1023, maxSeq - 1} {
			var lengths []int32
			for at := start; at < maxSeq; at += rows {
				lengths = append(lengths, prefillSeqLens(at, min(rows, maxSeq-at), qtok)...)
			}
			if len(lengths) > capacity {
				t.Errorf("rows=%d start=%d: %d lengths exceed %d", rows, start, len(lengths), capacity)
			}
			if rows == 1 && start == 0 && len(lengths) <= legacy {
				t.Fatal("regression no longer exercises the original undersized allocation")
			}
		}
	}
}

func TestPrefillSeqLenCapacityErrors(t *testing.T) {
	for _, args := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 1, 1}, {1, -1, 1}, {1, 1, -1}, {int(^uint(0) >> 1), 1, 1}} {
		if _, err := prefillSeqLenCapacity(args[0], args[1], args[2]); err == nil {
			t.Errorf("accepted invalid or overflowing sizes %v", args)
		}
	}
}

func TestParityZeroReferenceRowJSON(t *testing.T) {
	for _, tt := range []struct {
		ours []float32
		pass bool
		rel  float64
	}{
		{[]float32{0, 0}, true, 0},
		{[]float32{0.2, 0}, false, math.MaxFloat64},
	} {
		var r parityResult
		r.add(0, tt.ours, []float32{0, 0}, 0, parityTolerance{ATol: 0.1}, true)
		if r.Passed != tt.pass || r.MaxRowRel != tt.rel {
			t.Fatalf("ours=%v: passed=%v relative=%v; want %v,%v", tt.ours, r.Passed, r.MaxRowRel, tt.pass, tt.rel)
		}
		if _, err := json.Marshal(r); err != nil {
			t.Fatalf("ours=%v: JSON report: %v", tt.ours, err)
		}
	}
}
