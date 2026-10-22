package dsv4bench

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func frontierPair() ([]FrontierLogits, []FrontierLogits) {
	base := []FrontierLogits{
		{Source: "ds4-bench", Model: "m", Backend: "cuda", Quality: "fast", QuantBits: 8, PromptTokens: 32, FrontierTokens: 2048, Context: 2048, Vocab: 2, ArgmaxID: 0, ArgmaxLogit: 2, Logits: []float32{2, 1}},
		{Source: "ds4-bench", Model: "m", Backend: "cuda", Quality: "fast", QuantBits: 8, PromptTokens: 32, FrontierTokens: 4096, Context: 4096, Vocab: 2, ArgmaxID: 1, ArgmaxLogit: 3, Logits: []float32{1, 3}},
	}
	target := append([]FrontierLogits(nil), base...)
	target[0] = base[0]
	target[0].Source, target[0].Backend, target[0].Logits = "ds4-min", "cuda-go", []float32{2.00001, 1}
	target[0].ArgmaxLogit = target[0].Logits[0]
	target[1] = base[1]
	target[1].Source, target[1].Backend = "ds4-min", "cuda-go"
	return base, target
}

func TestWriteFrontierJSON(t *testing.T) {
	f := FrontierLogits{
		Source: "ds4-bench", Model: "model.gguf", Backend: "cuda", Quality: "fast",
		QuantBits: 8, PromptTokens: 32, FrontierTokens: 64, Context: 128,
		Vocab: 3, ArgmaxID: 1, ArgmaxLogit: 2, Logits: []float32{1, 2, 0.5},
	}
	var out bytes.Buffer
	if err := WriteFrontierJSON(&out, f); err != nil {
		t.Fatal(err)
	}
	want := `{"source":"ds4-bench","model":"model.gguf","backend":"cuda","quality":"fast","quant_bits":8,"prompt_tokens":32,"frontier_tokens":64,"ctx":128,"vocab":3,"argmax_id":1,"argmax_logit":2,"logits":[1,2,0.5]}` + "\n"
	if out.String() != want {
		t.Fatalf("JSON = %q, want %q", out.String(), want)
	}
}

func TestWriteFrontierJSONRejectsMutations(t *testing.T) {
	base := FrontierLogits{Source: "s", Model: "m", Backend: "b", Quality: "q", QuantBits: 8, PromptTokens: 1, FrontierTokens: 2, Context: 4, Vocab: 2, ArgmaxID: 0, ArgmaxLogit: 2, Logits: []float32{2, 1}}
	tests := []struct {
		name   string
		mutate func(*FrontierLogits)
	}{
		{"wrong vocab", func(f *FrontierLogits) { f.Vocab = 3 }},
		{"wrong argmax", func(f *FrontierLogits) { f.ArgmaxID = 1 }},
		{"wrong logit", func(f *FrontierLogits) { f.ArgmaxLogit = 3 }},
		{"nonfinite", func(f *FrontierLogits) { f.Logits[1] = float32(math.NaN()) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := base
			f.Logits = append([]float32(nil), base.Logits...)
			tt.mutate(&f)
			if err := WriteFrontierJSON(new(bytes.Buffer), f); err == nil {
				t.Fatal("invalid frontier accepted")
			}
		})
	}
	if err := WriteFrontierJSON(new(strings.Builder), base); err != nil {
		t.Fatal(err)
	}
}

func TestReadAndCompareFrontier(t *testing.T) {
	base, target := frontierPair()
	var left, right bytes.Buffer
	for i := range base {
		if err := WriteFrontierJSON(&left, base[i]); err != nil {
			t.Fatal(err)
		}
		if err := WriteFrontierJSON(&right, target[i]); err != nil {
			t.Fatal(err)
		}
	}
	got, err := ReadFrontierJSON(&left)
	if err != nil {
		t.Fatal(err)
	}
	other, err := ReadFrontierJSON(&right)
	if err != nil {
		t.Fatal(err)
	}
	report, err := CompareFrontier(got, other, FrontierTolerance{Abs: 1e-4, Rel: 1e-4})
	if err != nil || !report.Pass() || report.Compared != 2 || report.Top1Matches != 2 {
		t.Fatalf("report=%+v err=%v", report, err)
	}
}

func TestCompareFrontierMutationsFailClosed(t *testing.T) {
	base, target := frontierPair()
	tests := []struct {
		name     string
		mutate   func([]FrontierLogits)
		wantErr  string
		wantPass bool
	}{
		{"logit tolerance", func(rows []FrontierLogits) { rows[0].Logits[0] = 2.1; rows[0].ArgmaxLogit = 2.1 }, "", false},
		{"workload", func(rows []FrontierLogits) { rows[0].Context = 4096 }, "workload differs", false},
		{"argmax", func(rows []FrontierLogits) {
			rows[0].Logits = []float32{1, 3}
			rows[0].ArgmaxID = 1
			rows[0].ArgmaxLogit = 3
		}, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := append([]FrontierLogits(nil), target...)
			candidate[0].Logits = append([]float32(nil), target[0].Logits...)
			tt.mutate(candidate)
			report, err := CompareFrontier(base, candidate, FrontierTolerance{Abs: 1e-4, Rel: 1e-4})
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("err=%v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || report.Pass() {
				t.Fatalf("report=%+v err=%v, want failed comparison", report, err)
			}
		})
	}
}

func TestReadFrontierRejectsMutations(t *testing.T) {
	base, _ := frontierPair()
	var encoded bytes.Buffer
	if err := WriteFrontierJSON(&encoded, base[0]); err != nil {
		t.Fatal(err)
	}
	for _, input := range []string{"", encoded.String() + "{}\n", strings.Replace(encoded.String(), "\n", "\n\n", 1)} {
		if _, err := ReadFrontierJSON(strings.NewReader(input)); err == nil {
			t.Fatalf("accepted malformed frontier %q", input)
		}
	}
}
