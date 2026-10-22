package dsv4tools

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestSummarizeVector(t *testing.T) {
	stats, err := SummarizeVector([]float32{-4, 1, 3, -2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Min != -4 || stats.Max != 3 || math.Abs(stats.RMS-math.Sqrt(6)) > 1e-12 {
		t.Fatalf("stats = %+v", stats)
	}
	if len(stats.Top) != 5 || stats.Top[0] != (StatValue{Index: 0, Value: -4}) || stats.Top[1] != (StatValue{Index: 2, Value: 3}) {
		t.Fatalf("top = %+v", stats.Top)
	}
	var out bytes.Buffer
	if err := WriteVectorStats(&out, "logits", []float32{-4, 1, 3, -2, 0}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "logits: min=-4 max=3") || !strings.Contains(out.String(), "top[0]=-4") {
		t.Fatalf("output = %q", out.String())
	}
}

func TestSummarizeVectorRejectsMutations(t *testing.T) {
	for name, values := range map[string][]float32{
		"empty": nil,
		"nan":   {1, float32(math.NaN())},
		"inf":   {1, float32(math.Inf(1))},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := SummarizeVector(values); err == nil {
				t.Fatal("accepted invalid vector")
			}
		})
	}
	if err := WriteVectorStats(&bytes.Buffer{}, "", []float32{1}); err == nil {
		t.Fatal("accepted empty diagnostic name")
	}
}
