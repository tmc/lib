package main

import (
	"encoding/json"
	"math"
	"os"
	"slices"
	"testing"
)

func TestAfmoeIndependentNormAndEmbedding(t *testing.T) {
	// Expected arrays are generated independently from the pinned official
	// Python semantics; the first norm case distinguishes a late BF16 cast.
	var fixture struct {
		RMS struct {
			Input, Weight, Expected struct{ Data []float32 }
			Eps                     float32
		} `json:"rms_norm"`
		Embedding struct {
			Embedding, Expected struct{ Data []float32 }
			Hidden              int `json:"hidden_size"`
		} `json:"mup_embedding"`
	}
	b, err := os.ReadFile("testdata/afmoe-mini/reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &fixture); err != nil {
		t.Fatal(err)
	}
	got, err := afmoeRMSNorm(fixture.RMS.Input.Data, fixture.RMS.Weight.Data, fixture.RMS.Eps)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, fixture.RMS.Expected.Data) {
		t.Fatalf("norm %v, want %v", got, fixture.RMS.Expected.Data)
	}
	got, err = afmoeScaleEmbedding(fixture.Embedding.Embedding.Data, fixture.Embedding.Hidden)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, fixture.Embedding.Expected.Data) {
		t.Fatalf("embedding %v, want %v", got, fixture.Embedding.Expected.Data)
	}
}

func TestAfmoeMathRejectsInvalid(t *testing.T) {
	for _, eps := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := afmoeRMSNorm([]float32{1}, []float32{1}, eps); err == nil {
			t.Fatalf("accepted epsilon %g", eps)
		}
	}
	if _, err := afmoeRMSNorm([]float32{1}, nil, 1e-5); err == nil {
		t.Fatal("accepted missing weights")
	}
	if _, err := afmoeScaleEmbedding([]float32{1}, 2); err == nil {
		t.Fatal("accepted incomplete embedding row")
	}
}
