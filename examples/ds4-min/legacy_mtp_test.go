package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestLegacyMTPHeadFloat64AndOwnership(t *testing.T) {
	var data []byte
	m := &legacyMTPModel{weights: &dsparkModel{tensors: make(map[string]gguf.Tensor)}}
	add := func(name string, dims []uint64, values []float32) {
		m.weights.tensors[name] = gguf.Tensor{Name: name, TypeID: 0, Dimensions: dims, Offset: uint64(len(data))}
		for _, v := range values {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
	}
	add("mtp.0.hc_head_fn.weight", []uint64{uint64(Flash.HC * Flash.Embed), uint64(Flash.HC)}, make([]float32, Flash.HC*Flash.Embed*Flash.HC))
	add("mtp.0.hc_head_scale.weight", []uint64{1}, []float32{1})
	base := []float32{-2, -1, 1, 2}
	add("mtp.0.hc_head_base.weight", []uint64{4}, base)
	norm := make([]float32, Flash.Embed)
	for i := range norm {
		norm[i] = 1
	}
	add("mtp.0.norm.weight", []uint64{uint64(Flash.Embed)}, norm)
	m.weights.reader = bytes.NewReader(data)
	hidden := make([]float32, Flash.HC*Flash.Embed)
	for lane := 0; lane < Flash.HC; lane++ {
		for i := 0; i < Flash.Embed; i++ {
			hidden[lane*Flash.Embed+i] = float32(lane+1) * float32(i%5-2)
		}
	}
	original := append([]float32(nil), hidden...)
	got, err := m.headInput(hidden)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float64, Flash.Embed)
	for lane, b := range base {
		weight := 1/(1+math.Exp(-float64(b))) + float64(Flash.HCEps)
		for i := range want {
			want[i] += weight * float64(hidden[lane*Flash.Embed+i])
		}
	}
	var sum float64
	for _, v := range want {
		sum += v * v
	}
	scale := 1 / math.Sqrt(sum/float64(len(want))+float64(Flash.RMSEps))
	for i, v := range got {
		if math.Abs(float64(v)-want[i]*scale) > 2e-6 {
			t.Fatalf("head[%d]=%g, want %g", i, v, want[i]*scale)
		}
	}
	for i, v := range hidden {
		if v != original[i] {
			t.Fatal("head mutated hidden state")
		}
	}
	got[0] = 999
	if hidden[0] != original[0] {
		t.Fatal("head output aliases hidden state")
	}
}

func TestLegacyMTPDraftLength(t *testing.T) {
	for _, tt := range []struct{ in, want int }{{-1, 1}, {0, 1}, {1, 1}, {2, 2}, {16, 16}, {17, 16}} {
		if got := legacyMTPDraftLength(tt.in); got != tt.want {
			t.Fatalf("length(%d)=%d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestLegacyMTPSamplingDispatch(t *testing.T) {
	for _, tt := range []struct {
		temperature float32
		want        bool
	}{{0, true}, {.7, false}, {1, false}} {
		got, err := legacyMTPMayDraft(tt.temperature)
		if err != nil || got != tt.want {
			t.Fatalf("dispatch(%g)=%v,%v", tt.temperature, got, err)
		}
	}
	for _, v := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := legacyMTPMayDraft(v); err == nil {
			t.Fatal("invalid sampling temperature accepted")
		}
	}
}

func TestLegacyMTPMargin(t *testing.T) {
	for _, tt := range []struct {
		name      string
		logits    []float32
		drafts    int
		threshold float32
		want      bool
	}{
		{"below", []float32{1, 3.9}, 2, 3, false}, {"equal", []float32{1, 4}, 2, 3, true},
		{"above", []float32{1, 5}, 2, 3, true}, {"tie", []float32{4, 4}, 2, 3, false},
		{"one", nil, 1, 3, true}, {"three", nil, 3, 3, true}, {"disabled", nil, 2, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := legacyMTPMargin(tt.logits, tt.drafts, tt.threshold)
			if err != nil || got != tt.want {
				t.Fatalf("margin=%v,%v, want %v", got, err, tt.want)
			}
		})
	}
	if _, err := legacyMTPMargin([]float32{1}, 2, 3); err == nil {
		t.Fatal("accepted malformed logits")
	}
	if _, err := legacyMTPMargin(nil, 1, -1); err == nil {
		t.Fatal("accepted negative threshold")
	}
}

func TestLegacyMTPMissingWeights(t *testing.T) {
	if _, err := admitLegacyMTPModel(nil); err == nil {
		t.Fatal("accepted missing model")
	}
	var m *legacyMTPModel
	if _, err := m.input(nil, nil); err == nil {
		t.Fatal("accepted missing weights")
	}
}

func TestLegacyMTPRejectsDSparkAndDuplicates(t *testing.T) {
	for _, tensors := range [][]gguf.Tensor{
		{{Name: "mtp.0.main_proj.weight"}},
		{{Name: "mtp.0.e_proj.weight"}, {Name: "mtp.0.e_proj.weight"}},
	} {
		if _, err := admitLegacyMTPModel(&gguf.File{Tensors: tensors}); err == nil {
			t.Fatal("accepted incompatible directory")
		}
	}
}
