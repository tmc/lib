package cudamega

import (
	"math"
	"testing"
)

func mixtureInput() MoEInput {
	c := MoEWeightConfig{3, 2, 8, 17}
	rows := c.Layers * c.Experts * 2 * c.Intermediate
	v := MoEInput{Config: c, Layer: 2, Weights: make([]byte, rows*c.Hidden), Scales: make([]float32, rows), Hidden: make([]float32, c.Hidden), Routing: [2]float32{0.25, 0.75}}
	for i := range v.Weights {
		v.Weights[i] = 0x38
	}
	for i := range v.Scales {
		v.Scales[i] = 0.1
	}
	for i := range v.Hidden {
		v.Hidden[i] = 0.1
	}
	return v
}

func TestMoEBindingOwnership(t *testing.T) {
	v := mixtureInput()
	b, err := BindMoE(v)
	if err != nil {
		t.Fatal(err)
	}
	v.Weights[0] = 0
	v.Scales[0] = 99
	v.Hidden[0] = 99
	got, err := b.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got.Weights[0] != 0x38 || got.Scales[0] != 0.1 || got.Hidden[0] != 0.1 {
		t.Fatal("binding retained caller memory")
	}
	got.Weights[0] = 0
	got.Scales[0] = 99
	got.Hidden[0] = 99
	again, _ := b.Snapshot()
	if again.Weights[0] != 0x38 || again.Scales[0] != 0.1 || again.Hidden[0] != 0.1 {
		t.Fatal("snapshot aliases binding")
	}
	if _, err := (MoEBinding{}).Snapshot(); err == nil {
		t.Fatal("zero binding accepted")
	}
}

func TestMoEBindingRefusals(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*MoEInput)
	}{
		{"shape", func(v *MoEInput) { v.Config.Hidden = 18 }},
		{"layer", func(v *MoEInput) { v.Layer = 1 }},
		{"weights", func(v *MoEInput) { v.Weights = v.Weights[1:] }},
		{"scales", func(v *MoEInput) { v.Scales = v.Scales[1:] }},
		{"hidden", func(v *MoEInput) { v.Hidden = v.Hidden[1:] }},
		{"positive fp8 nan", func(v *MoEInput) { v.Weights[0] = 0x7f }},
		{"negative fp8 nan", func(v *MoEInput) { v.Weights[0] = 0xff }},
		{"nonfinite", func(v *MoEInput) { v.Routing[1] = float32(math.Inf(1)) }},
		{"overflow", func(v *MoEInput) {
			for i := range v.Hidden {
				v.Hidden[i] = math.MaxFloat32
			}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			v := mixtureInput()
			tt.edit(&v)
			if _, err := BindMoE(v); err == nil {
				t.Fatal("accepted invalid binding")
			}
		})
	}
}
