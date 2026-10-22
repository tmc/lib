package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestDSparkHCAdmittedWeightForward(t *testing.T) {
	m := &dsparkModel{stages: 3, tensors: map[string]gguf.Tensor{}}
	var data []byte
	add := func(name string, dims []uint64, values []float32) {
		t := gguf.Tensor{Name: name, Dimensions: dims, Offset: uint64(len(data)), TypeID: 0}
		m.tensors[name] = t
		for _, v := range values {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			data = append(data, b[:]...)
		}
	}
	mixWidth := Flash.HC * (Flash.HC + 2)
	add("mtp.0.hc_attn_fn.weight", []uint64{uint64(Flash.Embed * Flash.HC), uint64(mixWidth)}, make([]float32, Flash.Embed*Flash.HC*mixWidth))
	add("mtp.0.hc_attn_scale.weight", []uint64{3}, []float32{0, 0, 0})
	add("mtp.0.hc_attn_base.weight", []uint64{uint64(mixWidth)}, make([]float32, mixWidth))
	norm := make([]float32, Flash.Embed)
	for i := range norm {
		norm[i] = 1
	}
	add("mtp.0.attn_norm.weight", []uint64{uint64(Flash.Embed)}, norm)
	m.reader = bytes.NewReader(data)
	state := make([]float32, Flash.Embed*Flash.HC)
	for lane := 0; lane < Flash.HC; lane++ {
		for i := 0; i < Flash.Embed; i++ {
			state[lane*Flash.Embed+i] = float32(lane + 1)
		}
	}
	normalized, post, combine, err := m.hcPre(0, state, false)
	if err != nil {
		t.Fatal(err)
	}
	value := float64(10) * (0.5 + float64(Flash.HCEps))
	want := value / math.Sqrt(value*value+float64(Flash.RMSEps))
	for _, v := range normalized {
		if math.Abs(float64(v)-want) > 2e-7 {
			t.Fatalf("normalized%g want%g", v, want)
		}
	}
	for _, v := range post {
		if v != 1 {
			t.Fatalf("post%g want1", v)
		}
	}
	for _, v := range combine {
		if math.Abs(float64(v)-0.25) > 1e-5 {
			t.Fatalf("combine%g", v)
		}
	}
	// Perturbing only actual norm bytes must change the output, proving the
	// reference consumes admitted weights rather than a constant oracle.
	tensor := m.tensors["mtp.0.attn_norm.weight"]
	binary.LittleEndian.PutUint32(data[tensor.Offset:], math.Float32bits(2))
	changed, _, _, err := m.hcPre(0, state, false)
	if err != nil {
		t.Fatal(err)
	}
	if changed[0] != 2*normalized[0] || changed[1] != normalized[1] {
		t.Fatal("same-site weight mutation inactive")
	}
}
