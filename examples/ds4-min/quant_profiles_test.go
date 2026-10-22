package main

import (
	"strings"
	"testing"

	"github.com/tmc/lib/gguf"
)

func syntheticExpertFile(gate, up, down uint32) *gguf.File {
	g := &gguf.File{}
	for layer := 0; layer < 43; layer++ {
		prefix := "blk." + fixtureItoa(layer) + "."
		g.Tensors = append(g.Tensors,
			gguf.Tensor{Name: prefix + "ffn_gate_exps.weight", TypeID: gate, Dimensions: []uint64{4096, 2048, 256}},
			gguf.Tensor{Name: prefix + "ffn_up_exps.weight", TypeID: up, Dimensions: []uint64{4096, 2048, 256}},
			gguf.Tensor{Name: prefix + "ffn_down_exps.weight", TypeID: down, Dimensions: []uint64{2048, 4096, 256}},
		)
	}
	return g
}

// The test fixture has only 43 names, so keeping the conversion local avoids
// making the production profile code depend on strconv formatting details.
func fixtureItoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func TestLayoutOfRegisteredExpertPairs(t *testing.T) {
	for _, tc := range []struct {
		name           string
		gate, up, down uint32
		want           ExpertFormat
	}{
		{"q2", typeIQ2XXS, typeIQ2XXS, typeQ2K, ExpertsIQ2Q2K},
		{"q4", typeQ4K, typeQ4K, typeQ4K, ExpertsQ4K},
		{"mxfp4", typeMXFP4, typeMXFP4, typeMXFP4, ExpertsMXFP4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := LayoutOf(syntheticExpertFile(tc.gate, tc.up, tc.down))
			if err != nil {
				t.Fatal(err)
			}
			for i, format := range got {
				if format != tc.want {
					t.Fatalf("layer %d format %v, want %v", i, format, tc.want)
				}
			}
		})
	}
}

func TestExpertFormatForTypes(t *testing.T) {
	for _, tc := range []struct {
		name      string
		gate, up  uint32
		down      uint32
		want      ExpertFormat
		wantError string
	}{
		{"iq2-q2k", typeIQ2XXS, typeIQ2XXS, typeQ2K, ExpertsIQ2Q2K, ""},
		{"q4k", typeQ4K, typeQ4K, typeQ4K, ExpertsQ4K, ""},
		{"mxfp4", typeMXFP4, typeMXFP4, typeMXFP4, ExpertsMXFP4, ""},
		{"gate-up mismatch", typeIQ2XXS, typeQ4K, typeQ2K, 0, "gate/up types"},
		{"unsupported mixed", typeQ4K, typeQ4K, typeQ2K, 0, "unsupported expert types"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExpertFormatForTypes(tc.gate, tc.up, tc.down)
			if tc.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantError) {
					t.Fatalf("error = %v, want %q", err, tc.wantError)
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("format = %v, error = %v, want %v", got, err, tc.want)
			}
		})
	}
}

func TestLayoutOfRejectsUnqualifiedPairs(t *testing.T) {
	tests := []struct {
		name string
		file *gguf.File
		want string
	}{
		{"missing layer", syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K), "missing gate expert tensor"},
		{"gate up mismatch", syntheticExpertFile(typeIQ2XXS, typeQ4K, typeQ2K), "gate/up types"},
		{"unsupported pair", syntheticExpertFile(typeQ4K, typeQ4K, typeQ2K), "unsupported expert types"},
		{"duplicate", syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K), ""},
	}
	// Make the first case genuinely partial and the last genuinely duplicate.
	tests[0].file.Tensors = tests[0].file.Tensors[:len(tests[0].file.Tensors)-3]
	tests[3].file.Tensors = append(tests[3].file.Tensors, tests[3].file.Tensors[0])
	tests[3].want = "duplicate expert tensor"
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LayoutOf(tc.file)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("LayoutOf error %q, want substring %q", err, tc.want)
			}
		})
	}
}

func TestLayoutOfRejectsShapeMutation(t *testing.T) {
	g := syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K)
	g.Tensors[0].Dimensions[0]++
	if _, err := LayoutOf(g); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("LayoutOf mutated gate shape = %v, want dimensions error", err)
	}
	g = syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K)
	g.Tensors[2].Dimensions = []uint64{2048, 4096}
	if _, err := LayoutOf(g); err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Fatalf("LayoutOf mutated down shape = %v, want dimensions error", err)
	}
}

func TestLayoutOfRejectsNonExpertTypeMutation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		tensor gguf.Tensor
	}{
		{name: "F16 tensor", tensor: gguf.Tensor{Name: "blk.0.hc_ffn_fn.weight", TypeID: typeQ4K}},
		{name: "F32 tensor", tensor: gguf.Tensor{Name: "blk.0.ffn_norm.weight", TypeID: typeQ4K}},
		{name: "I32 tensor", tensor: gguf.Tensor{Name: "blk.0.ffn_gate_tid2eid.weight", TypeID: typeQ4K}},
		{name: "dense tensor", tensor: gguf.Tensor{Name: "blk.0.attn_q_a.weight", TypeID: typeMXFP4}},
		{name: "indexer tensor", tensor: gguf.Tensor{Name: "blk.0.indexer.attn_q_b.weight", TypeID: typeMXFP4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K)
			g.Tensors = append(g.Tensors, tc.tensor)
			if _, err := LayoutOf(g); err == nil || !strings.Contains(err.Error(), "unsupported type") {
				t.Fatalf("LayoutOf error = %v, want unsupported type", err)
			}
		})
	}
}

func TestLayoutOfAcceptsDeclaredNonExpertTypes(t *testing.T) {
	g := syntheticExpertFile(typeIQ2XXS, typeIQ2XXS, typeQ2K)
	g.Tensors = append(g.Tensors,
		gguf.Tensor{Name: "blk.0.hc_ffn_fn.weight", TypeID: 1},
		gguf.Tensor{Name: "blk.0.ffn_norm.weight", TypeID: 0},
		gguf.Tensor{Name: "blk.0.ffn_gate_tid2eid.weight", TypeID: 26},
		gguf.Tensor{Name: "blk.0.attn_q_a.weight", TypeID: 8},
		gguf.Tensor{Name: "blk.0.indexer.attn_q_b.weight", TypeID: 1},
	)
	if _, err := LayoutOf(g); err != nil {
		t.Fatal(err)
	}
}

func TestMatchProfileRejectsUnknownManifest(t *testing.T) {
	if _, err := MatchProfile(&gguf.File{Size: pinnedModelBytes}); err == nil {
		t.Fatal("unknown profile matched with empty manifest")
	}
	if _, err := MatchProfile(nil); err == nil {
		t.Fatal("nil GGUF matched")
	}
	profiles := Profiles()
	if len(profiles) != 2 || profiles[0].Name != "ds4f-q2" || profiles[1].Name != "ds4f-q2-q4" {
		t.Fatalf("profiles = %#v", profiles)
	}
	profiles[0].Name = "mutated"
	if Profiles()[0].Name != "ds4f-q2" {
		t.Fatal("Profiles returned mutable registry")
	}
}
