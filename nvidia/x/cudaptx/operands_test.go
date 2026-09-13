package cudaptx_test

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestSplitOperands(t *testing.T) {
	tests := []struct {
		name, input string
		want        []string
	}{
		{"vector", "{%f1, %f2, %f3, %f4}, [%rd5]", []string{"{%f1, %f2, %f3, %f4}", "[%rd5]"}},
		{"registers", "%r1, %r2", []string{"%r1", "%r2"}},
		{"nested", "[%r1+%r2], (%r3, %r4)", []string{"[%r1+%r2]", "(%r3, %r4)"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cudaptx.SplitOperands(tt.input); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("SplitOperands(%q) = %#v, want %#v", tt.input, got, tt.want)
			}
		})
	}
}

func TestExplainVectorFixtureOperands(t *testing.T) {
	h := cudaptx.Explain("ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd5];")
	if len(h.Operands) != 2 || h.Operands[0].Text != "{%f1, %f2, %f3, %f4}" || h.Operands[1].Text != "[%rd5]" {
		t.Fatalf("operands = %#v, want grouped vector and address", h.Operands)
	}
	if h.Operands[0].Doc != nil {
		t.Fatalf("vector operand unexpectedly resolved as register: %#v", h.Operands[0].Doc)
	}
	plain := cudaptx.Explain("mov.u32 %r0, %tid.x;")
	if len(plain.Operands) < 2 || plain.Operands[1].Doc == nil {
		t.Fatal("plain register operand did not resolve")
	}
}

func TestVectorFixtureParserOperands(t *testing.T) {
	m := parseFixture(t, "vector.sm90.ptx")
	k := m.Kernel("vec4")
	if k == nil {
		t.Fatal("vec4 kernel not found")
	}
	want := map[string][]string{
		"ld.global.v4.f32": {"{%f1, %f2, %f3, %f4}", "[%rd5]"},
		"st.global.v4.f32": {"[%rd7]", "{%f12, %f11, %f10, %f9}"},
	}
	for _, inst := range k.Instructions {
		if want, ok := want[inst.Mnemonic]; ok {
			if got := cudaptx.SplitOperands(inst.Operands); !reflect.DeepEqual(got, want) {
				t.Errorf("%s operands = %#v, want %#v", inst.Mnemonic, got, want)
			}
		}
	}
}
