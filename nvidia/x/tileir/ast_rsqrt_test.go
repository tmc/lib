package tileir_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func rsqrtAST() *tileir.Module {
	typ := tileir.NewTile([]int64{16}, tileir.Float32)
	return &tileir.Module{Functions: []*tileir.Function{{Name: "rsqrt", Params: []*tileir.Param{{Name: "%x", TypeVal: typ}}, Returns: []string{typ.String()}, Operations: []*tileir.Operation{
		{OpCode: "cuda_tile.rsqrt", Results: []string{"%r"}, Operands: []string{"%x"}, TypeVal: typ},
		{OpCode: "cuda_tile.return", Operands: []string{"%r"}},
	}}}}
}

func TestASTRsqrtBytes(t *testing.T) {
	for _, ftz := range []bool{false, true} {
		m := rsqrtAST()
		flag := byte(0)
		if ftz {
			m.Functions[0].Operations[0].Attributes = "{flush_to_zero}"
			flag = 1
		}
		b, err := m.ToBytecode()
		if err != nil {
			t.Fatal(err)
		}
		// Pinned opspec: 0x5D, result type index1, optional unit flag, operand0.
		// Return: opcode0x5C, zero results, one operand, SSA result index1.
		want := []byte{0x5d, 0x01, flag, 0x00, 0x5c, 0x00, 0x01, 0x01}
		if !bytes.Equal(b.Functions[0].BodyBytes, want) {
			t.Fatalf("body %x, want %x", b.Functions[0].BodyBytes, want)
		}
		wire, err := m.EncodeBytecode()
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := tileir.DecodeBytecodeBytes(wire)
		if err != nil {
			t.Fatal(err)
		}
		op := decoded.Functions[0].Operations[0]
		if op.OpCode != "cuda_tile.rsqrt" || op.TypeVal.String() != m.Functions[0].Params[0].TypeVal.String() || op.Attributes != m.Functions[0].Operations[0].Attributes || len(op.Operands) != 1 || op.Operands[0] != "%arg0" {
			t.Fatalf("decoded rsqrt: %+v", op)
		}
		round, err := decoded.ToModule().EncodeBytecode()
		if err != nil || !bytes.Equal(round, wire) {
			t.Fatalf("roundtrip: %v", err)
		}
	}
}

func TestASTRsqrtRejectsMalformed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*tileir.Module)
	}{
		{"integer", func(m *tileir.Module) {
			typ := tileir.NewTile([]int64{16}, tileir.Int32)
			m.Functions[0].Params[0].TypeVal = typ
			m.Functions[0].Operations[0].TypeVal = typ
			m.Functions[0].Returns = []string{typ.String()}
		}},
		{"shape", func(m *tileir.Module) {
			m.Functions[0].Operations[0].TypeVal = tileir.NewTile([]int64{8}, tileir.Float32)
		}},
		{"zero operands", func(m *tileir.Module) { m.Functions[0].Operations[0].Operands = nil }},
		{"extra operand", func(m *tileir.Module) { m.Functions[0].Operations[0].Operands = []string{"%x", "%x"} }},
		{"ftz payload", func(m *tileir.Module) { m.Functions[0].Operations[0].Attributes = "{flush_to_zero = false}" }},
		{"unknown rounding", func(m *tileir.Module) { m.Functions[0].Operations[0].Attributes = "{rounding_mode = nearest_even}" }},
		{"f64 outside subset", func(m *tileir.Module) {
			typ := tileir.NewTile([]int64{16}, tileir.Float64)
			m.Functions[0].Params[0].TypeVal = typ
			m.Functions[0].Operations[0].TypeVal = typ
			m.Functions[0].Returns = []string{typ.String()}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := rsqrtAST()
			tc.change(m)
			if wire, err := m.EncodeBytecode(); wire != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
				t.Fatalf("malformed accepted: %x %v", wire, err)
			}
		})
	}
}
