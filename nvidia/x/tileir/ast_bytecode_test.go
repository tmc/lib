package tileir_test

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func addAST() *tileir.Module {
	typ := tileir.NewTile([]int64{4}, tileir.Float32)
	return &tileir.Module{Functions: []*tileir.Function{{
		Name: "add", Params: []*tileir.Param{{Name: "%x", TypeVal: typ}, {Name: "%y", TypeVal: typ}},
		Returns: []string{typ.String()}, Operations: []*tileir.Operation{
			{OpCode: "cuda_tile.addf", Results: []string{"%sum"}, Operands: []string{"%x", "%y"},
				TypeVal: typ, Attributes: "{rounding_mode = nearest_even, flush_to_zero}"},
			{OpCode: "cuda_tile.return", Operands: []string{"%sum"}},
		},
	}}}
}

func TestASTBytecodeOperandAttributeBytes(t *testing.T) {
	m := addAST()
	bm, err := m.ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	// NVIDIA BytecodeWriter.cpp:1090-1163 and BytecodeGen.cpp:660-666:
	// addf opcode, tile type index, ftz flags, nearest-even enum, x/y indices;
	// return opcode, result count0, operand count, sum index. Parameters0/1.
	want := []byte{0x02, 0x01, 0x01, 0x00, 0x00, 0x01, 0x5c, 0x00, 0x01, 0x02}
	if !bytes.Equal(bm.Functions[0].BodyBytes, want) {
		t.Fatalf("body = %x, want primary-format bytes %x", bm.Functions[0].BodyBytes, want)
	}
	encoded, err := m.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tileir.DecodeBytecodeBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	op := decoded.Functions[0].Operations[0]
	if strings.Join(op.Operands, ",") != "%arg0,%arg1" || op.TypeVal.String() != m.Functions[0].Params[0].TypeVal.String() ||
		op.Attributes != "{rounding_mode = nearest_even, flush_to_zero}" {
		t.Fatalf("decoded typed operation = %+v", op)
	}
	if strings.Join(decoded.Functions[0].Operations[1].Operands, ",") != "%0" {
		t.Fatal("return operand was lost")
	}
	round, err := decoded.ToModule().EncodeBytecode()
	if err != nil || !bytes.Equal(round, encoded) {
		t.Fatalf("AST round trip differs: %v", err)
	}
}

func TestASTBytecodeVendorConstantGoldens(t *testing.T) {
	files, err := filepath.Glob("testdata/bytecodegoldens/*.tileirbc")
	if err != nil || len(files) == 0 {
		t.Fatalf("golden files: %v (%d files)", err, len(files))
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			want, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			bm, err := tileir.DecodeBytecodeBytes(want)
			if err != nil {
				t.Fatal(err)
			}
			m := bm.ToModule()
			if m == nil {
				t.Fatalf("vendor constant unsupported: %v", bm.Functions[0].BodyDecodeError)
			}
			ops := m.Functions[0].Operations
			if len(ops) != 2 || ops[0].OpCode != "cuda_tile.constant" || ops[1].OpCode != "cuda_tile.return" ||
				!strings.Contains(ops[0].Attributes, "dense<") {
				t.Fatalf("constant framing lost: %+v", ops)
			}
			got, err := m.EncodeBytecode()
			if err != nil || !bytes.Equal(got, want) {
				t.Fatalf("AST encoding differs from independently vendor-produced bytes: %v\ngot %x\nwant %x", err, got, want)
			}
		})
	}
}

func TestASTBytecodeHostOperandOracle(t *testing.T) {
	data, err := addAST().EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	bm, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{1, -2, 100, -0.5}
	y := []float32{8, 3, -99, 0.25}
	values := map[string][]float32{"%arg0": x, "%arg1": y}
	var got []float32
	for _, op := range bm.Functions[0].Operations {
		switch op.OpCode {
		case "cuda_tile.addf":
			a, b := values[op.Operands[0]], values[op.Operands[1]]
			out := make([]float32, len(a))
			for i := range out {
				out[i] = a[i] + b[i]
			}
			values[op.Results[0]] = out
		case "cuda_tile.return":
			got = values[op.Operands[0]]
		default:
			t.Fatalf("unexpected decoded operation %s", op.OpCode)
		}
	}
	want := []float32{9, 1, 1, -0.25}
	for i, value := range want {
		if got[i] != value {
			t.Fatalf("decoded operand oracle[%d] = %g, want %g", i, got[i], value)
		}
	}
}

func TestASTBytecodeConstantBits(t *testing.T) {
	for _, value := range []string{"-0.0", `dense<"0x0100c07f">`} {
		m := &tileir.Module{Functions: []*tileir.Function{{Name: "constant", Entry: true,
			Operations: []*tileir.Operation{
				{OpCode: "cuda_tile.constant", Results: []string{"%c"}, TypeVal: tileir.NewTile([]int64{4}, tileir.Float32), Attributes: "{value = " + value + "}"},
				{OpCode: "cuda_tile.return"},
			},
		}}}
		bm, err := m.ToBytecode()
		if err != nil {
			t.Fatal(err)
		}
		want := []byte{4, 0, 0, 0, 0x80}
		if value != "-0.0" {
			want = []byte{4, 1, 0, 0xc0, 0x7f}
		}
		if !bytes.Equal(bm.Constants[0], want) {
			t.Fatalf("constant bits = %x, want %x", bm.Constants[0], want)
		}
	}
}

func TestASTBytecodeUnsupportedClosed(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*tileir.Module)
	}{
		{"attention", func(m *tileir.Module) { m.Functions[0].Operations[0].OpCode = "cuda_tile.online_attention" }},
		{"region", func(m *tileir.Module) { m.Functions[0].Operations[0].OpCode = "cuda_tile.loop" }},
		{"memory", func(m *tileir.Module) { m.Functions[0].Operations[0].OpCode = "cuda_tile.load_view_tko" }},
		{"undefined_operand", func(m *tileir.Module) { m.Functions[0].Operations[0].Operands[0] = "%missing" }},
		{"type_mismatch", func(m *tileir.Module) { m.Functions[0].Params[1].TypeVal = tileir.NewTile([]int64{4}, tileir.Float64) }},
		{"duplicate_ssa", func(m *tileir.Module) { m.Functions[0].Operations[0].Results[0] = "%x" }},
		{"missing_rounding", func(m *tileir.Module) { m.Functions[0].Operations[0].Attributes = "{}" }},
		{"invalid_rounding", func(m *tileir.Module) { m.Functions[0].Operations[0].Attributes = "{rounding_mode = unknown}" }},
		{"unknown_attribute", func(m *tileir.Module) {
			m.Functions[0].Operations[0].Attributes = "{rounding_mode = nearest_even, semantic_id = 4}"
		}},
		{"unit_payload", func(m *tileir.Module) {
			m.Functions[0].Operations[0].Attributes = "{rounding_mode = nearest_even, flush_to_zero = false}"
		}},
		{"duplicate_attribute", func(m *tileir.Module) {
			m.Functions[0].Operations[0].Attributes = "{rounding_mode = zero, rounding_mode = nearest_even}"
		}},
		{"location", func(m *tileir.Module) { m.Functions[0].Operations[0].Loc = tileir.Location{File: "a.go", Line: 1} }},
		{"global", func(m *tileir.Module) { m.Globals = []*tileir.Global{{Name: "g"}} }},
		{"typed_nil", func(m *tileir.Module) { m.Functions[0].Params[0].TypeVal = (*tileir.TileType)(nil) }},
		{"huge_shape", func(m *tileir.Module) {
			m.Functions[0].Params[0].TypeVal = tileir.NewTile([]int64{1 << 62, 8}, tileir.Float32)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := addAST()
			tt.mutate(m)
			data, err := m.EncodeBytecode()
			if data != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
				t.Fatalf("partial encoding = %x, %v", data, err)
			}
		})
	}
}

func TestASTBytecodeMalformedBody(t *testing.T) {
	for _, body := range [][]byte{{0x80}, {0x02, 1, 2}, {0x02, 1, 0, 8, 0, 1}, {0x5c, 1, 99}} {
		bm, err := addAST().ToBytecode()
		if err != nil {
			t.Fatal(err)
		}
		bm.Functions[0].BodyBytes = body
		data, err := tileir.EncodeBytecode(bm)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tileir.DecodeBytecodeBytes(data); err == nil {
			t.Fatalf("malformed body %x accepted", body)
		}
	}
}

func TestASTBytecodeOpaqueUnsupportedBody(t *testing.T) {
	bm, err := addAST().ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	bm.Functions[0].BodyBytes = []byte{0x41} // Region opcode; never a guessed one-result op.
	data, err := tileir.EncodeBytecode(bm)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Functions[0].BodyDecodeError == nil || decoded.Functions[0].Operations != nil || decoded.ToModule() != nil {
		t.Fatal("opaque body published a partial AST")
	}
	round, err := tileir.EncodeBytecode(decoded)
	if err != nil || !bytes.Equal(round, data) {
		t.Fatalf("raw body preservation failed: %v", err)
	}
}

func TestASTBytecodeTokenOperands(t *testing.T) {
	m := &tileir.Module{Functions: []*tileir.Function{{Name: "tokens",
		Returns: []string{tileir.Token.String()}, Operations: []*tileir.Operation{
			{OpCode: "cuda_tile.make_token", Results: []string{"%a"}, TypeVal: tileir.Token},
			{OpCode: "cuda_tile.make_token", Results: []string{"%b"}, TypeVal: tileir.Token},
			{OpCode: "cuda_tile.join_tokens", Results: []string{"%joined"}, Operands: []string{"%a", "%b"}, TypeVal: tileir.Token},
			{OpCode: "cuda_tile.return", Operands: []string{"%joined"}},
		},
	}}}
	data, err := m.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	bm, err := tileir.DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if got := bm.Functions[0].Operations[2].Operands; strings.Join(got, ",") != "%0,%1" {
		t.Fatalf("join operands = %v", got)
	}
	round, err := bm.ToModule().EncodeBytecode()
	if err != nil || !bytes.Equal(data, round) {
		t.Fatalf("token round trip: %v", err)
	}
	m.Functions[0].Operations[2].Operands = []string{"%a", "%a"}
	if data, err := m.EncodeBytecode(); data != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatalf("multiply consumed token accepted: %x, %v", data, err)
	}
}

func TestASTBytecodeInvalidFunctionReferences(t *testing.T) {
	for _, mutate := range []func(*tileir.BytecodeModule){
		func(bm *tileir.BytecodeModule) { bm.Functions[0] = nil },
		func(bm *tileir.BytecodeModule) { bm.Functions[0].Name = "missing" },
		func(bm *tileir.BytecodeModule) { bm.Functions[0].SignatureIndex = len(bm.Types) },
		func(bm *tileir.BytecodeModule) { bm.Functions[0].SignatureIndex = 0 },
	} {
		bm, err := addAST().ToBytecode()
		if err != nil {
			t.Fatal(err)
		}
		mutate(bm)
		if data, err := tileir.EncodeBytecode(bm); data != nil || err == nil {
			t.Fatalf("invalid function reference accepted: %x, %v", data, err)
		}
	}
}

func TestASTBytecodeNumericSubset(t *testing.T) {
	// Opcode and attribute bytes are pinned from NVIDIA Ops.td/AttrDefs.td,
	// independently of this package's generated opSpec table.
	tests := []struct {
		name    string
		opcode  byte
		integer bool
		arity   int
		attrs   string
		wire    []byte
	}{
		{"absf", 0x00, false, 1, "", nil},
		{"absi", 0x01, true, 1, "", nil},
		{"negf", 0x4f, false, 1, "", nil},
		{"negi", 0x50, true, 1, "{overflow = no_signed_wrap}", []byte{1}},
		{"addf", 0x02, false, 2, "{rounding_mode = zero, flush_to_zero}", []byte{1, 1}},
		{"subf", 0x67, false, 2, "{rounding_mode = negative_inf}", []byte{0, 2}},
		{"mulf", 0x4c, false, 2, "{rounding_mode = positive_inf}", []byte{0, 3}},
		{"divf", 0x14, false, 2, "{rounding_mode = nearest_even}", []byte{0, 0}},
		{"addi", 0x03, true, 2, "{overflow = no_unsigned_wrap}", []byte{2}},
		{"subi", 0x68, true, 2, "{overflow = no_wrap}", []byte{3}},
		{"muli", 0x4e, true, 2, "{overflow = none}", []byte{0}},
		{"maxf", 0x45, false, 2, "{propagate_nan, flush_to_zero}", []byte{3}},
		{"minf", 0x47, false, 2, "{flush_to_zero}", []byte{2}},
		{"maxi", 0x46, true, 2, "{signedness = signed}", []byte{1}},
		{"mini", 0x48, true, 2, "{signedness = unsigned}", []byte{0}},
		{"andi", 0x04, true, 2, "", nil},
		{"ori", 0x52, true, 2, "", nil},
		{"xori", 0x6c, true, 2, "", nil},
		{"cmpf", 0x0e, false, 2, "{comparison_predicate = less_than, comparison_ordering = ordered}", []byte{2, 1}},
		{"cmpi", 0x0f, true, 2, "{comparison_predicate = greater_than_or_equal, signedness = signed}", []byte{5, 1}},
		{"select", 0x5f, false, 3, "", nil},
		{"mmaf", 0x49, false, 3, "{fast_acc}", []byte{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			element := tileir.Float32
			if tt.integer {
				element = tileir.Int32
			}
			typ := tileir.NewTile([]int64{4}, element)
			if tt.name == "mmaf" {
				typ = tileir.NewTile([]int64{2, 2}, element)
			}
			result := typ
			if tt.name == "cmpf" || tt.name == "cmpi" {
				result = tileir.NewTile([]int64{4}, tileir.Int1)
			}
			fn := &tileir.Function{Name: tt.name, Returns: []string{result.String()}}
			op := &tileir.Operation{OpCode: "cuda_tile." + tt.name, Results: []string{"%r"}, TypeVal: result, Attributes: tt.attrs}
			for i := 0; i < tt.arity; i++ {
				name := []string{"%a", "%b", "%c"}[i]
				paramType := typ
				if tt.name == "select" && i == 0 {
					paramType = tileir.NewTile([]int64{4}, tileir.Int1)
				}
				fn.Params = append(fn.Params, &tileir.Param{Name: name, TypeVal: paramType})
				op.Operands = append(op.Operands, name)
			}
			fn.Operations = []*tileir.Operation{op, {OpCode: "cuda_tile.return", Operands: []string{"%r"}}}
			bm, err := (&tileir.Module{Functions: []*tileir.Function{fn}}).ToBytecode()
			if err != nil {
				t.Fatal(err)
			}
			want := []byte{tt.opcode, bm.Functions[0].BodyBytes[1]}
			want = append(want, tt.wire...)
			for i := 0; i < tt.arity; i++ {
				want = append(want, byte(i))
			}
			want = append(want, 0x5c, 0, 1, byte(tt.arity))
			if !bytes.Equal(bm.Functions[0].BodyBytes, want) {
				t.Fatalf("body = %x, want %x", bm.Functions[0].BodyBytes, want)
			}
			data, err := tileir.EncodeBytecode(bm)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := tileir.DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			round, err := decoded.ToModule().EncodeBytecode()
			if err != nil || !bytes.Equal(data, round) {
				t.Fatalf("round trip: %v", err)
			}
		})
	}
}

func TestASTBytecodeMmaDtypes(t *testing.T) {
	// The explicit tuples come from NVIDIA Ops.td:1489-1534, independently
	// of the encoder's dtype dispatch. Shapes are otherwise valid 2x2 MMA.
	tests := []struct {
		lhs, rhs, acc tileir.Type
		want          bool
	}{
		{tileir.Float16, tileir.Float16, tileir.Float16, true},
		{tileir.Float16, tileir.Float16, tileir.Float32, true},
		{tileir.BFloat16, tileir.BFloat16, tileir.Float32, true},
		{tileir.TFloat32, tileir.TFloat32, tileir.Float32, true},
		{tileir.Float32, tileir.Float32, tileir.Float32, true},
		{tileir.Float64, tileir.Float64, tileir.Float64, true},
		{tileir.Float8E4M3, tileir.Float8E4M3, tileir.Float16, true},
		{tileir.Float8E4M3, tileir.Float8E4M3, tileir.Float32, true},
		{tileir.Float8E5M2, tileir.Float8E5M2, tileir.Float16, true},
		{tileir.Float8E5M2, tileir.Float8E5M2, tileir.Float32, true},
		{tileir.Float16, tileir.Float32, tileir.Float32, false},
		{tileir.Float32, tileir.Float16, tileir.Float32, false},
		{tileir.Float16, tileir.Float16, tileir.BFloat16, false},
		{tileir.Float64, tileir.Float64, tileir.Float32, false},
		{tileir.Float32, tileir.Float32, tileir.Float64, false},
		{tileir.BFloat16, tileir.BFloat16, tileir.Float16, false},
		{tileir.Float8E8M0, tileir.Float8E8M0, tileir.Float32, false},
	}
	for _, tt := range tests {
		t.Run(strings.Join([]string{tt.lhs.String(), tt.rhs.String(), tt.acc.String()}, "_"), func(t *testing.T) {
			a := tileir.NewTile([]int64{2, 2}, tt.lhs)
			b := tileir.NewTile([]int64{2, 2}, tt.rhs)
			c := tileir.NewTile([]int64{2, 2}, tt.acc)
			m := &tileir.Module{Functions: []*tileir.Function{{Name: "mma", Params: []*tileir.Param{
				{Name: "%a", TypeVal: a}, {Name: "%b", TypeVal: b}, {Name: "%c", TypeVal: c},
			}, Returns: []string{c.String()}, Operations: []*tileir.Operation{
				{OpCode: "cuda_tile.mmaf", Results: []string{"%r"}, Operands: []string{"%a", "%b", "%c"}, TypeVal: c},
				{OpCode: "cuda_tile.return", Operands: []string{"%r"}},
			}}}}
			data, err := m.EncodeBytecode()
			if !tt.want {
				if data != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
					t.Fatalf("unsupported dtype tuple encoded: %x, %v", data, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := tileir.DecodeBytecodeBytes(data)
			if err != nil {
				t.Fatal(err)
			}
			round, err := decoded.ToModule().EncodeBytecode()
			if err != nil || !bytes.Equal(data, round) {
				t.Fatalf("supported tuple round trip: %v", err)
			}
		})
	}
}

func TestASTBytecodeKernelReturnInterface(t *testing.T) {
	bm, err := addAST().ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	// First construct valid device-function bytes, then change only the
	// function-kind byte. This exercises decode independently of admission.
	data, err := tileir.EncodeBytecode(bm)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) <= 19 || data[16] != 1 || data[17] != 0 || data[18] != byte(bm.Functions[0].SignatureIndex) || data[19] != 0 {
		t.Fatal("unexpected single-function framing")
	}
	data[19] = tileir.FuncKindKernel
	if decoded, err := tileir.DecodeBytecodeBytes(data); err == nil || decoded != nil {
		t.Fatal("kernel returning values decoded")
	}
	bm.Functions[0].EntryFlag = tileir.FuncKindKernel
	if data, err := tileir.EncodeBytecode(bm); err == nil || data != nil {
		t.Fatal("kernel returning values encoded")
	}
	m := addAST()
	m.Functions[0].Entry = true
	if data, err := m.EncodeBytecode(); data != nil || !errors.Is(err, tileir.ErrUnsupportedASTEncoding) {
		t.Fatal("kernel returning values admitted through AST")
	}
}
