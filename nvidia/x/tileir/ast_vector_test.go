package tileir

import (
	"bytes"
	"testing"
)

func TestASTVectorPrimaryFrames(t *testing.T) {
	bm, err := vectorLowering(t, 5).Module.ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	// Primary opcode/result-type/attribute/SSA fields, independently written
	// from Ops.td and BytecodeGen.cpp. Block ID has THREE fixed results,
	// no axis attribute or invented result count. All shape changes are explicit.
	want := []byte{
		0x30, 5, 5, 5, 0x25, 7, 0, 3, 0x10, 7, 0, 0x4e, 7, 1, 6, 7,
		0x5b, 8, 8, 0x0b, 9, 9, 0x3a, 9, 0x03, 9, 1, 10, 11,
		0x10, 9, 1, 0x0f, 11, 2, 0, 12, 13, 0x10, 9, 2, 0x5f, 9, 14, 12, 15,
		0x5b, 12, 0, 0x0b, 13, 17, 0x51, 13, 18, 16,
		0x5b, 12, 1, 0x0b, 13, 20, 0x51, 13, 21, 16,
		0x5b, 12, 2, 0x0b, 13, 23, 0x51, 13, 24, 16,
		0x10, 14, 3, 0x44, 15,
		0x3d, 14, 15, 28, 0, 19, 14, 26, 27,
		0x3d, 14, 15, 28, 0, 22, 14, 26, 29,
		0x02, 14, 0, 0, 28, 30,
		0x65, 15, 12, 0, 25, 32, 14, 31, 0x5c, 0, 0,
	}
	if !bytes.Equal(bm.Functions[0].BodyBytes, want) {
		t.Fatalf("vector body=%x\nwant=%x", bm.Functions[0].BodyBytes, want)
	}
	if len(bm.Functions[0].Operations[0].Results) != 3 {
		t.Fatal("block ID results lost")
	}
}

func TestASTVectorRefusals(t *testing.T) {
	for _, mutate := range []func(*LoweringResult){
		func(r *LoweringResult) { r.Module.Functions[0].Operations[0].Results = []string{"%block_x"} },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[0].Attributes = "{axis = 0}" },
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[0].TypeVal = NewFunction(nil, []Type{NewTile(nil, Int64), NewTile(nil, Int32), NewTile(nil, Int32)})
		},
		func(r *LoweringResult) { r.Module.Functions[0].Operations[1].TypeVal = NewTile(nil, Int32) },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[5].Operands[0] = "%start" }, // rank-changing broadcast
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[12].TypeVal = NewTile([]int64{2}, NewPointer(Float32))
		},
		func(r *LoweringResult) { r.Module.Functions[0].Operations[14].Operands[0] = "%arg0" },  // scalar/vector pointer mismatch
		func(r *LoweringResult) { r.Module.Functions[0].Operations[14].Operands[1] = "%start" }, // scalar/vector offset mismatch
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[26].Operands[1] = "%padding"
			r.Module.Functions[0].Operations[21].TypeVal = NewTile(nil, Float32)
		},
	} {
		r := vectorLowering(t, 5)
		mutate(r)
		if data, err := r.Module.EncodeBytecode(); data != nil || err == nil {
			t.Fatalf("invalid vector AST encoded: %x %v", data, err)
		}
	}
}

func TestASTVectorMalformedCompleteBody(t *testing.T) {
	bm, err := vectorLowering(t, 5).Module.ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func([]byte){
		func(b []byte) { b[1] = 7 },  // wrong first block ID type
		func(b []byte) { b[6] = 2 },  // invalid signedness enum
		func(b []byte) { b[21] = 3 }, // broadcast source points to scalar blockID
	} {
		body := bytes.Clone(bm.Functions[0].BodyBytes)
		mutate(body)
		if ops, err := decodeASTBody(body, bm.Types, bm.Constants, bm.Functions[0].Signature); ops != nil || err == nil {
			t.Fatal("malformed complete vector body published")
		}
	}
}
