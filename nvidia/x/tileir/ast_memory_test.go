package tileir

import (
	"bytes"
	"errors"
	"testing"
)

func TestASTMemoryPrimaryBody(t *testing.T) {
	r := memoryLowering(t)
	bm, err := r.Module.ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	// Independent primary-format frame: fixed two load result references,
	// flags28/12, weak0, no operand/result count for optional groups.
	want := []byte{0x10, 5, 0, 0x10, 6, 1, 0x44, 7,
		0x3d, 6, 7, 28, 0, 0, 3, 4, 5,
		0x3d, 6, 7, 28, 0, 1, 3, 4, 7,
		0x02, 6, 0, 0, 6, 8,
		0x65, 7, 12, 0, 2, 10, 3, 9,
		0x5c, 0, 0}
	if !bytes.Equal(bm.Functions[0].BodyBytes, want) {
		t.Fatalf("memory body = %x, want primary-format %x", bm.Functions[0].BodyBytes, want)
	}
	if pointer := encodeSingleType(NewPointer(Float32), []Type{Float32}, BytecodeVersion{Major: 13, Minor: 4}); !bytes.Equal(pointer, []byte{12, 0, 0}) {
		t.Fatalf("pointer frame = %x", pointer)
	}
	data, err := EncodeBytecode(bm)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeBytecodeBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded.Functions[0].Operations[3].Results) != 2 || decoded.Functions[0].Operations[3].Operands[3] != "%2" {
		t.Fatal("load/token framing lost")
	}
	round, err := decoded.ToModule().EncodeBytecode()
	if err != nil || !bytes.Equal(data, round) {
		t.Fatalf("memory AST round trip: %v", err)
	}
}

func TestASTMemoryOptionalSegments(t *testing.T) {
	ptr, mask, padding := NewTile(nil, NewPointer(Float32)), NewTile(nil, Int1), NewTile(nil, Float32)
	for bits := uint64(0); bits < 8; bits++ {
		operands := []Type{ptr}
		for i, typ := range []Type{mask, padding, Token} {
			if bits&(1<<i) != 0 {
				operands = append(operands, typ)
			}
		}
		flags, err := memoryFlags("cuda_tile.load_ptr_tko", operands)
		if err != nil || flags != bits<<2 || memoryOperandCount("cuda_tile.load_ptr_tko", flags) != uint64(len(operands)) {
			t.Fatalf("load flags for %d: %d, %v", bits, flags, err)
		}
		var names []string
		values := make(map[string]int)
		for i := range operands {
			name := string(rune('a' + i))
			names = append(names, name)
			values[name] = i
		}
		frame, err := encodeMemoryAttrs(nil, &Operation{OpCode: "cuda_tile.load_ptr_tko", Operands: names}, map[string]string{"memory_ordering_semantics": "weak"}, values, operands)
		if err != nil || !bytes.Equal(frame, []byte{byte(bits << 2), 0}) {
			t.Fatalf("load attr frame = %x, %v", frame, err)
		}
	}
	for bits := uint64(0); bits < 4; bits++ {
		operands := []Type{ptr, padding}
		for i, typ := range []Type{mask, Token} {
			if bits&(1<<i) != 0 {
				operands = append(operands, typ)
			}
		}
		flags, err := memoryFlags("cuda_tile.store_ptr_tko", operands)
		if err != nil || flags != bits<<2 || memoryOperandCount("cuda_tile.store_ptr_tko", flags) != uint64(len(operands)) {
			t.Fatalf("store flags for %d: %d, %v", bits, flags, err)
		}
	}
	// Full admitted entries cover both optional data groups on loads and
	// the store mask. Tokens stay present under this route's chain policy.
	for bits := 0; bits < 8; bits++ {
		r := memoryLowering(t)
		for _, index := range []int{3, 4} {
			op := r.Module.Functions[0].Operations[index]
			operands := []string{op.Operands[0]}
			if bits&1 != 0 {
				operands = append(operands, "%mask")
			}
			if bits&2 != 0 {
				operands = append(operands, "%padding")
			}
			operands = append(operands, op.Operands[3])
			op.Operands = operands
		}
		if bits&4 == 0 {
			r.Module.Functions[0].Operations[6].Operands = []string{"%arg2", "%sum", "%loaded_y"}
		}
		data, err := r.Module.EncodeBytecode()
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytecodeBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		round, err := decoded.ToModule().EncodeBytecode()
		if err != nil || !bytes.Equal(data, round) {
			t.Fatalf("optional group entry %d: %v", bits, err)
		}
	}
}

func TestASTMemoryRefusals(t *testing.T) {
	for _, mutate := range []func(*LoweringResult){
		func(r *LoweringResult) { r.Module.Functions[0].Params[0].TypeVal = NewPointer(Float32) },
		func(r *LoweringResult) { r.Module.Functions[0].Params[0].TypeVal = NewTile(nil, NewPointer(Float64)) },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[0].TypeVal = NewTile([]int64{1}, Int1) },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[1].TypeVal = NewTile(nil, Float64) },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[4].Operands[3] = "%start" },
		func(r *LoweringResult) { r.Module.Functions[0].Operations[7].Operands = []string{"%stored"} },
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[3].Attributes = "{memory_ordering_semantics = release}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[6].Attributes = "{memory_ordering_semantics = acquire}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[3].Attributes = "{memory_ordering_semantics = relaxed}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[3].Attributes = "{memory_ordering_semantics = acquire}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[6].Attributes = "{memory_ordering_semantics = release}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[3].Attributes = "{memory_ordering_semantics = weak, memory_scope = device}"
		},
		func(r *LoweringResult) {
			r.Module.Functions[0].Operations[3].TypeVal = NewFunction(nil, []Type{NewTile(nil, Float32)})
		},
		func(r *LoweringResult) { r.Module.Functions[0].Entry = false },
	} {
		r := memoryLowering(t)
		mutate(r)
		if data, err := r.Module.EncodeBytecode(); data != nil || !errors.Is(err, ErrUnsupportedASTEncoding) {
			t.Fatalf("unsupported memory AST encoded: %x, %v", data, err)
		}
	}
}

func TestASTMemoryMalformedFrames(t *testing.T) {
	bm, err := memoryLowering(t).Module.ToBytecode()
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range [][]byte{
		{0x3d, 6, 7, 32},                // unknown flag
		{0x3d, 6, 7, 1, 0},              // unsupported scope
		{0x3d, 6, 7, 28, 3, 0, 3, 4, 5}, // load release
		{0x65, 7, 12, 2, 2, 10, 3, 9},   // store acquire
		{0x3d, 6, 7, 28, 0, 0, 3},       // truncated optional groups
		{0x3d, 2, 6, 7, 28, 0},          // invented result count
	} {
		ops, err := decodeASTBody(body, bm.Types, bm.Constants, bm.Functions[0].Signature)
		if err == nil || ops != nil {
			t.Fatalf("malformed memory body published: %x", body)
		}
	}
	// Mutate one field in an otherwise complete, admitted entry. Failure
	// cannot be explained just by a missing terminal return or token chain.
	for _, mutate := range []func([]byte){
		func(body []byte) { body[11] |= 32 }, // unknown load flag
		func(body []byte) { body[11] |= 1 },  // scope outside this subset
		func(body []byte) { body[12] = 3 },   // illegal load release
		func(body []byte) { body[35] = 2 },   // illegal store acquire
		func(body []byte) { body[12] = 1 },   // relaxed load missing scope
		func(body []byte) { body[12] = 2 },   // acquire load missing scope
		func(body []byte) { body[35] = 3 },   // release store missing scope
		func(body []byte) { body[10] = 6 },   // second load result is not token
		func(body []byte) { body[16] = 6 },   // predecessor field is numeric
	} {
		body := bytes.Clone(bm.Functions[0].BodyBytes)
		mutate(body)
		if ops, err := decodeASTBody(body, bm.Types, bm.Constants, bm.Functions[0].Signature); err == nil || ops != nil {
			t.Fatal("mutated complete frame published")
		}
	}
	for _, payload := range [][]byte{{}, {0}, {0, 99}, {0, 0, 1}, {1, 0}} {
		if _, err := decodeMemoryPointer(payload, []Type{Float32}, BytecodeVersion{Major: 13, Minor: 4}); err == nil {
			t.Fatalf("malformed pointer accepted: %x", payload)
		}
	}
}
