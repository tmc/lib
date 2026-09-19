package tileirdisasm_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileirdisasm"
)

const sampleModule = `// Producer: Cuda compilation tools, release 13.4, V13.4.59
entry @bf16_splat() {
  %cst_bf16 = constant <bf16: 5.500000e+00> : tile<bf16> loc(#loc)
  return loc(#loc)
} loc(#loc)
#loc = loc("kernel.cu":5:2)
`

const sampleMatrixModule = `// Producer: Cuda compilation tools, release 13.4
entry @matrix(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>) {
  %cst_16_i32 = constant <i32: 16> : tile<i32> loc(#loc2)
  %0 = make_token : token loc(#loc3)
  %assume = assume div_by<16>, %arg0 : tile<ptr<f32>> loc(#loc4)
  %for:5 = for %loopIdx in (%cst_0_i32 to %cst_2_i32, step %cst_1_i32) : tile<i32> iter_values(%iterArg0 = %bcast) -> (tile<16x16xf32>, token) {
    %31 = join_tokens %iterArg2, %iterArg1 : token loc(#loc5)
    %sum = cuda_tile.mmaf_scaled %arg0, %arg1 : tile<16x16xf32> loc(#loc6)
  }
  return loc(#loc7)
}
#loc2 = loc("tile_matrix.cu":10:37)
#loc3 = loc("tile_matrix.cu":11:2)
#loc4 = loc("tile_matrix.cu":12:15)
#loc5 = loc("tile_matrix.cu":14:5)
#loc6 = loc("tile_matrix.cu":15:10)
#loc7 = loc("tile_matrix.cu":18:1)
`

func TestParseModule(t *testing.T) {
	mod, err := tileirdisasm.ParseString(sampleModule)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	if !strings.Contains(mod.Producer, "13.4") {
		t.Errorf("Producer = %q, want version 13.4", mod.Producer)
	}

	fn := mod.FindFunction("bf16_splat")
	if fn == nil {
		t.Fatal("expected to find function @bf16_splat")
	}

	if !fn.Entry {
		t.Error("expected fn.Entry to be true")
	}

	if len(fn.Operations) != 2 {
		t.Fatalf("len(fn.Operations) = %d, want 2", len(fn.Operations))
	}

	op0 := fn.Operations[0]
	if len(op0.Results) != 1 || op0.Results[0] != "%cst_bf16" {
		t.Errorf("op0.Results = %v, want ['%%cst_bf16']", op0.Results)
	}
	if op0.OpCode != "constant" {
		t.Errorf("op0.OpCode = %q, want 'constant'", op0.OpCode)
	}
	if op0.Type != "tile<bf16>" {
		t.Errorf("op0.Type = %q, want 'tile<bf16>'", op0.Type)
	}
	if op0.Loc.File != "kernel.cu" || op0.Loc.Line != 5 || op0.Loc.Column != 2 {
		t.Errorf("op0.Loc = %v, want kernel.cu:5:2", op0.Loc)
	}

	op1 := fn.Operations[1]
	if op1.OpCode != "return" {
		t.Errorf("op1.OpCode = %q, want 'return'", op1.OpCode)
	}
}

func TestParseMatrixModule(t *testing.T) {
	mod, err := tileirdisasm.ParseString(sampleMatrixModule)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	fn := mod.FindFunction("matrix")
	if fn == nil {
		t.Fatal("expected function @matrix")
	}

	if len(fn.Params) != 3 {
		t.Fatalf("len(fn.Params) = %d, want 3", len(fn.Params))
	}
	if fn.Params[0].Name != "%arg0" || fn.Params[0].Type != "tile<ptr<f32>>" {
		t.Errorf("param[0] = %+v, want %%arg0 : tile<ptr<f32>>", fn.Params[0])
	}

	var mmaOp *tileirdisasm.Operation
	for _, op := range fn.Operations {
		if op.OpCode == "cuda_tile.mmaf_scaled" {
			mmaOp = op
			break
		}
	}
	if mmaOp == nil {
		t.Fatal("missing cuda_tile.mmaf_scaled operation")
	}

	if len(mmaOp.Results) != 1 || mmaOp.Results[0] != "%sum" {
		t.Errorf("mmaOp.Results = %v, want ['%%sum']", mmaOp.Results)
	}
	if mmaOp.Loc.File != "tile_matrix.cu" || mmaOp.Loc.Line != 15 {
		t.Errorf("mmaOp.Loc = %v, want tile_matrix.cu:15", mmaOp.Loc)
	}

	// Verify OperationAtLine
	targetOp := fn.OperationAtLine(mmaOp.Line)
	if targetOp != mmaOp {
		t.Errorf("OperationAtLine(%d) = %v, want %v", mmaOp.Line, targetOp, mmaOp)
	}
}

func TestExtractFromELFInvalid(t *testing.T) {
	_, err := tileirdisasm.ExtractFromELF([]byte("not an elf"))
	if err == nil {
		t.Error("expected error for invalid ELF bytes")
	}
}

func TestDisassembleEmpty(t *testing.T) {
	ctx := context.Background()
	_, err := tileirdisasm.Disassemble(ctx, nil)
	if err == nil {
		t.Error("expected error on nil bytecode")
	}
}

func TestTileDisasmBytecodeExtraction(t *testing.T) {
	// Build a valid synthetic ELF64 containing a .nv.tileir section
	expectedPayload := []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00, 13, 4, 0, 0, 0x00}
	var buf bytes.Buffer

	// ELF Header (64-bit little endian)
	// e_ident
	ident := [16]byte{0x7F, 'E', 'L', 'F', 2, 1, 1, 0}
	buf.Write(ident[:])
	binary.Write(&buf, binary.LittleEndian, uint16(1))  // e_type = ET_REL
	binary.Write(&buf, binary.LittleEndian, uint16(62)) // e_machine = EM_X86_64
	binary.Write(&buf, binary.LittleEndian, uint32(1))  // e_version
	binary.Write(&buf, binary.LittleEndian, uint64(0))  // e_entry
	binary.Write(&buf, binary.LittleEndian, uint64(0))  // e_phoff
	// e_shoff will be updated: 64 header + payload + shstrtab
	shoffOffset := buf.Len()
	binary.Write(&buf, binary.LittleEndian, uint64(0))  // placeholder e_shoff
	binary.Write(&buf, binary.LittleEndian, uint32(0))  // e_flags
	binary.Write(&buf, binary.LittleEndian, uint16(64)) // e_ehsize
	binary.Write(&buf, binary.LittleEndian, uint16(0))  // e_phentsize
	binary.Write(&buf, binary.LittleEndian, uint16(0))  // e_phnum
	binary.Write(&buf, binary.LittleEndian, uint16(64)) // e_shentsize
	binary.Write(&buf, binary.LittleEndian, uint16(3))  // e_shnum: 0(null), 1(.shstrtab), 2(.nv.tileir)
	binary.Write(&buf, binary.LittleEndian, uint16(1))  // e_shstrndx = 1

	// Section 2 data: .nv.tileir payload
	payloadOffset := uint64(buf.Len())
	buf.Write(expectedPayload)

	// Section 1 data: .shstrtab
	// Strings: "\x00.shstrtab\x00.nv.tileir\x00"
	shstrtabData := []byte("\x00.shstrtab\x00.nv.tileir\x00")
	shstrtabOffset := uint64(buf.Len())
	buf.Write(shstrtabData)

	// Align to 8 bytes for section headers
	for buf.Len()%8 != 0 {
		buf.WriteByte(0)
	}
	shoff := uint64(buf.Len())

	// Write section headers
	// 0: Null section
	buf.Write(make([]byte, 64))

	// 1: .shstrtab (offset 1 in shstrtabData)
	binary.Write(&buf, binary.LittleEndian, uint32(1))                 // sh_name (".shstrtab")
	binary.Write(&buf, binary.LittleEndian, uint32(3))                 // sh_type (SHT_STRTAB)
	binary.Write(&buf, binary.LittleEndian, uint64(0))                 // sh_flags
	binary.Write(&buf, binary.LittleEndian, uint64(0))                 // sh_addr
	binary.Write(&buf, binary.LittleEndian, shstrtabOffset)            // sh_offset
	binary.Write(&buf, binary.LittleEndian, uint64(len(shstrtabData))) // sh_size
	binary.Write(&buf, binary.LittleEndian, uint32(0))                 // sh_link
	binary.Write(&buf, binary.LittleEndian, uint32(0))                 // sh_info
	binary.Write(&buf, binary.LittleEndian, uint64(1))                 // sh_addralign
	binary.Write(&buf, binary.LittleEndian, uint64(0))                 // sh_entsize

	// 2: .nv.tileir (offset 11 in shstrtabData)
	binary.Write(&buf, binary.LittleEndian, uint32(11))                   // sh_name (".nv.tileir")
	binary.Write(&buf, binary.LittleEndian, uint32(1))                    // sh_type (SHT_PROGBITS)
	binary.Write(&buf, binary.LittleEndian, uint64(0))                    // sh_flags
	binary.Write(&buf, binary.LittleEndian, uint64(0))                    // sh_addr
	binary.Write(&buf, binary.LittleEndian, payloadOffset)                // sh_offset
	binary.Write(&buf, binary.LittleEndian, uint64(len(expectedPayload))) // sh_size
	binary.Write(&buf, binary.LittleEndian, uint32(0))                    // sh_link
	binary.Write(&buf, binary.LittleEndian, uint32(0))                    // sh_info
	binary.Write(&buf, binary.LittleEndian, uint64(1))                    // sh_addralign
	binary.Write(&buf, binary.LittleEndian, uint64(0))                    // sh_entsize

	// Backpatch e_shoff
	elfBytes := buf.Bytes()
	binary.LittleEndian.PutUint64(elfBytes[shoffOffset:], shoff)

	// Stage 1: ExtractFromELF extracts raw bytecode []byte
	extracted, err := tileirdisasm.ExtractFromELF(elfBytes)
	if err != nil {
		t.Fatalf("ExtractFromELF failed: %v", err)
	}
	if !bytes.Equal(extracted, expectedPayload) {
		t.Fatalf("ExtractFromELF got %x, want %x", extracted, expectedPayload)
	}

	// Stage 2: Verification that extracted bytecode is verified independently and separate from CUBIN
	if len(extracted) < 8 || string(extracted[1:7]) != "TileIR" {
		t.Errorf("extracted bytecode magic invalid: %x", extracted)
	}
}
