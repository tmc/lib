package tileir

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

// Synthetic metadata-only ELF. It contains one .text name and a zero byte,
// not assembled GPU instructions. It must never be loaded or executed.
func syntheticMemoryELF(t *testing.T, entry string, sm uint32) []byte {
	t.Helper()
	var ident [16]byte
	copy(ident[:], "\x7fELF")
	ident[elf.EI_CLASS] = byte(elf.ELFCLASS64)
	ident[elf.EI_DATA] = byte(elf.ELFDATA2LSB)
	ident[elf.EI_VERSION] = 1
	ident[elf.EI_OSABI] = 0x33
	ident[elf.EI_ABIVERSION] = 7
	names := []byte("\x00.shstrtab\x00.text." + entry + "\x00")
	header := elf.Header64{Ident: ident, Type: uint16(elf.ET_EXEC), Machine: uint16(elf.EM_CUDA), Version: 1, Flags: sm, Ehsize: 64, Shoff: 64, Shentsize: 64, Shnum: 3, Shstrndx: 1}
	sections := []elf.Section64{{}, {Name: 1, Type: uint32(elf.SHT_STRTAB), Off: 256, Size: uint64(len(names)), Addralign: 1},
		{Name: 11, Type: uint32(elf.SHT_PROGBITS), Flags: uint64(elf.SHF_ALLOC | elf.SHF_EXECINSTR), Off: 256 + uint64(len(names)), Size: 1, Addralign: 1}}
	// .shstrtab has nine characters; the next name starts at byte11.
	var b bytes.Buffer
	for _, value := range []any{header, sections} {
		if err := binary.Write(&b, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	b.Write(names)
	b.WriteByte(0)
	return b.Bytes()
}

func TestAssembleSemanticMemoryComposition(t *testing.T) {
	r := memoryLowering(t)
	want, err := r.Module.EncodeBytecode()
	if err != nil {
		t.Fatal(err)
	}
	result := &tileiras.Result{CUBIN: syntheticMemoryELF(t, "singleton", 90), Log: "synthetic metadata fixture; no compiler executed",
		Remarks: []tileiras.Remark{{Kind: "Analysis", Pass: "HostFixture", Name: "Synthetic", Function: "singleton", Args: map[string]string{"fixture": "true"}}}}
	calls := 0
	got, err := assembleSemanticMemory(context.Background(), r, "sm_90", func(ctx context.Context, data []byte, target string) (*tileiras.Result, error) {
		calls++
		if target != "sm_90" || !bytes.Equal(data, want) || !tileiras.IsBytecode(data) {
			t.Fatal("assembler did not receive exact typed bytecode/target")
		}
		return result, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || got.Artifact.Kind != cudakernel.KindCUBIN || got.Artifact.Target != "sm_90" || got.Artifact.EntryPoint != "singleton" || got.Artifact.Schema.TotalParamBytes != 24 || got.BytecodeDigest != memoryDigest(want) {
		t.Fatal("artifact composition lost contract")
	}
	if got.Artifact.Toolchain != "tileiras (version unobserved)" || got.Log != result.Log || got.Remarks[0].Args["fixture"] != "true" {
		t.Fatal("diagnostic/provenance lost")
	}
	result.CUBIN[0] = 0
	result.Remarks[0].Args["fixture"] = "changed"
	r.Schema.Params[0].Name = "changed"
	if got.Artifact.Raw[0] != 0x7f || got.Remarks[0].Args["fixture"] != "true" || got.Artifact.Schema.Params[0].Name != "x" {
		t.Fatal("assembler output or schema retained mutable aliases")
	}
}

func TestAssembleSemanticMemoryRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*tileiras.Result)
	}{
		{"empty", func(r *tileiras.Result) { r.CUBIN = nil }},
		{"fatbin", func(r *tileiras.Result) { r.Fatbin = []byte{1} }},
		{"bad_elf", func(r *tileiras.Result) { r.CUBIN = []byte("not ELF") }},
		{"different_target", func(r *tileiras.Result) { r.CUBIN = syntheticMemoryELF(t, "singleton", 100) }},
		{"unknown_target", func(r *tileiras.Result) { r.CUBIN = syntheticMemoryELF(t, "singleton", 0) }},
		{"different_entry", func(r *tileiras.Result) { r.CUBIN = syntheticMemoryELF(t, "other", 90) }},
		{"diagnostic_kind", func(r *tileiras.Result) { r.Remarks = []tileiras.Remark{{Kind: "bogus", Pass: "Memory", Name: "bad"}} }},
		{"diagnostic_function", func(r *tileiras.Result) {
			r.Remarks = []tileiras.Remark{{Kind: "Analysis", Pass: "Memory", Name: "bad", Function: "other"}}
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result := &tileiras.Result{CUBIN: syntheticMemoryELF(t, "singleton", 90)}
			tt.mutate(result)
			got, err := assembleSemanticMemory(context.Background(), memoryLowering(t), "sm_90", func(context.Context, []byte, string) (*tileiras.Result, error) { return result, nil })
			if got != nil || err == nil {
				t.Fatal("invalid assembler result accepted")
			}
		})
	}
	called := false
	fake := func(context.Context, []byte, string) (*tileiras.Result, error) {
		called = true
		return nil, tileiras.ErrNotFound
	}
	if got, err := assembleSemanticMemory(context.Background(), memoryLowering(t), "sm_90a", fake); got != nil || !errors.Is(err, tileiras.ErrArchSuffix) || called {
		t.Fatal("suffix was normalized or invoked assembler")
	}
	if got, err := assembleSemanticMemory(context.Background(), memoryLowering(t), "sm_90", fake); got != nil || !errors.Is(err, tileiras.ErrNotFound) || !called {
		t.Fatal("missing-tool error lost")
	}
	// Public suffix preflight is intrinsically zero-child and does not look
	// up vendor tools. The missing-tool path above is a private fake control.
	if got, err := AssembleSemanticMemory(context.Background(), nil, "sm_90a"); got != nil || !errors.Is(err, tileiras.ErrArchSuffix) {
		t.Fatal("public suffix preflight changed")
	}
	called = false
	r := memoryLowering(t)
	r.Grid[0] = 2
	if got, err := assembleSemanticMemory(context.Background(), r, "sm_90", fake); got != nil || err == nil || called {
		t.Fatal("binding failure invoked assembler")
	}
}
