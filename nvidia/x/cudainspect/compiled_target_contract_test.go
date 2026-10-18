package cudainspect_test

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudainspect"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

// This constructs bounded metadata and producer receipts, not a compiled kernel
// or authenticated NVIDIA output. Its numeric architecture and allocation are
// read by the existing production owners through the public inspector API.
func compiledTargetFixture(t *testing.T, numeric uint32, declared string) (*cudakernel.Artifact, cudainspect.CompiledResourceInput) {
	t.Helper()
	names := []byte("\x00.shstrtab\x00.text.kernel\x00.nv.info.kernel\x00.nv.shared.kernel\x00")
	const sections = 5
	stringOffset := 64 + sections*64
	textOffset := stringOffset + len(names)
	infoOffset := textOffset + 16
	image := make([]byte, infoOffset+8)
	copy(image, "\x7fELF")
	image[4], image[5], image[6] = 2, 1, 1
	image[elf.EI_OSABI], image[elf.EI_ABIVERSION] = 0x33, 7
	order := binary.LittleEndian
	order.PutUint16(image[16:], uint16(elf.ET_EXEC))
	order.PutUint16(image[18:], uint16(elf.EM_CUDA))
	order.PutUint32(image[20:], 1)
	order.PutUint64(image[40:], 64)
	order.PutUint32(image[48:], numeric)
	order.PutUint16(image[52:], 64)
	order.PutUint16(image[58:], 64)
	order.PutUint16(image[60:], sections)
	order.PutUint16(image[62:], 1)
	section := func(index int, name string, kind elf.SectionType, offset, size uint64) {
		header := image[64+index*64:]
		order.PutUint32(header, uint32(bytes.Index(names, []byte(name))))
		order.PutUint32(header[4:], uint32(kind))
		order.PutUint64(header[24:], offset)
		order.PutUint64(header[32:], size)
	}
	section(1, ".shstrtab", elf.SHT_STRTAB, uint64(stringOffset), uint64(len(names)))
	section(2, ".text.kernel", elf.SHT_PROGBITS, uint64(textOffset), 16)
	section(3, ".nv.info.kernel", elf.SHT_PROGBITS, uint64(infoOffset), 8)
	section(4, ".nv.shared.kernel", elf.SHT_NOBITS, 0, 0)
	copy(image[stringOffset:], names)
	order.PutUint64(image[textOffset:], 1)
	order.PutUint64(image[textOffset+8:], 2)
	copy(image[infoOffset:], []byte{4, byte(cudaoccupancy.AttrRegCount), 4, 0, 24, 0, 0, 0})
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	options := []string{"--gpu-name=" + declared}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "kernel", schema, declared,
		cudakernel.WithToolchain("constructed compiler receipt"), cudakernel.WithCompilerOptions(options))
	if err != nil {
		t.Fatal(err)
	}
	in := cudainspect.CompiledResourceInput{
		Source: []byte("constructed host target control"), CompilerVersion: artifact.Toolchain, CompilerOptions: options,
		DiagnosticImage: artifact.Digest, SASSImage: artifact.Digest, DisassemblerVersion: "constructed disassembler receipt",
		Diagnostics: fmt.Sprintf("Compiling entry function 'kernel' for '%s'\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n", declared),
		SASS:        fmt.Sprintf(".target sm_%d\nFunction : kernel\n/*0000*/ NOP; /* 0x0000000000000001 */\n/* 0x0000000000000002 */\n", numeric),
	}
	return artifact, in
}

func TestCompiledTargetNumericCompatibility(t *testing.T) {
	for _, test := range []struct {
		name, declared string
		numeric        uint32
		instructions   bool
	}{
		{"base90", "sm_90", 90, true},
		{"sm90a", "sm_90a", 90, false},
		{"base100", "sm_100", 100, true},
		{"sm100a", "sm_100a", 100, false},
		{"unknown_numeric", "sm_250a", 250, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			artifact, in := compiledTargetFixture(t, test.numeric, test.declared)
			inspector, err := cudainspect.NewInspector(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			r, err := inspector.InspectCompiledResources(artifact, in)
			if err != nil {
				t.Fatalf("numeric-compatible declared target refused: %v", err)
			}
			if r.Target != test.declared || r.DecodedTarget != fmt.Sprintf("sm_%d", test.numeric) {
				t.Fatalf("declared and decoded evidence collapsed: declared=%q decoded=%q", r.Target, r.DecodedTarget)
			}
			if r.Allocated.Value == nil || *r.Allocated.Value != 24 || r.Allocated.Category != "decoded" || r.StaticShared.Value == nil || *r.StaticShared.Value != 0 {
				t.Fatal("actual metadata allocation/explicit zero was lost")
			}
			if r.ImageDigest != artifact.Digest || r.Entry != artifact.EntryPoint || r.CompilerVersion != in.CompilerVersion || r.CompilerOptions[0] != in.CompilerOptions[0] {
				t.Fatal("declared compilation identity changed")
			}
			if r.StackBytes.Value == nil || *r.StackBytes.Value != 0 || r.StackBytes.Category != "compiler reported" {
				t.Fatal("exact declared-target diagnostic association lost")
			}
			if r.InstructionsKnown != test.instructions || r.PeakLive.Value != nil {
				t.Fatal("SASS or peak association was broadened by numeric compatibility")
			}
			if !test.instructions && (r.InstructionsReason == "" || r.PeakLive.Reason == "") {
				t.Fatal("unavailable suffixed SASS/peak evidence has no reason")
			}
		})
	}
}

func TestCompiledTargetContradictions(t *testing.T) {
	for _, name := range []string{"numeric", "minor", "malformed", "small_numeric", "image_receipt", "sass_image_receipt", "compiler_version", "compiler_suffix_option", "wrong_entry"} {
		t.Run(name, func(t *testing.T) {
			artifact, in := compiledTargetFixture(t, 90, "sm_90a")
			switch name {
			case "numeric":
				artifact.Target = "sm_100a"
			case "minor":
				artifact.Target = "sm_91a"
			case "malformed":
				artifact.Target = "sm_not-a-number"
			case "small_numeric":
				artifact.Target = "sm_9a"
			case "image_receipt":
				in.DiagnosticImage = strings.Repeat("a", 64)
			case "sass_image_receipt":
				in.SASSImage = strings.Repeat("a", 64)
			case "compiler_version":
				in.CompilerVersion = "different compiler"
			case "compiler_suffix_option":
				in.CompilerOptions = []string{"--gpu-name=sm_90"}
			case "wrong_entry":
				artifact.EntryPoint = "other"
			}
			inspector, _ := cudainspect.NewInspector(t.TempDir())
			if r, err := inspector.InspectCompiledResources(artifact, in); err == nil || r != nil {
				t.Fatal("inconsistent target/receipt/entry admitted")
			}
		})
	}
}

func TestCompiledTargetDiagnosticSuffixIsNotInterchangeable(t *testing.T) {
	for _, diagnosticTarget := range []string{"sm_90", "sm_90f"} {
		t.Run(diagnosticTarget, func(t *testing.T) {
			artifact, in := compiledTargetFixture(t, 90, "sm_90a")
			in.Diagnostics = strings.ReplaceAll(in.Diagnostics, "sm_90a", diagnosticTarget)
			in.LifeRanges = "Function : kernel\n/*0000*/ NOP; // | 2 | | | |\n"
			inspector, _ := cudainspect.NewInspector(t.TempDir())
			r, err := inspector.InspectCompiledResources(artifact, in)
			if err != nil {
				t.Fatal(err)
			}
			if r.Target != "sm_90a" || r.DecodedTarget != "sm_90" || r.StackBytes.Value != nil || r.SpillStoreBytes.Value != nil || r.SpillLoadBytes.Value != nil {
				t.Fatal("different diagnostic suffix was treated as exact compiler evidence")
			}
			if r.InstructionsKnown || r.PeakLive.Value != nil || r.PeakLive.Reason == "" {
				t.Fatal("numeric SASS target established suffixed pressure association")
			}
		})
	}
}
