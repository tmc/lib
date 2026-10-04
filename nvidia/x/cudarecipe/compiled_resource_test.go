package cudarecipe

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func compiledControl(t *testing.T) (*CompiledResources, LoadedResources) {
	t.Helper()
	r := &CompiledResources{SchemaVersion: "compiled-resources/v1", SourceDigest: strings.Repeat("a", 64), ImageDigest: strings.Repeat("b", 64), Entry: "compute", Target: "sm_90",
		Allocated: resourceFact(24, "decoded"), PeakLive: resourceFact(8, "derived"), RegisterCap: resourceFact(64, "compiler condition"), ModelRegisters: resourceFact(12, "modeled"),
		StackBytes: resourceFact(0, "compiler reported"), SpillLoadBytes: resourceFact(0, "compiler reported"), SpillStoreBytes: resourceFact(0, "compiler reported"), InstructionsKnown: true}
	l := LoadedResources{ImageDigest: r.ImageDigest, Entry: r.Entry, Target: r.Target, Registers: resourceFact(24, "runtime attribute"), LocalBytes: resourceFact(0, "runtime attribute")}
	return r, l
}

func claimFor(r *CompiledResources) RegisterResidencyClaim {
	return RegisterResidencyClaim{r.SourceDigest, r.ImageDigest, r.Entry, r.Target}
}

func TestCompiledResources_SeparatesAllocatedPeakLiveAndCap(t *testing.T) {
	r, l := compiledControl(t)
	if err := r.CheckRegisterResidency(claimFor(r), l); err != nil {
		t.Fatal(err)
	}
	if *r.Allocated.Value != 24 || *r.PeakLive.Value != 8 || *r.RegisterCap.Value != 64 || *r.ModelRegisters.Value != 12 {
		t.Fatal("resource categories collapsed")
	}
	r.Allocated = r.ModelRegisters
	if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
		t.Fatal("model substituted for allocation")
	}
}

func TestCompiledResources_MissingAttributeIsUnmeasured(t *testing.T) {
	for _, name := range []string{"allocated", "local", "spill", "instructions", "reported spill"} {
		t.Run(name, func(t *testing.T) {
			r, l := compiledControl(t)
			switch name {
			case "allocated":
				r.Allocated = ResourceFact{Category: "decoded", Reason: "absent"}
			case "local":
				l.LocalBytes = ResourceFact{Reason: "driver attribute unavailable"}
			case "spill":
				r.SpillLoadBytes = ResourceFact{Reason: "summary absent"}
			case "instructions":
				r.InstructionsKnown = false
			case "reported spill":
				r.SpillLoadBytes = resourceFact(8, "compiler reported")
			}
			if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
				t.Fatal("absence accepted as zero")
			}
		})
	}
}

func TestCompiledResources_LocalArrayAloneIsNotSpillProof(t *testing.T) {
	r, l := compiledControl(t)
	r.LocalLoads = []uint64{16}
	r.LocalStores = []uint64{32}
	if r.CorroboratedSpills() {
		t.Fatal("local array declared a spill")
	}
	if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
		t.Fatal("local use accepted as residency")
	}
	r.SpillStoreBytes = resourceFact(4, "compiler reported")
	if !r.CorroboratedSpills() {
		t.Fatal("matched spill rejected")
	}
	if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
		t.Fatal("compiler spill ignored")
	}
}

func TestCompiledResources_RejectsStaleImageAndWrongEntry(t *testing.T) {
	for _, name := range []string{"source", "image", "entry", "target", "loaded image", "loaded entry", "loaded target"} {
		t.Run(name, func(t *testing.T) {
			r, l := compiledControl(t)
			c := claimFor(r)
			switch name {
			case "source":
				c.SourceDigest = "stale"
			case "image":
				c.ImageDigest = "stale"
			case "entry":
				c.Entry = "other"
			case "target":
				c.Target = "sm_121"
			case "loaded image":
				l.ImageDigest = "stale"
			case "loaded entry":
				l.Entry = "other"
			case "loaded target":
				l.Target = "sm_121"
			}
			if err := r.CheckRegisterResidency(c, l); err == nil {
				t.Fatal("wrong identity accepted")
			}
		})
	}
}

func TestCompiledResources_MatchedCompilerAndSASS(t *testing.T) {
	hexImage, err := os.ReadFile("../cudaoccupancy/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	image, err := hex.DecodeString(strings.Join(strings.Fields(string(hexImage)), ""))
	if err != nil {
		t.Fatal(err)
	}
	info, err := cudakernel.ParseCUBIN(image)
	if err != nil {
		t.Fatal(err)
	}
	schema, _ := cudakernel.NewSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "compute", schema, info.Arch)
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	for _, tc := range []struct {
		name, entry string
		stores      uint64
		known       bool
	}{{"exact", "compute", 4, true}, {"wrong kernel", "other", 4, false}, {"zero known", "compute", 0, true}} {
		t.Run(tc.name, func(t *testing.T) {
			in := CompiledResourceInput{Source: []byte("synthetic host control"), CompilerVersion: "ptxas test fixture", DisassemblerVersion: "nvdisasm test fixture", DiagnosticImage: a.Digest, SASSImage: a.Digest,
				Diagnostics: "ptxas info : Compiling entry function '" + tc.entry + "' for '" + a.Target + "'\nptxas info : 0 bytes stack frame, " + strconv.FormatUint(tc.stores, 10) + " bytes spill stores, 0 bytes spill loads\n",
				SASS:        ".target " + a.Target + "\nFunction : compute\n/*0000*/ STL [R1], R2;\n/*0010*/ EXIT;\n"}
			modeled := uint64(12)
			in.ModelRegisters = &modeled
			in.LifeRanges = "Function : compute\n/*0000*/ STL [R1], R2; // | 2 | | | |\n/*0010*/ EXIT; // | 1 | | | |\n"
			r, err := i.InspectCompiledResources(a, in)
			if err != nil {
				t.Fatal(err)
			}
			if r.Allocated.Value == nil || *r.Allocated.Value != 8 {
				t.Fatalf("actual allocation lost: %+v", r.Allocated)
			}
			if r.PeakLive.Value != nil || r.PeakLive.Reason == "" || r.PeakLive.Category != "derived" {
				t.Fatal("partial listing established full-kernel pressure")
			}
			wrongLife := in
			wrongLife.LifeRanges = strings.ReplaceAll(in.LifeRanges, "compute", "other")
			if _, err := i.InspectCompiledResources(a, wrongLife); err == nil {
				t.Fatal("wrong life range entry accepted through fallback")
			}
			if (r.SpillStoreBytes.Value != nil) != tc.known {
				t.Fatal("wrong diagnostic association")
			}
			if r.CorroboratedSpills() != (tc.known && tc.stores > 0) {
				t.Fatalf("spill corroboration wrong: %+v", r)
			}
			for _, target := range []string{"", "sm_80"} {
				wrongTarget := in
				wrongTarget.SASS = "Function : compute\n/*0000*/ STL [R1], R2;\n/*0010*/ EXIT;\n"
				if target != "" {
					wrongTarget.SASS = ".target " + target + "\n" + wrongTarget.SASS
				}
				observed, err := i.InspectCompiledResources(a, wrongTarget)
				if err != nil {
					t.Fatal(err)
				}
				if observed.InstructionsKnown || len(observed.LocalStores) > 0 || observed.CorroboratedSpills() || observed.InstructionsReason == "" {
					t.Fatal("unmatched SASS target attributed to image")
				}
			}
			in.DiagnosticImage = "wrong"
			if _, err := i.InspectCompiledResources(a, in); err == nil {
				t.Fatal("wrong compiler image accepted")
			}
		})
	}
}

func TestCompiledResources_AbsentOwnerMetadata(t *testing.T) {
	attribute := []byte{4, cudaoccupancy.AttrMaxRegCount, 4, 0, 64, 0, 0, 0}
	image := registerELF(attribute)
	image[8] = 7
	binary.LittleEndian.PutUint32(image[48:52], 90)
	schema, _ := cudakernel.NewSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "kernel", schema, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	in := CompiledResourceInput{Source: []byte("synthetic register-ceiling-only ELF"), CompilerVersion: "host control", DisassemblerVersion: "host control", DiagnosticImage: a.Digest, SASSImage: a.Digest, SASS: "Function : kernel\n/*0000*/ EXIT;\n"}
	r, err := i.InspectCompiledResources(a, in)
	if err != nil {
		t.Fatal(err)
	}
	if r.Allocated.Value != nil || r.Allocated.Reason == "" || r.SpillLoadBytes.Value != nil || r.LocalMemory.Value != nil {
		t.Fatalf("absent facts became values: %+v", r)
	}
	for _, diag := range []string{
		"ptxas info : Compiling entry function 'kernel' for 'sm_90'\nptxas info : Function properties for other\n0 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n",
		"ptxas info : Compiling entry function 'kernel' for 'sm_80'\n0 bytes stack frame, 4 bytes spill stores, 4 bytes spill loads\n",
	} {
		in.Diagnostics = diag
		r, err := i.InspectCompiledResources(a, in)
		if err != nil {
			t.Fatal(err)
		}
		if r.SpillLoadBytes.Value != nil {
			t.Fatal("unmatched properties or target accepted")
		}
	}
	cap := uint64(32)
	in.RegisterCap = &cap
	if _, err := i.InspectCompiledResources(a, in); err == nil {
		t.Fatal("cap absent from actual command accepted")
	}
	in.CompilerOptions = []string{"--maxrregcount=32"}
	if _, err := i.InspectCompiledResources(a, in); err != nil {
		t.Fatal(err)
	}
	in.CompilerOptions = []string{"--maxrregcount=64"}
	if _, err := i.InspectCompiledResources(a, in); err == nil {
		t.Fatal("different actual cap accepted")
	}
}

func FuzzCompiledResourceDiagnostics(f *testing.F) {
	raw, err := os.ReadFile("../cudaoccupancy/testdata/kernels.cubin.hex")
	if err != nil {
		f.Fatal(err)
	}
	image, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		f.Fatal(err)
	}
	info, err := cudakernel.ParseCUBIN(image)
	if err != nil {
		f.Fatal(err)
	}
	schema, _ := cudakernel.NewSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "compute", schema, info.Arch)
	if err != nil {
		f.Fatal(err)
	}
	f.Add("ptxas info : Compiling entry function 'compute' for '" + a.Target + "'\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
	f.Add("ptxas info : Compiling entry function 'other' for '" + a.Target + "'\n0 bytes stack frame, 18446744073709551616 bytes spill stores, 0 bytes spill loads\n")
	f.Fuzz(func(t *testing.T, diag string) {
		if len(diag) > 4096 {
			return
		}
		i, _ := NewInspector(t.TempDir())
		r, err := i.InspectCompiledResources(a, CompiledResourceInput{Source: []byte("fuzz host control"), CompilerVersion: "host control", DisassemblerVersion: "host control", DiagnosticImage: a.Digest, SASSImage: a.Digest, Diagnostics: diag, SASS: "Function : compute\n/*0000*/ EXIT;\n"})
		if err == nil && (r.Allocated.Value == nil || *r.Allocated.Value != 8 || r.ImageDigest != a.Digest) {
			t.Fatal("diagnostic text altered artifact allocation or identity")
		}
	})
}

func TestCompiledResources_InstructionCoverage(t *testing.T) {
	ops := []cudasass.Instruction{
		{Address: 0, Opcode: "MOV", Encoding: cudasass.Encoding{Lo: 1, Hi: 2}},
		{Address: 16, Opcode: "NOP", Encoding: cudasass.Encoding{Lo: 3, Hi: 4}},
		{Address: 32, Opcode: "EXIT", Encoding: cudasass.Encoding{Lo: 5, Hi: 6}},
	}
	var image []byte
	for _, op := range ops {
		b := op.Encoding.Bytes()
		image = append(image, b[:]...)
	}
	sections := []SectionInfo{{Name: ".text.kernel", Size: 48}}
	if reason := compiledInstructionCoverage(sections, image, "kernel", &cudasass.Function{Instructions: ops}); reason != "" {
		t.Fatal(reason)
	}
	for _, tc := range []struct {
		name string
		ops  []cudasass.Instruction
	}{
		{"missing beginning", ops[1:]}, {"interior gap", []cudasass.Instruction{ops[0], ops[2]}}, {"truncated tail", ops[:2]},
		{"sparse terminated", []cudasass.Instruction{ops[0], {Address: 16, Opcode: ".word"}, ops[2]}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if compiledInstructionCoverage(sections, image, "kernel", &cudasass.Function{Instructions: tc.ops}) == "" {
				t.Fatal("partial listing accepted")
			}
		})
	}
	wrong := append([]cudasass.Instruction(nil), ops...)
	wrong[1].Encoding.Lo++
	if compiledInstructionCoverage(sections, image, "kernel", &cudasass.Function{Instructions: wrong}) == "" {
		t.Fatal("changed encoding accepted")
	}
}

func TestCompiledResources_PartialSpillCategory(t *testing.T) {
	r, _ := compiledControl(t)
	r.InstructionsKnown = false
	r.LocalStores = []uint64{16}
	r.SpillStoreBytes = resourceFact(4, "compiler reported")
	if !r.CorroboratedSpills() {
		t.Fatal("positive partial observation lost")
	}
	r.SpillStoreBytes.Category = "modeled"
	if r.CorroboratedSpills() {
		t.Fatal("modeled spills accepted")
	}
}

func TestCompiledResources_PartialDisassemblyRefusesResidency(t *testing.T) {
	raw, err := os.ReadFile("../cudaoccupancy/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	image, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		t.Fatal(err)
	}
	info, err := cudakernel.ParseCUBIN(image)
	if err != nil {
		t.Fatal(err)
	}
	schema, _ := cudakernel.NewSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "compute", schema, info.Arch)
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	in := CompiledResourceInput{Source: []byte("source receipt"), CompilerVersion: "fixture", DisassemblerVersion: "fixture", DiagnosticImage: a.Digest, SASSImage: a.Digest,
		Diagnostics: "Compiling entry function 'compute' for '" + a.Target + "'\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n",
		SASS:        ".target " + a.Target + "\nFunction : compute\n/*0000*/ MOV R0, 0x1;\n", LifeRanges: "Function : compute\n/*0000*/ MOV R0, 0x1; // | 2 | | | |\n"}
	r, err := i.InspectCompiledResources(a, in)
	if err != nil {
		t.Fatal(err)
	}
	if r.PeakLive.Value != nil || r.PeakLive.Reason == "" {
		t.Fatal("partial peak admitted")
	}
	l := LoadedResources{ImageDigest: a.Digest, Entry: a.EntryPoint, Target: a.Target, Registers: r.Allocated, LocalBytes: resourceFact(0, "runtime attribute")}
	l.Registers.Category = "runtime attribute"
	if err := r.CheckRegisterResidency(claimFor(r), l); err == nil {
		t.Fatal("partial disassembly admitted as complete no-local residency evidence")
	}
}

func TestCompiledResources_PeakRequiresCompletePressure(t *testing.T) {
	fn := &cudasass.Function{Name: "kernel", Instructions: []cudasass.Instruction{{Address: 0, Opcode: "MOV"}, {Address: 16, Opcode: "NOP"}, {Address: 32, Opcode: "EXIT"}}}
	p := map[uint64]cudasass.RegisterPressure{0: {GPR: 2}, 16: {GPR: 5}, 32: {GPR: 1}}
	f := compiledPeakLive(true, fn, p)
	if f.Value == nil || *f.Value != 5 || f.Category != "derived" {
		t.Fatalf("complete peak lost: %+v", f)
	}
	if f = compiledPeakLive(false, fn, p); f.Value != nil || f.Reason == "" {
		t.Fatal("partial artifact admitted")
	}
	delete(p, 16)
	if f = compiledPeakLive(true, fn, p); f.Value != nil || f.Reason == "" {
		t.Fatal("missing interior NOP pressure admitted")
	}
	p[48] = cudasass.RegisterPressure{GPR: 9}
	if f = compiledPeakLive(true, fn, p); f.Value != nil || f.Reason == "" {
		t.Fatal("wrong pressure addresses admitted")
	}
}

// The listing is a synthetic producer receipt constructed from retained section
// bytes. This tests admission coverage, not actual disassembler semantics.
func TestCompiledResources_CompletePeakProducerControl(t *testing.T) {
	raw, err := os.ReadFile("../cudaoccupancy/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	image, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		t.Fatal(err)
	}
	info, err := cudakernel.ParseCUBIN(image)
	if err != nil {
		t.Fatal(err)
	}
	schema, _ := cudakernel.NewSchema()
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, "compute", schema, info.Arch)
	if err != nil {
		t.Fatal(err)
	}
	i, _ := NewInspector(t.TempDir())
	base, err := i.Inspect(image, a.EntryPoint)
	if err != nil {
		t.Fatal(err)
	}
	var sass, life strings.Builder
	fmt.Fprintf(&sass, ".target %s\nFunction : compute\n", a.Target)
	life.WriteString("Function : compute\n")
	var count int
	for _, section := range base.SectionDetails {
		if section.Name != ".text.compute" {
			continue
		}
		for pc := uint64(0); pc < section.Size; pc += 16 {
			b := image[section.Offset+pc : section.Offset+pc+16]
			lo := binary.LittleEndian.Uint64(b[:8])
			hi := binary.LittleEndian.Uint64(b[8:])
			fmt.Fprintf(&sass, "/*%04x*/ MOV R0, R1; /* 0x%016x */\n/* 0x%016x */\n", pc, lo, hi)
			fmt.Fprintf(&life, "/*%04x*/ MOV R0, R1; // | 2 | | | |\n", pc)
			count++
		}
	}
	if count == 0 {
		t.Fatal("missing fixture text")
	}
	in := CompiledResourceInput{Source: []byte("synthetic full producer receipt"), CompilerVersion: "fixture", DisassemblerVersion: "fixture", DiagnosticImage: a.Digest, SASSImage: a.Digest, SASS: sass.String(), LifeRanges: life.String()}
	r, err := i.InspectCompiledResources(a, in)
	if err != nil {
		t.Fatal(err)
	}
	if !r.InstructionsKnown || r.PeakLive.Value == nil || *r.PeakLive.Value != 2 {
		t.Fatalf("complete producer receipt unavailable: instructions=%v reason=%s peak=%+v", r.InstructionsKnown, r.InstructionsReason, r.PeakLive)
	}
}
