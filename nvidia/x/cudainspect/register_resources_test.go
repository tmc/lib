package cudainspect_test

import (
	"encoding/binary"
	"encoding/hex"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudainspect"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

// registerReport submits S1's synthetic image to the public inspector with
// explicitly supplied receipts. No compiler or disassembler is invoked.
func registerReport(t *testing.T) *cudainspect.CompiledResources {
	t.Helper()
	a, in := compiledTargetFixture(t, 90, "sm_90a")
	image := append([]byte(nil), a.Raw...)
	binary.LittleEndian.PutUint32(image[len(image)-4:], 64)
	in.CompilerOptions = append(in.CompilerOptions, "--maxrregcount=64")
	cap, modeled := uint64(64), uint64(12)
	in.RegisterCap, in.ModelRegisters = &cap, &modeled
	a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, image, a.EntryPoint, a.Schema, a.Target,
		cudakernel.WithToolchain(in.CompilerVersion), cudakernel.WithCompilerOptions(in.CompilerOptions))
	if err != nil {
		t.Fatal(err)
	}
	in.DiagnosticImage, in.SASSImage = a.Digest, a.Digest
	i, err := cudainspect.NewInspector(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r, err := i.InspectCompiledResources(a, in)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestRegisterResourcesPublication(t *testing.T) {
	r := registerReport(t)
	resources, err := r.RegisterResources()
	if err != nil {
		t.Fatal(err)
	}
	if resources.Target != "sm_90a" || resources.DecodedTarget != "sm_90" || resources.ImageDigest != r.ImageDigest || resources.Entry != r.Entry {
		t.Fatal("selected identities changed")
	}
	if resources.Allocated != (cudareg.Fact{Count: 64, Status: cudareg.FactAvailable}) || resources.Cap.Count != 64 || resources.Cap.Status != cudareg.FactAvailable || resources.Modeled.Count != 12 || resources.Modeled.Status != cudareg.FactAvailable {
		t.Fatalf("resource facts: %+v", resources)
	}
	if resources.PeakLive.Status != cudareg.FactUnavailable || resources.PeakLive.Reason != r.PeakLive.Reason || resources.PeakLive.Reason == "" || r.InstructionsKnown {
		t.Fatal("unavailable suffixed analysis fabricated")
	}
	s, _, _ := cudareg.GenerateFixture()
	c, err := cudareg.PublishCandidate(&resources, s)
	if err != nil || c.Status != cudareg.ProtocolFeasible {
		t.Fatalf("publication %+v %v", c, err)
	}
	if c.TargetEvidence.Target != "sm_90a" || !strings.Contains(c.TargetReason, "does not establish") || c.Schedule.Groups["consumer"].Cap != 96 || c.Schedule.Groups["consumer"].Dynamic != 64 {
		t.Fatal("capability or ownership boundary changed")
	}
	r.DecodedTarget = ""
	resources, err = r.RegisterResources()
	if err != nil {
		t.Fatal(err)
	}
	c, err = cudareg.PublishCandidate(&resources, s)
	if err != nil || c.Status != cudareg.PublicationUnverified {
		t.Fatal("empty decoded target qualified publication")
	}
	if driver.Loaded() || nvrtc.Loaded() {
		t.Fatal("resource bridge loaded a CUDA library")
	}
}

func TestRegisterResourcesRefusals(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*cudainspect.CompiledResources)
	}{
		{"schema", func(r *cudainspect.CompiledResources) { r.SchemaVersion = "future" }},
		{"source_digest", func(r *cudainspect.CompiledResources) { r.SourceDigest = "bad" }},
		{"image_digest", func(r *cudainspect.CompiledResources) { r.ImageDigest = "" }},
		{"entry", func(r *cudainspect.CompiledResources) { r.Entry = "" }},
		{"compiler", func(r *cudainspect.CompiledResources) { r.CompilerVersion = " " }},
		{"disassembler", func(r *cudainspect.CompiledResources) { r.DisassemblerVersion = "" }},
		{"compiler_budget", func(r *cudainspect.CompiledResources) { r.CompilerVersion = strings.Repeat("v", 4097) }},
		{"disassembler_budget", func(r *cudainspect.CompiledResources) { r.DisassemblerVersion = strings.Repeat("v", 4097) }},
		{"option_count", func(r *cudainspect.CompiledResources) { r.CompilerOptions = make([]string, 129) }},
		{"option_budget", func(r *cudainspect.CompiledResources) { r.CompilerOptions = []string{strings.Repeat("v", 4097)} }},
		{"zero_allocated", func(r *cudainspect.CompiledResources) { *r.Allocated.Value = 0 }},
		{"zero_cap", func(r *cudainspect.CompiledResources) { *r.RegisterCap.Value = 0 }},
		{"decoded_suffix", func(r *cudainspect.CompiledResources) { r.DecodedTarget = r.Target }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := registerReport(t)
			tt.edit(r)
			resources, err := r.RegisterResources()
			if err == nil || !reflect.DeepEqual(resources, cudareg.Resources{}) {
				t.Fatalf("nontransactional conversion %+v %v", resources, err)
			}
		})
	}
	var absent *cudainspect.CompiledResources
	if _, err := absent.RegisterResources(); err == nil {
		t.Fatal("nil report accepted")
	}
}

func TestRegisterResourcesFacts(t *testing.T) {
	for _, name := range []string{"allocated", "peak", "cap", "modeled"} {
		for _, state := range []string{"overflow", "nil", "category", "caveat", "maximum"} {
			t.Run(name+"/"+state, func(t *testing.T) {
				r := registerReport(t)
				peak := uint64(7)
				r.PeakLive = cudainspect.ResourceFact{Value: &peak, Category: "derived"}
				facts := map[string]*cudainspect.ResourceFact{"allocated": &r.Allocated, "peak": &r.PeakLive, "cap": &r.RegisterCap, "modeled": &r.ModelRegisters}
				fact := facts[name]
				switch state {
				case "overflow":
					*fact.Value = uint64(1) << 32
				case "nil":
					fact.Value = nil
					fact.Reason = "not recorded"
				case "category":
					fact.Category = "invented"
				case "caveat":
					fact.Reason = "unmatched receipt"
				case "maximum":
					*fact.Value = uint64(^uint32(0))
				}
				resources, err := r.RegisterResources()
				if state == "overflow" {
					if err == nil {
						t.Fatal("overflow was truncated")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				converted := map[string]cudareg.Fact{"allocated": resources.Allocated, "peak": resources.PeakLive, "cap": resources.Cap, "modeled": resources.Modeled}[name]
				want := cudareg.FactAvailable
				if state == "nil" {
					want = cudareg.FactUnavailable
				}
				if state == "category" || state == "caveat" {
					want = cudareg.FactUnverified
				}
				if converted.Status != want || (want != cudareg.FactAvailable && converted.Reason == "") || (state == "nil" && converted.Count != 0) {
					t.Fatalf("fact %+v want %s", converted, want)
				}
			})
		}
	}
	r := registerReport(t)
	zero := uint64(0)
	r.PeakLive = cudainspect.ResourceFact{Value: &zero, Category: "derived"}
	resources, err := r.RegisterResources()
	if err != nil || resources.PeakLive.Status != cudareg.FactAvailable || resources.PeakLive.Count != 0 {
		t.Fatal("known zero peak lost", err)
	}
}

func TestRegisterResourcesRecordedRefusal(t *testing.T) {
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
	i, _ := cudainspect.NewInspector(t.TempDir())
	r, err := i.InspectCompiledResources(a, cudainspect.CompiledResourceInput{Source: []byte("retained image, supplied host receipts"), CompilerVersion: "supplied", DisassemblerVersion: "supplied", DiagnosticImage: a.Digest, SASSImage: a.Digest, Diagnostics: "Compiling entry function 'compute' for '" + info.Arch + "'\n0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n", SASS: ".target " + info.Arch + "\nFunction : compute\n/*0000*/ EXIT;\n"})
	if err != nil {
		t.Fatal(err)
	}
	resources, err := r.RegisterResources()
	if err != nil {
		t.Fatal(err)
	}
	if resources.Allocated.Count != 8 || resources.ImageDigest != a.Digest {
		t.Fatal("real retained metadata not read")
	}
	s, _, _ := cudareg.GenerateFixture()
	if _, err := cudareg.PublishCandidate(&resources, s); err == nil {
		t.Fatal("recorded target/allocation mismatch published")
	}
	resources.Target, resources.DecodedTarget = s.Target.Name, "sm_90"
	if _, err := cudareg.PublishCandidate(&resources, s); err == nil {
		t.Fatal("recorded allocation mismatch published")
	}
}

func TestRegisterResourcesDetachedConcurrent(t *testing.T) {
	r := registerReport(t)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Go(func() {
			resources, err := r.RegisterResources()
			if err != nil {
				t.Error(err)
				return
			}
			resources.Allocated.Count = 1
		})
	}
	wg.Wait()
	resources, err := r.RegisterResources()
	if err != nil {
		t.Fatal(err)
	}
	*r.Allocated.Value = 24
	r.CompilerOptions[0] = "changed"
	if resources.Allocated.Count != 64 {
		t.Fatal("conversion aliases input")
	}
}
