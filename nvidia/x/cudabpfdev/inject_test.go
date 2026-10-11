package cudabpfdev

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

const injectionPTX = `.version 9.0
.target sm_121a
.address_size 64
.visible .entry affine(
    .param .u64 out
)
{
    .reg .b64 %addr;
    .reg .b32 %value;
    ld.param.u64 %addr, [out];
    mov.u32 %value, 7;
    st.global.u32 [%addr], %value;
    ret;
}
`

func injectionFixture(t *testing.T, text string) (*ArtifactSource, InjectionPolicy, Program, Budget) {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "u64", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(text), "affine", schema, "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewArtifactSource(a, Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	policy := InjectionPolicy{Enabled: true, Allowlist: []KernelIdentity{{a.Digest, a.EntryPoint, a.Target}}, Entry: true, Exit: true, MemoryInstructions: []int{2}}
	p := program(Instruction{Op: Const, Dst: 1, Imm: 1}, Instruction{Op: MapAdd, B: 1}, Instruction{Op: Exit})
	b := budget()
	b.MaxRegisters = 128
	return s, policy, p, b
}

func TestPrepareInjectionPreservesOriginal(t *testing.T) {
	s, policy, p, b := injectionFixture(t, injectionPTX)
	i, err := PrepareInjection(s, p, b, policy)
	if err != nil {
		t.Fatal(err)
	}
	a, reason := i.Candidate()
	if a == nil {
		t.Fatal(reason)
	}
	if a.Digest == s.launch.Artifact.Digest || !reflect.DeepEqual(a.Schema, s.launch.Artifact.Schema) {
		t.Fatal("candidate identity or ABI wrong")
	}
	mod, err := cudaptx.ParseBytes(a.Raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(mod.Kernels) != 1 || mod.Kernels[0].Name != "affine" {
		t.Fatal("kernel identity changed")
	}
	// Remove only inserted blocks/globals and require every original byte back.
	var restored strings.Builder
	inBlock := false
	for _, line := range strings.SplitAfter(string(a.Raw), "\n") {
		if strings.HasPrefix(line, ".visible .global ") {
			continue
		}
		if line == "{\n" && !strings.HasSuffix(restored.String(), ")\n") {
			inBlock = true
			continue
		}
		if inBlock {
			if line == "}\n" {
				inBlock = false
			}
			continue
		}
		restored.WriteString(line)
	}
	if restored.String() != injectionPTX {
		t.Fatalf("original code changed:\n%s", restored.String())
	}
	for _, name := range []string{"site0_", "site1_", "site2_"} {
		if !strings.Contains(string(a.Raw), "__cudabpfdev_"+name+"r0") {
			t.Fatalf("missing isolated register namespace %s", name)
		}
	}
	if strings.Count(string(a.Raw), "vote.sync.ballot.b32") != 3 || strings.Count(string(a.Raw), "activemask.b32") != 3 {
		t.Fatal("missing active-lane synchronized site membership")
	}
	original, err := i.Original()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(original.launch, s.launch) {
		t.Fatal("fallback source changed")
	}
	a.Raw[0] = '!'
	second, _ := i.Candidate()
	if second.Raw[0] == '!' {
		t.Fatal("candidate aliases internal image")
	}
	j, err := PrepareInjection(s, p, b, policy)
	if err != nil {
		t.Fatal(err)
	}
	other, why := j.Candidate()
	if other == nil || !bytes.Equal(second.Raw, other.Raw) {
		t.Fatalf("nondeterministic candidate: %s", why)
	}
}

func TestInjectionAdmissionFallback(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(string) string
	}{
		{"branch", func(s string) string { return strings.Replace(s, "mov.u32 %value, 7;", "bra end;\nend:", 1) }},
		{"predicate", func(s string) string { return strings.Replace(s, "st.global", "@%p st.global", 1) }},
		{"barrier", func(s string) string { return strings.Replace(s, "mov.u32 %value, 7;", "bar.sync 0;", 1) }},
		{"multiple instructions", func(s string) string {
			return strings.Replace(s, "mov.u32 %value, 7;", "mov.u32 %value, 7; mov.u32 %value, 8;", 1)
		}},
		{"block comment", func(s string) string { return "/* comment */\n" + s }},
		{"reserved name", func(s string) string { return "// __cudabpfdev_map\n" + s }},
		{"global", func(s string) string { return strings.Replace(s, ".visible", ".global .u32 app;\n.visible", 1) }},
		{"ABI", func(s string) string { return strings.Replace(s, ".param .u64", ".param .u32", 1) }},
		{"early return", func(s string) string { return strings.Replace(s, "mov.u32 %value, 7;", "ret;", 1) }},
		{"old version", func(s string) string { return strings.Replace(s, "9.0", "8.8", 1) }},
		{"clock", func(s string) string { return strings.Replace(s, "mov.u32 %value, 7;", "mov.u32 %value, %clock;", 1) }},
		{"scheduling identity", func(s string) string { return strings.Replace(s, "mov.u32 %value, 7;", "mov.u32 %value, %smid;", 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, policy, p, b := injectionFixture(t, tc.change(injectionPTX))
			i, err := PrepareInjection(s, p, b, policy)
			if err != nil {
				t.Fatal(err)
			}
			if a, reason := i.Candidate(); a != nil || reason == "" {
				t.Fatalf("unsafe candidate=%v reason=%q", a != nil, reason)
			}
			original, err := i.Original()
			if err != nil || !reflect.DeepEqual(original.launch, s.launch) {
				t.Fatal("fallback modified original")
			}
		})
	}
}

func TestInjectionPolicyAndBudget(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*InjectionPolicy, *Program, *Budget, *ArtifactSource)
	}{
		{"disabled", func(p *InjectionPolicy, _ *Program, _ *Budget, _ *ArtifactSource) { p.Enabled = false }},
		{"allowlist miss", func(p *InjectionPolicy, _ *Program, _ *Budget, _ *ArtifactSource) { p.Allowlist[0].Digest = "wrong" }},
		{"no sites", func(p *InjectionPolicy, _ *Program, _ *Budget, _ *ArtifactSource) {
			p.Entry = false
			p.Exit = false
			p.MemoryInstructions = nil
		}},
		{"bad site", func(p *InjectionPolicy, _ *Program, _ *Budget, _ *ArtifactSource) { p.MemoryInstructions = []int{999} }},
		{"not memory", func(p *InjectionPolicy, _ *Program, _ *Budget, _ *ArtifactSource) { p.MemoryInstructions = []int{1} }},
		{"budget", func(_ *InjectionPolicy, _ *Program, b *Budget, _ *ArtifactSource) { b.MaxRegisters = 64 }},
		{"bytecode", func(_ *InjectionPolicy, p *Program, _ *Budget, _ *ArtifactSource) { p.Version = 99 }},
		{"ABI offset", func(_ *InjectionPolicy, _ *Program, _ *Budget, s *ArtifactSource) {
			s.launch.Artifact.Schema.Params[0].Offset = 8
			s.launch.Artifact.Schema.TotalParamBytes = 16
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, policy, p, b := injectionFixture(t, injectionPTX)
			tc.change(&policy, &p, &b, s)
			i, err := PrepareInjection(s, p, b, policy)
			if err != nil {
				t.Fatal(err)
			}
			if a, reason := i.Candidate(); a != nil || reason == "" {
				t.Fatalf("accepted: %s", reason)
			}
		})
	}
}

// Compile-only acceptance; never loads or launches a CUDA kernel.
func TestInjectionAssembler(t *testing.T) {
	tool, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Skipf("assembler qualification unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	version, err := cudatool.Run(ctx, tool, "", nil, "--version")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tool=%s version=%s%s", version.Path, version.Stdout, version.Stderr)
	s, policy, p, b := injectionFixture(t, injectionPTX)
	i, err := PrepareInjection(s, p, b, policy)
	if err != nil {
		t.Fatal(err)
	}
	a, reason := i.Candidate()
	if a == nil {
		t.Fatal(reason)
	}
	dir := t.TempDir()
	src, dst := filepath.Join(dir, "injected.ptx"), filepath.Join(dir, "injected.cubin")
	if err := os.WriteFile(src, a.Raw, 0600); err != nil {
		t.Fatal(err)
	}
	r, err := cudatool.Run(ctx, tool, "", nil, "-arch="+a.Target, "-v", src, "-o", dst)
	t.Logf("args=%v stdout=%s stderr=%s", r.Args, r.Stdout, r.Stderr)
	if err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(dst); err != nil || len(data) == 0 {
		t.Fatalf("missing CUBIN: %v", err)
	}
}

func TestStubBodyAdmission(t *testing.T) {
	const definition = ".func __bpftime_cuda__kernel_trace\n{\n    ret;\n}"
	cases := []struct{ name, replacement string }{
		{"valid", definition},
		{"duplicate", definition + "\n" + definition},
		{"predicated return", ".func __bpftime_cuda__kernel_trace\n{\n    @%p ret;\n}"},
		{"hidden shared storage", ".func __bpftime_cuda__kernel_trace\n{\n    .shared .b32 hidden[1024];\n    ret;\n}"},
		{"label", ".func __bpftime_cuda__kernel_trace\n{\nlabel:\n    ret;\n}"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, policy, p, b := injectionFixture(t, strings.Replace(stubPTX, definition, tc.replacement, 1))
			policy.StubFunctions = []string{"__bpftime_cuda__kernel_trace"}
			policy.Entry, policy.Exit = false, false
			policy.MemoryInstructions = nil
			i, err := PrepareInjection(s, p, b, policy)
			if err != nil {
				t.Fatal(err)
			}
			a, reason := i.Candidate()
			if tc.name == "valid" {
				if a == nil {
					t.Fatalf("valid stub refused: %s", reason)
				}
			} else if a != nil || reason == "" {
				t.Fatalf("unsafe stub admitted: %q", reason)
			}
		})
	}
}

// This gate compiles both original and rewritten stubs without loading them.
func TestStubInjectionAssembler(t *testing.T) {
	tool, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Skipf("assembler qualification unavailable: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	version, err := cudatool.Run(ctx, tool, "", nil, "--version")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tool=%s version=%s%s", version.Path, version.Stdout, version.Stderr)
	for _, target := range []string{"sm_100a", "sm_121a"} {
		t.Run(target, func(t *testing.T) {
			s, p, b, policy, err := nativeStubInjectionFixture(target, 17)
			if err != nil {
				t.Fatal(err)
			}
			i, err := PrepareInjection(s, p, b, policy)
			if err != nil {
				t.Fatal(err)
			}
			candidate, reason := i.Candidate()
			if candidate == nil {
				t.Fatal(reason)
			}
			original, err := s.DescribeLaunch()
			if err != nil {
				t.Fatal(err)
			}
			for _, a := range []*cudakernel.Artifact{original.Artifact, candidate} {
				dir := t.TempDir()
				src, dst := filepath.Join(dir, "stub.ptx"), filepath.Join(dir, "stub.cubin")
				if err := os.WriteFile(src, a.Raw, 0600); err != nil {
					t.Fatal(err)
				}
				r, err := cudatool.Run(ctx, tool, "", nil, "-arch="+target, "-v", src, "-o", dst)
				t.Logf("digest=%s args=%v stdout=%s stderr=%s", a.Digest, r.Args, r.Stdout, r.Stderr)
				if err != nil {
					t.Fatal(err)
				}
				if data, err := os.ReadFile(dst); err != nil || len(data) == 0 {
					t.Fatalf("missing CUBIN: %v", err)
				}
			}
		})
	}
}
