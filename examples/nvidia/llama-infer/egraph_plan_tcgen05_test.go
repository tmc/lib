package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// TestBuildEgraphPlansTCGen05 builds Llama-3-8B's plans for the datacenter
// Blackwell targets and checks that every projection, decode and prefill,
// is a tcgen05 UMMAGEMM.
func TestBuildEgraphPlansTCGen05(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_103a"} {
		t.Run(target, func(t *testing.T) {
			p, err := buildEgraphPlans(&llama3_8B, false, egraphOptions{Attn: true, PrefillRows: -1, Profile: targetProfile(t, target)})
			if err != nil {
				t.Fatal(err)
			}
			tests := []struct {
				plan         *hopper.Plan
				decode, fill int // UMMAGEMMs in the decode and prefill forms
			}{
				{p.attnIn, 1, 0},
				{p.post, 3, 0},
				{p.head, 1, 0},
				{p.pf.attnIn, 0, 1},
				{p.pf.post, 0, 3},
			}
			for _, tt := range tests {
				var decode, fill int
				for _, k := range tt.plan.Kernels {
					switch {
					case k.UMMA != nil && k.UMMA.MaxM > 0:
						decode++
					case k.UMMA != nil:
						fill++
					case k.GEMM != nil, k.Tile != nil:
						t.Errorf("%s: gemm not lowered to tcgen05", tt.plan.Name)
					}
				}
				if decode != tt.decode || fill != tt.fill {
					t.Errorf("%s: %d decode and %d prefill UMMAGEMMs, want %d and %d", tt.plan.Name, decode, fill, tt.decode, tt.fill)
				}
				if _, err := tt.plan.Source(); err != nil {
					t.Errorf("%s: %v", tt.plan.Name, err)
				}
			}
		})
	}
}

// TestEgraphPlansTCGen05Compiled compiles the sm_100a and sm_103a plans of
// TestBuildEgraphPlansTCGen05 with nvcc (no GPU needed) and checks that
// every UMMAGEMM kernel compiles without spills to tcgen05 MMAs, TMA loads,
// commits and TMEM loads. Without nvcc it skips.
func TestEgraphPlansTCGen05Compiled(t *testing.T) {
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Skip("nvcc not on PATH: COMPILED evidence not produced")
	}
	for _, target := range []string{"sm_100a", "sm_103a"} {
		p, err := buildEgraphPlans(&llama3_8B, false, egraphOptions{Attn: true, PrefillRows: -1, Profile: targetProfile(t, target)})
		if err != nil {
			t.Fatal(err)
		}
		for _, pl := range []*hopper.Plan{p.attnIn, p.post, p.head, p.pf.attnIn, p.pf.post} {
			t.Run(target+"/"+pl.Name, func(t *testing.T) {
				src, err := pl.Source()
				if err != nil {
					t.Fatal(err)
				}
				dir := t.TempDir()
				cu, cubin := filepath.Join(dir, "plan.cu"), filepath.Join(dir, "plan.cubin")
				if err := os.WriteFile(cu, []byte(src), 0o644); err != nil {
					t.Fatal(err)
				}
				var stderr bytes.Buffer
				cmd := exec.CommandContext(t.Context(), nvcc, "-arch="+target, "-cubin", "-Xptxas", "-v", "-o", cubin, cu)
				cmd.Stderr = &stderr
				if err := cmd.Run(); err != nil {
					t.Fatalf("nvcc: %v\n%s", err, stderr.Bytes())
				}
				rep, err := ptxasinfo.Parse(&stderr)
				if err != nil {
					t.Fatal(err)
				}
				regs := make(map[string]int)
				for _, f := range rep.Functions() {
					if f.Spills() {
						t.Errorf("%s spills %d/%d bytes", f.Name, f.SpillStores, f.SpillLoads)
					}
					regs[f.Name] = f.Registers
				}
				bin, err := os.ReadFile(cubin)
				if err != nil {
					t.Fatal(err)
				}
				d, err := cudasass.DisassembleCUBIN(bin)
				if err != nil {
					t.Fatal(err)
				}
				for _, k := range pl.Kernels {
					if k.UMMA == nil {
						continue
					}
					var fn *cudasass.Function
					for _, f := range d.Functions {
						if strings.Contains(f.Name, k.UMMA.Name+"i14CUtensorMap") {
							fn = f
						}
					}
					if fn == nil {
						t.Errorf("no SASS for %s", k.UMMA.Name)
						continue
					}
					s := cudasass.SummarizeSM100Function(*fn)
					t.Logf("COMPILED %s %s: %d registers; %d MMAs (%d block-scaled), %d TMA loads, %d scale copies, %d commits, %d TMEM loads",
						target, k.UMMA.Name, regs[fn.Name], s.UTMMA, s.BlockScaled, s.UTMALDG, s.UTMACP, s.UTMACMDFLUSH, s.LDTM)
					if s.UTMMA == 0 || s.UTMALDG == 0 || s.UTMACMDFLUSH == 0 || s.LDTM == 0 {
						t.Errorf("%s: SASS lacks tcgen05 MMA, TMA, commit or TMEM load: %+v", k.UMMA.Name, s)
					}
					if f := cudasass.AuditSM100Function(*fn); len(f) != 0 {
						t.Errorf("%s: SASS audit: %v", k.UMMA.Name, f)
					}
				}
			})
		}
	}
}

func targetProfile(t *testing.T, target string) cudadevicetable.Profile {
	t.Helper()
	p, err := cudadevicetable.LookupProfile(target)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
