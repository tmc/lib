package hopper

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

// TestUMMAGEMMCompiled compiles UMMAGEMM and NVFP4Quant with nvcc, which
// needs no GPU, for every target [UMMATarget] accepts, and checks ptxas's report (no
// spills, at most 255 registers) and the SASS: each GEMM must issue TMA
// loads (UTMALDG), tcgen05 MMAs (UTCHMMA for BF16, block-scaled UTCOMMA for
// NVFP4), commits (UTCBAR) and TMEM loads (LDTM), and NVFP4 also scale
// copies (UTCCP). The shapes are Llama-3-8B projections. Without nvcc the
// test skips; nothing here runs a kernel.
//
// With UMMA_SASS_DIR set, each kernel's SASS is written there.
func TestUMMAGEMMCompiled(t *testing.T) {
	nvcc, err := cudatool.LookTool("nvcc", "")
	if err != nil {
		t.Skipf("nvcc not found (%v): COMPILED evidence not produced", err)
	}
	gemms := []UMMAGEMM{
		{Name: "qkv_decode_m1", N: 6144, K: 4096, Weight: WeightBF16, Out: BF16, MaxM: 1},
		{Name: "down_decode_m16", N: 4096, K: 14336, Weight: WeightBF16, Out: F32, MaxM: 16,
			Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{{Name: "res", DType: F32, Broadcast: Full}}},
		{Name: "gateup_prefill", N: 28672, K: 4096, Weight: WeightBF16, Out: BF16},
		{Name: "qkv_decode_nvfp4", N: 6144, K: 4096, Weight: WeightNVFP4, Out: BF16, MaxM: 4},
		{Name: "down_prefill_nvfp4", N: 4096, K: 14336, Weight: WeightNVFP4, Out: F32,
			Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{{Name: "res", DType: F32, Broadcast: Full}}},
	}
	for _, target := range cudadevicetable.Targets() {
		if !UMMATarget(target) {
			continue
		}
		for _, g := range gemms {
			t.Run(target+"/"+g.Name, func(t *testing.T) {
				src, err := g.Source()
				if err != nil {
					t.Fatal(err)
				}
				if g.Weight == WeightNVFP4 {
					q := NVFP4Quant{Name: g.Name + "_quant", K: g.K}
					qs, err := q.CUDA()
					if err != nil {
						t.Fatal(err)
					}
					ql, _ := q.Launcher()
					src += qs + ql
				}
				fn, sass := compileSM100(t, nvcc, target, g.Name, src)
				s := cudasass.SummarizeSM100Function(*sass)
				t.Logf("COMPILED %s %s: %d registers, %d B static + %d B dynamic shared, %d stages; SASS %s",
					target, g.Name, fn.Registers, fn.SharedMem, g.norm1(t).SharedBytes(), g.norm1(t).Stages, opcodeSummary(sass))
				want := map[cudasass.SM100Kind]int{
					cudasass.SM100UTMALDG:   2,
					cudasass.SM100UTMMA:     1,
					cudasass.SM100UTMAFlush: 1,
					cudasass.SM100LDTM:      1,
				}
				if g.Weight == WeightNVFP4 {
					want[cudasass.SM100UTMALDG] = 4
					want[cudasass.SM100UTMACP] = 1
				}
				for kind, n := range want {
					if got := s.Count(kind); got < n {
						t.Errorf("%s: %d %s instructions, want at least %d", g.Name, got, kind, n)
					}
				}
				if g.Weight == WeightNVFP4 && s.BlockScaled == 0 {
					t.Errorf("%s: no block-scaled MMA in SASS", g.Name)
				}
				if g.Weight == WeightBF16 && s.BlockScaled != 0 {
					t.Errorf("%s: BF16 GEMM has %d block-scaled MMAs", g.Name, s.BlockScaled)
				}
			})
		}
	}
}

func (g UMMAGEMM) norm1(t *testing.T) UMMAGEMM {
	t.Helper()
	n, err := g.norm()
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// compileSM100 compiles src for target and returns ptxas's report on the
// kernel named name and its SASS.
func compileSM100(t *testing.T, nvcc, target, name, src string) (*ptxasinfo.Function, *cudasass.Function) {
	t.Helper()
	dir := t.TempDir()
	cu := filepath.Join(dir, "k.cu")
	cubin := filepath.Join(dir, "k.cubin")
	if err := os.WriteFile(cu, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(t.Context(), nvcc, "-arch="+target, "-cubin", "-Xptxas", "-v", "-o", cubin, cu)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("nvcc -arch=%s: %v\n%s", target, err, stderr.Bytes())
	}
	rep, err := ptxasinfo.Parse(&stderr)
	if err != nil {
		t.Fatal(err)
	}
	var fn *ptxasinfo.Function
	for _, f := range rep.Functions() {
		if f.Entry && strings.Contains(f.Name, name) && !strings.Contains(f.Name, name+"_") {
			fn = f
		}
	}
	if fn == nil {
		t.Fatalf("ptxas report has no entry %s:\n%s", name, stderr.Bytes())
	}
	if fn.Spills() {
		t.Errorf("%s %s: spills %d bytes stored, %d loaded", target, name, fn.SpillStores, fn.SpillLoads)
	}
	if fn.Registers > 255 {
		t.Errorf("%s %s: %d registers", target, name, fn.Registers)
	}
	bin, err := os.ReadFile(cubin)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.DisassembleCUBIN(bin)
	if err != nil {
		t.Fatalf("disassemble: %v", err)
	}
	var sass *cudasass.Function
	for _, f := range d.Functions {
		if f.Name == fn.Name {
			sass = f
		}
	}
	if sass == nil {
		t.Fatalf("SASS has no function %s", fn.Name)
	}
	if dir := os.Getenv("UMMA_SASS_DIR"); dir != "" {
		var b strings.Builder
		for _, in := range sass.Instructions {
			b.WriteString(in.Opcode + " " + in.Operands + "\n")
		}
		if err := os.WriteFile(filepath.Join(dir, target+"_"+name+".sass"), []byte(b.String()), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return fn, sass
}

// opcodeSummary lists the counts of the tensor-memory, TMA and barrier
// opcodes in f, by base mnemonic.
func opcodeSummary(f *cudasass.Function) string {
	counts := make(map[string]int)
	var order []string
	for _, in := range f.Instructions {
		op := in.Opcode
		base, _, _ := strings.Cut(op, ".")
		switch {
		case strings.HasPrefix(base, "UTC"), strings.HasPrefix(base, "UTMA"), base == "LDTM", base == "STTM", base == "SYNCS":
		default:
			continue
		}
		if counts[op] == 0 {
			order = append(order, op)
		}
		counts[op]++
	}
	var parts []string
	for _, op := range order {
		parts = append(parts, fmt.Sprintf("%s×%d", op, counts[op]))
	}
	return strings.Join(parts, " ")
}
