package cudasass_test

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestCheckRegisterFile(t *testing.T) {
	for _, tc := range []struct {
		name   string
		groups []cudasass.WarpGroupRegs
		ok     bool
	}{
		// 32·4·40 + 32·8·232 = 5120 + 59392 = 64512.
		{"producer 40, consumers 232", []cudasass.WarpGroupRegs{{4, 40}, {8, 232}}, true},
		// 5120 + 32·8·240 = 66560.
		{"consumers 240", []cudasass.WarpGroupRegs{{4, 40}, {8, 240}}, false},
		// 41 rounds to 48: 32·4·48 + 59392 = 65536, exactly full.
		{"granularity rounds up to full", []cudasass.WarpGroupRegs{{4, 41}, {8, 232}}, true},
		// Unrounded, 32·4·49 + 32·8·229 = 64896 fits; rounded to 56 and
		// 232 it is 7168 + 59392 = 66560.
		{"granularity rounds over", []cudasass.WarpGroupRegs{{4, 49}, {8, 229}}, false},
		{"single group of 255", []cudasass.WarpGroupRegs{{8, 255}}, true},
		{"zero warps", []cudasass.WarpGroupRegs{{0, 32}}, false},
		{"256 registers", []cudasass.WarpGroupRegs{{4, 256}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := cudasass.CheckRegisterFile(tc.groups)
			if (err == nil) != tc.ok {
				t.Fatalf("CheckRegisterFile(%v) = %v, want ok=%v", tc.groups, err, tc.ok)
			}
		})
	}
}

// listing returns an sm_121a disassembly listing of one kernel from
// "OPCODE operands | control" lines in the format of [bundles], with
// encodings carrying the control codes.
func listing(t *testing.T, kernel string, lines ...string) string {
	t.Helper()
	var b strings.Builder
	fmt.Fprintf(&b, "\t.target\tsm_121a\n\n")
	fmt.Fprintf(&b, "\t.section\t.text.%s,\"ax\",@progbits\n%s:\n.text.%s:\n", kernel, kernel, kernel)
	for i, bd := range bundles(t, lines...) {
		e := bd.Inst.Encoding
		fmt.Fprintf(&b, "        /*%04x*/                   %s %s ;   /* %#016x */\n", 16*i, bd.Inst.Opcode, bd.Inst.Operands, e.Lo)
		fmt.Fprintf(&b, "                                                  /* %#016x */\n", e.Hi)
	}
	return b.String()
}

func parseAuditListing(t *testing.T, s string) *cudasass.Disassembly {
	t.Helper()
	d, err := cudasass.ParseString(s)
	if err != nil {
		t.Fatalf("parse listing: %v\n%s", err, s)
	}
	return d
}

func TestProveDisassemblyFixture(t *testing.T) {
	f, err := os.Open("testdata/roundtrip/sm121a-roundtrip.sass")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	d, err := cudasass.Parse(f)
	if err != nil {
		t.Fatal(err)
	}
	reps, err := cudasass.ProveDisassembly(d, nil, cudasass.CubinConstraint{MaxGPRs: 232, ForbidSpills: true, MaxRedundantWaits: 0})
	if err != nil {
		t.Fatalf("ProveDisassembly = %v, want nil; reports %+v", err, reps)
	}
	for _, r := range reps {
		if !r.GPRsFromSASS || r.StackFrame != -1 {
			t.Errorf("%s: without a ptxas log, want SASS-derived registers and unknown stack: %+v", r.Kernel, r)
		}
	}
	// intmix names R0 through R7.
	if reps[0].Kernel != "intmix" || reps[0].UsedGPRs != 8 {
		t.Errorf("intmix UsedGPRs = %d (%s), want 8", reps[0].UsedGPRs, reps[0].Kernel)
	}
}

func TestProveDisassembly(t *testing.T) {
	spillLog, err := os.ReadFile("../ptxasinfo/testdata/spill.txt")
	if err != nil {
		t.Fatal(err)
	}
	rep, err := ptxasinfo.Parse(strings.NewReader(string(spillLog)))
	if err != nil {
		t.Fatal(err)
	}
	strict := cudasass.CubinConstraint{MaxGPRs: 232, ForbidSpills: true, MaxRedundantWaits: 0}
	for _, tc := range []struct {
		name   string
		kernel string
		lines  []string
		rep    *ptxasinfo.Report
		c      cudasass.CubinConstraint
		want   string // substring of a violation, or "" for none
	}{
		{"clean", "k", []string{"FFMA R4, R1, R2, R3 | s=4", "EXIT | s=5"}, nil, strict, ""},
		{"one STL", "k", []string{"STL [R1], R4 | s=1", "EXIT | s=5"}, nil, strict, "1 STL and 0 LDL"},
		{"one LDL", "k", []string{"LDL R4, [R1] | s=1 wr=0", "EXIT | s=5"}, nil, strict, "0 STL and 1 LDL"},
		{"STL allowed", "k", []string{"STL [R1], R4 | s=1", "EXIT | s=5"}, nil, cudasass.CubinConstraint{MaxRedundantWaits: -1}, ""},
		{"too many registers from SASS", "k", []string{"FFMA R240, R1, R2, R3 | s=4"}, nil, strict, "uses 241 registers"},
		// ptxas reports spilling though this listing shows no STL: a stale
		// listing must not pass.
		{"ptxas spill bytes", "_Z5spillPfPKf", []string{"FFMA R4, R1, R2, R3 | s=4"}, rep, strict, "2232 bytes spill stores"},
		{"ptxas stack frame", "_Z5scaleILi128EEvPfPKfi", []string{"FFMA R4, R1, R2, R3 | s=4"}, rep, strict, "128 bytes stack frame"},
		{"tensor required, none", "k", []string{"FFMA R4, R1, R2, R3 | s=4"}, nil, cudasass.CubinConstraint{RequireTensorMMA: true, MaxRedundantWaits: -1}, "no tensor-core MMA"},
		{"tensor required, HMMA", "k", []string{"HMMA.16816.F32.BF16 R4, R8, R12, R4 | s=2"}, nil, cudasass.CubinConstraint{RequireTensorMMA: true, MaxRedundantWaits: -1}, ""},
		{"repeated DEPBAR", "k", []string{"DEPBAR.LE SB0, 0x0 | s=1", "FFMA R4, R1, R2, R3 | s=4", "DEPBAR.LE SB0, 0x0 | s=1"}, nil, strict, "1 redundant waits"},
		{"DEPBAR after a load", "k", []string{"DEPBAR.LE SB0, 0x0 | s=1", "LDS R4, [R1] | s=1 wr=0", "DEPBAR.LE SB0, 0x0 | s=1"}, nil, strict, ""},
		// The second wait on barrier 0 follows the first with no setter
		// between: it cannot delay anything.
		{"wait on a cleared scoreboard", "k", []string{
			"LDS R4, [R1] | s=1 wr=0",
			"FFMA R5, R4, R2, R3 | s=4 wait=0x1",
			"FFMA R6, R4, R2, R3 | s=4 wait=0x1",
		}, nil, strict, "1 redundant waits"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text := listing(t, tc.kernel, tc.lines...)
			if tc.rep != nil {
				// The ptxas log is for sm_90a; a listing that declares
				// sm_121a would not match it.
				text = strings.Replace(text, "\t.target\tsm_121a\n", "", 1)
			}
			d := parseAuditListing(t, text)
			reps, err := cudasass.ProveDisassembly(d, tc.rep, tc.c)
			if len(reps) != 1 {
				t.Fatalf("got %d reports, want 1", len(reps))
			}
			v := reps[0].Violations
			if tc.want == "" {
				if err != nil || len(v) != 0 {
					t.Fatalf("ProveDisassembly = %v, violations %q; want none", err, v)
				}
				return
			}
			if !errors.Is(err, cudasass.ErrConstraint) {
				t.Fatalf("ProveDisassembly error = %v, want ErrConstraint", err)
			}
			if !strings.Contains(strings.Join(v, "\n"), tc.want) {
				t.Fatalf("violations = %q, want one containing %q", v, tc.want)
			}
		})
	}
}

// TestProveCubinNative compiles the egraph/hopper golden kernels with nvcc
// and checks each with ProveCubin: no spills, no stack frame and at most 232
// registers. It runs only with CUDA_CUDASASS_SPILL_NATIVE=1 and needs nvcc
// and nvdisasm on PATH, but no GPU. CUDA_CUDASASS_SPILL_NATIVE_ARCHS lists
// the targets, comma-separated; the default is sm_90a,sm_100a,sm_121a.
func TestProveCubinNative(t *testing.T) {
	if os.Getenv("CUDA_CUDASASS_SPILL_NATIVE") != "1" {
		t.Skip("CUDA_CUDASASS_SPILL_NATIVE=1 not set")
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Fatal("nvcc not on PATH")
	}
	archs := "sm_90a,sm_100a,sm_121a"
	if s := os.Getenv("CUDA_CUDASASS_SPILL_NATIVE_ARCHS"); s != "" {
		archs = s
	}
	files, err := filepath.Glob("../egraph/hopper/testdata/golden/*.cu")
	if err != nil || len(files) == 0 {
		t.Fatalf("no golden kernels: %v", err)
	}
	for _, arch := range strings.Split(archs, ",") {
		for _, f := range files {
			t.Run(arch+"/"+filepath.Base(f), func(t *testing.T) {
				cubin := filepath.Join(t.TempDir(), "k.cubin")
				out, err := exec.Command(nvcc, "-std=c++20", "-O3", "-arch="+arch, "-cubin", "-Xptxas", "-v", "-o", cubin, f).CombinedOutput()
				if err != nil {
					t.Fatalf("nvcc: %v\n%s", err, out)
				}
				b, err := os.ReadFile(cubin)
				if err != nil {
					t.Fatal(err)
				}
				reps, err := cudasass.ProveCubin(b, string(out), cudasass.CubinConstraint{MaxGPRs: 232, ForbidSpills: true, MaxRedundantWaits: -1})
				for _, r := range reps {
					t.Logf("%s %s: %d GPRs, STL %d, LDL %d, MMA %d, redundant waits %d", r.Arch, r.Kernel, r.UsedGPRs, r.STL, r.LDL, r.TensorMMA, r.RedundantWaits)
				}
				if err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}
