package ptxasinfo

import (
	"fmt"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func parseFile(t *testing.T, name string) *Report {
	t.Helper()
	f, err := os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r, err := Parse(f)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestParseTwoArch(t *testing.T) {
	r := parseFile(t, "testdata/two-arch.txt")
	if len(r.Targets) != 2 || r.Targets[0].Arch != "sm_90a" || r.Targets[1].Arch != "sm_80" {
		t.Fatalf("targets = %v, want sm_90a then sm_80", r.Targets)
	}
	if got := r.Targets[1].ConstMem; !reflect.DeepEqual(got, map[int]int{3: 256}) {
		t.Errorf("sm_80 module cmem = %v, want map[3:256]", got)
	}
	tests := []struct {
		name, arch string
		want       Function
	}{
		{"_Z5scaleILi128EEvPfPKfi", "sm_90a", Function{
			Name: "_Z5scaleILi128EEvPfPKfi", Arch: "sm_90a", Entry: true,
			StackFrame: 128, Registers: 40, Barriers: 1, CumulativeStack: 128,
			SharedMem: 512, MaxRegisters: 64,
		}},
		{"_Z6helperfi", "sm_90a", Function{Name: "_Z6helperfi", Arch: "sm_90a"}},
		{"_Z5spillPfPKf", "sm_90a", Function{
			Name: "_Z5spillPfPKf", Arch: "sm_90a", Entry: true, Registers: 122,
		}},
		{"_Z5scaleILi128EEvPfPKfi", "sm_80", Function{
			Name: "_Z5scaleILi128EEvPfPKfi", Arch: "sm_80", Entry: true,
			StackFrame: 128, Registers: 32, Barriers: 1, CumulativeStack: 128,
			SharedMem: 512, ConstMem: map[int]int{0: 372}, MaxRegisters: 64,
		}},
		{"_Z5spillPfPKf", "sm_80", Function{
			Name: "_Z5spillPfPKf", Arch: "sm_80", Entry: true, Registers: 124,
			ConstMem: map[int]int{0: 368},
		}},
	}
	for _, tt := range tests {
		got := r.Lookup(tt.name, tt.arch)
		if got == nil {
			t.Errorf("Lookup(%s, %s) = nil", tt.name, tt.arch)
			continue
		}
		if !reflect.DeepEqual(*got, tt.want) {
			t.Errorf("Lookup(%s, %s) =\n%+v\nwant\n%+v", tt.name, tt.arch, *got, tt.want)
		}
	}
	if n := len(r.Functions()); n != 6 {
		t.Errorf("len(Functions()) = %d, want 6", n)
	}
}

func TestParseSpill(t *testing.T) {
	r := parseFile(t, "testdata/spill.txt")
	f := r.Lookup("_Z5spillPfPKf", "sm_90a")
	if f == nil {
		t.Fatal("spill kernel missing")
	}
	if !f.Spills() || f.SpillStores != 2232 || f.SpillLoads != 2660 || f.StackFrame != 992 || f.Registers != 32 {
		t.Errorf("spill = %+v, want 2232 B stores, 2660 B loads, 992 B frame, 32 registers", *f)
	}
	if g := r.Lookup("_Z5scaleILi128EEvPfPKfi", "sm_90a"); g == nil || g.Spills() {
		t.Errorf("scale = %+v, want no spills", g)
	}
}

// TestParseDebugTwoArch reads nvcc -G output, where each module reports a
// device function before its first Compiling line. The function belongs to
// the module that follows the gmem line, not to the previous architecture.
func TestParseDebugTwoArch(t *testing.T) {
	r := parseFile(t, "testdata/debug-two-arch.txt")
	if len(r.Targets) != 2 || r.Targets[0].Arch != "sm_90a" || r.Targets[1].Arch != "sm_100a" {
		t.Fatalf("targets = %v, want sm_90a then sm_100a", r.Targets)
	}
	for _, tt := range []struct {
		arch  string
		frame int
	}{{"sm_90a", 128}, {"sm_100a", 136}} {
		f := r.Lookup("_Z6helperfi", tt.arch)
		if f == nil || f.Entry || f.StackFrame != tt.frame {
			t.Errorf("%s helper = %+v, want device function with %d-byte frame", tt.arch, f, tt.frame)
		}
	}
	for _, tg := range r.Targets {
		if len(tg.Functions) != 3 || tg.Functions[0].Name != "_Z6helperfi" {
			t.Errorf("%s functions = %v, want helper first of 3", tg.Arch, tg.Functions)
		}
	}
}

func TestParseLines(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		check   func(*Report) bool
		wantErr string
	}{
		{
			name: "older toolkit usage line",
			in: "ptxas info    : Compiling entry function 'k' for 'sm_70'\n" +
				"ptxas info    : Function properties for k\n" +
				"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n" +
				"ptxas info    : Used 18 registers, 360 bytes cmem[0], 8 bytes lmem\n",
			check: func(r *Report) bool {
				f := r.Lookup("k", "sm_70")
				return f != nil && f.Registers == 18 && f.ConstMem[0] == 360 && f.LocalMem == 8
			},
		},
		{
			name: "other tools' lines and warnings",
			in: "k.cu(3): warning #177-D: variable \"x\" was declared but never referenced\n" +
				"ptxas warning : Registers are spilled to local memory in function 'k'\n" +
				"ptxas info    : (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized\n",
			check: func(r *Report) bool {
				return len(r.Targets) == 0 && len(r.Warnings) == 1 && len(r.Notes) == 1 &&
					strings.HasPrefix(r.Notes[0], "(C7515)")
			},
		},
		{
			name: "device function only: architecture unknown",
			in: "ptxas info    : 0 bytes gmem\n" +
				"ptxas info    : Function properties for d\n" +
				"    8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n",
			check: func(r *Report) bool {
				return len(r.Targets) == 1 && r.Targets[0].Arch == "" && r.Lookup("d", "") != nil
			},
		},
		{
			name: "fatal",
			in:   "ptxas fatal   : Unresolved extern function 'f'\n",
			check: func(r *Report) bool {
				return len(r.Errors) == 1 && r.Errors[0] == "Unresolved extern function 'f'"
			},
		},
		{
			name: "unknown usage field",
			in: "ptxas info    : Compiling entry function 'k' for 'sm_90a'\n" +
				"ptxas info    : Used 18 registers, 4 widgets\n",
			wantErr: `unknown resource "4 widgets"`,
		},
		{
			name: "missing frame line",
			in: "ptxas info    : Compiling entry function 'k' for 'sm_90a'\n" +
				"ptxas info    : Function properties for k\n" +
				"ptxas info    : Used 18 registers\n",
			wantErr: "want stack frame line",
		},
		{
			name:    "truncated after properties",
			in:      "ptxas info    : Compiling entry function 'k' for 'sm_90a'\nptxas info    : Function properties for k\n",
			wantErr: "input ends before stack frame line of k",
		},
		{
			name:    "usage before function",
			in:      "ptxas info    : Used 18 registers\n",
			wantErr: "resource usage before any function",
		},
		{
			name:    "overflowing count",
			in:      "ptxas info    : Compiling entry function 'k' for 'sm_90a'\nptxas info    : Used 9999999999999999999 registers\n",
			wantErr: "unknown resource",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := Parse(strings.NewReader(tt.in))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Parse() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !tt.check(r) {
				t.Errorf("Parse() = %+v", r)
			}
		})
	}
}

// TestParseWGMMASerialized checks that ptxas's wgmma serialization
// diagnostics reach the function they name. The C7519 and C7512 lines are
// copied from ptxas 13.4 output; C7513 follows their form.
func TestParseWGMMASerialized(t *testing.T) {
	const (
		c7512 = "(C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources for the function '%s'"
		c7513 = "(C7513) Potential Performance Loss: wgmma.mma_async instructions are serialized due to non wgmma instructions defining input registers of a wgmma between start and end of the pipeline stage in the function '%s'"
		c7519 = "(C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '%s' in line 1191"
		c7517 = "(C7517) wgmma.wait_group is injected in around line 1159 by compiler to allow use of registers defined by GMMA in function '%s'"
	)
	// module reports kernels k and clean for arch, after the lines ptxas
	// prints ahead of the module.
	module := func(arch string, before ...string) string {
		var b strings.Builder
		for _, l := range before {
			b.WriteString("ptxas info    : " + l + "\n")
		}
		for _, name := range []string{"k", "clean"} {
			fmt.Fprintf(&b, "ptxas info    : Compiling entry function '%s' for '%s'\n", name, arch)
			fmt.Fprintf(&b, "ptxas info    : Function properties for %s\n", name)
			b.WriteString("    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n")
			b.WriteString("ptxas info    : Used 74 registers, used 1 barriers, 16 bytes smem\n")
		}
		return b.String()
	}
	gmem := "0 bytes gmem"
	tests := []struct {
		name     string
		in       string
		want     map[string][]string // arch -> codes ptxas serialized k for
		wantDiag int                 // diagnostics on k, in total
		notes    int
	}{
		{"clean", module("sm_90a", gmem), map[string][]string{"sm_90a": nil}, 0, 0},
		{"C7512", module("sm_90a", fmt.Sprintf(c7512, "k"), gmem), map[string][]string{"sm_90a": {"C7512"}}, 1, 0},
		{"C7513", module("sm_90a", fmt.Sprintf(c7513, "k"), gmem), map[string][]string{"sm_90a": {"C7513"}}, 1, 0},
		{"C7519", module("sm_90a", fmt.Sprintf(c7519, "k"), gmem), map[string][]string{"sm_90a": {"C7519"}}, 1, 0},
		{"C7517 is not serialization", module("sm_90a", fmt.Sprintf(c7517, "k"), gmem), map[string][]string{"sm_90a": nil}, 1, 0},
		{
			"two codes",
			module("sm_90a", fmt.Sprintf(c7512, "k"), fmt.Sprintf(c7519, "k"), gmem),
			map[string][]string{"sm_90a": {"C7512", "C7519"}}, 2, 0,
		},
		{
			"first architecture only",
			module("sm_90a", fmt.Sprintf(c7519, "k"), gmem) + module("sm_100a", gmem),
			map[string][]string{"sm_90a": {"C7519"}, "sm_100a": nil}, 1, 0,
		},
		{
			"second architecture only",
			module("sm_90a", gmem) + module("sm_100a", fmt.Sprintf(c7519, "k"), gmem),
			map[string][]string{"sm_90a": nil, "sm_100a": {"C7519"}}, 1, 0,
		},
		{
			"after the function",
			module("sm_90a", gmem) + "ptxas info    : " + fmt.Sprintf(c7519, "k") + "\n",
			map[string][]string{"sm_90a": {"C7519"}}, 1, 0,
		},
		{
			"unnamed and unknown functions stay notes",
			module("sm_90a", "(C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized", fmt.Sprintf(c7519, "other"), gmem),
			map[string][]string{"sm_90a": nil}, 0, 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, err := Parse(strings.NewReader(tt.in))
			if err != nil {
				t.Fatal(err)
			}
			diags := 0
			for arch, want := range tt.want {
				k := r.Lookup("k", arch)
				if k == nil {
					t.Fatalf("no k for %s in %+v", arch, r)
				}
				diags += len(k.Diagnostics)
				var got []string
				for _, d := range k.WGMMASerialized() {
					got = append(got, d.Code)
				}
				if !slices.Equal(got, want) {
					t.Errorf("%s k: serialized codes %q, want %q", arch, got, want)
				}
				if c := r.Lookup("clean", arch); c == nil || len(c.Diagnostics) != 0 {
					t.Errorf("%s clean = %+v, want no diagnostics", arch, c)
				}
			}
			if diags != tt.wantDiag {
				t.Errorf("k diagnostics = %d, want %d", diags, tt.wantDiag)
			}
			if len(r.Notes) != tt.notes {
				t.Errorf("Notes = %q, want %d", r.Notes, tt.notes)
			}
			wantSerial := 0
			for _, codes := range tt.want {
				if len(codes) > 0 {
					wantSerial++
				}
			}
			if got := len(r.WGMMASerialized()); got != wantSerial {
				t.Errorf("Report.WGMMASerialized() has %d functions, want %d", got, wantSerial)
			}
		})
	}
}
