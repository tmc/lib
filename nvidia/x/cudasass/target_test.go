package cudasass

import (
	"strings"
	"testing"
)

func TestParseExplicitTarget(t *testing.T) {
	const body = "Function : kernel\n/*0000*/ EXIT;\n"
	tests := []struct{ name, header, want string }{
		{"directive", ".target sm_121\n", "sm_121"},
		{"code preamble", "code for sm_121\n", "sm_121"},
		{"both identical", ".target sm_121\ncode for sm_121\n", "sm_121"},
		{"duplicate identical", ".target sm_121\n.target sm_121\n", "sm_121"},
		{"suffix retained", ".target sm_121a\n", "sm_121a"},
		{"directive options", ".target sm_90, texmode_independent\n", "sm_90"},
		{"tab", ".target\tsm_121\n", "sm_121"},
		{"missing", "", ""},
		{"conflict forward", ".target sm_121\ncode for sm_90\n", ""},
		{"conflict reverse", "code for sm_90\n.target sm_121\n", ""},
		{"conflict cannot recover", ".target sm_121\n.target sm_90\n.target sm_121\n", ""},
		{"suffix conflict", ".target sm_121\n.target sm_121a\n", ""},
		{"empty directive", ".target\n.target sm_121\n", ""},
		{"empty code preamble", "code for\n.target sm_121\n", ""},
		{"malformed token", ".target sm_121!\n", ""},
		{"unicode token", ".target sm_１２１\n", ""},
		{"oversized token", ".target sm_" + strings.Repeat("1", 129) + "\n.target sm_121\n", ""},
		{"header word boundary", ".targeted sm_121\ncode format sm_121\n", ""},
		{"declaration budget", strings.Repeat(".target sm_121\n", 4097), ""},
		{"instruction text not target", "// .target sm_121\n// code for sm_121\n", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := ParseString(tt.header + body)
			if err != nil {
				t.Fatal(err)
			}
			if d.Target != tt.want {
				t.Fatalf("target=%q want%q", d.Target, tt.want)
			}
			fn := d.Function("kernel")
			if fn == nil || len(fn.Instructions) != 1 || fn.Instructions[0].Opcode != "EXIT" {
				t.Fatal("ordinary instruction parsing altered")
			}
		})
	}
}

func TestDisassemblyTargetTokenBound(t *testing.T) {
	for _, n := range []int{128, 129} {
		token := "sm_" + strings.Repeat("1", n-3)
		got, declared := disassemblyTarget(".target " + token)
		if !declared || (n == 128 && got != token) || (n == 129 && got != "") {
			t.Fatalf("n%d target=%q declared=%v", n, got, declared)
		}
	}
}

func FuzzDisassemblyTarget(f *testing.F) {
	for _, s := range []string{".target sm_121", "code for sm_121a", ".target", ".target " + strings.Repeat("x", 129), "// .target sm_121"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, line string) {
		target, declared := disassemblyTarget(line)
		if len(target) > 128 || (target != "" && (!declared || !strings.HasPrefix(target, "sm_"))) {
			t.Fatalf("invalid target result %q %v", target, declared)
		}
	})
}
