package cudalowering

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"strings"
	"testing"
)

// Transcribed from the retained SM121 opt0 listing; not binary attestation.
const observedSharedConversion = `code for sm_121
Function : kernel
/*0000*/ LDC R1, c[0x0][0x37c];
/*0010*/ S2R R3, SR_CgaCtaId;
/*0020*/ LDC R9, c[0x0][0x388];
/*0030*/ LDCU.64 UR4, c[0x0][0x358];
/*0040*/ MOV R0, 0x400;
/*0050*/ S2R R5, SR_SWINHI;
/*0060*/ LEA R0, R3, R0, 0x18;
/*0070*/ MOV R2, R0;
/*0080*/ MOV R3, R5;
/*0090*/ STS [R0], R9;
/*00a0*/ LD.E R2, desc[UR4][R2.64];
/*00b0*/ LDC.64 R4, c[0x0][0x380];
/*00c0*/ IADD R7, R2, 0x7;
/*00d0*/ STG.E desc[UR4][R4.64], R7;
/*00e0*/ EXIT;
/*00f0*/ BRA 0xf0;
/*0100*/ NOP;
`

func parseConversionFixture(t *testing.T, source string) *cudasass.Function {
	t.Helper()
	d, err := cudasass.Parse(strings.NewReader(source))
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Functions) != 1 {
		t.Fatalf("functions=%d", len(d.Functions))
	}
	return d.Functions[0]
}
func TestSharedConversionObservedChain(t *testing.T) {
	scan := scanSharedConversions(parseConversionFixture(t, observedSharedConversion))
	want := sharedConversionChain{window: 0x50, lowCopy: 0x70, highCopy: 0x80, addressDefinition: 0x60, sharedAccess: 0x90, genericAccess: 0xa0}
	if len(scan.refusals) != 0 || len(scan.chains) != 1 || scan.chains[0] != want {
		t.Fatalf("scan=%+v want%+v", scan, want)
	}
}
func TestSharedConversionRefusesUnsupportedProvenance(t *testing.T) {
	tests := []struct{ name, old, replacement string }{
		{"unrelated special register", "SR_SWINHI", "SR_CgaCtaId"},
		{"high source redefined", "/*0080*/ MOV R3, R5;", "/*0078*/ MOV R5, 0x7;\n/*0080*/ MOV R3, R5;"},
		{"low clobbered", "/*0090*/ STS", "/*0088*/ MOV R2, 0x7;\n/*0090*/ STS"},
		{"high clobbered", "/*0090*/ STS", "/*0088*/ MOV R3, 0x7;\n/*0090*/ STS"},
		{"shared address clobbered", "/*0090*/ STS", "/*0088*/ MOV R0, 0x7;\n/*0090*/ STS"},
		{"unrelated shared address", "STS [R0]", "STS [R9]"},
		{"unsupported shared value", "STS [R0], R9", "STS [R0], FOO"},
		{"unknown pair writer", "/*0090*/ STS", "/*0088*/ FOO.64 R2, R0;\n/*0090*/ STS"},
		{"known pair writer", "/*0090*/ STS", "/*0088*/ LDC.64 R2, c[0x0][0x380];\n/*0090*/ STS"},
		{"predication", "/*0080*/ MOV", "/*0080*/ @P0 MOV"},
		{"branch", "/*0080*/ MOV", "/*0078*/ BRA 0xa0;\n/*0080*/ MOV"},
		{"label boundary", "/*0080*/ MOV", ".L_split:\n/*0080*/ MOV"},
		{"ambiguous witnesses", "/*00a0*/ LD.E", "/*0098*/ STS [R0], R9;\n/*00a0*/ LD.E"},
		{"descriptor absent", "desc[UR4][R2.64]", "[R2]"},
		{"wrong pair", "desc[UR4][R2.64]", "desc[UR4][R4.64]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !strings.Contains(observedSharedConversion, tt.old) {
				t.Fatal("bad substitution")
			}
			scan := scanSharedConversions(parseConversionFixture(t, renumberConversionFixture(strings.Replace(observedSharedConversion, tt.old, tt.replacement, 1))))
			if len(scan.chains) != 0 || len(scan.refusals) == 0 {
				t.Fatalf("unsupported chain accepted: %+v", scan)
			}
		})
	}
}
func TestSharedConversionOriginalSourceClobberAfterCopy(t *testing.T) {
	source := strings.Replace(observedSharedConversion, "/*0090*/ STS", "/*0088*/ MOV R5, 0x7;\n/*0090*/ STS", 1)
	scan := scanSharedConversions(parseConversionFixture(t, renumberConversionFixture(source)))
	if len(scan.chains) != 1 || len(scan.refusals) != 0 {
		t.Fatalf("scan=%+v", scan)
	}
}

const observedSharedAfter = `code for sm_121
Function : kernel
/*0000*/ LDC R1, c[0x0][0x37c];
/*0010*/ S2UR UR5, SR_CgaCtaId;
/*0020*/ LDCU.64 UR6, c[0x0][0x358];
/*0030*/ LDC.64 R2, c[0x0][0x380];
/*0040*/ UMOV UR4, 0x400;
/*0050*/ LDC R0, c[0x0][0x388];
/*0060*/ ULEA UR4, UR5, UR4, 0x18;
/*0070*/ IADD R5, R0, 0x7;
/*0080*/ STS [UR4], R0;
/*0090*/ STG.E desc[UR6][R2.64], R5;
/*00a0*/ EXIT;
`

func TestSharedConversionAfterForwardedValue(t *testing.T) {
	source := observedSharedAfter
	fn := parseConversionFixture(t, source)
	scan := scanSharedConversions(fn)
	if len(scan.chains) != 0 || len(scan.refusals) != 0 {
		t.Fatalf("scan=%+v", scan)
	}
	for _, i := range fn.Instructions {
		if i.Opcode == "LDS" {
			t.Fatal("fixture must not invent LDS")
		}
	}
}
func TestSharedConversionBoundedScan(t *testing.T) {
	fn := &cudasass.Function{Instructions: make([]cudasass.Instruction, 65537)}
	if scan := scanSharedConversions(fn); len(scan.chains) != 0 || len(scan.refusals) == 0 {
		t.Fatalf("scan=%+v", scan)
	}
	fn = parseConversionFixture(t, observedSharedConversion)
	fn.Instructions[1].Address = 0
	if scan := scanSharedConversions(fn); len(scan.chains) != 0 || len(scan.refusals) == 0 {
		t.Fatalf("scan=%+v", scan)
	}
}
func TestSharedConversionCompareExactMechanismAndUnknown(t *testing.T) {
	m, err := cudaptx.Parse(strings.NewReader(".version 8.0\n.target sm_121\n.address_size 64\n.entry kernel() { ret; }\n"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, source string
		wantCount    int
		wantErr      bool
	}{
		{"observed", observedSharedConversion, 1, false},
		{"unknown", strings.Replace(observedSharedConversion, "MOV R3, R5", "FOO.64 R3, R5", 1), 0, true},
		{"PTX spelling is not SASS", "code for sm_121\nFunction : kernel\n/*0000*/ CVTA.SHARED R2, R0;\n/*0010*/ EXIT;\n", 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.Parse(strings.NewReader(tt.source))
			if err != nil {
				t.Fatal(err)
			}
			r, err := VerifyNoSharedConversions(m, d, "unproven slot label")
			if (err != nil) != tt.wantErr {
				t.Fatalf("error=%v", err)
			}
			got := 0
			for _, div := range r.Divergences {
				if div.Kind == SharedConversion {
					got += div.SASSCount
					if !strings.Contains(div.Detail, "not slot attribution") || !strings.Contains(div.Detail, "generic-access=0xa0") {
						t.Fatalf("detail=%q", div.Detail)
					}
				}
			}
			if got != tt.wantCount {
				t.Fatalf("count=%d want%d", got, tt.wantCount)
			}
		})
	}
}

// Synthetic insertions receive aligned PCs; the positive raw fixture retains
// its original addresses and has a separately pinned six-PC oracle.
func renumberConversionFixture(source string) string {
	lines := strings.Split(source, "\n")
	pc := 0
	for i, line := range lines {
		if strings.HasPrefix(line, "/*") {
			end := strings.Index(line, "*/")
			lines[i] = fmt.Sprintf("/*%04x*/", pc) + line[end+2:]
			pc += 16
		}
	}
	return strings.Join(lines, "\n")
}

func TestSharedConversionSymbolBoundaryAndTextBudget(t *testing.T) {
	for _, change := range []func(*cudasass.Function){
		func(fn *cudasass.Function) { fn.Symbols = []cudasass.Symbol{{Name: "callee", Address: 0x80}} },
		func(fn *cudasass.Function) { fn.Instructions[8].Opcode = strings.Repeat("X", 129) },
		func(fn *cudasass.Function) { fn.Instructions[8].Operands = strings.Repeat("X", 4097) },
	} {
		fn := parseConversionFixture(t, observedSharedConversion)
		change(fn)
		scan := scanSharedConversions(fn)
		if len(scan.chains) != 0 || len(scan.refusals) == 0 {
			t.Fatalf("scan=%+v", scan)
		}
	}
}

func TestSharedConversionEmptyAndUnterminatedCannotProveAbsence(t *testing.T) {
	for _, fn := range []*cudasass.Function{nil, {}, {Instructions: []cudasass.Instruction{{Opcode: "NOP"}}}} {
		scan := scanSharedConversions(fn)
		if len(scan.refusals) == 0 || len(scan.chains) != 0 {
			t.Fatalf("scan=%+v", scan)
		}
	}
}

func TestSharedConversionRequiresExactObservedTarget(t *testing.T) {
	d, err := cudasass.Parse(strings.NewReader(observedSharedAfter))
	if err != nil {
		t.Fatal(err)
	}
	for _, target := range []string{"sm_121", "sm_86", "sm_90", "", "sm_121a"} {
		t.Run(target, func(t *testing.T) {
			m, err := cudaptx.Parse(strings.NewReader(".version 8.0\n.target sm_121\n.address_size 64\n.entry kernel() { ret; }\n"))
			if err != nil {
				t.Fatal(err)
			}
			m.Target = target
			r, err := VerifyNoSharedConversions(m, d, "descriptive label")
			if target == "sm_121" {
				if err != nil || r == nil || r.Err() != nil {
					t.Fatalf("healthy observed target: report=%v err=%v", r, err)
				}
			} else {
				if err == nil || r == nil || !strings.Contains(err.Error(), "target") {
					t.Fatalf("unsupported target admitted: report=%v err=%v", r, err)
				}
			}
			// Target restrictions belong only to explicitly requested checks.
			if _, err := Compare(m, d, nil); err != nil {
				t.Fatalf("default Compare altered: %v", err)
			}
		})
	}
}

func TestSharedConversionRequiresRetainedSASSTarget(t *testing.T) {
	m, err := cudaptx.Parse(strings.NewReader(".version 8.0\n.target sm_121\n.address_size 64\n.entry kernel() { ret; }\n"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name, header string
		valid        bool
	}{
		{"actual observed", "code for sm_121", true},
		{"actual directive", ".target sm_121", true},
		{"wrong86", ".target sm_86", false},
		{"wrong90", ".target sm_90", false},
		{"missing", "", false},
		{"different suffix", ".target sm_121a", false},
		{"conflicting forward", ".target sm_121\ncode for sm_90", false},
		{"conflicting reverse", "code for sm_90\n.target sm_121", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := strings.Replace(observedSharedAfter, "code for sm_121", tt.header, 1)
			d, err := cudasass.Parse(strings.NewReader(source))
			if err != nil {
				t.Fatal(err)
			}
			r, err := VerifyNoSharedConversions(m, d, "descriptive label")
			if tt.valid {
				if err != nil || r == nil || r.Err() != nil {
					t.Fatalf("healthy supported pair: report=%v err=%v", r, err)
				}
			} else if err == nil || r == nil || !strings.Contains(err.Error(), "SASS target") {
				t.Fatalf("unsupported SASS target admitted: report=%v err=%v", r, err)
			}
			if _, err := Compare(m, d, nil); err != nil {
				t.Fatalf("default Compare altered: %v", err)
			}
		})
	}
}
