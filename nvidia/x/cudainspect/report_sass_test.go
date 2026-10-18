package cudainspect

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
)

// kernelSASS is a listing for metadataELF's kernel. Its encodings are absent,
// so it cannot be bound to the image's bytes.
func kernelSASS(target string, opcodes ...string) string {
	var b strings.Builder
	if target != "" {
		b.WriteString("\t.target\t" + target + "\n")
	}
	b.WriteString(".section .text.kernel,\"ax\",@progbits\n")
	for n, op := range opcodes {
		fmt.Fprintf(&b, "/*%04x*/ %s R1, [R2] ;\n", n*16, op)
	}
	return b.String()
}

func roundtripSASS(t testing.TB) string {
	t.Helper()
	b, err := os.ReadFile("../cudasass/testdata/roundtrip/sm121a-roundtrip.sass")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func value(t testing.TB, r *Report, name string) string {
	t.Helper()
	f := r.Feature(name)
	if f == nil {
		t.Fatalf("feature %s absent", name)
	}
	if !f.Known() {
		return f.State
	}
	var b bytes.Buffer
	if err := json.Compact(&b, f.Value); err != nil {
		t.Fatal(err)
	}
	return f.State + " " + b.String()
}

func TestReportSASSRecorded(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), roundtripCUBIN(t), Request{
		Symbol:   "addk",
		Required: []string{"opcode_histogram", "memory_access_widths", "local_memory_ops", "feature_suffix"},
		SASS:     roundtripSASS(t),
	})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"feature_suffix":       `inferred "a"`,
		"memory_access_widths": `observed {"global_load":{"32":2},"global_store":{"32":1}}`,
		"local_memory_ops":     `observed {"loads":0,"stores":0}`,
		"opcode_histogram":     `observed {"counted":32,"quarantined":0,"opcodes":{"BRA":1,"EXIT":2,"FADD":1,"IMAD":4,"ISETP":1,"LDC":5,"LDCU":2,"LDG":2,"NOP":11,"S2R":1,"S2UR":1,"STG":1}}`,
	} {
		if got := value(t, r, name); got != want {
			t.Errorf("%s = %s, want %s", name, got, want)
		}
	}
	if r.Result != ResultComplete || r.SASS == nil || r.SASS.SizeBytes != int64(len(roundtripSASS(t))) {
		t.Fatalf("result %s sass %+v", r.Result, r.SASS)
	}
}

func TestReportSASSUnboundIsInferred(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", SASS: kernelSASS("sm_121", "LDG.E.128", "STS")})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := value(t, r, "memory_access_widths"), `inferred {"global_load":{"128":1},"shared_store":{"32":1}}`; got != want {
		t.Fatalf("widths = %s, want %s", got, want)
	}
	if got := value(t, r, "feature_suffix"); got != StateUnknown {
		t.Fatalf("suffix from a suffixless target: %s", got)
	}
	if !strings.Contains(diagnostics(r, "opcode_histogram"), "not bound to image bytes") {
		t.Fatalf("diagnostics: %+v", r.Diagnostics)
	}
}

func diagnostics(r *Report, feature string) string {
	var out []string
	for _, d := range r.Diagnostics {
		if d.Feature == feature {
			out = append(out, d.Message)
		}
	}
	return strings.Join(out, "\n")
}

func TestReportSASSRefusals(t *testing.T) {
	other := strings.Replace(kernelSASS("", "NOP"), ".text.kernel", ".text.other", 1)
	for _, tt := range []struct {
		name, sass string
		want       error
	}{
		{"wrong_function", other, ErrSASS},
		{"recorded_wrong_function", "", ErrSASS},
		{"target_mismatch", kernelSASS("sm_90", "NOP"), ErrSASS},
		{"target_minor_mismatch", kernelSASS("sm_120", "NOP"), ErrSASS},
		{"suffix_z", kernelSASS("sm_121z", "NOP"), ErrSASS},
		{"suffix_garbage", kernelSASS("sm_121_garbage", "NOP"), ErrSASS},
		{"suffix_af", kernelSASS("sm_121af", "NOP"), ErrSASS},
		{"suffix_ab12", kernelSASS("sm_121ab12", "NOP"), ErrSASS},
		{"target_malformed", kernelSASS("sm_x", "NOP"), ErrSASS},
		{"no_instructions", kernelSASS(""), ErrSASS},
		{"oversize", kernelSASS("", "NOP") + strings.Repeat("\n", MaxSASSBytes), ErrInputTooLarge},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.sass == "" {
				tt.sass = roundtripSASS(t) // lists intmix and addk, not kernel
			}
			i, _ := NewInspector(".")
			r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", SASS: tt.sass})
			if !errors.Is(err, tt.want) || r != nil {
				t.Fatalf("Report = %v, %v; want nil, %v", r, err, tt.want)
			}
		})
	}
}

func TestReportFeatureSuffix(t *testing.T) {
	for _, tt := range []struct{ target, want string }{
		{"sm_121a", `inferred "a"`},
		{"sm_121f", `inferred "f"`},
		{"sm_121", StateUnknown},
		{"", StateUnknown},
	} {
		t.Run(tt.target, func(t *testing.T) {
			i, _ := NewInspector(".")
			r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", SASS: kernelSASS(tt.target, "NOP")})
			if err != nil {
				t.Fatal(err)
			}
			if got := value(t, r, "feature_suffix"); got != tt.want {
				t.Fatalf("feature_suffix = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestReportUnrecognizedWidthQuarantined(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{
		Symbol:   "kernel",
		Optional: []string{"memory_access_widths"},
		SASS:     kernelSASS("sm_121", "LDG.E.ENL2.256", "STG.E.ENL2.256", "LDG.E"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := value(t, r, "opcode_histogram"), `inferred {"counted":1,"quarantined":2,"opcodes":{"LDG":1}}`; got != want {
		t.Fatalf("histogram = %s, want %s", got, want)
	}
	if got := value(t, r, "memory_access_widths"); got != StateUnknown {
		t.Fatalf("memory_access_widths = %s, want unknown", got)
	}
	if !strings.Contains(diagnostics(r, "opcode_histogram"), `unrecognized size modifier "256"`) {
		t.Fatalf("diagnostics: %s", diagnostics(r, "opcode_histogram"))
	}
}

func TestReportUnknownOpcodeQuarantined(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{
		Symbol:   "kernel",
		Required: []string{"opcode_histogram"},
		Optional: []string{"local_memory_ops"},
		SASS:     kernelSASS("", "frob", "LDG.E", "???", "LDL.64.128"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := value(t, r, "opcode_histogram"), `inferred {"counted":1,"quarantined":3,"opcodes":{"LDG":1}}`; got != want {
		t.Fatalf("histogram = %s, want %s", got, want)
	}
	// A quarantined instruction may be a local access, so no count is claimed.
	for _, name := range []string{"memory_access_widths", "local_memory_ops"} {
		if got := value(t, r, name); got != StateUnknown {
			t.Errorf("%s = %s, want unknown", name, got)
		}
	}
	if r.Result != ResultPartial {
		t.Fatalf("result %s, want partial", r.Result)
	}
	if n := strings.Count(diagnostics(r, "opcode_histogram"), "quarantined instruction"); n != 3 {
		t.Fatalf("%d quarantine diagnostics, want 3:\n%s", n, diagnostics(r, "opcode_histogram"))
	}
}

func TestReportQuarantineDiagnosticsBounded(t *testing.T) {
	ops := strings.Fields(strings.Repeat("frob ", 3*maxQuarantineDiagnostics))
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", SASS: kernelSASS("", ops...)})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(diagnostics(r, "opcode_histogram"), "\n")
	want := fmt.Sprintf("%d more quarantined instructions", 2*maxQuarantineDiagnostics)
	if len(lines) != maxQuarantineDiagnostics+2 || lines[len(lines)-1] != want {
		t.Fatalf("%d diagnostics, last %q; want %d ending %q", len(lines), lines[len(lines)-1], maxQuarantineDiagnostics+2, want)
	}
}

func TestReportLocalIsNotSpill(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", SASS: kernelSASS("", "LDL", "STL.64", "STL.128")})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := value(t, r, "local_memory_ops"), `inferred {"loads":1,"stores":2}`; got != want {
		t.Fatalf("local = %s, want %s", got, want)
	}
	for _, f := range r.Features {
		if strings.Contains(f.Name, "spill") || bytes.Contains(f.Value, []byte("spill")) {
			t.Fatalf("local operations reported as spills: %+v", f)
		}
	}
}

func TestReportRequiredSASSFeatureFails(t *testing.T) {
	i, _ := NewInspector(".")
	r, err := i.Report(context.Background(), metadataELF(true, 0, elf.EM_CUDA), Request{Symbol: "kernel", Required: []string{"memory_access_widths"}})
	if !errors.Is(err, ErrRequiredFeature) || !strings.HasSuffix(err.Error(), ": memory_access_widths") || r == nil || r.Result != ResultFailed {
		t.Fatalf("Report = %+v, %v", r, err)
	}
	if !strings.Contains(diagnostics(r, "memory_access_widths"), "no SASS supplied") {
		t.Fatalf("diagnostics: %+v", r.Diagnostics)
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	r.Feature("memory_access_widths").State = StateObserved
	r.Feature("memory_access_widths").Value = json.RawMessage("{}")
	r.Feature("memory_access_widths").Evidence = []string{"claimed"}
	r.Result = ResultComplete
	if err := r.Validate(); !errors.Is(err, ErrInvalidReport) {
		t.Fatalf("Validate accepted an instruction feature without SASS: %v", err)
	}
}

func TestClassifyOpcode(t *testing.T) {
	for _, tt := range []struct {
		opcode, base, class string
		bits                int
		quarantined         bool
	}{
		{"LDG.E", "LDG", "global_load", 32, false},
		{"LDG.E.128", "LDG", "global_load", 128, false},
		{"LDG.E.64.CONSTANT", "LDG", "global_load", 64, false},
		{"LDG.E.LTC128B.128", "LDG", "global_load", 128, false},
		{"STG.E.U8", "STG", "global_store", 8, false},
		{"LDS.U16", "LDS", "shared_load", 16, false},
		{"STS.128", "STS", "shared_store", 128, false},
		{"LDL", "LDL", "local_load", 32, false},
		{"STL.64", "STL", "local_store", 64, false},
		{"LDC.64", "LDC", "", 0, false},
		{"IMAD.WIDE.U32", "IMAD", "", 0, false},
		{"LDG.E.64.128", "", "", 0, true},
		{"LDG.E.ENL2.256", "", "", 0, true},
		{"STG.E.ENL2.256", "", "", 0, true},
		{"LDS.U64", "", "", 0, true},
		{"STL.S128", "", "", 0, true},
		{"LDG.E.16", "", "", 0, true},
		{"IMAD.WIDE.U64", "IMAD", "", 0, false},
		{"frob", "", "", 0, true},
		{"LDG..E", "", "", 0, true},
		{"", "", "", 0, true},
	} {
		base, class, bits, reason := classifyOpcode(tt.opcode)
		if base != tt.base || class != tt.class || bits != tt.bits || (reason != "") != tt.quarantined {
			t.Errorf("classifyOpcode(%q) = %q, %q, %d, %q", tt.opcode, base, class, bits, reason)
		}
	}
}
