package cudasass

import (
	"context"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

const sampleLifeRangeText = `
	.target	sm_121
	.section	.text.compute,"ax",@progbits
// +-------+--------+--------+---------+
// | GPR   | PRED   | UGPR   | UPRED   |
// |   #   |   #    |    #   |    #    |
// +-------+--------+--------+---------+
compute:
.text.compute:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                                                                                         // |   1   |        |        |         |
        /*0010*/                   S2R R0, SR_TID.X ;                                                                                              // |   2   |        |        |         |
        /*0020*/                   S2UR UR4, SR_CTAID.X ;                                                                                          // |   2   |        |    1   |         |
        /*0030*/                   LDCU UR5, c[0x0][0x388] ;                                                                                       // |   2   |        |    2   |         |
        /*0040*/                   LDC R5, c[0x0][0x360] ;                                                                                         // |   3   |        |    2   |         |
        /*0060*/                   ISETP.GE.AND P0, PT, R5, UR5, PT ;                                                                              // |   2   |   1    |    1   |         |
        /*00d0*/                   UISETP.GE.AND UP0, UPT, UR5, 0x1, UPT ;                                                                         // |   4   |        |    3   |    1    |
        /*0180*/                   UISETP.GT.AND UP1, UPT, UR8, 0xc, UPT ;                                                                         // |   4   |        |    5   |    2    |
        /*0500*/                   EXIT ;                                                                                                          // |   1   |        |        |         |

	.section	.text.vector_add,"ax",@progbits
vector_add:
        /*0000*/                   MOV R1, c[0x0][0x28] ;                                                                                          // |   1   |        |        |         |
        /*0010*/                   ISETP.GE.AND P0, PT, R1, R8, PT ;                                                                               // |   2   |   1    |        |         |
        /*0020*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`

const samplePass1Text = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
	//## File "kernels.cu", line 10
	//## File ".nv_debug_ptx_txt", line 25
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
	//## File "kernels.cu", line 11
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   S2UR UR4, SR_CTAID.X ;
        /*0030*/                   LDCU UR5, c[0x0][0x388] ;
        /*0040*/                   LDC R5, c[0x0][0x360] ;
        /*0060*/                   ISETP.GE.AND P0, PT, R5, UR5, PT ;
        /*00d0*/                   UISETP.GE.AND UP0, UPT, UR5, 0x1, UPT ;
        /*0180*/                   UISETP.GT.AND UP1, UPT, UR8, 0xc, UPT ;
        /*0500*/                   EXIT ;
        /*0510*/                   NOP ;
        /*0520*/                   NOP ;

	.section	.text.vector_add,"ax",@progbits
vector_add:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   ISETP.GE.AND P0, PT, R1, R8, PT ;
        /*0020*/                   EXIT ;
`

func TestParseLifeRangeCount(t *testing.T) {
	pressure, err := ParseLifeRangeCount(strings.NewReader(sampleLifeRangeText))
	if err != nil {
		t.Fatalf("ParseLifeRangeCount failed: %v", err)
	}

	if len(pressure) != 2 {
		t.Fatalf("expected 2 kernels, got %d", len(pressure))
	}

	compute, ok := pressure["compute"]
	if !ok {
		t.Fatal("compute kernel missing from pressure map")
	}

	tests := []struct {
		addr uint64
		want RegisterPressure
		desc string
	}{
		{0x0000, RegisterPressure{GPR: 1, Pred: 0, UGPR: 0, UPred: 0}, "initial load"},
		{0x0020, RegisterPressure{GPR: 2, Pred: 0, UGPR: 1, UPred: 0}, "first uniform register"},
		{0x0060, RegisterPressure{GPR: 2, Pred: 1, UGPR: 1, UPred: 0}, "predicate set"},
		{0x00d0, RegisterPressure{GPR: 4, Pred: 0, UGPR: 3, UPred: 1}, "uniform predicate set"},
		{0x0180, RegisterPressure{GPR: 4, Pred: 0, UGPR: 5, UPred: 2}, "multiple uniform predicates"},
		{0x0500, RegisterPressure{GPR: 1, Pred: 0, UGPR: 0, UPred: 0}, "exit instruction"},
	}

	for _, tt := range tests {
		got, ok := compute[tt.addr]
		if !ok {
			t.Errorf("address 0x%x (%s) missing in compute", tt.addr, tt.desc)
			continue
		}
		if got != tt.want {
			t.Errorf("address 0x%x (%s): got %+v, want %+v", tt.addr, tt.desc, got, tt.want)
		}
	}

	// Positive control: verify nonzero GPR on a known kernel
	var hasNonZeroGPR bool
	for _, p := range compute {
		if p.GPR > 0 {
			hasNonZeroGPR = true
			break
		}
	}
	if !hasNonZeroGPR {
		t.Error("positive control failed: compute has zero GPR across all instructions")
	}

	// Verify vector_add
	vadd, ok := pressure["vector_add"]
	if !ok {
		t.Fatal("vector_add kernel missing from pressure map")
	}
	if got := vadd[0x0010]; got != (RegisterPressure{GPR: 2, Pred: 1, UGPR: 0, UPred: 0}) {
		t.Errorf("vector_add 0x0010: got %+v, want GPR=2 Pred=1", got)
	}
}

func TestJoinPressure(t *testing.T) {
	disasm, err := ParseString(samplePass1Text)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	pressure, err := ParseLifeRangeCount(strings.NewReader(sampleLifeRangeText))
	if err != nil {
		t.Fatalf("ParseLifeRangeCount failed: %v", err)
	}

	if err := JoinPressure(disasm, pressure); err != nil {
		t.Fatalf("JoinPressure failed: %v", err)
	}

	compute := disasm.Function("compute")
	if compute == nil {
		t.Fatal("compute function missing")
	}

	// Verify instruction pressure population
	inst0 := compute.Instructions[0]
	if inst0.Address != 0x0000 || inst0.LiveGPR != 1 || inst0.LivePred != 0 {
		t.Errorf("inst 0x0000: got GPR=%d, Pred=%d; want 1, 0", inst0.LiveGPR, inst0.LivePred)
	}
	if inst0.Pressure() != (RegisterPressure{GPR: 1, Pred: 0, UGPR: 0, UPred: 0}) {
		t.Errorf("inst 0x0000 Pressure(): got %+v", inst0.Pressure())
	}

	// Verify uniform registers on 0x0180
	var inst180 *Instruction
	for i := range compute.Instructions {
		if compute.Instructions[i].Address == 0x0180 {
			inst180 = &compute.Instructions[i]
			break
		}
	}
	if inst180 == nil {
		t.Fatal("instruction 0x0180 missing")
	}
	if inst180.LiveGPR != 4 || inst180.LiveUGPR != 5 || inst180.LiveUPred != 2 {
		t.Errorf("inst 0x0180: got GPR=%d UGPR=%d UPred=%d; want 4, 5, 2",
			inst180.LiveGPR, inst180.LiveUGPR, inst180.LiveUPred)
	}

	// Verify trailing NOPs get 0 pressure without join error
	nop1 := compute.Instructions[len(compute.Instructions)-2]
	if nop1.Opcode != "NOP" || nop1.LiveGPR != 0 || nop1.LiveUGPR != 0 {
		t.Errorf("trailing NOP got nonzero pressure: %+v", nop1)
	}
}

func TestJoinPressureTotalFailures(t *testing.T) {
	t.Run("missing_non_nop_instruction_in_pass2", func(t *testing.T) {
		disasm, err := ParseString(`
.section .text.k,"ax",@progbits
k:
/*0000*/ MOV R1, R2 ;
/*0010*/ FFMA R0, R1, R2, R0 ;
`)
		if err != nil {
			t.Fatal(err)
		}

		// Pass 2 omits 0x0010 FFMA
		pressure := map[string]map[uint64]RegisterPressure{
			"k": {
				0x0000: {GPR: 2},
			},
		}

		err = JoinPressure(disasm, pressure)
		if err == nil {
			t.Fatal("expected total join failure when non-NOP instruction is missing from Pass 2")
		}
		if !strings.Contains(err.Error(), "total join failed") || !strings.Contains(err.Error(), "0x10") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("orphan_address_in_pass2", func(t *testing.T) {
		disasm, err := ParseString(`
.section .text.k,"ax",@progbits
k:
/*0000*/ MOV R1, R2 ;
`)
		if err != nil {
			t.Fatal(err)
		}

		// Pass 2 has unexpected instruction at 0x0020
		pressure := map[string]map[uint64]RegisterPressure{
			"k": {
				0x0000: {GPR: 2},
				0x0020: {GPR: 3},
			},
		}

		err = JoinPressure(disasm, pressure)
		if err == nil {
			t.Fatal("expected total join failure when Pass 2 has extra address")
		}
		if !strings.Contains(err.Error(), "0x20") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("missing_kernel_in_pass2", func(t *testing.T) {
		disasm, err := ParseString(`
.section .text.k1,"ax",@progbits
k1:
/*0000*/ MOV R1, R2 ;
.section .text.k2,"ax",@progbits
k2:
/*0000*/ MOV R1, R2 ;
`)
		if err != nil {
			t.Fatal(err)
		}

		pressure := map[string]map[uint64]RegisterPressure{
			"k1": {0x0000: {GPR: 2}},
		}

		err = JoinPressure(disasm, pressure)
		if err == nil {
			t.Fatal("expected error when kernel is missing from Pass 2")
		}
		if !strings.Contains(err.Error(), "k2") {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

func TestDisassembleWithPressureFakeTool(t *testing.T) {
	fakeScript := `for arg do
    if [ "$arg" = "--print-life-ranges" ]; then
        printf '.section .text.k,"ax",@progbits\nk:\n/*0000*/ MOV R1, R2 ; // | 2 | | |\n/*0010*/ EXIT ; // | 1 | | |\n'
        exit 0
    fi
done
printf '.section .text.k,"ax",@progbits\nk:\n//## File "k.cu", line 5\n/*0000*/ MOV R1, R2 ;\n/*0010*/ EXIT ;\n'
`
	p := fakeTool(t, fakeScript)
	d := Disassembler{Path: p, TempDir: t.TempDir()}

	disasm, err := d.DisassembleWithPressure(context.Background(), testCUBIN())
	if err != nil {
		t.Fatalf("DisassembleWithPressure failed: %v", err)
	}

	if len(disasm.Functions) != 1 {
		t.Fatalf("expected 1 function, got %d", len(disasm.Functions))
	}
	k := disasm.Function("k")
	if k == nil {
		t.Fatal("function k missing")
	}
	if len(k.Instructions) != 2 {
		t.Fatalf("expected 2 instructions, got %d", len(k.Instructions))
	}
	if k.Instructions[0].LiveGPR != 2 {
		t.Errorf("inst 0 LiveGPR: got %d, want 2", k.Instructions[0].LiveGPR)
	}
	if k.Instructions[0].SourceLine != 5 {
		t.Errorf("inst 0 SourceLine: got %d, want 5", k.Instructions[0].SourceLine)
	}
	if k.Instructions[1].LiveGPR != 1 {
		t.Errorf("inst 1 LiveGPR: got %d, want 1", k.Instructions[1].LiveGPR)
	}
}

// TestLifeRangeOptions covers register-life option validation and the command
// line it produces. It inspects the flags rather than running a stub tool,
// because Disassemble now checks the flags against what the resolved nvdisasm
// accepts, and a stub accepts nothing.
func TestLifeRangeOptions(t *testing.T) {
	flags, err := Options{Mode: RegisterLife, LifeRangeMode: "count", PrintCode: true}.flags()
	if err != nil {
		t.Fatalf("flags: %v", err)
	}
	got := cudatool.Args(flags...)
	want := []string{"--print-life-ranges", "--life-range-mode", "count", "--print-code"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("args = %q, want %q", got, want)
	}
	rejected := []struct {
		name string
		o    Options
	}{
		{"life range mode without register life mode", Options{Mode: Text, LifeRangeMode: "count"}},
		{"source lines with register life mode", Options{Mode: RegisterLife, SourceLines: true}},
	}
	for _, tt := range rejected {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tt.o.flags(); err == nil {
				t.Errorf("flags accepted %+v", tt.o)
			}
		})
	}
}

func TestInstalledDisassembleWithPressure(t *testing.T) {
	p := os.Getenv("CUDASASS_TEST_CUBIN")
	if p == "" {
		t.Skip("set CUDASASS_TEST_CUBIN to a real compiled cubin")
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	d := Disassembler{Path: os.Getenv("NVDISASM"), TempDir: t.TempDir()}
	disasm, err := d.DisassembleWithPressure(ctx, b)
	if err != nil {
		t.Fatalf("DisassembleWithPressure failed: %v", err)
	}

	if len(disasm.Functions) == 0 {
		t.Fatal("expected at least one function in disassembly")
	}

	var totalGPR int
	var totalInstructions int
	for _, fn := range disasm.Functions {
		for _, inst := range fn.Instructions {
			totalInstructions++
			totalGPR += inst.LiveGPR
		}
	}

	// Positive control: verify nonzero GPR across instructions
	if totalInstructions == 0 {
		t.Fatal("zero instructions found")
	}
	if totalGPR == 0 {
		t.Fatalf("positive control failed: total GPR is 0 across %d instructions", totalInstructions)
	}
	t.Logf("DisassembleWithPressure passed across %d functions, %d instructions, total live GPR sum %d",
		len(disasm.Functions), totalInstructions, totalGPR)
}

func BenchmarkParseLifeRangeCount(b *testing.B) {
	data := sampleLifeRangeText
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := ParseLifeRangeCount(strings.NewReader(data))
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJoinPressure(b *testing.B) {
	disasm, err := ParseString(samplePass1Text)
	if err != nil {
		b.Fatal(err)
	}
	pressure, err := ParseLifeRangeCount(strings.NewReader(sampleLifeRangeText))
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Re-create disasm copy for each iteration
		d := &Disassembly{
			Functions: make([]*Function, len(disasm.Functions)),
		}
		for fi, fn := range disasm.Functions {
			fnCopy := &Function{
				Name:         fn.Name,
				BaseAddress:  fn.BaseAddress,
				Instructions: make([]Instruction, len(fn.Instructions)),
			}
			copy(fnCopy.Instructions, fn.Instructions)
			d.Functions[fi] = fnCopy
		}

		if err := JoinPressure(d, pressure); err != nil {
			b.Fatal(err)
		}
	}
}
