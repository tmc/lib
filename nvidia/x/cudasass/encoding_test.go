package cudasass_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// encodingFixture is nvdisasm --print-code --print-instruction-encoding output.
// See testdata/README.md for its provenance.
const encodingFixture = "testdata/sm121a-encoding.sass"

// fixtureArch is the --binary value naming the architecture encodingFixture was
// compiled for. nvdisasm spells it SM121a there and sm_121a in .target.
const fixtureArch = "SM121a"

func readEncodingFixture(t *testing.T) *cudasass.Disassembly {
	t.Helper()
	data, err := os.ReadFile(encodingFixture)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.Parse(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// TestParseEncoding checks the words parsed out of the fixture against the
// words nvdisasm printed in it, including which of the two lines each came
// from. The expectations are transcribed from the fixture, not from the parser.
func TestParseEncoding(t *testing.T) {
	d := readEncodingFixture(t)
	fn := d.Function("addk")
	if fn == nil {
		t.Fatalf("no addk function in %s", encodingFixture)
	}

	for _, tt := range []struct {
		addr   uint64
		opcode string
		want   cudasass.Encoding
		// bytes is the instruction as it occupies instruction memory.
		bytes string
	}{
		// /*0000*/ LDC R1, c[0x0][0x37c] ; /* 0x0000df00ff017b82 */
		//                                 /* 0x000fe20000000800 */
		{0x00, "LDC", cudasass.Encoding{Lo: 0x0000df00ff017b82, Hi: 0x000fe20000000800}, "827b01ff00df00000008000000e20f00"},
		// /*0010*/ S2R R0, SR_TID.X ;      /* 0x0000000000007919 */
		//                                 /* 0x000e2e0000002100 */
		{0x10, "S2R", cudasass.Encoding{Lo: 0x0000000000007919, Hi: 0x000e2e0000002100}, "197900000000000000210000002e0e00"},
		// /*0070*/ @P0 EXIT ;              /* 0x000000000000094d */
		//                                 /* 0x000fea0003800000 */
		{0x70, "EXIT", cudasass.Encoding{Lo: 0x000000000000094d, Hi: 0x000fea0003800000}, "4d090000000000000000800300ea0f00"},
	} {
		t.Run(tt.opcode, func(t *testing.T) {
			var inst cudasass.Instruction
			for _, i := range fn.Instructions {
				if i.Address == tt.addr {
					inst = i
					break
				}
			}
			if inst.Opcode != tt.opcode {
				t.Fatalf("instruction at %#x is %q, want %q", tt.addr, inst.Opcode, tt.opcode)
			}
			if inst.Encoding != tt.want {
				t.Errorf("encoding = %v, want %v", inst.Encoding, tt.want)
			}
			if !inst.Encoding.Present() {
				t.Error("Present() = false")
			}
			if got := hex.EncodeToString(sliceOf(inst.Encoding.Bytes())); got != tt.bytes {
				t.Errorf("Bytes() = %s, want %s", got, tt.bytes)
			}
			if inst.Comment != "" {
				t.Errorf("Comment = %q, want the encoding not to land in it", inst.Comment)
			}
		})
	}
}

// TestParseEncodingCoverage checks that every instruction in the fixture, and
// no non-instruction line, carries a complete encoding.
func TestParseEncodingCoverage(t *testing.T) {
	d := readEncodingFixture(t)
	if len(d.Functions) != 1 {
		t.Fatalf("got %d functions, want 1", len(d.Functions))
	}
	fn := d.Functions[0]
	if len(fn.Instructions) == 0 {
		t.Fatal("no instructions")
	}
	for _, inst := range fn.Instructions {
		if !inst.Encoding.Present() {
			t.Errorf("%#x %s: no encoding", inst.Address, inst.Opcode)
		}
		if inst.Encoding.Hi == 0 {
			t.Errorf("%#x %s: high word missing", inst.Address, inst.Opcode)
		}
	}
	// One encoding per 16-byte slot: the addresses nvdisasm printed and the
	// encodings it printed must agree on how many instructions there are.
	for i, inst := range fn.Instructions {
		if want := fn.BaseAddress + uint64(16*i); inst.Address != want {
			t.Fatalf("instruction %d at %#x, want %#x: an encoding line was taken for an instruction or dropped", i, inst.Address, want)
		}
	}
}

// TestParseWithoutEncoding checks that the zero Encoding marks output that
// carried none, which is what distinguishes "not captured" from a real value.
func TestParseWithoutEncoding(t *testing.T) {
	const sass = `
.section .text.compute,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x28] ;    // a real comment
        /*0010*/                   EXIT ;
`
	d, err := cudasass.ParseString(sass)
	if err != nil {
		t.Fatal(err)
	}
	fn := d.Function("compute")
	if fn == nil || len(fn.Instructions) != 2 {
		t.Fatalf("unexpected functions: %+v", d.Functions)
	}
	for _, inst := range fn.Instructions {
		if inst.Encoding.Present() {
			t.Errorf("%s: Present() = true, want false", inst.Opcode)
		}
	}
	if got := fn.Instructions[0].Comment; got != "a real comment" {
		t.Errorf("Comment = %q, want %q", got, "a real comment")
	}
}

// TestEncodingRoundTrip feeds the captured encodings back to nvdisasm through
// its --binary mode and requires the instructions it reports to be the ones the
// fixture was parsed from. nvdisasm, not this package, decides whether the
// bytes are right: it rejects an all-zero word, a length that is not a multiple
// of sixteen, and, as the control below shows, the two words in the wrong order.
func TestEncodingRoundTrip(t *testing.T) {
	tool, err := exec.LookPath("nvdisasm")
	if err != nil {
		t.Skipf("nvdisasm not in PATH: %v", err)
	}

	want := readEncodingFixture(t).Functions[0].Instructions
	var raw []byte
	for _, inst := range want {
		b := inst.Encoding.Bytes()
		raw = append(raw, b[:]...)
	}

	out, err := runBinaryMode(t, tool, raw)
	if err != nil {
		t.Fatalf("nvdisasm --binary: %v\n%s", err, out)
	}
	got, err := cudasass.Parse(bytes.NewReader(out))
	if err != nil {
		t.Fatal(err)
	}
	var flat []cudasass.Instruction
	for _, fn := range got.Functions {
		flat = append(flat, fn.Instructions...)
	}
	if len(flat) != len(want) {
		t.Fatalf("round trip produced %d instructions, want %d", len(flat), len(want))
	}
	for i := range want {
		if flat[i].Address != want[i].Address {
			t.Errorf("instruction %d: address %#x, want %#x", i, flat[i].Address, want[i].Address)
		}
		// A branch into an ELF section is printed as `(label); the same
		// encoding in a raw binary, which has no symbols, is printed as the
		// address that label resolves to. Only the operands differ, so those
		// instructions are compared without them.
		if strings.Contains(want[i].Operands, "`(") {
			if flat[i].Predicate != want[i].Predicate || flat[i].Opcode != want[i].Opcode {
				t.Errorf("instruction %d: round trip %s %s, fixture %s %s", i, flat[i].Predicate, flat[i].Opcode, want[i].Predicate, want[i].Opcode)
			}
			continue
		}
		w, g := cudasass.NormalizeInstruction(want[i]), cudasass.NormalizeInstruction(flat[i])
		if w != g {
			t.Errorf("instruction %d: round trip %q, fixture %q", i, g, w)
		}
	}

	// Control: with the words swapped the same bytes must not disassemble, so
	// a passing round trip is evidence about the order and not only the values.
	swapped := make([]byte, len(raw))
	for i := 0; i < len(raw); i += 16 {
		copy(swapped[i:], raw[i+8:i+16])
		copy(swapped[i+8:], raw[i:i+8])
	}
	if out, err := runBinaryMode(t, tool, swapped); err == nil {
		t.Errorf("nvdisasm accepted word-swapped encodings:\n%s", out)
	}
}

func runBinaryMode(t *testing.T, tool string, raw []byte) ([]byte, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "raw.bin")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	return exec.CommandContext(context.Background(), tool, "--binary", fixtureArch, path).CombinedOutput()
}

func sliceOf(b [16]byte) []byte { return b[:] }

// A legal instruction may in principle encode to a zero low word. Parse must
// attach the high word on the strength of having seen an encoding comment, not
// on the low word being nonzero, or such an instruction silently loses half its
// encoding and the orphaned high word attaches to nothing.
func TestZeroLowWordStillTakesHighWord(t *testing.T) {
	const src = `		.headerflags	@"EF_CUDA_SM121"
		.text.k:
        /*0000*/                   MOV R1, c[0x0][0x28] ;    /* 0x0000000000000000 */
                                                             /* 0x000fc40000000f00 */
        /*0010*/                   EXIT ;                    /* 0x000000000000794d */
                                                             /* 0x000fea0003800000 */
`
	d, err := cudasass.ParseString(src)
	if err != nil {
		t.Fatal(err)
	}
	var got []cudasass.Encoding
	for _, fn := range d.Functions {
		for _, inst := range fn.Instructions {
			got = append(got, inst.Encoding)
		}
	}
	if len(got) != 2 {
		t.Fatalf("parsed %d instructions, want 2", len(got))
	}
	want := cudasass.Encoding{Lo: 0, Hi: 0x000fc40000000f00}
	if got[0] != want {
		t.Errorf("zero-low-word instruction: got %v, want %v", got[0], want)
	}
	if !got[1].Present() {
		t.Errorf("second instruction lost its encoding: %v", got[1])
	}
}
