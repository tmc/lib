package cudacapture_test

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// loadFixtureCapture returns the committed kernels.* fixture with ptx and sass
// optionally replaced, so a control can change one tier and leave the rest.
func loadFixtureCapture(t *testing.T, ptx, sass string) *cudacapture.Capture {
	t.Helper()
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("kernels.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	ptxBytes := read("kernels.ptx")
	if ptx != "" {
		ptxBytes = []byte(ptx)
	}
	sassBytes := read("kernels.sass")
	if sass != "" {
		sassBytes = []byte(sass)
	}
	return &cudacapture.Capture{
		Code: cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.cu":    read("kernels.cu"),
			"kernels.ptx":   ptxBytes,
			"kernels.cubin": cubin,
			"kernels.sass":  sassBytes,
		},
	}
}

var ptxLoc = regexp.MustCompile(`(?m)^(\t\.loc\t1 )(\d+)( .*)$`)

// movePTXLines returns testdata/kernels.ptx with every source line the line
// table names replaced by move(line).
//
// The controls below are transformations of the committed compiler output
// rather than second, hand-written PTX files. A PTX nobody compiled cannot
// show that the binding check reads the one the CUBIN was built from: it would
// differ from the fixture in its header, its register declarations and its
// instruction text as well as in the line table under test.
func movePTXLines(t *testing.T, move func(int) int) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "kernels.ptx"))
	if err != nil {
		t.Fatalf("read kernels.ptx: %v", err)
	}
	moved := ptxLoc.ReplaceAllStringFunc(string(b), func(m string) string {
		g := ptxLoc.FindStringSubmatch(m)
		n, err := strconv.Atoi(g[2])
		if err != nil {
			t.Fatalf("unreadable .loc line %q", m)
		}
		return g[1] + strconv.Itoa(move(n)) + g[3]
	})
	if moved == string(b) {
		t.Fatal("movePTXLines changed nothing; the .loc spelling in the fixture has moved")
	}
	return moved
}

func TestPTXLineUnknownWhenSourceLineIsAmbiguous(t *testing.T) {
	// The fixture supplies the ambiguity itself: nvcc puts the three parameter
	// loads of scale on the kernel's declaration line, so that line names no
	// one PTX instruction. The SASS carries no PTX line of its own, so nothing
	// identifies which of the three this first instruction came from. Choosing
	// by opcode text would name PTX line 27 with no more evidence than line 29.
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, "", ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel("_Z5scalePfPKff")
	if k == nil {
		t.Fatal("_Z5scalePfPKff not decoded")
	}
	inst := k.Instruction(0)
	if inst == nil {
		t.Fatal("no instruction at offset 0")
	}
	if inst.SourceLine != 11 {
		t.Errorf("SourceLine = %d, want 11; the source tier is unaffected", inst.SourceLine)
	}
	if inst.PTXLine != 0 {
		t.Errorf("PTXLine = %d, want 0 (unknown)", inst.PTXLine)
	}
	// The same kernel's exit sits on a source line exactly one PTX
	// instruction claims, and is identified. Without this the test above
	// would also pass on a loader that never reported a PTX line at all.
	exit := k.Instruction(0xe0)
	if exit == nil {
		t.Fatal("no instruction at offset 0xe0")
	}
	if exit.PTXLine != 45 {
		t.Errorf("PTXLine at 0xe0 = %d, want 45", exit.PTXLine)
	}
}

func TestUnclaimedSourceLineIsRefused(t *testing.T) {
	// The SASS attributes scale's exit to kernels.cu:14, where the PTX puts
	// its "ret". Moving that one .loc back onto line 13 leaves line 14 claimed
	// by the disassembly and by nothing in the PTX: the two cannot be one
	// compilation. The previous behaviour was to fuse them and report those
	// instructions at the PTX line of the last "ret" it could find.
	ptx := movePTXLines(t, func(n int) int {
		if n == 14 {
			return 13
		}
		return n
	})
	_, err := cudacapture.LoadCode(loadFixtureCapture(t, ptx, ""))
	if err == nil {
		t.Fatal("LoadCode accepted a SASS claiming a source line the PTX never names")
	}
	for _, want := range []string{"not from one compilation", "kernels.cu", "14"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// shiftedPTX returns the fixture PTX with every source line moved by two, as a
// second compilation of a file of the same name would have.
func shiftedPTX(t *testing.T) string {
	t.Helper()
	return movePTXLines(t, func(n int) int { return n + 2 })
}

func TestFusedPairIsRefused(t *testing.T) {
	_, err := cudacapture.LoadCode(loadFixtureCapture(t, shiftedPTX(t), ""))
	if err == nil {
		t.Fatal("LoadCode fused a PTX and SASS whose line tables disagree")
	}
	if !strings.Contains(err.Error(), "not from one compilation") {
		t.Errorf("error = %v, want a binding failure", err)
	}
}

func TestMatchedPairLoads(t *testing.T) {
	// The gate must not refuse the pair it is meant to admit.
	if _, err := cudacapture.LoadCode(loadFixtureCapture(t, "", "")); err != nil {
		t.Fatalf("LoadCode on the matched fixture: %v", err)
	}
}
