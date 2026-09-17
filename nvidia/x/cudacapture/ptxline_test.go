package cudacapture_test

import (
	"encoding/hex"
	"os"
	"path/filepath"
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

// ambiguousPTX puts two instructions on source line 2, so the source line no
// longer identifies one of them.
const ambiguousPTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"

.visible .entry compute(
	.param .u64 a
)
{
	.loc 1 2 0;
	mov.u64 %rd1, %rd2;
	add.u64 %rd1, %rd1, 1;
	.loc 1 3 0;
	ret;
}
`

func TestPTXLineUnknownWhenSourceLineIsAmbiguous(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, ambiguousPTX, ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	inst := code.Kernels[0].Instruction(0)
	if inst == nil {
		t.Fatal("no instruction at offset 0")
	}
	if inst.SourceLine != 2 {
		t.Errorf("SourceLine = %d, want 2; the source tier is unaffected", inst.SourceLine)
	}
	// Two PTX instructions claim line 2 and the SASS carries no PTX line, so
	// nothing identifies which one this MOV came from. Choosing by opcode text
	// would name line 12 with no more evidence than line 13.
	if inst.PTXLine != 0 {
		t.Errorf("PTXLine = %d, want 0 (unknown)", inst.PTXLine)
	}
}

// unmappedRetPTX leaves source line 3, where the SASS RET sits, with no PTX
// instruction at all.
const unmappedRetPTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"

.visible .entry compute(
	.param .u64 a
)
{
	.loc 1 2 0;
	mov.u64 %rd1, %rd2;
	ret;
}
`

func TestUnclaimedSourceLineIsRefused(t *testing.T) {
	// The PTX names only line 2; the SASS attributes its RET to line 3. The
	// two cannot be one compilation, and the previous behaviour was to fuse
	// them and report the RET at the PTX line of the last "ret" it could find.
	_, err := cudacapture.LoadCode(loadFixtureCapture(t, unmappedRetPTX, ""))
	if err == nil {
		t.Fatal("LoadCode accepted a SASS claiming a source line the PTX never names")
	}
	for _, want := range []string{"not from one compilation", "kernels.cu", "[3]"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// shiftedPTX is the fixture PTX with every source line moved by two, as a
// second compilation of a file of the same name would have.
const shiftedPTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"

.visible .entry compute(
	.param .u64 a
)
{
	.loc 1 4 0;
	mov.u64 %rd1, %rd2;
	.loc 1 5 0;
	ret;
}
`

func TestFusedPairIsRefused(t *testing.T) {
	_, err := cudacapture.LoadCode(loadFixtureCapture(t, shiftedPTX, ""))
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
