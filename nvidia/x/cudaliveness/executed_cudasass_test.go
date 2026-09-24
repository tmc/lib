package cudaliveness_test

import (
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// The cudasass round-trip fixture: kernel intmix, built by CUDA 13.4 ptxas
// for sm_121a. See ../cudasass/testdata/roundtrip/README.md.
const (
	roundTripSASS  = "../cudasass/testdata/roundtrip/sm121a-roundtrip.sass"
	roundTripCubin = "../cudasass/testdata/roundtrip/sm121a-roundtrip.cubin.hex"
)

// TestExecutedFromPatchedImage feeds the mirrors a real patch reports into
// the control. The fixture is expected to carry a mirror, and the test fails
// rather than skips if it stops doing so, because the refusal it covers
// would then go untested.
func TestExecutedFromPatchedImage(t *testing.T) {
	sass, err := os.ReadFile(roundTripSASS)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.ParseString(string(sass))
	if err != nil {
		t.Fatalf("parse %s: %v", roundTripSASS, err)
	}
	fn := d.Function("intmix")
	if fn == nil {
		t.Fatalf("no intmix function in %s", roundTripSASS)
	}
	text, err := os.ReadFile(roundTripCubin)
	if err != nil {
		t.Fatal(err)
	}
	b, err := hex.DecodeString(strings.Join(strings.Fields(string(text)), ""))
	if err != nil {
		t.Fatalf("decode %s: %v", roundTripCubin, err)
	}
	im, err := cudasass.OpenCubin(b)
	if err != nil {
		t.Fatal(err)
	}

	index := -1
	for i, inst := range fn.Instructions {
		if inst.Address == 0x10 {
			index = i
		}
	}
	if index < 0 {
		t.Fatalf("no instruction at 0x10 in %s", fn.Name)
	}
	plan, err := cudasass.CheckMutation(fn, cudasass.SM121a,
		cudasass.Mutation{Index: index, Replacement: cudasass.Instruction{Opcode: "S2R", Operands: "R7, SR_TID.Y"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	patched, err := im.Apply("intmix", []cudasass.MutationPlan{plan})
	if err != nil {
		t.Fatal(err)
	}
	if len(patched.UnpatchedMirrors) == 0 {
		t.Fatalf("Apply reported no unpatched mirrors for %s; the fixture no longer exercises the mirror refusal", roundTripCubin)
	}

	// The oracle: cudasass's own load gate must agree the image is mirrored.
	if err := patched.Loadable(); !errors.Is(err, cudasass.ErrMirroredText) {
		t.Fatalf("Loadable() = %v, want %v; the control and the load gate disagree", err, cudasass.ErrMirroredText)
	}

	e := holding()
	e.Mirrors = patched.UnpatchedMirrors
	err = e.Check()
	if !errors.Is(err, cudaliveness.ErrNotLive) {
		t.Fatalf("Check() with mirrors %q = %v, want ErrNotLive", patched.UnpatchedMirrors, err)
	}
	for _, m := range patched.UnpatchedMirrors {
		if !strings.Contains(err.Error(), m) {
			t.Errorf("Check() = %q, want it to name mirror %q", err, m)
		}
	}

	// The same values without the mirrors hold, so the failure above was
	// caused by the mirrors and nothing else.
	e.Mirrors = nil
	if err := e.Check(); err != nil {
		t.Errorf("Check() without mirrors = %v, want nil", err)
	}
	t.Logf("mirrors: %q", patched.UnpatchedMirrors)
}
