package cudacapture_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestFusionFailureKeepsTheKernelTier is the control on tiering: a PTX and SASS
// that disagree about the line table must still yield the tier that does not
// depend on them agreeing. Refusing outright made cudaprof unusable on profiles
// that pprof itself reads correctly, and the kernel ranking a reader wants from
// such a profile is built from the CUBIN sections and the disassembly alone.
func TestFusionFailureKeepsTheKernelTier(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, shiftedPTX(t), ""))
	var fusion *cudacapture.SourceFusionError
	if !errors.As(err, &fusion) {
		t.Fatalf("LoadCode error = %v, want *SourceFusionError", err)
	}
	if code == nil {
		t.Fatal("LoadCode withheld the kernel tier on a source-fusion failure")
	}
	if len(code.Kernels) == 0 {
		t.Fatal("no kernels; a ranking cannot be rendered from this")
	}
	if fusion.Source != "kernels.cu" {
		t.Errorf("Source = %q, want kernels.cu", fusion.Source)
	}
	if len(fusion.Unclaimed) == 0 {
		t.Error("Unclaimed is empty; the error names nothing the caller can check")
	}

	// The other half of the contract: what the failure invalidates must be
	// absent, not merely doubted. A retained SourceLine here is a guess in the
	// shape of a measurement, which is the defect the fusion check exists for.
	for _, k := range code.Kernels {
		if len(k.Instructions) == 0 {
			t.Errorf("%s: no instructions", k.Name)
		}
		for _, inst := range k.Instructions {
			if inst.Text == "" {
				t.Errorf("%s+%#x: empty instruction text", k.Name, inst.Offset)
			}
			if inst.SourceLine != 0 || inst.SourceFile != "" {
				t.Errorf("%s+%#x: source attribution %s:%d survived an unfused load",
					k.Name, inst.Offset, inst.SourceFile, inst.SourceLine)
			}
			if inst.PTXLine != 0 {
				t.Errorf("%s+%#x: PTXLine %d survived an unfused load", k.Name, inst.Offset, inst.PTXLine)
			}
			if len(inst.InlinedAt) != 0 {
				t.Errorf("%s+%#x: inline chain survived an unfused load", k.Name, inst.Offset)
			}
		}
	}
}

// TestFusedLoadKeepsAttribution is the near-miss half of the dissociation: the
// degrade must not fire on a pair that does fuse, or every profile would lose
// source attribution and the test above would pass vacuously.
func TestFusedLoadKeepsAttribution(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, "", ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	attributed := 0
	for _, k := range code.Kernels {
		for _, inst := range k.Instructions {
			if inst.SourceLine > 0 {
				attributed++
			}
		}
	}
	if attributed == 0 {
		t.Error("a matched pair produced no source attribution at all")
	}
}
