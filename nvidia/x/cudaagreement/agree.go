package cudaagreement

import (
	"errors"
	"fmt"
)

// An Arm pairs one executor's record with the bytes its run left in the plan's
// output buffer. Output is read back by the caller after the stream drains;
// this package never reads device memory.
type Arm struct {
	Record *Record
	Output []byte
}

// Agree reports whether every arm left identical bytes in its output buffer.
// The comparison is bitwise and has no tolerance, which is the only comparison
// an integer path admits. A float path would need a tolerance derived from the
// kernel's own rounding, and this package does not supply one.
//
// Agree also refuses a comparison with fewer than two arms or with an empty
// output: identical emptiness is not agreement.
func Agree(arms []Arm) error {
	if len(arms) < 2 {
		return fmt.Errorf("cudaagreement: %d arms, need at least 2", len(arms))
	}
	first := arms[0]
	if len(first.Output) == 0 {
		return fmt.Errorf("cudaagreement: arm %s produced no output bytes", armName(first))
	}
	for _, arm := range arms[1:] {
		if len(arm.Output) != len(first.Output) {
			return fmt.Errorf("cudaagreement: arm %s produced %d output bytes, arm %s produced %d",
				armName(arm), len(arm.Output), armName(first), len(first.Output))
		}
		for i := range arm.Output {
			if arm.Output[i] != first.Output[i] {
				return fmt.Errorf("cudaagreement: arms %s and %s differ at output byte %d: 0x%02x vs 0x%02x",
					armName(first), armName(arm), i, first.Output[i], arm.Output[i])
			}
		}
	}
	return nil
}

// Verify is the gate. It holds when every arm ran every step on the backend
// that arm requested, every arm ran the same steps over the same artifacts,
// and all arms left identical output bytes.
//
// A step that fell back fails Verify even when the outputs match. Outputs that
// agree only where an arm declined the work are evidence about the arm that
// did it, not about the one that did not.
func Verify(arms []Arm) error {
	if len(arms) < 2 {
		return fmt.Errorf("cudaagreement: %d arms, need at least 2", len(arms))
	}
	for _, arm := range arms {
		if arm.Record == nil {
			return errors.New("cudaagreement: arm has no execution record")
		}
		if err := arm.Record.ExecutedOn(arm.Record.Requested); err != nil {
			return err
		}
	}
	if err := sameWork(arms); err != nil {
		return err
	}
	return Agree(arms)
}

// sameWork checks that the arms ran the same plan: the same step names in the
// same order over the same artifact digests, at the same geometry wherever
// both arms launched the step's artifact. Two arms that agree on the output
// of different work have not agreed on anything.
func sameWork(arms []Arm) error {
	first := arms[0].Record
	for _, arm := range arms[1:] {
		r := arm.Record
		if len(r.Steps) != len(first.Steps) {
			return fmt.Errorf("cudaagreement: arm %s ran %d steps, arm %s ran %d", r.Arm, len(r.Steps), first.Arm, len(first.Steps))
		}
		for i := range r.Steps {
			a, b := first.Steps[i], r.Steps[i]
			if a.Step != b.Step {
				return fmt.Errorf("cudaagreement: step %d is %s in arm %s and %s in arm %s", i, a.Step, first.Arm, b.Step, r.Arm)
			}
			if a.Digest != b.Digest {
				return fmt.Errorf("cudaagreement: step %s used artifact %s in arm %s and %s in arm %s", a.Step, a.Digest, first.Arm, b.Digest, r.Arm)
			}
			// Geometry identifies a launch of the step's own artifact. A
			// step run inside another program has that program's geometry,
			// so geometry is compared only where both arms ran the artifact.
			if a.Program == "" && b.Program == "" && (a.Grid != b.Grid || a.Block != b.Block) {
				return fmt.Errorf("cudaagreement: step %s ran %dx%d in arm %s and %dx%d in arm %s", a.Step, a.Grid, a.Block, first.Arm, b.Grid, b.Block, r.Arm)
			}
		}
	}
	return nil
}

func armName(a Arm) string {
	if a.Record == nil {
		return "<unrecorded>"
	}
	return a.Record.Arm
}
