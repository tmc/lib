package cudaliveness

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// ErrUnexplainedPrediction reports an [Executed] control whose Predicted
// output was not stated or not derived. An output matching a prediction
// nobody can account for confirms only that the two digests agree.
var ErrUnexplainedPrediction = errors.New("cudaliveness: predicted output has no derivation")

// Executed is a control on a patched kernel image: it asks whether the
// patched bytes executed, or whether the patch was applied to bytes the
// device never ran.
//
// Three things must hold, and each fails for its own reason. Predicted is
// the digest the patched launch must produce, stated before the run, and Why
// records how it was derived; both are required. Unpatched is the digest of
// the same launch with the unpatched image, and Patched must differ from it:
// a patch whose output equals the unpatched output was inert or never ran,
// and that is a failure even when Patched also equals Predicted. Mirrors
// lists copies of the kernel text the patch path did not update, such as
// cudasass.PatchedImage.UnpatchedMirrors (written as a plain name, not a
// doc link, since this package does not import cudasass). With any mirror
// present no run can show which copy a later load will use, so the control
// fails without comparing outputs.
//
// Digests are compared as given, as in [Undisturbed]. Each digest must cover
// the same projection of the output: the whole output, or the part the
// derivation says the patch changes. An empty digest on either side is an
// unrun control, never a match.
//
// The prediction and the mirrors are known before any launch and no run can
// repair either, so they are checked first: a control with no derivation or
// with a mirror fails whether or not it ran. The run state is checked next,
// then the two comparisons.
//
// Executed says the patched bytes ran. It does not say the patch is correct.
type Executed struct {
	Site      string   // the patched kernel and patch site, for the failure message
	Predicted string   // digest the patched launch must produce, stated before the run
	Why       string   // how Predicted was derived; required
	Unpatched string   // digest of the same launch with the unpatched image
	Patched   string   // digest of the launch with the patched image
	Mirrors   []string // copies of the kernel text the patch did not update
	Ran       bool     // whether both launches were made at all
}

// Name returns the control's identity in a failure message.
func (e Executed) Name() string { return "patch execution at " + e.Site }

// Check implements [Control].
func (e Executed) Check() error {
	if e.Predicted == "" {
		return fmt.Errorf("%s: no predicted output: %w", e.Name(), ErrUnexplainedPrediction)
	}
	if e.Why == "" {
		return fmt.Errorf("%s: predicted %s: %w", e.Name(), e.Predicted, ErrUnexplainedPrediction)
	}
	if len(e.Mirrors) > 0 {
		return fmt.Errorf("%s: the image keeps unpatched copies of the text %q, so no run can show which copy a later load will use: %w",
			e.Name(), e.Mirrors, ErrNotLive)
	}
	if !e.Ran || e.Patched == "" || e.Unpatched == "" {
		return fmt.Errorf("%s: %w", e.Name(), cudaerrors.ErrHardwarePending)
	}
	if e.Patched == e.Unpatched {
		return fmt.Errorf("%s: patched output %s equals the unpatched output: the patch was inert or never ran: %w",
			e.Name(), e.Patched, ErrNotLive)
	}
	if e.Patched != e.Predicted {
		return fmt.Errorf("%s: patched output %s, predicted %s (%s): %w",
			e.Name(), e.Patched, e.Predicted, e.Why, ErrNotLive)
	}
	return nil
}
