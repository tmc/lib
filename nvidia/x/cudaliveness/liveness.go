package cudaliveness

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// ErrNoControls reports that liveness was asserted without any control. A
// probe reporting nothing, with nothing to show it would have reported
// something, is the failure this package exists to catch.
var ErrNoControls = errors.New("cudaliveness: no controls: a clean report with no control is not a measurement")

// ErrUnexplainedCount reports a [Counted] control whose Want was not derived.
// A count matching a number nobody can account for confirms only that the two
// numbers agree.
var ErrUnexplainedCount = errors.New("cudaliveness: expected count has no derivation")

// ErrNotLive reports that a control ran and failed, so the probe did not
// observe what it was pointed at. Results from the same run are void, not
// negative.
var ErrNotLive = errors.New("cudaliveness: probe was not live")

// A Control is one prediction about a live probe's report, paired with the
// outcome of the run that tested it.
//
// Check reports why the control failed, or nil if it held. A control that did
// not run reports an error matching [cudaerrors.ErrHardwarePending]; one that
// ran and failed reports an error matching [ErrNotLive].
type Control interface {
	Name() string
	Check() error
}

// Planted is a control in which a defect is deliberately introduced into the
// subject at a known site. A live probe must report it.
//
// Planted is the cheapest control and the weakest. It establishes that the
// probe reports something, not that it observed the intended kernel: a probe
// attached to the wrong target may still report a defect planted broadly
// enough. Pair it with [Counted].
type Planted struct {
	Site     string // where the defect was planted, for the failure message
	Reported int    // defects the probe reported for this run
	Ran      bool   // whether the planted run was made at all
}

// Name returns the control's identity in a failure message.
func (p Planted) Name() string { return "planted defect at " + p.Site }

// Check implements [Control].
func (p Planted) Check() error {
	if !p.Ran {
		return fmt.Errorf("%s: %w", p.Name(), cudaerrors.ErrHardwarePending)
	}
	if p.Reported < 1 {
		return fmt.Errorf("%s: probe reported %d defects, want at least 1: %w",
			p.Name(), p.Reported, ErrNotLive)
	}
	return nil
}

// Counted is a control on the probe's own instrumentation counter, checked
// against a count derived from the workload before the run.
//
// Want must be exact. "Greater than zero" is not a prediction: a counter can
// be non-zero because the probe instrumented a warm-up launch, a runtime's own
// kernels, or a different target entirely. Why records the derivation and is
// required, because a count nobody can account for is not evidence even when
// it matches.
//
// If the probe exposes no counter, do not use Counted. An absent counter is
// not a passing control.
type Counted struct {
	Site string // what was counted, for the failure message
	Want int64  // the derived count a live probe must report
	Got  int64  // the count it did report
	Why  string // how Want was derived; required
	Ran  bool   // whether the counted run was made at all
}

// Name returns the control's identity in a failure message.
func (c Counted) Name() string { return "instrumentation count at " + c.Site }

// Check implements [Control].
func (c Counted) Check() error {
	if !c.Ran {
		return fmt.Errorf("%s: %w", c.Name(), cudaerrors.ErrHardwarePending)
	}
	if c.Why == "" {
		return fmt.Errorf("%s: want %d: %w", c.Name(), c.Want, ErrUnexplainedCount)
	}
	if c.Got != c.Want {
		return fmt.Errorf("%s: probe counted %d, want %d (%s): %w",
			c.Name(), c.Got, c.Want, c.Why, ErrNotLive)
	}
	return nil
}

// Refused is a control in which the probe is handed an instrument it must
// reject, such as an object built for the wrong architecture.
//
// It tests the half the other controls cannot reach. [Planted] and [Counted]
// both run through the accepting path; if that path silently does nothing,
// they fail without saying why. A probe that accepts an instrument it should
// have rejected never consulted the instrument at all.
type Refused struct {
	Site     string // what was offered and must be rejected
	Rejected bool   // whether the probe rejected it
	Ran      bool   // whether the refusal run was made at all
}

// Name returns the control's identity in a failure message.
func (r Refused) Name() string { return "refusal of " + r.Site }

// Check implements [Control].
func (r Refused) Check() error {
	if !r.Ran {
		return fmt.Errorf("%s: %w", r.Name(), cudaerrors.ErrHardwarePending)
	}
	if !r.Rejected {
		return fmt.Errorf("%s: probe accepted an instrument it must reject: %w",
			r.Name(), ErrNotLive)
	}
	return nil
}

// Assert reports whether the probe was live for the run that produced the
// given controls. It returns nil only when every control ran and held.
//
// Assert reports [ErrNoControls] when given none. Otherwise it joins every
// failure rather than returning the first, because an unrun control and a
// failed one call for different work and a caller seeing only the first will
// do the wrong one.
//
// A non-nil error voids the run's other results. It does not make them
// negative: nothing was measured.
func Assert(controls ...Control) error {
	if len(controls) == 0 {
		return ErrNoControls
	}
	var errs []error
	for _, c := range controls {
		if err := c.Check(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Adjacent is a control in which a mutation is made at a site the probe must
// not see: outside the instrumented window, or at a neighbouring site.
// A live probe with a correct window reports nothing for it.
//
// Adjacent is the negative half of [Planted] and the two are only meaningful
// together. Planted alone passes for a probe whose window is too wide -- one
// watching the whole module reports the planted defect and would report
// anything else as well, so its clean results carry no information about the
// site. A probe that reports both mutations has not localized anything.
type Adjacent struct {
	Site     string // the site that must not be observed
	Reported int    // records the probe produced for it
	Ran      bool   // whether the adjacent run was made at all
}

// Name returns the control's identity in a failure message.
func (a Adjacent) Name() string { return "adjacent mutation at " + a.Site }

// Check implements [Control].
func (a Adjacent) Check() error {
	if !a.Ran {
		return fmt.Errorf("%s: %w", a.Name(), cudaerrors.ErrHardwarePending)
	}
	if a.Reported != 0 {
		return fmt.Errorf("%s: probe reported %d records for a site outside its window, want 0: %w",
			a.Name(), a.Reported, ErrNotLive)
	}
	return nil
}

// Undisturbed is a control comparing the subject's own output with the probe
// attached and detached. The two digests must be identical.
//
// It answers a different question from the others. They ask whether the probe
// observed the subject; this asks whether what it observed was still the
// subject. A probe that changes the computation it watches -- by perturbing
// register pressure, occupancy, or launch order -- reports faithfully about a
// program the caller did not run, and every other control here passes while it
// does so.
//
// Digests are compared as given. An empty digest on either side is an unrun
// control, not a match.
type Undisturbed struct {
	Site        string // what was compared, for the failure message
	ObserverOff string // endpoint digest with the probe detached
	ObserverOn  string // endpoint digest with the probe attached
	Ran         bool   // whether both runs were made at all
}

// Name returns the control's identity in a failure message.
func (u Undisturbed) Name() string { return "observer-off endpoint identity at " + u.Site }

// Check implements [Control].
func (u Undisturbed) Check() error {
	if !u.Ran || u.ObserverOff == "" || u.ObserverOn == "" {
		return fmt.Errorf("%s: %w", u.Name(), cudaerrors.ErrHardwarePending)
	}
	if u.ObserverOff != u.ObserverOn {
		return fmt.Errorf("%s: observer-off %s, observer-on %s: the probe changed the subject: %w",
			u.Name(), u.ObserverOff, u.ObserverOn, ErrNotLive)
	}
	return nil
}
