package cudadevicetable

import "fmt"

// Answer is what is known about one feature at one target, together with
// whether anything actually looked.
//
// A bool cannot hold this. "The assembler rejected it" and "no assembler has
// been pointed at it" are different facts with different right responses, and
// collapsing them into false is how a target nobody has probed acquires a
// confident row of negatives. [Target.Supports] keeps them apart with an
// error; Answer keeps them apart in a value, so a caller can carry the
// distinction instead of handling it at the point of lookup.
type Answer struct {
	Target  string
	Feature string

	// Supported is meaningful only when Measured is true.
	Supported bool

	// Measured reports whether an instrument produced Supported. When it is
	// false, Supported is false because a zero value has to be something,
	// not because anything said so.
	Measured bool

	// Source names the instrument, empty when Measured is false.
	Source string

	// Diagnostic is the assembler's rejection message, when there is one.
	Diagnostic string
}

// Fallback policies for [Answer.Or]. Which one is right is decided by what
// catches the mistake, not by optimism.
const (
	// AssumeSupported is for a feature whose misuse the assembler catches
	// downstream: emit the instruction, and a target that does not have it
	// fails to compile, loudly, with the real diagnostic. Refusing instead
	// would make every new GPU less capable than an old one until someone
	// regenerates a table, which is the wrong failure for hardware that is
	// almost certainly a superset of what came before.
	AssumeSupported = true

	// AssumeAbsent is for a feature whose absence would otherwise flow
	// onward silently -- anything feeding a geometry, occupancy or roofline
	// number, where a wrong value is wrong in the way that reads as right.
	AssumeAbsent = false
)

// Ask returns what the table knows about feature at target. It never guesses:
// a target the table does not cover, or a feature it never probed, comes back
// with Measured false. Use [Answer.Or] to state the fallback explicitly.
func Ask(target, feature string) Answer {
	a := Answer{Target: target, Feature: feature}
	t, ok := LookupTarget(target)
	if !ok {
		return a
	}
	f, ok := t.Features[feature]
	if !ok || f.Source != Probed {
		return a
	}
	a.Supported, a.Measured, a.Diagnostic = f.Supported, true, f.Diagnostic
	if t.Probe != nil {
		a.Source = fmt.Sprintf("%s %s (%s, %s)", t.Probe.Tool, t.Probe.ToolVersion, t.Probe.Host, t.Probe.ProbedAt)
	}
	return a
}

// Or returns the measured answer, or assume when nothing measured it. The
// argument is the policy, written at the call site, so that reading the call
// tells you what happens on hardware that did not exist when the table was
// generated.
func (a Answer) Or(assume bool) bool {
	if a.Measured {
		return a.Supported
	}
	return assume
}

// Why explains the answer in one clause, suitable for embedding in an error.
func (a Answer) Why() string {
	switch {
	case !a.Measured:
		return fmt.Sprintf("no instrument has probed %q at %s", a.Feature, a.Target)
	case a.Supported:
		return fmt.Sprintf("%s accepts %q (%s)", a.Target, a.Feature, a.Source)
	case a.Diagnostic != "":
		return fmt.Sprintf("%s rejects %q: %s", a.Target, a.Feature, a.Diagnostic)
	default:
		return fmt.Sprintf("%s rejects %q (%s)", a.Target, a.Feature, a.Source)
	}
}

func (a Answer) String() string { return a.Why() }
