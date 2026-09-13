package cudatool

import (
	"errors"
	"fmt"
	"strings"
)

// Flag is one option a command line means to emit, with the argument that
// follows it when the option takes one.
//
// An empty Value means the flag stands alone, so a Flag describes its own
// arity: the command line built from a Flag and the arity checked against the
// tool cannot disagree.
type Flag struct {
	Name  string
	Value string
}

// Arity reports how the flag is spelled: [Value] when an argument follows it,
// [Bare] otherwise.
func (f Flag) Arity() Arity {
	if f.Value != "" {
		return Value
	}
	return Bare
}

// Args returns the command-line form of flags: a bare flag contributes its
// name, and a flag with a value contributes its name followed by that value as
// a separate argument.
func Args(flags ...Flag) []string {
	var args []string
	for _, f := range flags {
		args = append(args, f.Name)
		if f.Value != "" {
			args = append(args, f.Value)
		}
	}
	return args
}

// ErrUnsupportedOption means the resolved tool does not accept an option as the
// command line spells it, either because the option is absent from that toolkit
// or because its arity differs.
var ErrUnsupportedOption = errors.New("option not supported by this toolkit")

// Check reports whether the tool accepts every flag as spelled. The error wraps
// [ErrUnsupportedOption] and names the tool, the toolkit version, where the
// option set came from, and every flag at fault, so one error describes the
// whole difference rather than the first one found.
//
// An arity mismatch is a failure in both directions: a tool that spells an
// option bare rejects the whole command line when an argument follows it, and
// one that requires an argument consumes the next word when none is given.
func (c Caps) Check(flags ...Flag) error {
	var bad []string
	for _, f := range flags {
		arity, ok := c.Arity(f.Name)
		switch {
		case !ok:
			bad = append(bad, f.Name+" is not accepted")
		case arity == Value && f.Arity() == Bare:
			bad = append(bad, f.Name+" takes an argument, the command line emits it bare")
		case arity == Bare && f.Arity() == Value:
			bad = append(bad, fmt.Sprintf("%s takes no argument, the command line gives it %q", f.Name, f.Value))
		}
	}
	return c.unsupported(bad)
}

// CheckNames reports whether the tool accepts every named option, saying
// nothing about arity. It is for options a caller passes through as text, where
// the argument form is the caller's business but the name is still either
// accepted or rejected. Arguments that do not begin with a dash are ignored,
// since in an argument list they are the values of the options before them.
func (c Caps) CheckNames(args ...string) error {
	var bad []string
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			continue
		}
		name, _, _ := strings.Cut(arg, "=")
		if !c.Accepts(name) {
			bad = append(bad, name+" is not accepted")
		}
	}
	return c.unsupported(bad)
}

// unsupported reports the faults found in one command line, naming the tool,
// the toolkit version and where the option set came from.
func (c Caps) unsupported(bad []string) error {
	if len(bad) == 0 {
		return nil
	}
	return fmt.Errorf("%s %v (%s): %w: %s", c.Tool, c.Version, c.origin(), ErrUnsupportedOption, strings.Join(bad, "; "))
}

// origin names where Options came from, so an error a caller reads says whether
// the answer was measured from this tool or recorded for another version.
func (c Caps) origin() string {
	switch {
	case c.Probed:
		return "from --help"
	case c.Source.Compare(c.Version) != 0:
		return "recorded for " + c.Source.String()
	}
	return "recorded"
}
