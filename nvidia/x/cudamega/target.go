package cudamega

import (
	"fmt"
	"strconv"
	"strings"
)

// A Target is the device architecture a program is emitted for, spelled the
// way the toolchain spells it.
//
// It is an input to program construction, not a property of the host this
// process runs on. Emitted text and PTX need no device: a program can be
// generated for sm_80 on a machine that has no GPU at all. The authorized
// host says where a program may execute, which is a separate question and a
// separate artifact.
//
// The zero Target is not usable; [NewProgram] refuses it.
type Target string

// The targets this package has goldens for. Others parse and are accepted;
// these are the ones something in testdata was built from.
const (
	SM80   Target = "sm_80"   // the floor the emitted primitives require
	SM90   Target = "sm_90"   // the floor barrier.cluster requires
	SM121a Target = "sm_121a" // the qualification host, arch-specific
)

// Capability returns the target's compute capability.
//
// The spelling is "sm_" followed by the capability digits, with the last
// digit the minor version, and optionally a trailing architecture-specific
// suffix: sm_80 is 8.0, sm_121a is 12.1 with suffix "a". The suffix selects
// features within one capability and does not change the number, so it is
// reported separately rather than folded into the comparison.
func (t Target) Capability() (major, minor int, suffix string, err error) {
	s, ok := strings.CutPrefix(string(t), "sm_")
	if !ok {
		return 0, 0, "", fmt.Errorf("target %q is not spelled sm_<capability>", t)
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	digits, suffix := s[:i], s[i:]
	if len(digits) < 2 {
		return 0, 0, "", fmt.Errorf("target %q has no capability digits", t)
	}
	for _, r := range suffix {
		if r < 'a' || r > 'z' {
			return 0, 0, "", fmt.Errorf("target %q has an unrecognized suffix %q", t, suffix)
		}
	}
	major, err = strconv.Atoi(digits[:len(digits)-1])
	if err != nil {
		return 0, 0, "", fmt.Errorf("target %q: %w", t, err)
	}
	minor, _ = strconv.Atoi(digits[len(digits)-1:])
	return major, minor, suffix, nil
}

// atLeast reports whether the target is at or above a capability.
func (t Target) atLeast(major, minor int) (bool, error) {
	ma, mi, _, err := t.Capability()
	if err != nil {
		return false, err
	}
	return ma > major || ma == major && mi >= minor, nil
}
