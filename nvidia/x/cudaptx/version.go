package cudaptx

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrVersionUnaudited reports that a module's .version is newer than
// [PTXISAVersion], the ISA version the generated tables describe. The
// instruction tables record the version each instruction was introduced in, so
// they can place a lower bound on a module they cover and can say nothing about
// one they do not: a mnemonic added after [PTXISAVersion] is absent from the
// table, and an absent mnemonic is indistinguishable from an unconstrained one.
//
// [Module.AuditVersion] returns an error wrapping ErrVersionUnaudited rather
// than reporting success, so that a module the tables cannot cover is an
// explicit unknown instead of a pass. Callers that want to proceed anyway test
// for it with [errors.Is].
var ErrVersionUnaudited = errors.New("cudaptx: module .version is newer than the generated tables")

// isaVersion is a PTX ISA version as written in a .version directive.
type isaVersion struct {
	major int
	minor int
}

func (v isaVersion) String() string { return fmt.Sprintf("%d.%d", v.major, v.minor) }

// less reports whether v precedes w. Minor numbers are compared as integers,
// not as decimal fractions: PTX ISA 8.10 would follow 8.9.
func (v isaVersion) less(w isaVersion) bool {
	if v.major != w.major {
		return v.major < w.major
	}
	return v.minor < w.minor
}

// parseISAVersion parses a .version argument such as "8.8".
func parseISAVersion(s string) (isaVersion, error) {
	s = strings.TrimSpace(s)
	major, minor, ok := strings.Cut(s, ".")
	if !ok {
		return isaVersion{}, fmt.Errorf("malformed PTX ISA version %q: want major.minor", s)
	}
	maj, err := strconv.Atoi(major)
	if err != nil || maj < 0 {
		return isaVersion{}, fmt.Errorf("malformed PTX ISA version %q: bad major number", s)
	}
	min, err := strconv.Atoi(minor)
	if err != nil || min < 0 {
		return isaVersion{}, fmt.Errorf("malformed PTX ISA version %q: bad minor number", s)
	}
	return isaVersion{maj, min}, nil
}

// tablesISAVersion is [PTXISAVersion] parsed, or the zero value if the
// generated constant is malformed, in which case every module counts as
// unaudited.
var tablesISAVersion, _ = parseISAVersion(PTXISAVersion)

// targetArchitecture returns the architecture named by a .target argument.
// The directive takes a comma-separated list whose first entry is the
// architecture and whose remaining entries are extension targets such as
// "debug" or "map_f64_to_f32"; the extensions are not interpreted here.
// The second result reports whether the architecture is well formed, meaning
// "sm_" followed by digits and an optional feature suffix such as "sm_121a".
func targetArchitecture(target string) (string, bool) {
	arch, _, _ := strings.Cut(target, ",")
	arch = strings.ToLower(strings.TrimSpace(arch))
	rest, ok := strings.CutPrefix(arch, "sm_")
	if !ok || rest == "" || rest[0] < '0' || rest[0] > '9' {
		return arch, false
	}
	digits := 0
	for digits < len(rest) && rest[digits] >= '0' && rest[digits] <= '9' {
		digits++
	}
	suffix := rest[digits:]
	if suffix != "" && !isTargetSuffix(suffix) {
		return arch, false
	}
	return arch, true
}

// isTargetSuffix reports whether s is a feature suffix on an sm_ architecture,
// such as the "a" of sm_121a or the "f" of sm_103f. Any all-letter suffix is
// accepted: the set grows with each architecture, and rejecting an unfamiliar
// one would fail a module that is merely newer than this code.
func isTargetSuffix(s string) bool {
	for _, c := range s {
		if c < 'a' || c > 'z' {
			return false
		}
	}
	return true
}

// AuditVersion checks the module header and the PTX ISA version its
// instructions require. If kernelName is empty it checks every kernel.
//
// It reports three kinds of defect, all of them decidable from the module text:
// a missing or malformed .version or .target directive; a .version newer than
// the generated tables describe, which is returned wrapping
// [ErrVersionUnaudited]; and an instruction introduced in a PTX ISA version
// later than the .version the module declares.
//
// AuditVersion is the .version counterpart of [Module.AuditArchitecture],
// which checks the same instructions against the .target architecture. Neither
// predicts whether a given driver will accept the module: the PTX ISA version
// a driver supports is a property of the loading toolkit, not of the module,
// and is not recorded here. Use [Module.ExplainLoadError] to interpret a load
// that the driver has already rejected.
func (m *Module) AuditVersion(kernelName string) error {
	if m == nil {
		return fmt.Errorf("cudaptx: nil module")
	}
	if m.Version == "" {
		return fmt.Errorf("cudaptx: module has no .version directive")
	}
	version, err := parseISAVersion(m.Version)
	if err != nil {
		return fmt.Errorf("cudaptx: module .version: %w", err)
	}
	if m.Target == "" {
		return fmt.Errorf("cudaptx: module has no .target directive")
	}
	if arch, ok := targetArchitecture(m.Target); !ok {
		return fmt.Errorf("cudaptx: module .target %q: malformed architecture %q", m.Target, arch)
	}
	if tablesISAVersion.less(version) {
		return fmt.Errorf("cudaptx: module .version %s, tables describe PTX ISA %s: %w",
			version, PTXISAVersion, ErrVersionUnaudited)
	}

	kernels := m.Kernels
	if kernelName != "" {
		k := m.Kernel(kernelName)
		if k == nil {
			return fmt.Errorf("cudaptx: kernel %q not found", kernelName)
		}
		kernels = []*Kernel{k}
	}
	for _, k := range kernels {
		if err := k.auditVersion(version); err != nil {
			return err
		}
	}
	return nil
}

// auditVersion checks that every instruction in the kernel was introduced no
// later than version. A mnemonic the tables do not record is skipped: it
// carries no lower bound, the same treatment [Kernel.AuditArchitecture] gives
// an instruction with no recorded minimum target.
func (k *Kernel) auditVersion(version isaVersion) error {
	if k == nil {
		return nil
	}
	for _, inst := range k.Instructions {
		min := InstructionMinPTX(inst.Mnemonic)
		if min == "" {
			continue
		}
		req, err := parseISAVersion(min)
		if err != nil {
			continue
		}
		if version.less(req) {
			return fmt.Errorf("cudaptx: kernel %q: instruction %q (%s) requires PTX ISA %s, module declares %s",
				k.Name, inst.Opcode, inst.Mnemonic, req, version)
		}
	}
	return nil
}

// ExplainLoadError wraps err with the header a driver rejected. It is meant
// for CUDA_ERROR_UNSUPPORTED_PTX_VERSION from a module load, which names
// neither the version that was rejected nor the version the driver wanted:
//
//	if err := mod.LoadData(src); errors.Is(err, driver.ErrUnsupportedPtxVersion) {
//		return parsed.ExplainLoadError(err)
//	}
//
// The rejected .version and .target come from the module text. The cause does
// not: a driver reports no PTX ISA version of its own, so the message names
// the toolkit mismatch as the usual explanation without asserting it. Nothing
// here is a check, and ExplainLoadError makes no claim about modules that
// load. err is returned unchanged if it is nil.
func (m *Module) ExplainLoadError(err error) error {
	if err == nil {
		return nil
	}
	version, target := m.headerText()
	return fmt.Errorf("cudaptx: module .version %s .target %s rejected: the loading toolkit's PTX compiler predates this .version; "+
		"pin LD_LIBRARY_PATH and PATH to the toolkit that produced the module, or compile it for an older .version: %w",
		version, target, err)
}

// headerText returns the module's .version and .target for a message,
// substituting a placeholder for a directive the module does not carry.
func (m *Module) headerText() (version, target string) {
	version, target = "<none>", "<none>"
	if m == nil {
		return version, target
	}
	if m.Version != "" {
		version = m.Version
	}
	if m.Target != "" {
		target = m.Target
	}
	return version, target
}
