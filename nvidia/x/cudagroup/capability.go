package cudagroup

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

var (
	// ErrCapabilityAbsent means a required record or attribute is missing.
	ErrCapabilityAbsent = errors.New("capability is absent")
	// ErrCapabilityUnverified means a required limit was not directly probed.
	ErrCapabilityUnverified = errors.New("capability is not directly probed")
	// ErrCapabilityUnsupported means probed limits cannot admit the fixture.
	ErrCapabilityUnsupported = errors.New("capability does not support the fixture")
	// ErrCapabilityInvalid means a record is malformed or names another
	// device architecture.
	ErrCapabilityInvalid = errors.New("capability record is invalid")
)

// GenerateReductionFixtureWithCapability generates the real reduction fixture
// and checks its declared launch against record before returning it. Refusal
// returns an empty Fixture. The caller must supply device facts; a target's
// assembler feature record cannot supply device limits.
//
// This checks the recorded warp size and block-thread limit, not the current
// device, toolchain, compiled kernel resources, or native execution. The older
// GenerateReductionFixture functions generate source without this admission.
func GenerateReductionFixtureWithCapability(group Group, target string, record cudadevicetable.Architecture) (Fixture, error) {
	f, err := GenerateReductionFixtureForTarget(group, target)
	if err != nil {
		return Fixture{}, err
	}
	if err := f.ValidateCapability(record); err != nil {
		return Fixture{}, err
	}
	return f, nil
}

// ValidateCapability checks this fixture's launch geometry and the supplied
// device record. Required limits must be positive, directly probed driver
// attributes with valid provenance. Missing and unverified limits do not imply
// unsupported hardware; use errors.Is with the ErrCapability errors to
// distinguish them.
//
// Architecture suffixes select instruction sets, not different device limits:
// sm_121a uses a sm_121 Architecture, while the exact fixture target must still
// be admitted by ValidateLaunch. This does not validate arbitrary source or
// prove instruction support, current-device availability, or native execution.
// The record is read only and must not be concurrently mutated by its caller.
func (f Fixture) ValidateCapability(record cudadevicetable.Architecture) error {
	if err := f.ValidateLaunch(); err != nil {
		return fmt.Errorf("validate reduction launch: %w", err)
	}
	if record.Architecture == "" && record.Probe == nil && len(record.Attributes) == 0 {
		return fmt.Errorf("%s: %w", f.Target, ErrCapabilityAbsent)
	}
	arch := cudadevicetable.Normalize(f.Target)
	if record.Architecture != arch {
		return fmt.Errorf("%s: record architecture %q, want %q: %w", f.Target, record.Architecture, arch, ErrCapabilityInvalid)
	}
	if err := (cudadevicetable.Table{Schema: 1, Architectures: map[string]cudadevicetable.Architecture{arch: record}}).Validate(); err != nil {
		return fmt.Errorf("%s: %w: %w", f.Target, ErrCapabilityInvalid, err)
	}
	if record.Probe != nil {
		// The emitter's exact target allowlist has at least two decimal digits.
		cc := strings.TrimPrefix(arch, "sm_")
		cc = cc[:len(cc)-1] + "." + cc[len(cc)-1:]
		if record.Probe.ComputeCapability != cc {
			return fmt.Errorf("%s: probe compute capability %q, want %q: %w", f.Target, record.Probe.ComputeCapability, cc, ErrCapabilityInvalid)
		}
	}
	var values [2]int64
	for i, attr := range []struct{ name, instrument string }{
		{"warpSize", "DevAttrWarpSize"},
		{"maxThreadsPerBlock", "DevAttrMaxThreadsPerBlock"},
	} {
		name := attr.name
		v, ok := record.Attr(name)
		if !ok {
			return fmt.Errorf("%s: %s: %w", f.Target, name, ErrCapabilityAbsent)
		}
		if v.Source != cudadevicetable.Probed {
			return fmt.Errorf("%s: %s is %s (%s): %w", f.Target, name, v.Source, v.Note, ErrCapabilityUnverified)
		}
		if v.Instrument != attr.instrument {
			return fmt.Errorf("%s: %s names instrument %q, want %q: %w", f.Target, name, v.Instrument, attr.instrument, ErrCapabilityInvalid)
		}
		if v.Value <= 0 {
			return fmt.Errorf("%s: %s has nonpositive probed value %d: %w", f.Target, name, v.Value, ErrCapabilityInvalid)
		}
		values[i] = v.Value
	}
	if values[0] != 32 {
		return fmt.Errorf("%s: probed warpSize %d, want 32: %w", f.Target, values[0], ErrCapabilityUnsupported)
	}
	// ValidateLaunch already established the exact 32x1x1 reduction block.
	if values[1] < int64(f.Launch.Block.X) {
		return fmt.Errorf("%s: reduction block requires %d threads, probed limit is %d: %w", f.Target, f.Launch.Block.X, values[1], ErrCapabilityUnsupported)
	}
	return nil
}
