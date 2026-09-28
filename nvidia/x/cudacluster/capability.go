package cudacluster

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// The capability sentinels live in cudaerrors, because four lanes in three
// unrelated areas answer the same question and none of them should import a
// cluster package for an error value. These aliases exist so that callers of
// this package can match without a second import; the definitions and the
// contract are there, not here.
var (
	// ErrUnsupported reports that an instrument examined the target and
	// refused. It is a measurement.
	ErrUnsupported = cudaerrors.ErrUnsupported

	// ErrHardwarePending reports that no instrument has examined the
	// target, so support is unknown. It is the absence of a measurement and
	// must never be reported as a negative one.
	ErrHardwarePending = cudaerrors.ErrHardwarePending
)

// minClusterCapability is the compute capability at which thread block
// clusters first appear. The fence is the capability, not a vendor tier: the
// Blackwell consumer parts are above it.
const minClusterCapability = 90

// clusterFeature is the feature key recorded in the device table.
//
// Its fixture asserts barrier.cluster.arrive, barrier.cluster.wait and
// %cluster_ctarank. It does not assert mapa.shared::cluster, so a true answer
// here establishes clusters and not distributed shared memory. See
// [CheckDSMEM].
const clusterFeature = "cluster"

// Capability is a compute capability scaled by ten, so sm_121a is 121 and
// sm_90 is 90. Comparisons are on this integer; the dotted form is
// presentation.
type Capability int

// String returns the dotted form, such as "12.1".
func (c Capability) String() string { return fmt.Sprintf("%d.%d", int(c)/10, int(c)%10) }

// CapabilityOf parses the compute capability out of an exact PTX target such
// as "sm_121a".
//
// This is a lexical fact about the target name and nothing more. It does not
// consult the device table and it does not mean a device of that capability is
// present. Use it to apply the documented capability fence; use [Check] to ask
// what an assembler actually reported.
func CapabilityOf(target string) (Capability, error) {
	t := strings.ToLower(strings.TrimSpace(target))
	if !strings.HasPrefix(t, "sm_") {
		return 0, fmt.Errorf("cudacluster: %q is not an sm_ target", target)
	}
	digits := t[len("sm_"):]
	end := 0
	for end < len(digits) && digits[end] >= '0' && digits[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0, fmt.Errorf("cudacluster: %q has no capability digits", target)
	}
	n, err := strconv.Atoi(digits[:end])
	if err != nil {
		return 0, fmt.Errorf("cudacluster: %q: %w", target, err)
	}
	return Capability(n), nil
}

// Check reports whether target may be used for cluster work.
//
// It requires two independent things to agree: the capability fence, which is
// documented and lexical, and the recorded assembler fact, which is measured.
// Either one alone has been wrong in this corpus, so neither is trusted alone.
//
// It returns [ErrUnsupported] when an instrument refused the target,
// [ErrHardwarePending] when none has looked, and nil when the target is
// recorded as accepting clusters. A nil result is permission to attempt a run,
// not evidence that a run succeeded.
func Check(target string) error {
	return check(target, cudadevicetable.LookupTarget)
}

func check(target string, lookup func(string) (cudadevicetable.Target, bool)) error {
	cap, err := CapabilityOf(target)
	if err != nil {
		return err
	}
	if cap < minClusterCapability {
		if cap == 86 {
			return fmt.Errorf("cudacluster: %s is below the cluster capability fence, but has no recorded refusal: %w", target, ErrHardwarePending)
		}
		return fmt.Errorf("cudacluster: %s is compute capability %s, below %s: %w",
			target, cap, Capability(minClusterCapability), ErrUnsupported)
	}
	rec, ok := lookup(target)
	if !ok {
		return fmt.Errorf("cudacluster: no feature record for %s: %w", target, ErrHardwarePending)
	}
	ok, err = rec.Supports(clusterFeature)
	if err != nil {
		return fmt.Errorf("cudacluster: %s: %w: %w", target, ErrHardwarePending, err)
	}
	if !ok {
		return fmt.Errorf("cudacluster: assembler refused clusters for %s: %w", target, ErrUnsupported)
	}
	return nil
}

// CheckDSMEM reports whether target may be used for distributed shared memory
// specifically, and it currently returns [ErrHardwarePending] for every target.
//
// This is not an oversight. The device table's "cluster" fixture asserts the
// cluster barrier and %cluster_ctarank; it contains no mapa.shared::cluster and
// no ld.shared::cluster. So the recorded fact establishes that a target has
// clusters, and says nothing about whether its assembler accepts a remote
// shared-memory window. Reporting the cluster fact as a DSMEM answer would be
// substituting a neighbouring measurement for a missing one.
//
// testdata/dsmem.ptx is the fixture that would settle it, written to the same
// shape as the existing probes. Assembling it per target and recording the
// result belongs to the owner of cudadevicetable, not here.
func CheckDSMEM(target string) error {
	if err := Check(target); err != nil {
		return err
	}
	return fmt.Errorf("cudacluster: %s has clusters, but no instrument has assembled "+
		"mapa.shared::cluster for it (see testdata/dsmem.ptx): %w", target, ErrHardwarePending)
}
