package cudadevicetable

import (
	"fmt"
	"sort"
	"strings"
)

// A Profile is what a code generator needs to know about the GPU it emits
// for: the PTX target, the resource limits that decide tile shapes and
// grid sizes, and the bandwidths a cost model divides by.
//
// Every number is a [Value] and keeps its provenance. A profile for a
// probed device carries the probed values (see [LookupProfile]); every
// other value is a datasheet figure, [Unverified], whose Note cites where
// it came from. A generator that must not act on a datasheet figure can
// check Source, and one that plans offline for hardware it cannot attach
// can still plan.
type Profile struct {
	Target  string // PTX target the generator emits for, such as "sm_103a"
	Product string // product the per-product values describe, such as "GB300"

	SMs                 Value // streaming multiprocessors
	SharedPerBlock      Value // bytes of shared memory a block may use without opt-in, the static limit
	SharedPerBlockOptin Value // bytes of dynamic shared memory a block may opt in to
	SharedPerSM         Value // bytes of shared memory per SM
	ThreadsPerSM        Value // resident threads per SM
	BlocksPerSM         Value // resident blocks per SM
	TMEMPerSM           Value // bytes of Tensor Memory per SM; 0 without tcgen05
	ClusterSize         Value // largest portable thread block cluster
	ClusterSizeOptin    Value // largest cluster with cudaFuncAttributeNonPortableClusterSizeAllowed
	MemoryGBps          Value // device memory bandwidth, 10⁹ bytes per second
	NVLinkGBps          Value // GPU-to-GPU NVLink bandwidth per GPU, both directions; 0 without NVLink
}

// Supports reports whether ptxas accepts the named feature for p.Target, as
// recorded by the feature probe (see [Target.Supports]). It is an error
// for a target or feature that was never probed.
func (p Profile) Supports(feature string) (bool, error) {
	t, ok := LookupTarget(p.Target)
	if !ok {
		return false, fmt.Errorf("cudadevicetable: target %s was never probed", p.Target)
	}
	return t.Supports(feature)
}

// Validate reports whether every value in p obeys the provenance rules of
// [Table.Validate]: a probed value names its instrument, and any other
// value explains itself in a note.
func (p Profile) Validate() error {
	for name, v := range p.values() {
		switch v.Source {
		case Probed:
			if !knownInstrument(v.Instrument) {
				return fmt.Errorf("%s/%s: %s is probed but names instrument %q", p.Target, p.Product, name, v.Instrument)
			}
		case Derived, Unverified:
			if v.Instrument != "" || v.Note == "" {
				return fmt.Errorf("%s/%s: %s is %s and must have a note and no instrument", p.Target, p.Product, name, v.Source)
			}
		default:
			return fmt.Errorf("%s/%s: %s has source %q", p.Target, p.Product, name, v.Source)
		}
	}
	return nil
}

// values maps each value in p to the attribute name devices.json uses for
// it, or to a descriptive name where the driver has no attribute.
func (p *Profile) values() map[string]*Value {
	return map[string]*Value{
		"multiprocessorCount":              &p.SMs,
		"maxSharedMemoryPerBlock":          &p.SharedPerBlock,
		"maxSharedMemoryPerBlockOptin":     &p.SharedPerBlockOptin,
		"maxSharedMemoryPerMultiprocessor": &p.SharedPerSM,
		"maxThreadsPerMultiprocessor":      &p.ThreadsPerSM,
		"maxBlocksPerMultiprocessor":       &p.BlocksPerSM,
		"tensorMemoryPerMultiprocessor":    &p.TMEMPerSM,
		"clusterSize":                      &p.ClusterSize,
		"clusterSizeOptin":                 &p.ClusterSizeOptin,
		"memoryBandwidthGBps":              &p.MemoryGBps,
		"nvlinkBandwidthGBps":              &p.NVLinkGBps,
	}
}

// LookupProfile returns the profile named by name: a PTX target such as
// "sm_103a", which selects that target's first listed product, or a
// product such as "GB300" or "NVIDIA GB10", in any case. Where the table
// holds a probe of that product (see [Lookup]), the probed values replace
// the datasheet ones.
func LookupProfile(name string) (Profile, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	key = strings.TrimPrefix(key, "nvidia ")
	key = strings.TrimPrefix(key, "geforce ")
	for _, p := range datasheet {
		if strings.EqualFold(p.Target, key) || strings.EqualFold(p.Product, key) {
			return withProbe(p), nil
		}
	}
	return Profile{}, fmt.Errorf("cudadevicetable: no profile for %q (have %s)", name, strings.Join(ProfileNames(), ", "))
}

// ProfileNames returns the targets and products [LookupProfile] accepts,
// sorted.
func ProfileNames() []string {
	seen := map[string]bool{}
	for _, p := range datasheet {
		seen[p.Target] = true
		seen[p.Product] = true
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// TargetProfiles returns the profile of every listed product for target,
// in table order. The products of one target can differ in SM count and
// bandwidth but not in instruction set or per-SM limits.
func TargetProfiles(target string) []Profile {
	var out []Profile
	for _, p := range datasheet {
		if strings.EqualFold(p.Target, target) {
			out = append(out, withProbe(p))
		}
	}
	return out
}

// withProbe overlays the probed attributes of p's architecture, if the
// probed device is p's product. A probe of one product does not describe
// another product of the same architecture: GB10 and RTX PRO 6000 are both
// sm_12x and differ in SM count by a factor of four.
func withProbe(p Profile) Profile {
	a, ok := Lookup(p.Target)
	if !ok || a.Probe == nil || !strings.EqualFold(strings.TrimPrefix(a.Probe.Device, "NVIDIA "), p.Product) {
		return p
	}
	for name, v := range p.values() {
		if pv, ok := a.Attributes[name]; ok && pv.Source == Probed {
			*v = pv
		}
	}
	return p
}
