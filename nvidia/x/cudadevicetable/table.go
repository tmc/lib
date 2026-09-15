package cudadevicetable

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

//go:embed devices.json
var devicesJSON []byte

// Source records how a [Value] was obtained. The zero value is invalid: a
// value with no stated source is exactly the thing this package exists to
// make impossible.
type Source string

const (
	// Probed means the value was read from a live device through the driver
	// attribute named in Value.Attribute.
	Probed Source = "probed"

	// Derived means the value was computed from probed values in the same
	// record. Value.Note states the arithmetic.
	Derived Source = "derived"

	// Unverified means the value was hand-entered and no device has
	// confirmed it. It may still be correct; nothing here says it is.
	Unverified Source = "unverified"
)

// Value is one device fact together with how it was obtained.
type Value struct {
	Value  int64  `json:"value"`
	Source Source `json:"source"`

	// Instrument names what the value was read through: a driver attribute
	// ("DevAttrMaxSharedMemoryPerMultiprocessor") or a CUDA runtime device
	// property ("cudaDeviceProp.l2CacheSize"). Required when Source is
	// [Probed], empty otherwise.
	//
	// Not every hardware fact has a driver attribute behind it -- this
	// module's bindings expose no L2 cache size, for one -- so requiring a
	// driver attribute specifically would force a real measurement to be
	// recorded as unverified. The requirement is that the instrument be
	// named and recognizable, not that it be of one kind.
	Instrument string `json:"instrument,omitempty"`

	// Note explains a derivation or records why a value could not be
	// probed. Required when Source is [Derived] or [Unverified].
	Note string `json:"note,omitempty"`
}

// Probe identifies the device a record was read from. Without it a probed
// value is unreproducible, which makes it indistinguishable from a datasheet
// figure that merely claims to be a measurement.
type Probe struct {
	Device            string `json:"device"`
	ComputeCapability string `json:"computeCapability"`
	DriverVersion     string `json:"driverVersion"`
	Host              string `json:"host"`
	ProbedAt          string `json:"probedAt"`
	Tool              string `json:"tool"`
}

// Architecture is the set of known facts for one GPU architecture.
type Architecture struct {
	Architecture string `json:"architecture"`

	// Probe is nil when no value in Attributes is [Probed].
	Probe *Probe `json:"probe,omitempty"`

	Attributes map[string]Value `json:"attributes"`
}

// Table is the full set of architectures with recorded facts.
type Table struct {
	Schema        int                     `json:"schema"`
	Architectures map[string]Architecture `json:"architectures"`
}

var table Table

func init() {
	if err := json.Unmarshal(devicesJSON, &table); err != nil {
		panic("cudadevicetable: embedded devices.json: " + err.Error())
	}
	if err := table.Validate(); err != nil {
		panic("cudadevicetable: embedded devices.json: " + err.Error())
	}
}

// Lookup returns the record for arch, which may be written "sm_121" or
// "sm121" in any case. The architecture suffix used for feature-specific
// codegen ("sm_121a") resolves to its base architecture, because the suffix
// selects an instruction set and does not describe different hardware.
//
// A miss means no device with that architecture has been probed. It does not
// mean the architecture is unsupported, and callers must not substitute a
// neighbouring architecture's record for a missing one: an occupancy or
// roofline figure computed from the wrong architecture is wrong in exactly
// the way that reads as right.
func Lookup(arch string) (Architecture, bool) {
	a, ok := table.Architectures[Normalize(arch)]
	return a, ok
}

// Normalize canonicalizes an architecture string to the "sm_NNN" form used as
// a table key, dropping any feature suffix.
func Normalize(arch string) string {
	s := strings.ToLower(strings.TrimSpace(arch))
	s = strings.TrimPrefix(s, "sm_")
	s = strings.TrimPrefix(s, "sm")
	s = strings.TrimLeft(s, "_")
	// Drop a trailing feature suffix such as the "a" of sm_121a.
	s = strings.TrimRight(s, "abcdefghijklmnopqrstuvwxyz")
	if s == "" {
		return ""
	}
	return "sm_" + s
}

// Architectures returns the probed architecture keys in sorted order.
func Architectures() []string {
	keys := make([]string, 0, len(table.Architectures))
	for k := range table.Architectures {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Attr returns the named attribute.
func (a Architecture) Attr(name string) (Value, bool) {
	v, ok := a.Attributes[name]
	return v, ok
}

// instrumentPrefixes are the recognized ways a value can have been read from
// a device. A value claiming [Probed] must name one of them, so that the
// claim is reproducible by whoever doubts it.
var instrumentPrefixes = []string{"DevAttr", "cudaDeviceProp."}

func knownInstrument(s string) bool {
	for _, p := range instrumentPrefixes {
		if strings.HasPrefix(s, p) && len(s) > len(p) {
			return true
		}
	}
	return false
}

// ProbedAttr returns the named attribute only if it was read from a device.
// Use it wherever a wrong value would flow onward silently: it converts a
// datasheet figure from a plausible number into an error.
func (a Architecture) ProbedAttr(name string) (int64, error) {
	v, ok := a.Attributes[name]
	if !ok {
		return 0, fmt.Errorf("cudadevicetable: %s: no attribute %q", a.Architecture, name)
	}
	if v.Source != Probed {
		return 0, fmt.Errorf("cudadevicetable: %s: %q is %s, not probed: %s", a.Architecture, name, v.Source, v.Note)
	}
	return v.Value, nil
}

// Validate reports whether the table's provenance rules hold. It is run
// against the embedded table at init, so a malformed devices.json fails at
// program start rather than at the first lookup.
func (t Table) Validate() error {
	if t.Schema != 1 {
		return fmt.Errorf("unsupported schema %d", t.Schema)
	}
	for key, arch := range t.Architectures {
		if key != Normalize(key) {
			return fmt.Errorf("architecture key %q is not normalized (want %q)", key, Normalize(key))
		}
		if arch.Architecture != key {
			return fmt.Errorf("architecture %q records Architecture %q", key, arch.Architecture)
		}
		if len(arch.Attributes) == 0 {
			return fmt.Errorf("architecture %q has no attributes", key)
		}
		probed := false
		for name, v := range arch.Attributes {
			switch v.Source {
			case Probed:
				probed = true
				// The rule the package rests on: a claim to have measured
				// something must name the instrument it was measured with.
				if v.Instrument == "" {
					return fmt.Errorf("%s: %q is probed but names no instrument", key, name)
				}
				if !knownInstrument(v.Instrument) {
					return fmt.Errorf("%s: %q names instrument %q, want a %s prefix", key, name, v.Instrument, strings.Join(instrumentPrefixes, " or "))
				}
			case Derived, Unverified:
				if v.Instrument != "" {
					return fmt.Errorf("%s: %q is %s but names instrument %q", key, name, v.Source, v.Instrument)
				}
				if v.Note == "" {
					return fmt.Errorf("%s: %q is %s with no note explaining why", key, name, v.Source)
				}
			default:
				return fmt.Errorf("%s: %q has source %q, want probed, derived or unverified", key, name, v.Source)
			}
		}
		switch {
		case probed && arch.Probe == nil:
			return fmt.Errorf("%s: has probed attributes but no probe record", key)
		case !probed && arch.Probe != nil:
			return fmt.Errorf("%s: has a probe record but no probed attributes", key)
		}
		if arch.Probe != nil {
			for field, s := range map[string]string{
				"device":            arch.Probe.Device,
				"computeCapability": arch.Probe.ComputeCapability,
				"host":              arch.Probe.Host,
				"probedAt":          arch.Probe.ProbedAt,
				"tool":              arch.Probe.Tool,
			} {
				if s == "" {
					return fmt.Errorf("%s: probe record has empty %s", key, field)
				}
			}
		}
	}
	return nil
}
