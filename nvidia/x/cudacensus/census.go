package cudacensus

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// UnknownResidency is the zero Residency: the cost model did not say which
// memory path the bytes travel. It is spelled out rather than left as an
// untyped empty string because a byte count whose residency is unstated must
// not silently acquire one. Points refuses to evaluate it.
const UnknownResidency = cudaroofline.Residency("")

// Cost is what one launch moves, computes, and where the bytes it moves live.
// A capture records none of the three, so a Cost always comes from a model the
// caller supplies.
//
// Residency is part of the cost, not a detail of the ceiling. On a measured
// GB10 the same bytes attain 239.6 GB/s device-resident and 162.4 GB/s read
// through host page tables, a factor of 1.43. A byte count divided into the
// wrong one of those produces a headroom that cannot be collected, and the
// arithmetic gives no sign of it. Weights mmap'd from a model file are
// HostPageTableResident unless something copied them to the device.
//
// Leaving Residency unset yields UnknownResidency, which Points refuses rather
// than guesses.
//
// FLOPs may be left zero when only bandwidth is modeled. A roofline point
// built from a zero FLOP count reports AttainedGBps correctly but classifies
// as unconstrained, because intensity is then zero; read the bandwidth, not
// the classification.
type Cost struct {
	Bytes     uint64
	FLOPs     uint64
	Residency cudaroofline.Residency
}

// A Coster reports the cost of one launch. It returns false for a launch the
// model does not cover, which Census records rather than hides.
type Coster interface {
	Cost(Launch) (Cost, bool)
}

// CosterFunc adapts a function to Coster.
type CosterFunc func(Launch) (Cost, bool)

// Cost calls f.
func (f CosterFunc) Cost(l Launch) (Cost, bool) { return f(l) }

// Entry aggregates every launch of one kernel within a window.
type Entry struct {
	Name     string
	Launches int
	DeviceNs int64

	// Bytes and FLOPs are the modeled totals over all Launches. They are
	// zero when Modeled is false.
	Bytes uint64
	FLOPs uint64

	// Residency is the memory path every modeled launch of this kernel
	// agreed on. It is UnknownResidency when the model did not state one, or
	// when launches disagreed; either way the entry cannot be judged against
	// a ceiling, and Points says so instead of picking one.
	Residency cudaroofline.Residency

	// Modeled reports whether the Coster covered this kernel. An unmodeled
	// entry still carries its launch count and device time.
	Modeled bool

	// Grids and Blocks are the distinct launch geometries seen, sorted. More
	// than one geometry under one name means a single Cost per name cannot
	// describe every launch.
	Grids  []string
	Blocks []string
}

// Census aggregates launches by kernel name, sorted by descending device time.
// A nil Coster leaves every entry unmodeled, which is the right way to take a
// first look at a capture.
func Census(launches []Launch, c Coster) []Entry {
	type acc struct {
		e         Entry
		grids     map[string]bool
		blocks    map[string]bool
		residency map[cudaroofline.Residency]bool
	}
	byName := make(map[string]*acc)
	var order []string
	for _, l := range launches {
		a := byName[l.Name]
		if a == nil {
			a = &acc{
				e:         Entry{Name: l.Name, Modeled: true},
				grids:     map[string]bool{},
				blocks:    map[string]bool{},
				residency: map[cudaroofline.Residency]bool{},
			}
			byName[l.Name] = a
			order = append(order, l.Name)
		}
		a.e.Launches++
		a.e.DeviceNs += l.DurationNs()
		a.grids[l.Grid] = true
		a.blocks[l.Block] = true
		if c == nil {
			a.e.Modeled = false
			continue
		}
		cost, ok := c.Cost(l)
		if !ok {
			a.e.Modeled = false
			continue
		}
		a.e.Bytes += cost.Bytes
		a.e.FLOPs += cost.FLOPs
		a.residency[cost.Residency] = true
	}
	out := make([]Entry, 0, len(order))
	for _, n := range order {
		a := byName[n]
		// A kernel is modeled only if every one of its launches was: a
		// partial total would understate bytes while looking complete.
		if !a.e.Modeled {
			a.e.Bytes, a.e.FLOPs = 0, 0
		}
		// One residency, agreed on by every launch, or none. Two launches of
		// one kernel reading different memory paths have no single ceiling.
		if len(a.residency) == 1 {
			for r := range a.residency {
				a.e.Residency = r
			}
		}
		a.e.Grids = sortedKeys(a.grids)
		a.e.Blocks = sortedKeys(a.blocks)
		out = append(out, a.e)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].DeviceNs > out[j].DeviceNs })
	return out
}

func sortedKeys(m map[string]bool) []string {
	s := make([]string, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	sort.Strings(s)
	return s
}

// Coverage reports the device time the cost model accounted for and the total
// device time in the census. Report the ratio next to any bandwidth derived
// from these entries: a bandwidth computed over a tenth of the device time
// describes a tenth of the workload.
func Coverage(entries []Entry) (modeledNs, totalNs int64) {
	for _, e := range entries {
		totalNs += e.DeviceNs
		if e.Modeled {
			modeledNs += e.DeviceNs
		}
	}
	return modeledNs, totalNs
}

// Points evaluates against m every modeled entry whose residency matches m's
// ceiling, one point per kernel name.
//
// Entries are skipped for three reasons, and the last two are reported as an
// error rather than passed over quietly:
//
//   - Unmodeled or zero-duration: no byte count to divide, nothing to say.
//   - UnknownResidency: the model did not state which memory path the bytes
//     travel, so no ceiling applies to them.
//   - A residency other than m's: judging these bytes against this ceiling
//     would misstate the headroom by the ratio between the two paths, which
//     on a measured GB10 is 1.43x.
//
// The points that did match are returned alongside the error, so a caller who
// has read it can proceed deliberately. A caller who ignores the error gets
// fewer points than entries, never a point computed against the wrong ceiling.
func Points(m *cudaroofline.Model, entries []Entry) ([]cudaroofline.KernelPoint, error) {
	want := m.Ceiling.Residency
	var out []cudaroofline.KernelPoint
	var unknown, mismatched []string
	for _, e := range entries {
		if !e.Modeled || e.DeviceNs <= 0 {
			continue
		}
		switch e.Residency {
		case want:
			out = append(out, m.Evaluate(e.Name, e.FLOPs, e.Bytes, uint64(e.DeviceNs)))
		case UnknownResidency:
			unknown = append(unknown, e.Name)
		default:
			mismatched = append(mismatched, fmt.Sprintf("%s (%s)", e.Name, e.Residency))
		}
	}
	var problems []string
	if len(unknown) > 0 {
		problems = append(problems, fmt.Sprintf("%d with unstated residency: %s", len(unknown), strings.Join(unknown, ", ")))
	}
	if len(mismatched) > 0 {
		problems = append(problems, fmt.Sprintf("%d not %s: %s", len(mismatched), want, strings.Join(mismatched, ", ")))
	}
	if len(problems) > 0 {
		return out, fmt.Errorf("cudacensus: %d of %d entries not evaluated against the %s ceiling: %s",
			len(unknown)+len(mismatched), len(entries), want, strings.Join(problems, "; "))
	}
	return out, nil
}

// EvaluatedCoverage reports the device time Points would actually evaluate
// against m, and the total device time in the census.
//
// Coverage answers "did the model state bytes for this kernel"; this answers
// the stricter question "did it state bytes on the memory path this ceiling
// describes". The two differ whenever a model covers a kernel but states a
// residency that does not match, and that gap is exactly where a 1.43x error
// would otherwise hide: Coverage would read high while every byte of it was
// unjudgeable. Report this ratio, not Coverage, beside any number derived from
// Points.
func EvaluatedCoverage(m *cudaroofline.Model, entries []Entry) (evaluatedNs, totalNs int64) {
	for _, e := range entries {
		totalNs += e.DeviceNs
		if e.Modeled && e.DeviceNs > 0 && e.Residency == m.Ceiling.Residency {
			evaluatedNs += e.DeviceNs
		}
	}
	return evaluatedNs, totalNs
}
