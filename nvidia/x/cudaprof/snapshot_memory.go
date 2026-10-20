package cudaprof

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"reflect"
	"sort"

	"github.com/tmc/lib/egraphcore"
)

// hotPCLimit is the number of PCs retained for each metric.
const hotPCLimit = 10

// GuidanceBounded is Guidance with caller-owned storage accounting.
//
// Identity, quality and unit checks read s without copying it. If admit is
// not nil it then runs with a view whose StallCounts and MetricUnits borrow
// s; admit must not retain or modify them. An admit error is returned
// unchanged and nothing has been copied. Only then is the result built: its
// bytes are reserved in result in one request before any map is copied, so
// the returned guidance is detached from later changes to s. The per-metric
// PC selection keeps at most ten candidates per metric in scratch and never
// sorts every PC. Strings are borrowed from s, as for egraphcore.MemoryScope.
// The caller keeps result open for as long as it uses the guidance and may
// close scratch on return.
//
// A nil scope is unlimited, so Guidance is GuidanceBounded with nil scopes.
// An ErrGuidanceUnavailable guidance and its copied sampling and units are
// also charged to result, which stays charged until closed.
func (s *ProfileSnapshot) GuidanceBounded(target ProfileTarget, admit func(*SearchGuidance) error, scratch, result *egraphcore.MemoryScope) (*SearchGuidance, error) {
	if s == nil {
		return nil, errors.New("cudaprof: nil profile snapshot")
	}
	if s.Version != ProfileSnapshotVersion {
		return nil, fmt.Errorf("%w: version %d", ErrInvalidSnapshotVersion, s.Version)
	}
	if err := validateSnapshot(s); err != nil {
		return nil, err
	}
	if target.CodeDigest != s.CodeDigest || target.Region != s.Region || target.CollectorMode != s.CollectorMode {
		return nil, fmt.Errorf("%w: code digest, region or collector mode differs", ErrProfileTargetMismatch)
	}
	targetWorkload, targetTokensHash, targetDevice := target.Workload, target.TokensHash, target.Device
	if targetWorkload == "" {
		return nil, errors.New("cudaprof: empty target workload")
	}
	if targetDevice == "" {
		return nil, errors.New("cudaprof: empty target device")
	}

	if targetWorkload != s.Workload {
		return nil, fmt.Errorf("%w: target workload %q does not match snapshot %q",
			ErrWorkloadMismatch, targetWorkload, s.Workload)
	}
	if targetTokensHash != s.TokensHash {
		return nil, fmt.Errorf("%w: target tokens hash %q does not match snapshot %q",
			ErrWorkloadMismatch, targetTokensHash, s.TokensHash)
	}
	if targetDevice != s.Device {
		return nil, fmt.Errorf("%w: target device %q does not match snapshot %q",
			ErrDeviceMismatch, targetDevice, s.Device)
	}

	header := SearchGuidance{Workload: s.Workload, Device: s.Device, CodeDigest: s.CodeDigest, Region: s.Region, CollectorMode: s.CollectorMode, TokensHash: s.TokensHash, Timestamp: s.Timestamp, Losses: s.Losses, Coverage: s.Coverage, UncertaintyNs: s.ClockUncertaintyNs}
	unavailable := func(reason string, units bool) (*SearchGuidance, error) {
		header.UnavailableReason = reason
		n := int64(reflect.TypeFor[SearchGuidance]().Size())
		if s.Sampling != nil {
			n += int64(reflect.TypeFor[SamplingProvenance]().Size())
		}
		if units && s.MetricUnits != nil {
			m, err := mapBytes[string, string](len(s.MetricUnits))
			if err != nil || m > math.MaxInt64-n {
				return nil, egraphcore.ErrMemorySize
			}
			n += m
		}
		if err := result.Reserve(n); err != nil {
			return nil, err
		}
		unavail := new(SearchGuidance)
		*unavail = header
		unavail.Sampling = cloneSampling(s.Sampling)
		if units {
			unavail.MetricUnits = maps.Clone(s.MetricUnits)
		}
		return unavail, ErrGuidanceUnavailable
	}
	if s.Coverage <= 0.0 || len(s.Metrics) == 0 {
		return unavailable("insufficient coverage or missing telemetry metrics", false)
	}
	for metric := range s.Metrics {
		if s.MetricUnits[metric] == "" {
			return unavailable("missing unit for metric "+metric, true)
		}
	}
	for _, counters := range s.PCStalls {
		for metric := range counters {
			if s.MetricUnits[metric] == "" {
				return unavailable("missing unit for pc metric "+metric, true)
			}
		}
	}
	if admit != nil {
		view := header
		view.Available = true
		view.StallCounts = s.Metrics
		view.MetricUnits = s.MetricUnits
		if err := admit(&view); err != nil {
			return nil, err
		}
	}
	if result != nil {
		n, err := guidanceBytes(s)
		if err != nil {
			return nil, err
		}
		if err := result.Reserve(n); err != nil {
			return nil, err
		}
	}
	guidance := new(SearchGuidance)
	*guidance = header
	guidance.Available = true
	guidance.Sampling = cloneSampling(s.Sampling)
	guidance.MetricUnits = maps.Clone(s.MetricUnits)
	guidance.StallCounts = cloneMetrics(s.Metrics)
	guidance.PCStalls = clonePCStalls(s.PCStalls)
	hot, err := hotPCs(s.PCStalls, scratch, result)
	if err != nil {
		return nil, err
	}
	guidance.HotPCs = hot
	return guidance, nil
}

func cloneSampling(p *SamplingProvenance) *SamplingProvenance {
	if p == nil {
		return nil
	}
	c := *p
	return &c
}

func cloneMetrics(m map[string]int64) map[string]int64 {
	c := make(map[string]int64, len(m))
	for k, v := range m {
		c[k] = v
	}
	return c
}

func clonePCStalls(m map[uint64]map[string]int64) map[uint64]map[string]int64 {
	if m == nil {
		return nil
	}
	c := make(map[uint64]map[string]int64, len(m))
	for pc, counters := range m {
		c[pc] = maps.Clone(counters)
	}
	return c
}

// guidanceBytes sizes the copies GuidanceBounded makes, excluding hot PCs.
// Units are Go objects, map headers and one key/value slot per map entry;
// allocator and map-bucket overhead are outside them.
func guidanceBytes(s *ProfileSnapshot) (int64, error) {
	var total int64
	add := func(n int64, err error) error {
		if err != nil || n > math.MaxInt64-total {
			return egraphcore.ErrMemorySize
		}
		total += n
		return nil
	}
	if err := add(int64(reflect.TypeFor[SearchGuidance]().Size()), nil); err != nil {
		return 0, err
	}
	if s.Sampling != nil {
		if err := add(int64(reflect.TypeFor[SamplingProvenance]().Size()), nil); err != nil {
			return 0, err
		}
	}
	if err := add(mapBytes[string, int64](len(s.Metrics))); err != nil {
		return 0, err
	}
	if s.MetricUnits != nil {
		if err := add(mapBytes[string, string](len(s.MetricUnits))); err != nil {
			return 0, err
		}
	}
	if s.PCStalls != nil {
		if err := add(mapBytes[uint64, map[string]int64](len(s.PCStalls))); err != nil {
			return 0, err
		}
		for _, counters := range s.PCStalls {
			if counters == nil {
				continue
			}
			if err := add(mapBytes[string, int64](len(counters))); err != nil {
				return 0, err
			}
		}
	}
	return total, nil
}

// mapBytes sizes a map header and entries key/value slots.
func mapBytes[K comparable, V any](entries int) (int64, error) {
	slot := int64(reflect.TypeFor[struct {
		key   K
		value V
	}]().Size())
	if slot > 0 && int64(entries) > (math.MaxInt64-int64(reflect.TypeFor[map[K]V]().Size()))/slot {
		return 0, egraphcore.ErrMemorySize
	}
	return int64(reflect.TypeFor[map[K]V]().Size()) + int64(entries)*slot, nil
}

type pcHit struct {
	pc    uint64
	count int64
}

// better orders hits by descending count, then ascending PC.
func (h pcHit) better(o pcHit) bool {
	return h.count > o.count || h.count == o.count && h.pc < o.pc
}

// insertHit keeps hits sorted best first and at most hotPCLimit long.
func insertHit(hits []pcHit, h pcHit) []pcHit {
	i := sort.Search(len(hits), func(i int) bool { return h.better(hits[i]) })
	if i >= hotPCLimit {
		return hits
	}
	if len(hits) < hotPCLimit {
		hits = append(hits, pcHit{})
	}
	copy(hits[i+1:], hits[i:len(hits)-1])
	hits[i] = h
	return hits
}

// hotPCs ranks each metric independently; it never infers exclusivity between
// samples. It keeps at most hotPCLimit candidates per metric in scratch and
// charges the returned map and slices to result before creating them. Both
// scopes may be nil.
func hotPCs(pcStalls map[uint64]map[string]int64, scratch, result *egraphcore.MemoryScope) (map[string][]uint64, error) {
	hdr, err := mapBytes[string, []uint64](0)
	if err != nil {
		return nil, err
	}
	if err := result.Reserve(hdr); err != nil {
		return nil, err
	}
	top, err := mapBytes[string, []pcHit](0)
	if err != nil {
		return nil, err
	}
	if err := scratch.Reserve(top); err != nil {
		return nil, err
	}
	byMetric := make(map[string][]pcHit)
	for pc, counters := range pcStalls {
		for metric, count := range counters {
			if count <= 0 {
				continue
			}
			hits, ok := byMetric[metric]
			if !ok {
				slot, err := mapBytes[string, []pcHit](1)
				if err != nil {
					return nil, err
				}
				if err := scratch.Reserve(slot - top + hotPCLimit*int64(reflect.TypeFor[pcHit]().Size())); err != nil {
					return nil, err
				}
				hits = make([]pcHit, 0, hotPCLimit)
			}
			byMetric[metric] = insertHit(hits, pcHit{pc, count})
		}
	}
	entries, err := mapBytes[string, []uint64](len(byMetric))
	if err != nil {
		return nil, err
	}
	if err := result.Reserve(entries - hdr); err != nil {
		return nil, err
	}
	hot := make(map[string][]uint64, len(byMetric))
	for metric, hits := range byMetric {
		if err := result.Reserve(int64(len(hits)) * int64(reflect.TypeFor[uint64]().Size())); err != nil {
			return nil, err
		}
		pcs := make([]uint64, len(hits))
		for i, h := range hits {
			pcs[i] = h.pc
		}
		hot[metric] = pcs
	}
	return hot, nil
}
