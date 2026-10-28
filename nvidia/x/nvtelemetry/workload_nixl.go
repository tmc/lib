package nvtelemetry

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// NIXLTelemetryBuild is the source profile used for the staging-drop counter
// and buffer-full behavior below. Other builds retain unknown semantics.
const NIXLTelemetryBuild = "7077d029467a02d6f5b23e7eb2de244dec8a0b6a"

// NIXLAgent identifies one expected application-supplied producer incarnation.
// Build is an exact source/build identity, not merely a major version. Mode is
// the selected sink mode. These identities are assertions by the caller, not
// discovery or authentication of a running producer.
type NIXLAgent struct {
	Agent        string
	Rank         int
	RestartEpoch string
	Build        string
	Mode         string
}

// NIXLAgentObservation records supplied initialization and scrape evidence.
// Sink and Scrape are "ready", "failed" or "unknown". Detail can describe a
// port-bind failure. EvidenceSHA256 names the original status artifact.
// LastObservationNanos uses the summary's declared clock and epoch, not wall
// time. Before/After are the supplied aggregate endpoints for this incarnation.
// Their RawSHA256 values bind retained bytes; this API does not parse the bytes
// or authenticate the caller's normalization into WorkloadMetric values.
type NIXLAgentObservation struct {
	Producer             NIXLAgent
	Sink, Scrape         string
	Detail               string
	EvidenceSHA256       string
	HasLastObservation   bool
	LastObservationNanos int64
	Before, After        WorkloadExport
}

// NIXLInventoryOptions bounds the supplied inventory and all export bodies and
// metric populations in aggregate. Limits uses WorkloadLimits defaults/caps;
// MaxMetrics also bounds each inventory's cardinality. MaxAgeNanos must be
// positive. All observations use ClockDomain/ClockEpochID. No clock mapping or
// calibration is inferred.
type NIXLInventoryOptions struct {
	RunID                     string
	StartNanos, EndNanos      int64
	ClockDomain, ClockEpochID string
	MaxAgeNanos               int64
	Limits                    WorkloadLimits
}

// NIXLAgentInventory is the detached result for one expected producer.
// Status is ready, missing, restarted, producer_changed, sink_failed,
// sink_unknown, scrape_failed, scrape_unknown, stale, or window_unavailable.
// A ready result establishes only the supplied inventory/window statement.
// StagingDrops is an exact interval count or empty when unavailable. It never
// counts sink failures or retention losses. RetentionLoss is always unknown;
// BufferBehavior describes only the pinned buffer mode's source contract.
type NIXLAgentInventory struct {
	Sink, Scrape              string
	HasLastObservation        bool
	LastObservationNanos      int64
	Producer                  NIXLAgent
	Observed                  *NIXLAgent
	Status, Detail            string
	EvidenceSHA256            string
	BeforeSHA256, AfterSHA256 string
	Windows                   []WorkloadDelta
	StagingDrops              string
	RetentionLoss             string
	BufferBehavior            string
}

// NIXLInventory compares an expected population with supplied observations.
// InventoryComplete does not mean complete telemetry, no loss, successful
// transfers, completed requests or authentic producer qualification.
type NIXLInventory struct {
	Options                   NIXLInventoryOptions
	Expected, Observed, Ready int
	InventoryComplete         bool
	Agents                    []NIXLAgentInventory
}

// SummarizeNIXLInventory compares every expected agent/rank and incarnation
// against supplied sink/scrape status, freshness and actual CounterWorkloadWindow
// results. A zero staging-drop counter cannot hide a missing or failed sink.
// The order of Agents follows expected. Unexpected or duplicate observations
// are refused, rather than silently assigned to an expected producer.
//
// This function opens no sink, port, endpoint or artifact and discovers no
// process. A rank's supplied windows must name provider nixl, its exact Build,
// Rank and RunID; every metric must carry the matching agent_name label.
// Missing/reset/incompatible window data withholds that producer's readiness.
func SummarizeNIXLInventory(expected []NIXLAgent, observed []NIXLAgentObservation, o NIXLInventoryOptions) (NIXLInventory, error) {
	l, err := workloadLimits(o.Limits)
	if err != nil {
		return NIXLInventory{}, err
	}
	if len(expected) == 0 || len(expected) > l.MaxMetrics || len(observed) > l.MaxMetrics || o.StartNanos < 0 || o.EndNanos <= o.StartNanos || o.MaxAgeNanos <= 0 {
		return NIXLInventory{}, errors.New("invalid NIXL inventory bounds")
	}
	check := func(ss ...string) error {
		for _, s := range ss {
			if strings.TrimSpace(s) == "" || len(s) > l.MaxStringBytes {
				return errors.New("invalid NIXL inventory identity")
			}
		}
		return nil
	}
	if err := check(o.RunID, o.ClockDomain, o.ClockEpochID); err != nil {
		return NIXLInventory{}, err
	}
	type key struct {
		agent string
		rank  int
	}
	keys := make(map[key]NIXLAgent, len(expected))
	validateAgent := func(a NIXLAgent) error {
		if a.Rank < 0 {
			return errors.New("negative NIXL rank")
		}
		return check(a.Agent, a.RestartEpoch, a.Build, a.Mode)
	}
	for _, a := range expected {
		if err := validateAgent(a); err != nil {
			return NIXLInventory{}, err
		}
		k := key{a.Agent, a.Rank}
		if _, ok := keys[k]; ok {
			return NIXLInventory{}, errors.New("duplicate expected NIXL producer")
		}
		keys[k] = a
	}
	// Validate population/storage limits before copying any caller metadata.
	bytesLeft, metricsLeft := l.MaxBytes, l.MaxMetrics
	index := make(map[key]int, len(observed))
	for i, b := range observed {
		if err := validateAgent(b.Producer); err != nil {
			return NIXLInventory{}, err
		}
		k := key{b.Producer.Agent, b.Producer.Rank}
		if _, ok := keys[k]; !ok {
			return NIXLInventory{}, errors.New("unexpected NIXL producer")
		}
		if _, ok := index[k]; ok {
			return NIXLInventory{}, errors.New("duplicate observed NIXL producer")
		}
		index[k] = i
		if b.Sink != "ready" && b.Sink != "failed" && b.Sink != "unknown" || b.Scrape != "ready" && b.Scrape != "failed" && b.Scrape != "unknown" || len(b.Detail) > l.MaxStringBytes || !nixlDigest(b.EvidenceSHA256) {
			return NIXLInventory{}, errors.New("invalid NIXL status evidence")
		}
		for _, e := range []WorkloadExport{b.Before, b.After} {
			if int64(len(e.Raw)) > bytesLeft || len(e.Metrics) > metricsLeft {
				return NIXLInventory{}, errors.New("NIXL inventory export limit exceeded")
			}
			bytesLeft -= int64(len(e.Raw))
			metricsLeft -= len(e.Metrics)
			if len(e.UnmappedNames) > l.MaxMetrics || len(e.Source.Units) > l.MaxMetrics || len(e.NextPageToken) > l.MaxStringBytes {
				return NIXLInventory{}, errors.New("NIXL inventory export metadata limit exceeded")
			}
			if err := nixlSourceBounds(e.Source, l); err != nil {
				return NIXLInventory{}, err
			}
			for _, m := range e.Metrics {
				if err := nixlSourceBounds(m.Source, l); err != nil {
					return NIXLInventory{}, err
				}
				if len(m.Labels) > 32 {
					return NIXLInventory{}, errors.New("NIXL inventory label limit exceeded")
				}
				for _, s := range []string{m.Name, m.Unit, m.Semantic, m.Kind, m.Temporality, m.RawValue, m.RawTime, m.RawTimeUnit, m.TimeSource, m.StepKind} {
					if len(s) > l.MaxStringBytes {
						return NIXLInventory{}, errors.New("NIXL inventory metric string limit exceeded")
					}
				}
				for k, v := range m.Labels {
					if len(k) > l.MaxStringBytes || len(v) > l.MaxStringBytes {
						return NIXLInventory{}, errors.New("NIXL inventory label limit exceeded")
					}
				}
			}
		}
	}
	o.Limits = l
	result := NIXLInventory{Options: o, Expected: len(expected), Observed: len(observed)}
	for _, a := range expected {
		row := NIXLAgentInventory{Producer: a, Status: "missing", RetentionLoss: "unknown", BufferBehavior: "unknown"}
		if a.Build == NIXLTelemetryBuild && a.Mode == "buffer" {
			row.BufferBehavior = "reject_when_full"
		}
		i, ok := index[key{a.Agent, a.Rank}]
		if !ok {
			result.Agents = append(result.Agents, row)
			continue
		}
		b := observed[i]
		got := b.Producer
		row.Observed = &got
		row.Sink = b.Sink
		row.Scrape = b.Scrape
		row.HasLastObservation = b.HasLastObservation
		row.LastObservationNanos = b.LastObservationNanos
		row.Detail = b.Detail
		row.EvidenceSHA256 = b.EvidenceSHA256
		if len(b.Before.Raw) > 0 {
			h := sha256.Sum256(b.Before.Raw)
			row.BeforeSHA256 = hex.EncodeToString(h[:])
		}
		if len(b.After.Raw) > 0 {
			h := sha256.Sum256(b.After.Raw)
			row.AfterSHA256 = hex.EncodeToString(h[:])
		}
		switch {
		case got.RestartEpoch != a.RestartEpoch:
			row.Status = "restarted"
		case got.Build != a.Build || got.Mode != a.Mode:
			row.Status = "producer_changed"
		case b.Sink == "failed":
			row.Status = "sink_failed"
		case b.Sink != "ready":
			row.Status = "sink_unknown"
		case b.Scrape == "failed":
			row.Status = "scrape_failed"
		case b.Scrape != "ready":
			row.Status = "scrape_unknown"
		case !b.HasLastObservation || b.LastObservationNanos < 0 || b.LastObservationNanos > o.EndNanos || o.EndNanos-b.LastObservationNanos > o.MaxAgeNanos:
			row.Status = "stale"
		default:
			row.Status = "window_unavailable"
			windows, err := nixlWindows(a, b, o)
			if err != nil {
				row.Detail = err.Error()
			} else {
				row.Status = "ready"
				result.Ready++
				row.Windows = windows
				if a.Build == NIXLTelemetryBuild && a.Mode == "prometheus" {
					drops := 0
					for _, w := range windows {
						if w.Name == "agent_telemetry_events_dropped_total" && w.Unit == "events" && w.Semantic == "telemetry_staging_drops" {
							drops++
							row.StagingDrops = w.Delta
						}
					}
					if drops != 1 {
						row.StagingDrops = ""
					}
				}
			}
		}
		result.Agents = append(result.Agents, row)
	}
	result.InventoryComplete = result.Ready == result.Expected
	return result, nil
}

func nixlDigest(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func nixlWindows(a NIXLAgent, b NIXLAgentObservation, o NIXLInventoryOptions) ([]WorkloadDelta, error) {
	for _, e := range []WorkloadExport{b.Before, b.After} {
		s := e.Source
		if s.Provider != "nixl" || s.Version != a.Build || s.RunID != o.RunID || s.Rank == nil || *s.Rank != a.Rank {
			return nil, errors.New("NIXL workload source identity mismatch")
		}
		h := sha256.Sum256(e.Raw)
		if len(e.Raw) == 0 || hex.EncodeToString(h[:]) != e.RawSHA256 {
			return nil, errors.New("NIXL workload raw digest mismatch")
		}
		for _, m := range e.Metrics {
			if m.Labels["agent_name"] != a.Agent {
				return nil, errors.New("NIXL metric agent identity mismatch")
			}
		}
	}
	windows, err := CounterWorkloadWindow(b.Before, b.After, o.StartNanos, o.EndNanos, o.ClockDomain, o.ClockEpochID)
	if err != nil {
		return nil, fmt.Errorf("NIXL workload window: %w", err)
	}
	for i := range windows {
		if n, err := workloadNumber(windows[i].Delta); err != nil || !n.IsInt() {
			return nil, errors.New("fractional NIXL counter")
		}
		labels := make(map[string]string, len(windows[i].Labels))
		for k, v := range windows[i].Labels {
			labels[k] = v
		}
		windows[i].Labels = labels
	}
	return windows, nil
}

func nixlSourceBounds(s WorkloadSource, l WorkloadLimits) error {
	for _, v := range []string{s.Provider, s.Version, s.RunID, s.Phase, s.ClockDomain, s.ClockEpochID} {
		if len(v) > l.MaxStringBytes {
			return errors.New("NIXL source string limit exceeded")
		}
	}
	// Units is not copied or interpreted by this counter-inventory consumer.
	return nil
}
