package counters

import (
	"encoding/hex"
	"errors"
	"math"
	"reflect"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// Selection retains the exact qualified launches selected for counter work.
// Capture identities and independent correctness receipts remain caller owned.
type Selection struct {
	Identity shape.Identity `json:"identity"`
	Base     shape.Launch   `json:"base"`
	Target   shape.Launch   `json:"target"`
}

// Select rejects failed shape qualification and nonunique selections. Matching
// uses every shape field, including optional fields; shortened names cannot
// select work. A selection identifies historical launches, not a guarantee that
// replay will execute equivalent work. Collection must verify replay separately.
func Select(report shape.Report, identity shape.Identity) (Selection, error) {
	if report.Version != 1 || !report.Pass || !qualifiedIdentity(identity) {
		return Selection{}, errors.New("counters: shape qualification failed")
	}
	var found *shape.Group
	for i := range report.Groups {
		g := &report.Groups[i]
		if !reflect.DeepEqual(g.Identity, identity) {
			continue
		}
		if found != nil {
			return Selection{}, errors.New("counters: ambiguous shape selection")
		}
		found = g
	}
	if found == nil {
		return Selection{}, errors.New("counters: shape not observed")
	}
	if len(found.Base) != 1 || len(found.Target) != 1 {
		return Selection{}, errors.New("counters: ambiguous launch selection")
	}
	for _, l := range []shape.Launch{found.Base[0], found.Target[0]} {
		if l.Activity < 0 || l.Start <= 0 || l.Duration <= 0 || l.Start > math.MaxInt64-l.Duration || l.Journal == "" || l.RawSymbol == "" || l.Identity == nil || !reflect.DeepEqual(*l.Identity, identity) || l.APIStatus != "matched" || l.MetadataStatus != "matched" || len(l.Missing) > 0 || l.Reason != "" {
			return Selection{}, errors.New("counters: launch qualification failed")
		}
	}
	return Selection{identity, found.Base[0], found.Target[0]}, nil
}

// Receipt describes one separately collected counter value. Properties must
// come from the native metric query, not a guess from its name. Value is nil
// when unavailable and Failure retains the collection error. Replay describes
// actual collection behavior, including state restoration. Overhead remains
// unmeasured unless measured under an independent controlled experiment.
type Receipt struct {
	Identity       shape.Identity              `json:"identity"`
	CaptureSHA256  string                      `json:"capture_sha256"`
	Journal        string                      `json:"journal"`
	RawSymbol      string                      `json:"raw_symbol"`
	Device         string                      `json:"device"`
	Chip           string                      `json:"chip"`
	DriverVersion  string                      `json:"driver_version"`
	ToolVersion    string                      `json:"tool_version"`
	Replay         string                      `json:"replay"`
	ReplayComplete bool                        `json:"replay_complete"`
	Properties     cuptirange.MetricProperties `json:"properties"`
	Value          *float64                    `json:"value,omitempty"`
	Failure        string                      `json:"failure,omitempty"`
	Overhead       string                      `json:"overhead"`
}

// Comparison retains both receipts, including unavailable values. Compatible
// does not imply that a counter caused a timing difference. Delta is target
// minus base and is absent unless definitions and collection context match.
type Comparison struct {
	Base       Receipt  `json:"base"`
	Target     Receipt  `json:"target"`
	Compatible bool     `json:"compatible"`
	Reason     string   `json:"reason,omitempty"`
	Delta      *float64 `json:"delta,omitempty"`
}

// Compare accepts only matching observed shapes, native metric definitions,
// device/chip identity, driver/tool versions and replay contracts. Missing
// identity, invalid native scope/type, incomplete replay, errors and nonfinite
// values remain unavailable. Capture hashes must be 64 hexadecimal characters.
func Compare(base, target Receipt) Comparison {
	r := Comparison{Base: base, Target: target}
	for _, v := range []Receipt{base, target} {
		switch {
		case !qualifiedIdentity(v.Identity):
			r.Reason = "shape identity unavailable"
		case !validSHA256(v.CaptureSHA256) || v.Journal == "" || v.RawSymbol == "":
			r.Reason = "capture identity unavailable"
		case v.Device == "" || v.Chip == "" || v.DriverVersion == "" || v.ToolVersion == "":
			r.Reason = "device or tool identity unavailable"
		case v.Properties.Scope != cupti.MetricCollectionScopeContext && v.Properties.Scope != cupti.MetricCollectionScopeDevice:
			r.Reason = "metric scope unavailable"
		case v.Properties.Type < cupti.MetricTypeCounter || v.Properties.Type >= cupti.MetricTypeCount:
			r.Reason = "metric type unavailable"
		case v.Properties.Name == "" || v.Properties.Description == "" || v.Properties.Dimension == "" || v.Properties.HardwareUnit == "" || v.Properties.CUPTIVersion == 0:
			r.Reason = "metric definition unavailable"
		case v.Replay == "" || !v.ReplayComplete:
			r.Reason = "replay incomplete or unspecified"
		case v.Overhead == "":
			r.Reason = "counter overhead disposition unavailable"
		case v.Value == nil || v.Failure != "":
			r.Reason = "counter value unavailable"
		case math.IsNaN(*v.Value) || math.IsInf(*v.Value, 0):
			r.Reason = "counter value nonfinite"
		}
		if r.Reason != "" {
			return r
		}
	}
	if !reflect.DeepEqual(base.Identity, target.Identity) || !reflect.DeepEqual(base.Properties, target.Properties) || base.Device != target.Device || base.Chip != target.Chip || base.DriverVersion != target.DriverVersion || base.ToolVersion != target.ToolVersion || base.Replay != target.Replay {
		r.Reason = "metric or collection context incompatible"
		return r
	}
	delta := *target.Value - *base.Value
	if math.IsNaN(delta) || math.IsInf(delta, 0) {
		r.Reason = "counter delta nonfinite"
		return r
	}
	r.Compatible, r.Delta = true, &delta
	return r
}

func qualifiedIdentity(id shape.Identity) bool {
	if (id.Phase != "decode" && id.Phase != "prefill") || id.Token < 0 || id.Operation == "" || id.DType == "" || id.CacheLength < 0 || (id.Layer != nil && *id.Layer < 0) {
		return false
	}
	for i := range id.Grid {
		if id.Grid[i] <= 0 || id.Block[i] <= 0 {
			return false
		}
	}
	return true
}

func validSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
