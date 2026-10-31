package advise

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// CounterReference locates producer-retained evidence. AnalyzeCounters validates
// its shape; it does not read the locator or attest to the producer's authority.
type CounterReference struct {
	Locator string `json:"locator"`
	SHA256  string `json:"sha256"`
}

// CounterIdentity binds one action to its producer's exact evidence scope.
// ImageSHA256 identifies the execution environment image, not a code module.
type CounterIdentity struct {
	SourceSelectionSHA256 string           `json:"source_selection_sha256"`
	CandidateSourceSHA256 string           `json:"candidate_source_sha256"`
	CaseConfigSHA256      string           `json:"case_config_sha256"`
	ImageSHA256           string           `json:"image_sha256"`
	DeviceUUID            string           `json:"device_uuid"`
	RawReport             CounterReference `json:"raw_report"`
	TypedCensus           CounterReference `json:"typed_census"`
	Admission             CounterReference `json:"admission"`
}

// CounterLaunch is sourced launch geometry, not observed residency.
type CounterLaunch struct {
	Grid   [3]uint64        `json:"grid"`
	Source CounterReference `json:"source"`
}

// CounterSMCount is a sourced device SM count in the action's identity scope.
type CounterSMCount struct {
	Count  uint64           `json:"count"`
	Source CounterReference `json:"source"`
}

// CounterAction is one action in one acquisition. IDs are opaque producer IDs;
// kernel display names are not identities. Observations must name this ID and
// acquisition, with no invented pass identifier. Definition may retain vendor
// native-kind evidence; reported replay counts are quantities, not PassIDs.
// Supported scalar NumericTypes are uint64, int64 and double. The explicit
// unknown type retains opaque JSON for unrecognized metric names without facts;
// it must not be used to bypass the numeric contract of a recognized metric.
type CounterAction struct {
	ID            string                         `json:"id"`
	Kernel        string                         `json:"kernel"`
	AcquisitionID string                         `json:"acquisition_id"`
	Identity      CounterIdentity                `json:"identity"`
	Descriptors   []cudacapture.MetricDescriptor `json:"descriptors"`
	Observations  []cudacapture.Observation      `json:"observations"`
	Launch        *CounterLaunch                 `json:"launch,omitempty"`
	SMCount       *CounterSMCount                `json:"sm_count,omitempty"`
}

// CounterFinding names the exact action and metric descriptors behind an item.
// SourceRefs additionally identify optional launch and device facts used by it.
type CounterFinding struct {
	Item       Item               `json:"item"`
	ActionID   string             `json:"action_id"`
	MetricIDs  []string           `json:"metric_ids"`
	SourceRefs []CounterReference `json:"source_refs,omitempty"`
}

// CounterReport preserves unconverted inputs in acquisition order. It has no
// profile, cost ranking, critical path, or advice-outcome attribution.
type CounterReport struct {
	Mode        string           `json:"mode"`
	Actions     []CounterAction  `json:"actions"`
	Findings    []CounterFinding `json:"findings"`
	Limitations []string         `json:"limitations"`
}

type counterMeaning struct{ unit, scope, denominator, aggregation string }

var counterMeanings = map[string]counterMeaning{
	"gpu__time_duration.sum":                                          {"ns", "ACTION", "action", "sum"},
	"dram__bytes_read.sum":                                            {"byte", "ACTION", "action", "sum"},
	"dram__bytes_write.sum":                                           {"byte", "ACTION", "action", "sum"},
	"launch__occupancy_limit_registers":                               {"block", "BLOCK_LIMIT", "blocks_per_sm", "value"},
	"launch__occupancy_limit_shared_mem":                              {"block", "BLOCK_LIMIT", "blocks_per_sm", "value"},
	"launch__occupancy_limit_barriers":                                {"block", "BLOCK_LIMIT", "blocks_per_sm", "value"},
	"launch__occupancy_limit_warps":                                   {"block", "BLOCK_LIMIT", "blocks_per_sm", "value"},
	"launch__occupancy_limit_blocks":                                  {"block", "BLOCK_LIMIT", "blocks_per_sm", "value"},
	"smsp__sass_thread_inst_executed_op_integer_pred_on.sum":          {"inst", "THREAD", "predicated_on", "sum"},
	"smsp__inst_executed_pipe_tensor.sum":                             {"inst", "WARP", "tensor_pipe", "sum"},
	"smsp__issue_active.avg.pct_of_peak_sustained_active":             {"%", "ACTION", "active", "avg"},
	"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": {"%", "ACTION", "elapsed", "avg"},
	"sm__warps_active.avg.pct_of_peak_sustained_active":               {"%", "ACTION", "active", "avg"},
}

func counterDigest(s string) bool {
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32 && s == strings.ToLower(s)
}
func counterRef(r CounterReference) bool {
	return strings.TrimSpace(r.Locator) != "" && counterDigest(r.SHA256)
}

// counterNumber preserves the JSON number as an exact rational. In particular,
// uint64 never passes through float64, even for a derived display quantity.
func counterNumber(raw json.RawMessage, kind string) (*big.Rat, error) {
	var number json.Number
	if !json.Valid(raw) || json.Unmarshal(raw, &number) != nil || len(raw) == 0 {
		return nil, fmt.Errorf("not a JSON number")
	}
	// json.Number's decoder also accepts numeric strings; explicitly exclude them.
	s := strings.TrimSpace(string(raw))
	if s == "" || s[0] == '"' || s == "null" {
		return nil, fmt.Errorf("not a JSON number")
	}
	switch kind {
	case "uint64":
		if _, err := strconv.ParseUint(s, 10, 64); err != nil {
			return nil, fmt.Errorf("invalid uint64")
		}
	case "int64":
		if _, err := strconv.ParseInt(s, 10, 64); err != nil {
			return nil, fmt.Errorf("invalid int64")
		}
	case "double":
		// Reject values outside the finite native double range, without using the
		// rounded conversion in arithmetic or replacing the retained raw JSON.
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid finite double")
		}
		if f == 0 {
			significand := strings.SplitN(strings.ToLower(s), "e", 2)[0]
			if strings.ContainsAny(significand, "123456789") {
				return nil, fmt.Errorf("double underflow")
			}
		}
	default:
		return nil, fmt.Errorf("unsupported numeric type %q", kind)
	}
	n, ok := new(big.Rat).SetString(s)
	if !ok {
		return nil, fmt.Errorf("invalid number")
	}
	return n, nil
}

type counterValue struct {
	descriptor cudacapture.MetricDescriptor
	raw        json.RawMessage
	number     *big.Rat
}

// AnalyzeCounters checks structural joins and interprets a small set of named
// counter domains. Upstream admission, native correctness and evidence authority
// remain producer guarantees. Missing or unknown metrics do not become zero.
// No profile analysis, time attribution, ranking, or baseline comparison runs.
func AnalyzeCounters(actions []CounterAction) (*CounterReport, error) {
	// Own a snapshot, including RawMessage bytes, so later caller changes do not
	// detach report facts from the retained evidence.
	r := &CounterReport{Mode: "COUNTER_ONLY", Findings: []CounterFinding{}, Limitations: []string{
		"Structural validation only; upstream admission and native correctness are producer guarantees.",
		"Own-action counters only; no runtime attribution, critical path, peak fraction, cost ranking, baseline hardware comparison or advice success.",
		"Unknown or unavailable metrics remain raw and uninterpreted; optional missing facts remain unknown.",
	}}
	r.Actions = slices.Clone(actions)
	for i := range r.Actions {
		a := &r.Actions[i]
		a.Descriptors = slices.Clone(a.Descriptors)
		a.Observations = slices.Clone(a.Observations)
		for j := range a.Observations {
			a.Observations[j].Value = slices.Clone(a.Observations[j].Value)
		}
		if a.Launch != nil {
			launch := *a.Launch
			a.Launch = &launch
		}
		if a.SMCount != nil {
			sm := *a.SMCount
			a.SMCount = &sm
		}
	}
	ids := make(map[string]bool)
	for _, a := range r.Actions {
		if strings.TrimSpace(a.ID) == "" || strings.TrimSpace(a.Kernel) == "" || strings.TrimSpace(a.AcquisitionID) == "" || ids[a.ID] {
			return nil, fmt.Errorf("missing or duplicate action identity %q", a.ID)
		}
		ids[a.ID] = true
		identity := a.Identity
		for _, digest := range []string{identity.SourceSelectionSHA256, identity.CandidateSourceSHA256, identity.CaseConfigSHA256, identity.ImageSHA256} {
			if !counterDigest(digest) {
				return nil, fmt.Errorf("action %s: invalid identity digest", a.ID)
			}
		}
		if strings.TrimSpace(identity.DeviceUUID) == "" || !counterRef(identity.RawReport) || !counterRef(identity.TypedCensus) || !counterRef(identity.Admission) {
			return nil, fmt.Errorf("action %s: missing evidence identity", a.ID)
		}
		if a.Launch != nil {
			if !counterRef(a.Launch.Source) {
				return nil, fmt.Errorf("action %s: missing launch source", a.ID)
			}
			for _, dimension := range a.Launch.Grid {
				if dimension == 0 {
					return nil, fmt.Errorf("action %s: zero grid dimension", a.ID)
				}
			}
		}
		if a.SMCount != nil && (a.SMCount.Count == 0 || !counterRef(a.SMCount.Source)) {
			return nil, fmt.Errorf("action %s: invalid sourced SM count", a.ID)
		}
		descriptors := make(map[string]cudacapture.MetricDescriptor)
		names := make(map[string]bool)
		for _, d := range a.Descriptors {
			if d.ID == "" || d.Name == "" || names[d.Name] {
				return nil, fmt.Errorf("action %s: missing or ambiguous metric name", a.ID)
			}
			if _, ok := descriptors[d.ID]; ok {
				return nil, fmt.Errorf("action %s: duplicate descriptor", a.ID)
			}
			descriptors[d.ID] = d
			names[d.Name] = true
			if m, ok := counterMeanings[d.Name]; ok && (d.Unit != m.unit || d.Scope != m.scope || d.Denominator != m.denominator || d.Aggregation != m.aggregation) {
				return nil, fmt.Errorf("action %s metric %s: incompatible unit/domain/denominator/aggregation", a.ID, d.ID)
			}
		}
		seen := make(map[string]bool)
		values := make(map[string]counterValue)
		for _, o := range a.Observations {
			if len(o.Value) > 0 && !json.Valid(o.Value) {
				return nil, fmt.Errorf("action %s: invalid raw metric JSON", a.ID)
			}
			d, ok := descriptors[o.DescriptorID]
			if !ok || seen[o.DescriptorID] || o.EntityID != a.ID || o.RunID != a.AcquisitionID || o.PassID != "" || o.Start != 0 || o.End != 0 || o.Instant != 0 {
				return nil, fmt.Errorf("action %s: ambiguous or mismatched observation join", a.ID)
			}
			seen[o.DescriptorID] = true
			if o.Availability == "" {
				return nil, fmt.Errorf("action %s: missing metric availability", a.ID)
			}
			if o.Availability != "MEASURED" {
				continue
			}
			if d.NumericType == "unknown" {
				if !json.Valid(o.Value) {
					return nil, fmt.Errorf("action %s: invalid opaque metric JSON", a.ID)
				}
				if _, known := counterMeanings[d.Name]; known {
					return nil, fmt.Errorf("action %s: known metric has unsupported numeric type", a.ID)
				}
				continue
			}
			n, err := counterNumber(o.Value, d.NumericType)
			if err != nil {
				return nil, fmt.Errorf("action %s metric %s: %w", a.ID, d.ID, err)
			}
			if m, known := counterMeanings[d.Name]; known {
				if n.Sign() < 0 || (m.scope == "BLOCK_LIMIT" && !n.IsInt()) {
					return nil, fmt.Errorf("action %s metric %s: invalid quantity", a.ID, d.ID)
				}
				values[d.Name] = counterValue{d, o.Value, n}
			}
		}
		add := func(level EvidenceLevel, finding, next string, metricNames []string, refs ...CounterReference) {
			metricIDs := make([]string, 0, len(metricNames))
			for _, name := range metricNames {
				metricIDs = append(metricIDs, values[name].descriptor.ID)
			}
			r.Findings = append(r.Findings, CounterFinding{ActionID: a.ID, MetricIDs: metricIDs, SourceRefs: refs, Item: Item{Severity: SeverityContext, Category: CategoryArchitecture, EvidenceLevel: level, Location: Location{Kernel: a.Kernel}, Finding: finding, NextMeasurement: next, NoRuntimeAttribution: true}})
		}
		// Stable descriptor order, never value or estimated-cost ranking.
		for _, d := range a.Descriptors {
			if v, ok := values[d.Name]; ok {
				add(EvidenceObserved, fmt.Sprintf("%s = %s %s; domain %s; denominator %s; aggregation %s", d.Name, v.raw, d.Unit, d.Scope, d.Denominator, d.Aggregation), "", []string{d.Name})
			}
		}
		limits := []string{"launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_barriers", "launch__occupancy_limit_warps", "launch__occupancy_limit_blocks"}
		complete := true
		var minimum *big.Rat
		for _, name := range limits {
			v, ok := values[name]
			if !ok {
				complete = false
				break
			}
			if minimum == nil || v.number.Cmp(minimum) < 0 {
				minimum = v.number
			}
		}
		if complete && minimum.Sign() > 0 {
			reg := values[limits[0]].number
			shared := values[limits[1]].number
			if reg.Cmp(minimum) == 0 && shared.Cmp(minimum) == 0 {
				add(EvidenceSuspected, fmt.Sprintf("Registers and shared memory jointly limit reported capacity to %s blocks/SM; this is not achieved residency. A register-only remedy is unsupported.", minimum.RatString()), "Measure a variant that clears both resource constraints while retaining exact source/case and numerical checks.", limits)
			}
			if a.Launch != nil && a.SMCount != nil {
				grid := big.NewInt(1)
				for _, dimension := range a.Launch.Grid {
					grid.Mul(grid, new(big.Int).SetUint64(dimension))
				}
				slots := new(big.Int).Mul(minimum.Num(), new(big.Int).SetUint64(a.SMCount.Count))
				relation := "at least"
				if grid.Cmp(slots) < 0 {
					relation = "fewer than"
				}
				add(EvidenceObserved, fmt.Sprintf("Grid %s CTAs is %s %s nominal capacity slots (%s blocks/SM × %d sourced SMs); geometry alone does not establish residency, waves, idle time or a performance limit.", grid, relation, slots, minimum.RatString(), a.SMCount.Count), "", limits, a.Launch.Source, a.SMCount.Source)
			}
		}
		read, rok := values["dram__bytes_read.sum"]
		write, wok := values["dram__bytes_write.sum"]
		duration, dok := values["gpu__time_duration.sum"]
		if rok && wok && dok && duration.number.Sign() > 0 {
			total := new(big.Rat).Add(read.number, write.number)
			rate := new(big.Rat).Quo(total, duration.number)
			add(EvidenceObserved, fmt.Sprintf("Same-action DRAM traffic is %s bytes over %s ns; %s decimal GB/s (rounded to 12 decimal places). Total action traffic is not weight-only traffic, a peak fraction or a critical-path rate.", total.RatString(), duration.number.RatString(), rate.FloatString(12)), "", []string{"dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum"})
		}
	}
	return r, nil
}
