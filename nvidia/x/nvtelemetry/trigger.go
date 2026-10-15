package nvtelemetry

import "math"

// TriggerPolicy evaluates telemetry readings against an escalation threshold.
type TriggerPolicy struct {
	MetricName string  `json:"metric_name"`
	Threshold  float64 `json:"threshold"`
	Comparison string  `json:"comparison"` // ">", ">=", "<", "<="
	TenantID   string  `json:"tenant_id,omitempty"`
	Reason     string  `json:"reason,omitempty"`
}

// Matches reports whether reading r satisfies the trigger condition.
// Unavailable, histogram and malformed scalar readings never match.
// Integer readings are converted to float64 for comparison with Threshold;
// integers outside [-2^53, 2^53] may round. The Reading retains its exact value.
func (tp TriggerPolicy) Matches(r Reading) bool {
	if r.Name != tp.MetricName || !r.IsAvailable() {
		return false
	}
	val, ok := triggerScalar(r)
	if !ok {
		return false
	}
	switch tp.Comparison {
	case ">":
		return val > tp.Threshold
	case ">=":
		return val >= tp.Threshold
	case "<":
		return val < tp.Threshold
	case "<=":
		return val <= tp.Threshold
	default:
		return val > tp.Threshold
	}
}

// EvaluateSample returns the first Reading in sample matching any policy in policies.
func EvaluateSample(sample Sample, policies []TriggerPolicy) (Reading, TriggerPolicy, bool) {
	for _, reading := range sample.Readings {
		for _, policy := range policies {
			if policy.Matches(reading) {
				return reading, policy, true
			}
		}
	}
	return Reading{}, TriggerPolicy{}, false
}

// triggerScalar returns the available scalar in the threshold's float64 domain.
func triggerScalar(r Reading) (float64, bool) {
	if r.Observation != nil && (r.Observation.Kind == "histogram" || r.Observation.Histogram != nil) {
		return 0, false
	}
	value := r.Value
	if n := r.Number; n != nil {
		if r.Value != 0 || r.IntValue != 0 || r.IsInt || validNumber(n) != nil {
			return 0, false
		}
		switch {
		case n.Float != nil:
			value = *n.Float
		case n.Int != nil:
			value = float64(*n.Int)
		case n.Uint != nil:
			value = float64(*n.Uint)
		}
	} else if r.Observation != nil {
		return 0, false
	} else if r.IsInt {
		value = float64(r.IntValue)
	}
	return value, !math.IsNaN(value) && !math.IsInf(value, 0)
}
