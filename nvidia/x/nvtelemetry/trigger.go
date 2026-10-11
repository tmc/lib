package nvtelemetry

// TriggerPolicy evaluates telemetry readings against an escalation threshold.
type TriggerPolicy struct {
	MetricName string  `json:"metric_name"`
	Threshold  float64 `json:"threshold"`
	Comparison string  `json:"comparison"` // ">", ">=", "<", "<="
	TenantID   string  `json:"tenant_id,omitempty"`
	Reason     string  `json:"reason,omitempty"`
}

// Matches reports whether reading r satisfies the trigger condition.
func (tp TriggerPolicy) Matches(r Reading) bool {
	if r.Name != tp.MetricName || !r.IsAvailable() {
		return false
	}
	val := r.Value
	if r.IsInt {
		val = float64(r.IntValue)
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
