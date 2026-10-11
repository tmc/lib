package nvtelemetry

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strings"
)

// AccountingClock pins the caller's mapping from a monotonic epoch to Unix
// nanoseconds. Billing admission requires an explicitly known zero uncertainty;
// this implementation does not discover or qualify that mapping.
type AccountingClock struct {
	Domain           string `json:"domain"`
	Epoch            string `json:"epoch"`
	UnixOffsetNanos  int64  `json:"unix_offset_nanos"`
	MappingKnown     bool   `json:"mapping_known"`
	UncertaintyKnown bool   `json:"uncertainty_known"`
	UncertaintyNanos int64  `json:"uncertainty_nanos"`
	ValidFrom        int64  `json:"valid_from"`
	ValidUntil       int64  `json:"valid_until"`
	Evidence         string `json:"evidence"`
}

// AccountingEnergy is one complete physical cumulative counter interval.
// Lineage identifies the original counter, not an ingestion path. Duplicate
// or overlapping intervals from the same physical device are rejected.
type AccountingEnergy struct {
	Evidence           string        `json:"evidence"` // measured, not derived
	ClockDomain        string        `json:"clock_domain"`
	ClockEpoch         string        `json:"clock_epoch"`
	Scope              string        `json:"scope"` // physical_device
	Summary            EnergySummary `json:"summary"`
	Lineage            string        `json:"lineage"`
	RawDigest          string        `json:"raw_digest"`
	ConfigurationEpoch string        `json:"configuration_epoch,omitempty"`
}

// AccountingAllocation pairs an authoritative assignment with an explicit
// modeled share. CapacityPermille is allocated capacity, not measured work or
// measured per-owner energy. Policy must be capacity_time_v1.
type AccountingAllocation struct {
	ID               string           `json:"id"`
	Assignment       EntityAssignment `json:"assignment"`
	CapacityPermille uint32           `json:"capacity_permille"`
	Policy           string           `json:"policy"`
	PolicyEvidence   string           `json:"policy_evidence"`
}

// AccountingRate is a caller-pinned, effective-dated price for one physical
// device. Price and DiscountFactor are exact decimal or rational numbers.
// Rounding is per allocation and rate segment, never an invoice reconstruction.
type AccountingRate struct {
	DeviceUUID              string `json:"device_uuid"`
	EffectiveFromUnixNanos  int64  `json:"effective_from_unix_nanos"`
	EffectiveUntilUnixNanos int64  `json:"effective_until_unix_nanos"`
	Currency                string `json:"currency"`
	MinorDigits             uint32 `json:"minor_digits"`
	PricePerGPUHour         string `json:"price_per_gpu_hour"`
	DiscountFactor          string `json:"discount_factor"`
	Region                  string `json:"region"`
	SKU                     string `json:"sku"`
	Scope                   string `json:"scope"` // device_usage_only
	BillingQuantumNanos     int64  `json:"billing_quantum_nanos"`
	CommitmentAssumption    string `json:"commitment_assumption"`
	Source                  string `json:"source"`
	Revision                string `json:"revision"`
}

// AccountingWork contains independently admitted successful, correct work.
// Aggregate serving token metrics must not be copied here without evidence
// that they obey the named outcome and correctness policy.
type AccountingWork struct {
	ID           string `json:"id"`
	AllocationID string `json:"allocation_id"`
	Units        uint64 `json:"units"`
	Unit         string `json:"unit"`
	Success      bool   `json:"success"`
	Correct      bool   `json:"correct"`
	Policy       string `json:"policy"`
	Evidence     string `json:"evidence"`
	ClockDomain  string `json:"clock_domain"`
	ClockEpoch   string `json:"clock_epoch"`
	StartNanos   int64  `json:"start_nanos"`
	EndNanos     int64  `json:"end_nanos"`
}

// AccountingInput is version 1 of the bounded accounting consumer format.
// Windows, allocations, rates and work each contain at most 128 entries.
// Baseline subtraction and whole-node energy are not inferred from device data.
type AccountingInput struct {
	Version          int                    `json:"version"`
	Nodes            []AccountingNode       `json:"nodes,omitempty"`
	Idle             []AccountingIdle       `json:"idle,omitempty"`
	Clock            AccountingClock        `json:"clock"`
	Energy           []AccountingEnergy     `json:"energy"`
	Baselines        []AccountingEnergy     `json:"baselines,omitempty"`
	BaselinePolicy   string                 `json:"baseline_policy,omitempty"`
	BaselineEvidence string                 `json:"baseline_evidence,omitempty"`
	WorkPolicy       string                 `json:"work_policy,omitempty"`
	Allocations      []AccountingAllocation `json:"allocations"`
	Rates            []AccountingRate       `json:"rates"`
	Work             []AccountingWork       `json:"work,omitempty"`
}

// AccountingShare records an exact rational estimate. Rational strings use
// numerator/denominator notation; exact physical totals remain integer MJ.
type AccountingShare struct {
	AllocationID          string `json:"allocation_id"`
	EnergyMillijoules     string `json:"energy_millijoules"`
	AllocatedGPUHours     string `json:"allocated_gpu_hours"`
	BilledGPUHours        string `json:"billed_gpu_hours"`
	CostMajor             string `json:"cost_major"`
	RoundedMinor          string `json:"rounded_minor"`
	Currency              string `json:"currency"`
	MinorDigits           uint32 `json:"minor_digits"`
	WorkUnit              string `json:"work_unit,omitempty"`
	WorkUnits             string `json:"work_units,omitempty"`
	WorkPerAllocatedJoule string `json:"work_per_allocated_joule,omitempty"`
	EfficiencyUnavailable string `json:"efficiency_unavailable,omitempty"`
}

// AccountingReport preserves input lineage and assumptions alongside modeled
// shares. Residual energy is unallocated, not an independently measured idle
// quantity. Cost excludes power infrastructure, taxes and other invoice scope.
type AccountingReport struct {
	Nodes                             []AccountingNodeTotal `json:"nodes,omitempty"`
	IdleEnergyMillijoules             string                `json:"idle_energy_millijoules,omitempty"`
	UnclassifiedResidualMillijoules   string                `json:"unclassified_residual_millijoules,omitempty"`
	EnergyAvailable                   bool                  `json:"energy_available"`
	EnergyUnavailableReason           string                `json:"energy_unavailable_reason,omitempty"`
	EnergyClass                       string                `json:"energy_class"`
	CostClass                         string                `json:"cost_class"`
	Version                           int                   `json:"version"`
	Input                             AccountingInput       `json:"input"`
	Method                            string                `json:"method"`
	PhysicalEnergyMillijoules         string                `json:"physical_energy_millijoules"`
	BaselineAdjustedEnergyMillijoules string                `json:"baseline_adjusted_energy_millijoules,omitempty"`
	ResidualEnergyMillijoules         string                `json:"residual_energy_millijoules"`
	Shares                            []AccountingShare     `json:"shares"`
}

func accountingNumber(s string) (*big.Rat, error) {
	if len(s) == 0 || len(s) > 64 {
		return nil, fmt.Errorf("accounting: invalid numeric precision")
	}
	for _, c := range s {
		if (c < '0' || c > '9') && c != '.' && c != '/' {
			return nil, fmt.Errorf("accounting: price must be exact decimal or rational")
		}
	}
	r, ok := new(big.Rat).SetString(s)
	if !ok || r.Sign() < 0 || r.Num().BitLen() > 256 || r.Denom().BitLen() > 256 {
		return nil, fmt.Errorf("accounting: invalid nonnegative number %q", s)
	}
	return r, nil
}
func accountingDuration(start, end int64) (int64, error) {
	if start < 0 || end <= start {
		return 0, fmt.Errorf("accounting: invalid interval")
	}
	return end - start, nil
}
func accountingPhysicalDevice(a EntityAssignment) string {
	if a.Sharing == "mig" {
		return a.Device.MIGParentUUID
	}
	return a.Device.UUID
}
func accountingOverlap(a, b, c, d int64) (int64, int64) {
	if a < c {
		a = c
	}
	if b > d {
		b = d
	}
	return a, b
}
func accountingRat(v int64) *big.Rat { return new(big.Rat).SetInt64(v) }
func accountingHours(nanos *big.Int, permille uint32) *big.Rat {
	return new(big.Rat).SetFrac(new(big.Int).Mul(nanos, new(big.Int).SetUint64(uint64(permille))), big.NewInt(3600000000000000))
}
func accountingRound(r *big.Rat, digits uint32) string {
	factor := new(big.Int).Exp(big.NewInt(10), new(big.Int).SetUint64(uint64(digits)), nil)
	n := new(big.Int).Mul(r.Num(), factor)
	q, rem := new(big.Int), new(big.Int)
	q.QuoRem(n, r.Denom(), rem)
	cmp := new(big.Int).Lsh(rem, 1).Cmp(r.Denom())
	if cmp > 0 || (cmp == 0 && q.Bit(0) != 0) {
		q.Add(q, big.NewInt(1))
	}
	return q.String()
}

// AccountEnergy apportions complete physical intervals by allocated capacity
// time under a uniform-energy-rate assumption, using exact rational arithmetic.
// It refuses unknown sharing, ambiguous clocks, missing coverage, resets,
// overlapping physical totals or capacity above one device. Idle and omitted
// capacity remain explicit residuals. Costs are model-only caller-rate estimates.
// With no supplied energy, allocated hours and costs remain available while
// physical energy, residuals and energy efficiency are explicitly unavailable.
// Malformed supplied counters are errors, never a cost-only fallback.
func AccountEnergy(in AccountingInput) (AccountingReport, error) {
	out := AccountingReport{EnergyClass: "capacity_time_allocation_estimate", CostClass: "caller_rate_model_only_not_invoice", Version: 1, Input: in, Method: "capacity_time_v1; uniform_energy_within_each_counter_interval; per_allocation_rate_segment_ceil; half_even_minor_rounding"}
	refuse := func(s string) (AccountingReport, error) { return out, fmt.Errorf("accounting: %s", s) }
	if len(in.Nodes) > 128 || len(in.Idle) > 128 {
		return refuse("scope evidence budget exceeded")
	}
	if in.Version != 1 || (len(in.Energy) == 0 && len(in.Allocations) == 0 && len(in.Nodes) == 0) || len(in.Energy) > 128 || len(in.Baselines) > 128 || len(in.Allocations) > 128 || len(in.Rates) > 128 || len(in.Work) > 128 {
		return refuse("unsupported version or input budget")
	}
	hasEnergy := len(in.Energy) > 0
	if !hasEnergy {
		out.EnergyClass = "physical_energy_unavailable"
		out.EnergyUnavailableReason = "physical_energy_unavailable"
	}
	clock := in.Clock
	if clock.Domain == "" || clock.Epoch == "" || clock.Evidence == "" || !clock.MappingKnown || !clock.UncertaintyKnown || clock.UncertaintyNanos != 0 || clock.ValidFrom < 0 || clock.ValidUntil <= clock.ValidFrom {
		return refuse("unknown or uncertain clock mapping")
	}
	physical := new(big.Int)
	seen := make(map[string]bool)
	for i, e := range in.Energy {
		s := e.Summary
		if raw, err := hex.DecodeString(e.RawDigest); err != nil || len(raw) != 32 {
			return refuse("invalid raw source digest")
		}
		if math.IsNaN(s.EnergyConsumedJoules) || math.IsInf(s.EnergyConsumedJoules, 0) || s.EnergyConsumedJoules != float64(s.EnergyConsumedMJ)/1000 {
			return refuse("contradictory derived joule copy")
		}
		duration, err := accountingDuration(s.StartMonoNanos, s.EndMonoNanos)
		if err != nil {
			return out, err
		}
		if e.ClockDomain != clock.Domain || e.ClockEpoch != clock.Epoch {
			return refuse("unknown or mismatched energy clock epoch")
		}
		if e.Evidence != EvidenceMeasured || e.Scope != "physical_device" || e.Lineage == "" || e.RawDigest == "" || s.DeviceUUID == "" || !s.Valid || s.ResetDetected || s.Reason != "" || s.Method != "cumulative_counter" || s.CoveredDurationNanos != duration || s.TotalDurationNanos != duration || s.CoverageRatio != 1 || s.EndEnergyMJ < s.StartEnergyMJ || s.EndEnergyMJ-s.StartEnergyMJ != s.EnergyConsumedMJ {
			return refuse("unavailable, partial, reset, or contradictory physical energy")
		}
		if s.StartMonoNanos < clock.ValidFrom || s.EndMonoNanos > clock.ValidUntil {
			return refuse("energy outside clock mapping")
		}
		id := fmt.Sprintf("%s/%d/%d", e.Lineage, s.StartMonoNanos, s.EndMonoNanos)
		if seen[id] {
			return refuse("duplicate counter lineage")
		}
		seen[id] = true
		for _, p := range in.Energy[:i] {
			a, b := accountingOverlap(s.StartMonoNanos, s.EndMonoNanos, p.Summary.StartMonoNanos, p.Summary.EndMonoNanos)
			if e.Lineage == p.Lineage && s.DeviceUUID != p.Summary.DeviceUUID {
				return refuse("counter lineage changes physical device")
			}
			if e.Lineage == p.Lineage {
				if s.StartMonoNanos >= p.Summary.EndMonoNanos && s.StartEnergyMJ < p.Summary.EndEnergyMJ {
					return refuse("counter reset between intervals")
				}
				if p.Summary.StartMonoNanos >= s.EndMonoNanos && p.Summary.StartEnergyMJ < s.EndEnergyMJ {
					return refuse("counter reset between intervals")
				}
			}
			if a < b && (s.DeviceUUID == p.Summary.DeviceUUID || e.Lineage == p.Lineage) {
				return refuse("overlapping physical counter lineage")
			}
		}
		physical.Add(physical, new(big.Int).SetUint64(s.EnergyConsumedMJ))
	}
	if len(in.Baselines) > 0 {
		if in.BaselinePolicy != "uniform_baseline_rate_v1" || in.BaselineEvidence == "" {
			return refuse("missing baseline policy")
		}
		baselineDevices := make(map[string]bool)
		adjusted := new(big.Rat).SetInt(physical)
		for _, b := range in.Baselines {
			s := b.Summary
			duration, err := accountingDuration(s.StartMonoNanos, s.EndMonoNanos)
			if err != nil {
				return out, err
			}
			if math.IsNaN(s.EnergyConsumedJoules) || math.IsInf(s.EnergyConsumedJoules, 0) || s.EnergyConsumedJoules != float64(s.EnergyConsumedMJ)/1000 {
				return refuse("contradictory baseline joule copy")
			}
			if s.StartMonoNanos < clock.ValidFrom || s.EndMonoNanos > clock.ValidUntil {
				return refuse("baseline outside clock mapping")
			}
			if raw, err := hex.DecodeString(b.RawDigest); err != nil || len(raw) != 32 {
				return refuse("invalid baseline source digest")
			}
			if b.ClockDomain != clock.Domain || b.ClockEpoch != clock.Epoch {
				return refuse("unknown or mismatched baseline clock epoch")
			}
			if b.Evidence != EvidenceMeasured || b.Scope != "physical_device" || baselineDevices[s.DeviceUUID] || b.Lineage == "" || b.RawDigest == "" || s.DeviceUUID == "" || !s.Valid || s.ResetDetected || s.Reason != "" || s.Method != "cumulative_counter" || s.TotalDurationNanos != duration || s.CoveredDurationNanos != duration || s.CoverageRatio != 1 || s.EndEnergyMJ < s.StartEnergyMJ || s.EndEnergyMJ-s.StartEnergyMJ != s.EnergyConsumedMJ {
				return refuse("invalid baseline counter")
			}
			baselineDevices[s.DeviceUUID] = true
			matched := false
			for _, e := range in.Energy {
				v := e.Summary
				if v.DeviceUUID != s.DeviceUUID {
					continue
				}
				matched = true
				if e.ConfigurationEpoch != b.ConfigurationEpoch {
					return refuse("baseline configuration differs")
				}
				x, y := accountingOverlap(v.StartMonoNanos, v.EndMonoNanos, s.StartMonoNanos, s.EndMonoNanos)
				if x < y {
					return refuse("baseline overlaps observation")
				}
				subtract := new(big.Rat).Mul(new(big.Rat).SetUint64(s.EnergyConsumedMJ), new(big.Rat).SetFrac(big.NewInt(v.EndMonoNanos-v.StartMonoNanos), big.NewInt(duration)))
				if subtract.Cmp(new(big.Rat).SetUint64(v.EnergyConsumedMJ)) > 0 {
					return refuse("negative baseline-adjusted estimate")
				}
				adjusted.Sub(adjusted, subtract)
			}
			if !matched {
				return refuse("baseline has unmatched physical scope")
			}
		}
		out.BaselineAdjustedEnergyMillijoules = adjusted.RatString()
	}
	workIDs := make(map[string]bool)
	for _, w := range in.Work {
		if w.ID == "" || workIDs[w.ID] || w.Unit == "" || w.Policy == "" || w.Policy != in.WorkPolicy || w.Evidence == "" {
			return refuse("missing, duplicate, or contradictory successful-work policy")
		}
		workIDs[w.ID] = true
	}

	ids := make(map[string]bool)
	for _, a := range in.Allocations {
		if a.ID == "" || ids[a.ID] || a.Policy != "capacity_time_v1" || a.PolicyEvidence == "" || a.CapacityPermille == 0 || a.CapacityPermille > 1000 {
			return refuse("invalid allocation identity, capacity, or policy")
		}
		ids[a.ID] = true
		if err := a.Assignment.Validate(); err != nil {
			return out, err
		}
		v := a.Assignment
		if v.StartNanos < clock.ValidFrom || v.EndNanos > clock.ValidUntil {
			return refuse("allocation outside clock mapping validity")
		}
		if v.Owner.AllocationID != a.ID {
			return refuse("allocation does not match authoritative owner identity")
		}
		if !v.Authoritative || v.State != "assigned" || v.Sharing == "unknown" || v.ClockDomain != clock.Domain || v.ClockEpoch != clock.Epoch {
			return refuse("unknown sharing, assignment, or clock")
		}
		matched := false
		for _, e := range in.Energy {
			if e.Summary.DeviceUUID != accountingPhysicalDevice(v) {
				continue
			}
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, e.Summary.StartMonoNanos, e.Summary.EndMonoNanos)
			if x < y {
				matched = true
				if v.Sharing == "mig" && (e.ConfigurationEpoch == "" || e.ConfigurationEpoch != v.MIGConfigurationEpoch) {
					return refuse("mig reconfiguration epoch mismatch")
				}
			}
		}
		if hasEnergy && !matched {
			return refuse("allocation lacks physical energy scope")
		}
	}
	for _, w := range in.Work {
		if !ids[w.AllocationID] {
			return refuse("unknown work allocation")
		}
	}
	// Capacity admission also applies when only financial estimates are requested.
	capacityWindows := make([]EnergySummary, 0, len(in.Energy)+len(in.Allocations))
	for _, e := range in.Energy {
		capacityWindows = append(capacityWindows, e.Summary)
	}
	if !hasEnergy {
		for _, a := range in.Allocations {
			v := a.Assignment
			capacityWindows = append(capacityWindows, EnergySummary{DeviceUUID: accountingPhysicalDevice(v), StartMonoNanos: v.StartNanos, EndMonoNanos: v.EndNanos})
		}
	}
	// Capacity constraints are evaluated at every interval boundary, not merely
	// as the average of overlapping allocations.
	for _, s := range capacityWindows {
		points := []int64{s.StartMonoNanos, s.EndMonoNanos}
		for _, a := range in.Allocations {
			v := a.Assignment
			if accountingPhysicalDevice(v) != s.DeviceUUID {
				continue
			}
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, s.StartMonoNanos, s.EndMonoNanos)
			if x < y {
				points = append(points, x, y)
			}
		}
		sort.Slice(points, func(i, j int) bool { return points[i] < points[j] })
		for _, at := range points[:len(points)-1] {
			var capacity uint64
			var active, exclusive int
			for _, a := range in.Allocations {
				v := a.Assignment
				if accountingPhysicalDevice(v) == s.DeviceUUID && v.StartNanos <= at && at < v.EndNanos {
					capacity += uint64(a.CapacityPermille)
					active++
					if v.Sharing == "exclusive" {
						exclusive++
					}
				}
			}
			if exclusive > 0 && active > 1 {
				return refuse("overlapping exclusive allocation claims")
			}
			if capacity > 1000 {
				return refuse("overallocated physical capacity")
			}
		}
	}
	prices := make([]*big.Rat, len(in.Rates))
	discounts := make([]*big.Rat, len(in.Rates))
	for i, r := range in.Rates {
		if r.DeviceUUID == "" || r.EffectiveUntilUnixNanos <= r.EffectiveFromUnixNanos || r.Currency == "" || r.MinorDigits > 6 || r.Region == "" || r.SKU == "" || r.Scope != "device_usage_only" || r.BillingQuantumNanos <= 0 || r.CommitmentAssumption == "" || r.Source == "" || r.Revision == "" {
			return refuse("incomplete rate provenance or billing scope")
		}
		var err error
		prices[i], err = accountingNumber(r.PricePerGPUHour)
		if err != nil {
			return out, err
		}
		discounts[i], err = accountingNumber(r.DiscountFactor)
		if err != nil {
			return out, err
		}
		if discounts[i].Cmp(accountingRat(1)) > 0 {
			return refuse("discount factor exceeds one")
		}
		for _, p := range in.Rates[:i] {
			x, y := accountingOverlap(r.EffectiveFromUnixNanos, r.EffectiveUntilUnixNanos, p.EffectiveFromUnixNanos, p.EffectiveUntilUnixNanos)
			if r.DeviceUUID == p.DeviceUUID && x < y {
				return refuse("overlapping rate schedules")
			}
		}
	}
	allocated := new(big.Rat)
	for _, a := range in.Allocations {
		energy, hours, billed, cost := new(big.Rat), new(big.Rat), new(big.Rat), new(big.Rat)
		currency := ""
		var digits uint32
		v := a.Assignment
		matchedDuration := int64(0)
		for _, e := range in.Energy {
			s := e.Summary
			if s.DeviceUUID != accountingPhysicalDevice(v) {
				continue
			}
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, s.StartMonoNanos, s.EndMonoNanos)
			if x >= y {
				continue
			}
			matchedDuration += y - x
			portion := new(big.Rat).SetFrac(new(big.Int).Mul(new(big.Int).SetUint64(s.EnergyConsumedMJ), new(big.Int).Mul(big.NewInt(y-x), new(big.Int).SetUint64(uint64(a.CapacityPermille)))), new(big.Int).Mul(big.NewInt(s.EndMonoNanos-s.StartMonoNanos), big.NewInt(1000)))
			energy.Add(energy, portion)
		}
		// Billing uses the full assignment interval, not the observed energy subset.
		// Refuse missing physical windows rather than hiding an allocation gap.
		duration, err := accountingDuration(v.StartNanos, v.EndNanos)
		if err != nil {
			return out, err
		}
		if hasEnergy && matchedDuration != duration {
			return refuse("allocation interval has missing energy coverage")
		}
		hours = accountingHours(big.NewInt(duration), a.CapacityPermille)
		mappedStart := new(big.Int).Add(big.NewInt(v.StartNanos), big.NewInt(clock.UnixOffsetNanos))
		mappedEnd := new(big.Int).Add(big.NewInt(v.EndNanos), big.NewInt(clock.UnixOffsetNanos))
		if !mappedStart.IsInt64() || !mappedEnd.IsInt64() {
			return refuse("mapped billing interval overflow")
		}
		start, end := mappedStart.Int64(), mappedEnd.Int64()
		covered := int64(0)
		for i, r := range in.Rates {
			if r.DeviceUUID != accountingPhysicalDevice(v) {
				continue
			}
			x, y := accountingOverlap(start, end, r.EffectiveFromUnixNanos, r.EffectiveUntilUnixNanos)
			if x >= y {
				continue
			}
			if currency != "" && (currency != r.Currency || digits != r.MinorDigits) {
				return refuse("currency or money precision changes within allocation")
			}
			currency = r.Currency
			digits = r.MinorDigits
			covered += y - x
			quantum := big.NewInt(r.BillingQuantumNanos)
			ns := big.NewInt(y - x)
			ns.Add(ns, new(big.Int).Sub(quantum, big.NewInt(1)))
			ns.Quo(ns, quantum)
			ns.Mul(ns, quantum)
			bh := accountingHours(ns, a.CapacityPermille)
			billed.Add(billed, bh)
			c := new(big.Rat).Mul(bh, prices[i])
			c.Mul(c, discounts[i])
			cost.Add(cost, c)
		}
		if covered != duration {
			return refuse("missing effective-dated rate coverage")
		}
		share := AccountingShare{AllocationID: a.ID, EnergyMillijoules: energy.RatString(), AllocatedGPUHours: hours.RatString(), BilledGPUHours: billed.RatString(), CostMajor: cost.RatString(), RoundedMinor: accountingRound(cost, digits), Currency: currency, MinorDigits: digits, EfficiencyUnavailable: "successful_correct_work_unavailable"}
		if !hasEnergy {
			share.EnergyMillijoules = ""
			share.EfficiencyUnavailable = "physical_energy_unavailable"
		}
		work := new(big.Int)
		unit := ""
		for _, w := range in.Work {
			if !ids[w.AllocationID] || w.Unit == "" || w.Policy == "" || w.Evidence == "" {
				return refuse("unknown work allocation or outcome policy")
			}
			if w.AllocationID == a.ID && (w.ClockDomain != clock.Domain || w.ClockEpoch != clock.Epoch || w.StartNanos < v.StartNanos || w.EndNanos > v.EndNanos || w.EndNanos < w.StartNanos) {
				return refuse("work outside allocation clock interval")
			}
			if w.AllocationID != a.ID || !w.Success || !w.Correct {
				continue
			}
			if unit != "" && unit != w.Unit {
				return refuse("mixed work units")
			}
			unit = w.Unit
			work.Add(work, new(big.Int).SetUint64(w.Units))
		}
		if unit != "" {
			share.WorkUnit = unit
			share.WorkUnits = work.String()
			if !hasEnergy {
				share.EfficiencyUnavailable = "physical_energy_unavailable"
			} else if work.Sign() > 0 && energy.Sign() > 0 {
				ratio := new(big.Rat).Quo(new(big.Rat).SetInt(work), new(big.Rat).Quo(energy, accountingRat(1000)))
				share.WorkPerAllocatedJoule = ratio.RatString()
				share.EfficiencyUnavailable = ""
			} else {
				share.EfficiencyUnavailable = "zero_work_or_allocated_energy"
			}
		}
		allocated.Add(allocated, energy)
		out.Shares = append(out.Shares, share)
	}
	if !hasEnergy {
		if err := accountingScopes(in, &out); err != nil {
			return out, err
		}
		return out, nil
	}
	out.EnergyAvailable = true
	out.PhysicalEnergyMillijoules = physical.String()
	residual := new(big.Rat).Sub(new(big.Rat).SetInt(physical), allocated)
	if residual.Sign() < 0 {
		return refuse("negative energy residual")
	}
	out.ResidualEnergyMillijoules = residual.RatString()
	if err := accountingScopes(in, &out); err != nil {
		return out, err
	}
	return out, nil
}

// ReadAccounting consumes one versioned JSON document with a caller byte
// budget, rejecting unknown fields, trailing documents and structural defects.
// maxBytes must be positive and at most 16 MiB.
func ReadAccounting(r io.Reader, maxBytes int64) (AccountingReport, error) {
	if r == nil || maxBytes <= 0 || maxBytes > 16<<20 {
		return AccountingReport{}, fmt.Errorf("accounting: invalid reader budget")
	}
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return AccountingReport{}, err
	}
	if int64(len(data)) > maxBytes {
		return AccountingReport{}, fmt.Errorf("accounting: input budget exceeded")
	}
	if err := accountingJSON(data, reflect.TypeOf(AccountingInput{})); err != nil {
		return AccountingReport{}, err
	}
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	var in AccountingInput
	if err = d.Decode(&in); err != nil {
		return AccountingReport{}, fmt.Errorf("accounting: decode: %w", err)
	}
	var extra any
	if err = d.Decode(&extra); err != io.EOF {
		return AccountingReport{}, fmt.Errorf("accounting: trailing document")
	}
	return AccountEnergy(in)
}

// WriteAccounting validates and writes a versioned input suitable for
// ReadAccounting. It does not persist data, bill a customer, or change rates.
func WriteAccounting(w io.Writer, in AccountingInput) error {
	if _, err := AccountEnergy(in); err != nil {
		return err
	}
	return writeAccountingJSON(w, in)
}

// WriteAccountingReport writes a model-only result, including its input
// provenance. CostMajor is exact; RoundedMinor uses half-even currency rounding.
func WriteAccountingReport(w io.Writer, report AccountingReport) error {
	expected, err := AccountEnergy(report.Input)
	if err != nil {
		return err
	}
	actualJSON, err := json.Marshal(report)
	if err != nil {
		return err
	}
	expectedJSON, err := json.Marshal(expected)
	if err != nil {
		return err
	}
	if !bytes.Equal(actualJSON, expectedJSON) {
		return fmt.Errorf("accounting: report contradicts retained input")
	}
	return writeAccountingJSON(w, report)
}
func writeAccountingJSON(w io.Writer, value any) error {
	if w == nil {
		return fmt.Errorf("accounting: nil writer")
	}
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(data) > 16<<20 {
		return fmt.Errorf("accounting: output budget exceeded")
	}
	data = append(data, '\n')
	n, err := w.Write(data)
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	return err
}

// AccountTelemetry reads a complete raw telemetry stream and feeds its exact
// cumulative counter delta into the accounting consumer. Boundary readings
// must have known source timestamps in the manifest clock, at the requested
// boundaries; query times never establish sensor timing. The caller supplies
// the explicit epoch/mapping evidence and allocation/rate/outcome policies.
// Native streams with unknown sensor timestamps are refused, not upgraded.
func AccountTelemetry(r io.Reader, in AccountingInput, window string, start, end, maxBytes int64) (AccountingReport, error) {
	if len(in.Energy) != 0 {
		return AccountingReport{}, fmt.Errorf("accounting: raw import requires no precomputed energy")
	}
	// Retain a bounded raw digest in the report without consuming a second source.
	if r == nil || maxBytes <= 0 || maxBytes > 16<<20 {
		return AccountingReport{}, fmt.Errorf("accounting: invalid raw import budget")
	}
	data, err := io.ReadAll(io.LimitReader(r, maxBytes+1))
	if err != nil {
		return AccountingReport{}, err
	}
	if int64(len(data)) > maxBytes {
		return AccountingReport{}, fmt.Errorf("accounting: raw import budget exceeded")
	}
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if err := accountingJSON(line, reflect.TypeOf(StreamRecord{})); err != nil {
			return AccountingReport{}, err
		}
	}
	manifest, _, samples, err := ReadStreamStrict(bytes.NewReader(data), maxBytes)
	if err != nil {
		return AccountingReport{}, err
	}
	if !manifest.HasClockCalibration || manifest.ClockCalibrationBoundsNanos != 0 || start < manifest.ClockMappingValidFrom || end > manifest.ClockMappingValidUntil || manifest.ClockMonotonicDomain != in.Clock.Domain || manifest.ClockMonotonicEpochID != in.Clock.Epoch {
		return AccountingReport{}, fmt.Errorf("accounting: raw source clock mapping unknown or inconsistent")
	}
	if len(samples) > 128 || len(manifest.Devices) != 1 {
		return AccountingReport{}, fmt.Errorf("accounting: raw import needs bounded single-device capture")
	}
	for _, sample := range samples {
		if sample.QueryStartNanos < start || sample.QueryStartNanos > end {
			continue
		}
		if sample.ClockDiscontinuity {
			return AccountingReport{}, fmt.Errorf("accounting: discontinuous raw source clock")
		}
		for _, reading := range sample.Readings {
			if reading.Name != MetricTotalEnergyMJ {
				continue
			}
			if reading.Evidence != EvidenceMeasured || !reading.IsInt || !reading.IsAvailable() || !reading.HasSourceTimestamp || reading.SourceTimestampDomain != manifest.ClockMonotonicDomain || reading.SourceTimestamp != sample.QueryStartNanos {
				return AccountingReport{}, fmt.Errorf("accounting: unknown or misaligned counter sensor timestamp")
			}
		}
	}
	_, energy := AggregateWindow(window, start, end, samples, WindowOptions{})
	if len(energy) != 1 {
		return AccountingReport{}, fmt.Errorf("accounting: missing or ambiguous cumulative counter")
	}
	sum := sha256.Sum256(data)
	in.Energy = []AccountingEnergy{{Evidence: EvidenceMeasured, ClockDomain: manifest.ClockMonotonicDomain, ClockEpoch: manifest.ClockMonotonicEpochID, Scope: "physical_device", Summary: energy[0], Lineage: manifest.RunID + "/" + energy[0].DeviceUUID + "/" + MetricTotalEnergyMJ, RawDigest: hex.EncodeToString(sum[:])}}
	return AccountEnergy(in)
}

// accountingJSON preserves canonical field spelling and rejects duplicate keys
// before encoding/json's permissive struct matching can discard contradictions.
func accountingJSON(data []byte, typ reflect.Type) error {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var read func(reflect.Type, int) error
	read = func(typ reflect.Type, depth int) error {
		if depth > 64 {
			return fmt.Errorf("accounting: json nesting budget exceeded")
		}
		nullable := false
		for typ != nil && typ.Kind() == reflect.Pointer {
			nullable = true
			typ = typ.Elem()
		}
		token, err := d.Token()
		if err != nil {
			return err
		}
		if token == nil && !nullable && typ != nil && typ.Kind() != reflect.Slice && typ.Kind() != reflect.Map && typ.Kind() != reflect.Interface {
			return fmt.Errorf("accounting: null scalar or required object")
		}
		delim, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delim {
		case '{':
			names := make(map[string]reflect.Type)
			if typ != nil && typ.Kind() == reflect.Struct {
				for i := 0; i < typ.NumField(); i++ {
					f := typ.Field(i)
					if !f.IsExported() {
						continue
					}
					name := strings.Split(f.Tag.Get("json"), ",")[0]
					if name == "-" {
						continue
					}
					if name == "" {
						name = f.Name
					}
					names[name] = f.Type
				}
			}
			seen := make(map[string]bool)
			for d.More() {
				keyToken, err := d.Token()
				if err != nil {
					return err
				}
				key, ok := keyToken.(string)
				if !ok || seen[key] {
					return fmt.Errorf("accounting: duplicate or invalid json key")
				}
				seen[key] = true
				var next reflect.Type
				if typ != nil && typ.Kind() == reflect.Struct {
					var known bool
					next, known = names[key]
					if !known {
						return fmt.Errorf("accounting: unknown or noncanonical json field %q", key)
					}
				} else if typ != nil && typ.Kind() == reflect.Map {
					next = typ.Elem()
				}
				if err := read(next, depth+1); err != nil {
					return err
				}
			}
			var required []string
			if typ == reflect.TypeOf(EnergySummary{}) {
				required = []string{"start_energy_mj", "end_energy_mj", "energy_consumed_mj", "energy_consumed_joules", "coverage_ratio", "covered_duration_nanos", "total_duration_nanos"}
			}
			if typ == reflect.TypeOf(AccountingRate{}) {
				required = []string{"minor_digits"}
			}
			if typ == reflect.TypeOf(AccountingWork{}) {
				required = []string{"units"}
			}
			for _, key := range required {
				if !seen[key] {
					return fmt.Errorf("accounting: missing required field %q", key)
				}
			}
			end, err := d.Token()
			if err != nil || end != json.Delim('}') {
				return fmt.Errorf("accounting: invalid json object")
			}
		case '[':
			var next reflect.Type
			if typ != nil && (typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array) {
				next = typ.Elem()
			}
			for d.More() {
				if err := read(next, depth+1); err != nil {
					return err
				}
			}
			end, err := d.Token()
			if err != nil || end != json.Delim(']') {
				return fmt.Errorf("accounting: invalid json array")
			}
		default:
			return fmt.Errorf("accounting: unexpected json delimiter")
		}
		return nil
	}
	if err := read(typ, 0); err != nil {
		return err
	}
	if _, err := d.Token(); err != io.EOF {
		return fmt.Errorf("accounting: trailing json token")
	}
	return nil
}
