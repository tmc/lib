package nvtelemetry

import (
	"fmt"
	"math/big"
)

// AccountingNode supplies a measured whole-node counter and an authoritative,
// complete device membership for exactly that interval. Energy uses scope
// whole_node and Summary.DeviceUUID equals NodeID. Evidence is caller-provided
// provenance, not authentication performed by this package.
type AccountingNode struct {
	NodeID             string           `json:"node_id"`
	Energy             AccountingEnergy `json:"energy"`
	DeviceUUIDs        []string         `json:"device_uuids"`
	MembershipEvidence string           `json:"membership_evidence"`
}

// AccountingNodeTotal separates measured node and GPU-component totals.
// Missing GPU counters leave component and other totals empty, not zero.
// OtherEnergyMillijoules is their difference, not measured CPU or idle energy.
// Node energy is never added to PhysicalEnergyMillijoules or allocation shares.
type AccountingNodeTotal struct {
	GPUComponentsAvailable         bool   `json:"gpu_components_available"`
	GPUComponentsUnavailableReason string `json:"gpu_components_unavailable_reason,omitempty"`
	NodeID                         string `json:"node_id"`
	StartNanos                     int64  `json:"start_nanos"`
	EndNanos                       int64  `json:"end_nanos"`
	NodeEnergyMillijoules          string `json:"node_energy_millijoules"`
	GPUComponentMillijoules        string `json:"gpu_component_millijoules"`
	OtherEnergyMillijoules         string `json:"other_energy_millijoules"`
}

// AccountingIdle is an authoritative idle-capacity interval. Capacity is a
// caller-evidenced policy input, never inferred from missing allocations.
// The resulting energy is a uniform-rate capacity estimate, not an idle sensor.
type AccountingIdle struct {
	DeviceUUID       string `json:"device_uuid"`
	ClockDomain      string `json:"clock_domain"`
	ClockEpoch       string `json:"clock_epoch"`
	StartNanos       int64  `json:"start_nanos"`
	EndNanos         int64  `json:"end_nanos"`
	CapacityPermille uint32 `json:"capacity_permille"`
	Authoritative    bool   `json:"authoritative"`
	Evidence         string `json:"evidence"`
	Policy           string `json:"policy"`
}

func accountingScopes(in AccountingInput, out *AccountingReport) error {
	fail := func(s string) error { return fmt.Errorf("accounting: %s", s) }
	nodes := make([]AccountingEnergy, 0, len(in.Nodes))
	out.Nodes = nil
	for i, n := range in.Nodes {
		e := n.Energy
		s := e.Summary
		if n.NodeID == "" || s.DeviceUUID != n.NodeID || e.Scope != "whole_node" || n.MembershipEvidence == "" || len(n.DeviceUUIDs) == 0 || len(n.DeviceUUIDs) > 128 {
			return fail("invalid node scope or membership evidence")
		}
		for _, p := range in.Nodes[:i] {
			x, y := accountingOverlap(s.StartMonoNanos, s.EndMonoNanos, p.Energy.Summary.StartMonoNanos, p.Energy.Summary.EndMonoNanos)
			if x < y {
				for _, d := range n.DeviceUUIDs {
					for _, pd := range p.DeviceUUIDs {
						if d == pd {
							return fail("device belongs to overlapping node scopes")
						}
					}
				}
			}
		}
		e.Scope = "physical_device"
		nodes = append(nodes, e)
		if _, err := AccountEnergy(AccountingInput{Version: 1, Clock: in.Clock, Energy: nodes}); err != nil {
			return err
		}
		gpu := new(big.Int)
		componentsAvailable := true
		seen := make(map[string]bool)
		for _, d := range n.DeviceUUIDs {
			if d == "" || seen[d] {
				return fail("duplicate or absent node device identity")
			}
			seen[d] = true
			covered := int64(0)
			provided := false
			for _, g := range in.Energy {
				if g.Summary.DeviceUUID == n.NodeID {
					return fail("node identity collides with physical GPU")
				}
				if g.Lineage == e.Lineage {
					return fail("node and GPU counter lineage overlap")
				}
				if g.Summary.DeviceUUID != d {
					continue
				}
				provided = true
				v := g.Summary
				x, y := accountingOverlap(s.StartMonoNanos, s.EndMonoNanos, v.StartMonoNanos, v.EndMonoNanos)
				if x >= y {
					continue
				}
				if x != v.StartMonoNanos || y != v.EndMonoNanos {
					return fail("node component interval requires exact containment")
				}
				covered += y - x
				gpu.Add(gpu, new(big.Int).SetUint64(v.EnergyConsumedMJ))
			}
			if covered == 0 && !provided {
				componentsAvailable = false
			} else if covered != s.EndMonoNanos-s.StartMonoNanos {
				return fail("node membership lacks complete GPU counter coverage")
			}
		}
		total := new(big.Int).SetUint64(s.EnergyConsumedMJ)
		other := new(big.Int).Sub(total, gpu)
		if other.Sign() < 0 {
			return fail("node total below measured GPU components")
		}
		entry := AccountingNodeTotal{NodeID: n.NodeID, StartNanos: s.StartMonoNanos, EndNanos: s.EndMonoNanos, NodeEnergyMillijoules: total.String(), GPUComponentsAvailable: componentsAvailable}
		if componentsAvailable {
			entry.GPUComponentMillijoules = gpu.String()
			entry.OtherEnergyMillijoules = other.String()
		} else {
			entry.GPUComponentsUnavailableReason = "gpu_counter_unavailable"
		}
		out.Nodes = append(out.Nodes, entry)
	}
	idle := new(big.Rat)
	for i, v := range in.Idle {
		if !v.Authoritative || v.Evidence == "" || v.Policy != "capacity_time_v1" || v.DeviceUUID == "" || v.ClockDomain != in.Clock.Domain || v.ClockEpoch != in.Clock.Epoch || v.CapacityPermille == 0 || v.CapacityPermille > 1000 || v.StartNanos < in.Clock.ValidFrom || v.EndNanos > in.Clock.ValidUntil {
			return fail("invalid idle evidence")
		}
		if _, err := accountingDuration(v.StartNanos, v.EndNanos); err != nil {
			return err
		}
		// Idle claims must be disjoint and cannot intersect any allocated capacity.
		// This deliberately refuses ambiguous mixed-idle shares instead of guessing.
		for _, p := range in.Idle[:i] {
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, p.StartNanos, p.EndNanos)
			if p.DeviceUUID == v.DeviceUUID && x < y {
				return fail("overlapping idle claims")
			}
		}
		for _, a := range in.Allocations {
			p := a.Assignment
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, p.StartNanos, p.EndNanos)
			if accountingPhysicalDevice(p) == v.DeviceUUID && x < y {
				return fail("idle claim overlaps allocation")
			}
		}
		covered := int64(0)
		for _, e := range in.Energy {
			s := e.Summary
			if s.DeviceUUID != v.DeviceUUID {
				continue
			}
			x, y := accountingOverlap(v.StartNanos, v.EndNanos, s.StartMonoNanos, s.EndMonoNanos)
			if x >= y {
				continue
			}
			covered += y - x
			portion := new(big.Rat).SetFrac(new(big.Int).Mul(new(big.Int).SetUint64(s.EnergyConsumedMJ), new(big.Int).Mul(big.NewInt(y-x), big.NewInt(int64(v.CapacityPermille)))), new(big.Int).Mul(big.NewInt(s.EndMonoNanos-s.StartMonoNanos), big.NewInt(1000)))
			idle.Add(idle, portion)
		}
		if covered != v.EndNanos-v.StartNanos {
			return fail("idle claim lacks measured energy coverage")
		}
	}
	if len(in.Idle) > 0 {
		residual, ok := new(big.Rat).SetString(out.ResidualEnergyMillijoules)
		if !ok || idle.Cmp(residual) > 0 {
			return fail("idle estimate exceeds residual")
		}
		out.IdleEnergyMillijoules = idle.RatString()
		out.UnclassifiedResidualMillijoules = new(big.Rat).Sub(residual, idle).RatString()
	}
	return nil
}
