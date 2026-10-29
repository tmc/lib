package cudagreen

import (
	"errors"
	"fmt"
	"testing"
)

func TestComputePlanSKUs(t *testing.T) {
	skus := []struct {
		name string
		arch string
		sms  int
	}{
		{"GB10", "sm_121a", 48},
		{"H100 SXM", "sm_90a", 132},
		{"B200", "sm_100a", 148},
		{"B200 full", "sm_100a", 160},
		{"Rubin 192", "sm_107a", 192},
		{"Rubin 224", "sm_107a", 224},
	}
	regimes := []struct {
		name  string
		specs []Spec
	}{
		{"prefill", PrefillHeavy()},
		{"decode", DecodeHeavy()},
		{"clusters", []Spec{
			{Role: RoleCompute, Fraction: 0.6, ClusterSize: 16},
			{Role: RoleDecode, Fraction: 0.3, MinSMs: 12, ClusterSize: 2},
			{Role: RoleFabric, Fraction: 0.1},
		}},
	}
	for _, sku := range skus {
		g, err := DefaultGranularity(sku.arch)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range regimes {
			t.Run(sku.name+"/"+r.name, func(t *testing.T) {
				plan, err := ComputePlan(sku.arch, sku.sms, g, r.specs)
				if err != nil {
					t.Fatal(err)
				}
				if err := VerifyPlan(plan); err != nil {
					t.Fatal(err)
				}
				if len(plan.Partitions) != len(r.specs) {
					t.Fatalf("%d partitions, want %d", len(plan.Partitions), len(r.specs))
				}
				if plan.Unassigned >= plan.TotalSMs/2 {
					t.Errorf("%d of %d SMs unassigned", plan.Unassigned, plan.TotalSMs)
				}
			})
		}
	}
}

func TestComputePlanSizes(t *testing.T) {
	g := Granularity{Min: 8, Alignment: 8}
	tests := []struct {
		name  string
		sms   int
		specs []Spec
		want  []int
	}{
		{"prefill 160", 160, PrefillHeavy(), []int{128, 16, 8, 8}},
		{"decode 160", 160, DecodeHeavy(), []int{32, 112, 8, 8}},
		// 132 SMs leave 128 usable: decode, the furthest above target, shrinks.
		{"prefill 132", 132, PrefillHeavy(), []int{104, 8, 8, 8}},
		{"minimum wins", 64, []Spec{{Role: RoleCompute, Fraction: 0.1, MinSMs: 20}}, []int{24}},
		{"cluster unit", 64, []Spec{{Role: RoleCompute, Fraction: 0.5, ClusterSize: 16}}, []int{32}},
		{"cluster rounds", 64, []Spec{{Role: RoleCompute, Fraction: 0.3, ClusterSize: 12}}, []int{24}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := ComputePlan("sm_100a", tt.sms, g, tt.specs)
			if err != nil {
				t.Fatal(err)
			}
			var got []int
			for _, p := range plan.Partitions {
				got = append(got, p.SMs)
			}
			if fmt.Sprint(got) != fmt.Sprint(tt.want) {
				t.Errorf("sizes = %v, want %v", got, tt.want)
			}
			if err := VerifyPlan(plan); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestComputePlanErrors(t *testing.T) {
	g := Granularity{Min: 8, Alignment: 8}
	tests := []struct {
		name  string
		sms   int
		g     Granularity
		specs []Spec
		inf   bool
	}{
		{"minimums exceed", 48, g, []Spec{{MinSMs: 32}, {MinSMs: 24}}, true},
		{"fractions over 1", 160, g, []Spec{{Fraction: 0.7}, {Fraction: 0.4}}, false},
		{"negative fraction", 160, g, []Spec{{Fraction: -0.1}}, false},
		{"too many SMs", 300, g, nil, false},
		{"zero granularity", 160, Granularity{}, nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ComputePlan("sm_100a", tt.sms, tt.g, tt.specs)
			if err == nil {
				t.Fatal("ComputePlan succeeded")
			}
			if got := errors.Is(err, ErrInfeasible); got != tt.inf {
				t.Errorf("errors.Is(%v, ErrInfeasible) = %v, want %v", err, got, tt.inf)
			}
		})
	}
}

// TestVerifyPlanMutants checks that each invariant VerifyPlan states is
// enforced: every mutation of a valid plan is rejected.
func TestVerifyPlanMutants(t *testing.T) {
	g := Granularity{Min: 8, Alignment: 8}
	good, err := ComputePlan("sm_100a", 160, g, []Spec{
		{Role: RoleCompute, Fraction: 0.5, ClusterSize: 16},
		{Role: RoleDecode, Fraction: 0.4, MinSMs: 16},
		{Role: RoleFabric, Fraction: 0.1},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyPlan(good); err != nil {
		t.Fatal(err)
	}
	mutants := []struct {
		name string
		mut  func(*Plan)
	}{
		{"overlap", func(p *Plan) {
			q := &p.Partitions[1]
			q.First--
			q.Mask = Range(q.First, q.SMs)
		}},
		{"extra bit", func(p *Plan) { p.Partitions[0].Mask[3] |= 1 }},
		{"missing bit", func(p *Plan) {
			q := &p.Partitions[2]
			q.Mask[q.First/64] &^= 1 << (q.First % 64)
		}},
		{"beyond device", func(p *Plan) {
			q := &p.Partitions[2]
			q.First = p.TotalSMs - 4
			q.Mask = Range(q.First, q.SMs)
		}},
		{"misaligned", func(p *Plan) {
			q := &p.Partitions[1]
			q.SMs -= 4
			q.Mask = Range(q.First, q.SMs)
			p.Unassigned += 4
		}},
		{"cluster misaligned", func(p *Plan) {
			q := &p.Partitions[0]
			q.SMs -= 8
			q.Mask = Range(q.First, q.SMs)
			q.Clusters = q.SMs / 16
			p.Unassigned += 8
		}},
		{"below minimum", func(p *Plan) { p.Partitions[1].Spec.MinSMs = p.Partitions[1].SMs + 8 }},
		{"cluster count", func(p *Plan) { p.Partitions[0].Clusters++ }},
		{"accounting", func(p *Plan) { p.Unassigned++ }},
		{"too many groups", func(p *Plan) { p.Granularity.Groups = 2 }},
	}
	for _, m := range mutants {
		t.Run(m.name, func(t *testing.T) {
			p := good
			p.Partitions = append([]Partition(nil), good.Partitions...)
			m.mut(&p)
			if err := VerifyPlan(p); err == nil {
				t.Errorf("VerifyPlan accepted mutant")
			}
		})
	}
}

func TestDefaultGranularity(t *testing.T) {
	tests := []struct {
		arch string
		want Granularity
		ok   bool
	}{
		{"sm_121a", Granularity{Min: 8, Alignment: 8}, true},
		{"sm_100f", Granularity{Min: 8, Alignment: 8}, true},
		{"sm_90a", Granularity{Min: 8, Alignment: 8}, true},
		{"sm_89", Granularity{Min: 2, Alignment: 2}, true},
		{"sm_60", Granularity{}, false},
		{"compute_90", Granularity{}, false},
	}
	for _, tt := range tests {
		got, err := DefaultGranularity(tt.arch)
		if (err == nil) != tt.ok || got != tt.want {
			t.Errorf("DefaultGranularity(%q) = %v, %v; want %v, ok=%v", tt.arch, got, err, tt.want, tt.ok)
		}
	}
}

func TestPartitionGrid(t *testing.T) {
	plan, err := ComputePlan("sm_121a", 48, Granularity{Min: 8, Alignment: 8}, []Spec{{Role: RoleCompute, Fraction: 1, ClusterSize: 2}})
	if err != nil {
		t.Fatal(err)
	}
	p, ok := plan.Partition(RoleCompute)
	if !ok {
		t.Fatal("no compute partition")
	}
	if p.Grid(2) != 96 || p.Clusters != 24 {
		t.Errorf("grid = %d, clusters = %d; want 96, 24", p.Grid(2), p.Clusters)
	}
	if _, ok := plan.Partition(RoleDecode); ok {
		t.Error("found decode partition")
	}
}
