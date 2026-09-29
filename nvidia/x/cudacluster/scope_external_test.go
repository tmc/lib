package cudacluster_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacluster"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

func TestUncheckedScopesCannotEmitOrMap(t *testing.T) {
	plan, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 4096, DynamicMax: 4096, TotalMax: 4096, Source: "external test"})
	if err != nil {
		t.Fatal(err)
	}
	region, err := plan.Add(cudashared.Request{Name: "word", Size: 4, Alignment: 4})
	if err != nil {
		t.Fatal(err)
	}
	var zero cudacluster.Cluster
	var nilScope *cudacluster.Cluster
	checked, err := cudacluster.Open("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	_, readOnly, err := checked.MapSharedRank("%r1", "%r2", plan, region, "%r3")
	if err != nil {
		t.Fatal(err)
	}
	_, writable, err := checked.MapSharedRankMut("%r1", "%r2", plan, region, "%r3")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		call func(*cudacluster.Cluster) error
	}{
		{"zero map", func(c *cudacluster.Cluster) error {
			_, _, err := c.MapSharedRank("%r1", "%r2", plan, region, "%r3")
			return err
		}},
		{"zero load", func(c *cudacluster.Cluster) error {
			_, err := c.EmitLoadU32("%r1", readOnly)
			return err
		}},
		{"zero store", func(c *cudacluster.Cluster) error {
			_, err := c.EmitStoreU32(writable, "%r1")
			return err
		}},
		{"zero read", func(c *cudacluster.Cluster) error {
			_, err := c.EmitReadU32("%r1", "%r2", "%r3", plan, region, "%r4")
			return err
		}},
		{"nil map", func(c *cudacluster.Cluster) error {
			_, _, err := c.MapSharedRank("%r1", "%r2", plan, region, "%r3")
			return err
		}},
		{"nil read", func(c *cudacluster.Cluster) error {
			_, err := c.EmitReadU32("%r1", "%r2", "%r3", plan, region, "%r4")
			return err
		}},
		{"nil load", func(c *cudacluster.Cluster) error {
			_, err := c.EmitLoadU32("%r1", readOnly)
			return err
		}},
		{"nil store", func(c *cudacluster.Cluster) error {
			_, err := c.EmitStoreU32(writable, "%r1")
			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &zero
			if tc.name == "nil map" || tc.name == "nil read" || tc.name == "nil load" || tc.name == "nil store" {
				c = nilScope
			}
			if err := tc.call(c); err == nil {
				t.Fatal("operation accepted an unchecked scope")
			}
		})
	}
}

func TestStoreRejectsPointerFromAnotherScope(t *testing.T) {
	plan, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 4096, DynamicMax: 4096, TotalMax: 4096, Source: "external test"})
	if err != nil {
		t.Fatal(err)
	}
	region, err := plan.Add(cudashared.Request{Name: "word", Size: 4, Alignment: 4})
	if err != nil {
		t.Fatal(err)
	}
	first, err := cudacluster.Open("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	second, err := cudacluster.Open("sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	_, ptr, err := first.MapSharedRankMut("%r1", "%r2", plan, region, "%r3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.EmitStoreU32(ptr, "%r4"); err == nil {
		t.Fatal("store through a pointer from another scope was accepted")
	}
}
