package cudagroup

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestGroupMembership(t *testing.T) {
	g, err := New(WarpGroup, driver.Dim3{X: 128, Y: 1, Z: 1})
	if err != nil || g.Size() != 128 || !g.Contains(driver.Dim3{X: 127}) {
		t.Fatalf("membership: %v size=%d", err, g.Size())
	}
}
func TestGroupRankBounds(t *testing.T) {
	g, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	if _, err := g.Rank(driver.Dim3{X: 32}); err == nil {
		t.Fatal("accepted out-of-range rank")
	}
}
func TestCollectiveParticipation(t *testing.T) {
	g, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	c, _ := NewCollective(ReduceSum, g, MemoryThread)
	if err := c.ValidateParticipation(31); err == nil {
		t.Fatal("accepted dropped participant")
	}
	if err := c.ValidateParticipation(32); err != nil {
		t.Fatal(err)
	}
	ranks := make([]uint64, 32)
	for i := range ranks {
		ranks[i] = uint64(i)
	}
	if err := c.ValidateRanks(ranks); err != nil {
		t.Fatal(err)
	}
	ranks[31] = ranks[30]
	if err := c.ValidateRanks(ranks); err == nil {
		t.Fatal("accepted duplicate rank")
	}
}

func TestTMEMPeerGroups(t *testing.T) {
	cluster, _ := New(Cluster, driver.Dim3{X: 4, Y: 1, Z: 1})
	if _, err := NewTMEMCollective("sm_100a", TMEMGroup1, cluster, driver.Dim3{X: 128, Y: 1, Z: 1}, 0, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := NewTMEMCollective("sm_100a", 3, cluster, driver.Dim3{X: 128, Y: 1, Z: 1}, 0, 0); err == nil {
		t.Fatal("accepted arbitrary tmem peer count")
	}
}
func TestWarpgroupUniformity(t *testing.T) {
	g, _ := New(WarpGroup, driver.Dim3{X: 128, Y: 1, Z: 1})
	c, _ := NewCollective(WarpgroupUniformity, g, MemoryCTA)
	if !c.Uniform {
		t.Fatal("reduction is not uniform")
	}
}
func TestCTAGroupNotClusterSize(t *testing.T) {
	cta, _ := New(CTA, driver.Dim3{X: 256, Y: 1, Z: 1})
	if cta.Size() == 512 {
		t.Fatal("cta confused with cluster")
	}
	if _, err := WarpGroupForCTA(cta, 0); err != nil {
		t.Fatal(err)
	}
}
func TestMemoryScopeDistinct(t *testing.T) {
	g, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	c, _ := NewCollective(ReduceSum, g, MemoryCTA)
	if c.Group.Kind() != Warp || c.Scope != MemoryCTA {
		t.Fatal("scope replaced membership")
	}
}
func TestFixtureSource(t *testing.T) {
	g, _ := New(Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	f, err := GenerateReductionFixture(g)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{"warp_reduce_sum<float>", "__shfl_down_sync", "threadIdx.x == 0"} {
		if !strings.Contains(f.Source, s) {
			t.Errorf("fixture missing %q", s)
		}
	}
	if err := f.ValidateLaunch(); err != nil {
		t.Fatal(err)
	}
}
