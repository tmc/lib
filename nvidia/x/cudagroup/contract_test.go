package cudagroup

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestWarpgroupParentRanks(t *testing.T) {
	cta, err := New(CTA, driver.Dim3{X: 32, Y: 4, Z: 2})
	if err != nil {
		t.Fatal(err)
	}
	for warp := uint32(0); warp < 8; warp++ {
		g, err := WarpGroupForCTA(cta, warp)
		if err != nil {
			t.Fatal(err)
		}
		for lane := uint32(0); lane < 128; lane++ {
			rank, err := g.Rank(driver.Dim3{X: lane})
			if err != nil || rank != uint64(warp/4)*128+uint64(lane) {
				t.Fatalf("warp %d lane %d: %d, %v", warp, lane, rank, err)
			}
		}
	}
	if _, err := WarpGroupForCTA(cta, 8); err == nil {
		t.Fatal("accepted missing warp")
	}
}

func TestLaunchShapeMutants(t *testing.T) {
	cta, _ := New(CTA, driver.Dim3{X: 32, Y: 4, Z: 1})
	req := LaunchRequirements{Grid: driver.Dim3{X: 4, Y: 1, Z: 1}, Block: cta.Dims()}
	if err := ValidateLaunch(req, cta); err != nil {
		t.Fatal(err)
	}
	for _, block := range []driver.Dim3{{X: 128, Y: 1, Z: 1}, {X: 32, Y: 3, Z: 1}, {X: math.MaxUint32, Y: math.MaxUint32, Z: math.MaxUint32}, {}} {
		bad := req
		bad.Block = block
		if err := ValidateLaunch(bad, cta); err == nil {
			t.Fatalf("accepted wrong block %v", block)
		}
	}
	cluster, _ := New(Cluster, driver.Dim3{X: 2, Y: 1, Z: 1})
	req.Cluster = cluster.Dims()
	if err := ValidateLaunch(req, cluster); err != nil {
		t.Fatal(err)
	}
	req.Cluster = driver.Dim3{X: 1, Y: 2, Z: 1}
	if err := ValidateLaunch(req, cluster); err == nil {
		t.Fatal("accepted mismatched cluster")
	}
	req.Cluster = cluster.Dims()
	req.Grid.X = 3
	if err := ValidateLaunch(req, cluster); err == nil {
		t.Fatal("accepted partial cluster")
	}
	if _, err := New(CTA, driver.Dim3{X: math.MaxUint32, Y: math.MaxUint32, Z: math.MaxUint32}); err == nil {
		t.Fatal("accepted overflowing extent")
	}
	if err := ValidateLaunch(req, Group{}); err == nil {
		t.Fatal("accepted zero group")
	}
}

func TestUniformIssueMutants(t *testing.T) {
	g, _ := New(WarpGroup, driver.Dim3{X: 128, Y: 1, Z: 1})
	c, _ := NewCollective(WarpgroupUniformity, g, MemoryCTA)
	ranks := make([]uint64, 128)
	args := make([]uint32, 128)
	for i := range ranks {
		ranks[i] = uint64(i)
		args[i] = 232
	}
	if err := c.ValidateUniform(ranks, args); err != nil {
		t.Fatal(err)
	}
	args[127] = 32
	if err := c.ValidateUniform(ranks, args); err == nil {
		t.Fatal("accepted nonuniform issue")
	}
	args[127] = 232
	ranks[127] = 126
	if err := c.ValidateUniform(ranks, args); err == nil {
		t.Fatal("accepted missing lane replaced by duplicate")
	}
	ranks[127] = 128
	if err := c.ValidateUniform(ranks, args); err == nil {
		t.Fatal("accepted wrong lane")
	}
	c.Kind = 255
	if err := c.ValidateParticipation(128); err == nil {
		t.Fatal("accepted mutated operation")
	}
	if _, err := NewCollective(ReduceSum, g, MemoryCTA); err == nil {
		t.Fatal("accepted unsupported warpgroup reduction")
	}
}

func TestTMEMParticipationContract(t *testing.T) {
	cluster, _ := New(Cluster, driver.Dim3{X: 4, Y: 1, Z: 1})
	block := driver.Dim3{X: 32, Y: 4, Z: 1}
	for _, peers := range []TMEMPeers{TMEMGroup1, TMEMGroup2} {
		c, err := NewTMEMCollective("sm_100a", peers, cluster, block, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		if c.Peers() != peers || uint64(c.Peers()) == cluster.Size() {
			t.Fatal("instruction group became cluster size")
		}
		req := LaunchRequirements{Grid: driver.Dim3{X: 4, Y: 1, Z: 1}, Block: block, Cluster: cluster.Dims()}
		if err := c.ValidateLaunch(req); err != nil {
			t.Fatal(err)
		}
		issuers := make([]Issuer, int(peers)*32)
		args := make([]uint32, len(issuers))
		for i := range issuers {
			issuers[i] = Issuer{uint64(2 + i/32), uint64(32 + i%32)}
			args[i] = 64
		}
		if err := c.ValidateParticipation(issuers, args); err != nil {
			t.Fatal(err)
		}
		if err := c.ValidateParticipation(issuers[:len(issuers)-1], args); err == nil {
			t.Fatal("accepted dropped issuer")
		}
		issuers[0].CTA = 0
		if err := c.ValidateParticipation(issuers, args); err == nil {
			t.Fatal("accepted wrong peer")
		}
		issuers[0].CTA = 2
		args[0] = 32
		if err := c.ValidateParticipation(issuers, args); err == nil {
			t.Fatal("accepted mismatched argument")
		}
		args[0] = 64
		issuers[0] = issuers[1]
		if err := c.ValidateParticipation(issuers, args); err == nil {
			t.Fatal("accepted duplicate issuer")
		}
	}
	for _, tc := range []struct {
		target string
		peers  TMEMPeers
		first  uint64
		warp   uint32
	}{
		{"sm_121a", TMEMGroup1, 0, 0}, {"sm_999a", TMEMGroup1, 0, 0}, {"sm_100a", 3, 0, 0},
		{"sm_100a", TMEMGroup2, 1, 0}, {"sm_100a", TMEMGroup2, 4, 0}, {"sm_100a", TMEMGroup1, 0, 4},
	} {
		if _, err := NewTMEMCollective(tc.target, tc.peers, cluster, block, tc.first, tc.warp); err == nil {
			t.Fatalf("accepted invalid tmem %+v", tc)
		}
	}
}

func FuzzGroupDimensions(f *testing.F) {
	f.Add(uint32(32), uint32(4), uint32(1))
	f.Add(uint32(math.MaxUint32), uint32(math.MaxUint32), uint32(math.MaxUint32))
	f.Fuzz(func(t *testing.T, x, y, z uint32) {
		g, err := New(CTA, driver.Dim3{X: x, Y: y, Z: z})
		if err != nil {
			return
		}
		rank, err := g.Rank(driver.Dim3{X: x - 1, Y: y - 1, Z: z - 1})
		if err != nil || rank+1 != g.Size() || g.Size() > 1024 {
			t.Fatalf("bad rank %d size %d: %v", rank, g.Size(), err)
		}
	})
}
