package egraph_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestCollectiveRewritePreservesEffectIdentity(t *testing.T) {
	before := egraph.CollectiveIntent{Communicator: "comm", Generation: 7, Rank: 0, WorldSize: 2, Operation: "allreduce", DType: "f32", Count: 4, Stream: 11, EffectToken: 9}
	for _, tt := range []struct {
		name   string
		change func(*egraph.CollectiveIntent)
	}{
		{"communicator", func(c *egraph.CollectiveIntent) { c.Communicator = "other" }},
		{"generation", func(c *egraph.CollectiveIntent) { c.Generation++ }},
		{"rank", func(c *egraph.CollectiveIntent) { c.Rank = 1 }},
		{"world-size", func(c *egraph.CollectiveIntent) { c.WorldSize = 3 }},
		{"operation", func(c *egraph.CollectiveIntent) { c.Operation = "allgather" }},
		{"dtype", func(c *egraph.CollectiveIntent) { c.DType = "f16" }},
		{"count", func(c *egraph.CollectiveIntent) { c.Count++ }},
		{"in-place", func(c *egraph.CollectiveIntent) { c.InPlace = true }},
		{"stream", func(c *egraph.CollectiveIntent) { c.Stream++ }},
		{"effect-token", func(c *egraph.CollectiveIntent) { c.EffectToken++ }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			after := before
			tt.change(&after)
			if err := after.Validate(); err != nil {
				t.Fatalf("changed intent must remain valid: %v", err)
			}
			if egraph.CanRewrite(before, after) || egraph.CanRewrite(after, before) {
				t.Fatal("rewrite accepted a changed collective effect identity")
			}
		})
	}
}

func TestCollectiveRewriteUnchangedAndRankParticipation(t *testing.T) {
	for _, operation := range []string{"allreduce", "reducescatter", "allgather", "alltoall", "broadcast"} {
		for _, inPlace := range []bool{false, true} {
			for _, stream := range []uint64{0, 11} {
				c := egraph.CollectiveIntent{Communicator: "comm", Generation: 7, Rank: 0, WorldSize: 2, Operation: operation, DType: "f32", Count: 4, InPlace: inPlace, Stream: stream, EffectToken: 9}
				if !egraph.CanRewrite(c, c) {
					t.Fatalf("unchanged valid intent rejected: %+v", c)
				}
				peer := c
				peer.Rank = 1
				if !c.SameEpoch(peer) {
					t.Fatal("rank participation lost its shared epoch")
				}
				if egraph.CanRewrite(c, peer) {
					t.Fatal("rewrite changed the participating rank")
				}
			}
		}
	}
}

func TestCollectiveRewriteRejectsInvalidIntent(t *testing.T) {
	valid := egraph.CollectiveIntent{Communicator: "comm", Generation: 7, WorldSize: 2, Operation: "allreduce", DType: "f32", Count: 4, EffectToken: 9}
	invalid := valid
	invalid.EffectToken = 0
	for _, pair := range [][2]egraph.CollectiveIntent{{valid, invalid}, {invalid, valid}, {invalid, invalid}} {
		if egraph.CanRewrite(pair[0], pair[1]) {
			t.Fatal("rewrite accepted an invalid intent")
		}
	}
}
