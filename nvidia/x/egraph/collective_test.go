package egraph

import "testing"

func TestCollectiveIntentValidationAndIdentity(t *testing.T) {
	c := CollectiveIntent{Communicator: "c", Generation: 1, Rank: 0, WorldSize: 2, Operation: "allreduce", DType: "f32", Count: 4, EffectToken: 9}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if !c.SameEpoch(c) || !CanRewrite(c, c) {
		t.Fatal("valid intent lost identity")
	}
	c.Generation++
	if c.SameEpoch(CollectiveIntent{Communicator: "c", Generation: 1, Rank: 0, WorldSize: 2, Operation: "allreduce", DType: "f32", Count: 4, EffectToken: 9}) {
		t.Fatal("generation reuse accepted")
	}
}
