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

func TestCollectiveIntentSupportedOperations(t *testing.T) {
	for _, operation := range []string{"allgather", "allreduce", "alltoall", "broadcast", "reducescatter"} {
		t.Run(operation, func(t *testing.T) {
			c := CollectiveIntent{
				Communicator: "c",
				Generation:   1,
				Rank:         0,
				WorldSize:    2,
				Operation:    operation,
				DType:        "f32",
				Count:        4,
				EffectToken:  9,
			}
			if err := c.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCollectiveIntentRejectsUnknownOperation(t *testing.T) {
	c := CollectiveIntent{
		Communicator: "c",
		Generation:   1,
		Rank:         0,
		WorldSize:    2,
		Operation:    "barrier",
		DType:        "f32",
		Count:        1,
		EffectToken:  9,
	}
	if err := c.Validate(); err == nil {
		t.Fatal("unknown collective operation accepted")
	}
}

func TestCollectiveIntentOperationDescriptor(t *testing.T) {
	c := CollectiveIntent{
		Communicator: "c",
		Generation:   1,
		Rank:         0,
		WorldSize:    2,
		Operation:    "allreduce",
		DType:        "bf16",
		Count:        4,
		InPlace:      true,
		EffectToken:  9,
	}
	desc, err := c.OperationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	if desc.Kind.String() != "AllReduce" || desc.DataType.String() != "bfloat16" || desc.Count != 4 || !desc.InPlace {
		t.Fatalf("unexpected descriptor: %+v", desc)
	}
}

func TestCollectiveIntentOperationDescriptorRejectsDtype(t *testing.T) {
	c := CollectiveIntent{Communicator: "c", Generation: 1, Rank: 0, WorldSize: 2, Operation: "allreduce", DType: "fp4", Count: 1, EffectToken: 9}
	if _, err := c.OperationDescriptor(); err == nil {
		t.Fatal("unsupported dtype accepted")
	}
}
