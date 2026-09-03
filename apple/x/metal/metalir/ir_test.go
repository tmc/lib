package metalir

import "testing"

func TestDataTypeAndNodeSizes(t *testing.T) {
	tests := []struct {
		typeName DataType
		name     string
		size     int
	}{
		{Float32, "float32", 4},
		{Float16, "float16", 2},
		{BFloat16, "bfloat16", 2},
		{Int32, "int32", 4},
		{Int64, "int64", 8},
		{Uint8, "uint8", 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.typeName.String(); got != test.name {
				t.Fatalf("String() = %q, want %q", got, test.name)
			}
			if got := test.typeName.SizeBytes(); got != test.size {
				t.Fatalf("SizeBytes() = %d, want %d", got, test.size)
			}
		})
	}
	if got := (&Node{Shape: nil, Type: Float32}).NumElements(); got != 1 {
		t.Fatalf("scalar elements = %d, want 1", got)
	}
	if got := (&Node{Shape: []int{2, 3, 4}, Type: Float16}).SizeBytes(); got != 48 {
		t.Fatalf("tensor bytes = %d, want 48", got)
	}
}

func TestGraphTopologyAndMemoryPlan(t *testing.T) {
	g := NewGraph()
	a := g.AddInput("a", []int{2, 3}, Float32)
	b := g.AddInput("b", []int{3, 4}, Float32)
	product := g.AddMatMul(a, b)
	activated := g.AddUnary("silu", product)
	sum := g.AddBinary("add", activated, activated)
	if product.ID != 2 || activated.ID != 3 || sum.ID != 4 {
		t.Fatalf("node IDs = %d,%d,%d", product.ID, activated.ID, sum.ID)
	}
	if got := product.Shape; len(got) != 2 || got[0] != 2 || got[1] != 4 {
		t.Fatalf("matmul shape = %v", got)
	}
	plan, err := PlanMemory(g, 256)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := plan.Offsets[a.ID]; ok {
		t.Fatal("input a received an arena offset")
	}
	if plan.Offsets[product.ID] != 0 || plan.Offsets[activated.ID] != 256 || plan.Offsets[sum.ID] != 0 {
		t.Fatalf("offsets = %v", plan.Offsets)
	}
	if plan.TotalBytes != 512 {
		t.Fatalf("total bytes = %d, want 512", plan.TotalBytes)
	}
}

func TestValidateRejectsMalformedGraphs(t *testing.T) {
	tests := []struct {
		name  string
		build func() *Graph
	}{
		{"negative dimension", func() *Graph { g := NewGraph(); g.AddInput("x", []int{-1}, Float32); return g }},
		{"invalid type", func() *Graph { g := NewGraph(); g.AddInput("x", []int{1}, DataType(99)); return g }},
		{"matmul rank", func() *Graph {
			g := NewGraph()
			a := g.AddInput("a", []int{2}, Float32)
			b := g.AddInput("b", []int{2}, Float32)
			g.AddMatMul(a, b)
			return g
		}},
		{"binary mismatch", func() *Graph {
			g := NewGraph()
			a := g.AddInput("a", []int{2}, Float32)
			b := g.AddInput("b", []int{3}, Float32)
			g.AddBinary("add", a, b)
			return g
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.build().Validate(); err == nil {
				t.Fatal("Validate succeeded")
			}
		})
	}
}

func TestPlanMemoryRejectsAlignment(t *testing.T) {
	g := NewGraph()
	g.AddInput("x", []int{1}, Float32)
	for _, alignment := range []int{-1, 3} {
		if _, err := PlanMemory(g, alignment); err == nil {
			t.Fatalf("alignment %d accepted", alignment)
		}
	}
}

func TestValidateAcceptsScalarRoPEPosition(t *testing.T) {
	g := NewGraph()
	x := g.AddInput("x", []int{2, 8}, Float32)
	position := g.AddInput("position", []int{1, 1}, Float32)
	g.AddBinary("rope", x, position)
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
}
