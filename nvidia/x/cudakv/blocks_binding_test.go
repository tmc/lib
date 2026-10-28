package cudakv

import (
	"fmt"
	"testing"
)

func TestBlockAllocationGenerationAndBinding(t *testing.T) {
	a := NewBlockAllocator(2, 4)
	if err := a.AllocSequence(1, nil, 3); err != nil {
		t.Fatal(err)
	}
	g, _ := a.AllocationGeneration(1)
	table := a.Table(1)
	if err := a.ValidateSequence(1, g, table, 3); err != nil {
		t.Fatal(err)
	}
	for _, blocks := range [][]int{{table[0], table[0]}, {table[0], 4}, {table[0]}} {
		if err := a.ValidateSequence(1, g, blocks, 3); err == nil {
			t.Fatal("corrupt table admitted", blocks)
		}
	}
	free := a.NumFreeBlocks()
	a.tables[1][0] = 4
	if err := a.AppendToken(1); err == nil || a.NumFreeBlocks() != free {
		t.Fatal("corrupt allocation mutated", err)
	}
	a.tables[1][0] = table[0]
	a.FreeSequence(1)
	if err := a.AllocSequence(1, nil, 3); err != nil {
		t.Fatal(err)
	}
	if err := a.ValidateSequence(1, g, a.Table(1), 3); err == nil {
		t.Fatal("reallocated owner admitted")
	}
}

func ExampleBlockAllocator_AllocationGeneration() {
	a := NewBlockAllocator(2, 1)
	_ = a.AllocSequence(7, nil, 1)
	g, ok := a.AllocationGeneration(7)
	fmt.Println(g > 0, ok)
	// Output: true true
}

func ExampleBlockAllocator_ValidateSequence() {
	a := NewBlockAllocator(2, 1)
	_ = a.AllocSequence(7, nil, 1)
	g, _ := a.AllocationGeneration(7)
	fmt.Println(a.ValidateSequence(7, g, a.Table(7), 1))
	// Output: <nil>
}
