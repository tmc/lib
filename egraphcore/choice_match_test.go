package egraphcore

import "testing"

type choiceMatchNode struct {
	label    string
	children []uint32
}

type choiceMatchValue struct {
	label    string
	children []*choiceMatchValue
}

func TestSelectedChoiceMatches(t *testing.T) {
	shared := &choiceMatchValue{label: "leaf"}
	cycle := &choiceMatchValue{label: "loop"}
	cycle.children = []*choiceMatchValue{cycle}
	tests := []struct {
		name         string
		root         uint32
		choice       map[uint32]int
		alternatives map[uint32][]choiceMatchNode
		value        *choiceMatchValue
		want         bool
	}{
		{
			name:   "shared child",
			root:   0,
			choice: map[uint32]int{0: 0, 1: 0},
			alternatives: map[uint32][]choiceMatchNode{
				0: {{label: "pair", children: []uint32{1, 1}}},
				1: {{label: "leaf"}},
			},
			value: &choiceMatchValue{label: "pair", children: []*choiceMatchValue{shared, shared}},
			want:  true,
		},
		{
			name:   "mismatched child",
			root:   0,
			choice: map[uint32]int{0: 0, 1: 0},
			alternatives: map[uint32][]choiceMatchNode{
				0: {{label: "pair", children: []uint32{1}}},
				1: {{label: "leaf"}},
			},
			value: &choiceMatchValue{label: "pair", children: []*choiceMatchValue{{label: "other"}}},
			want:  false,
		},
		{
			name:         "missing choice",
			root:         0,
			choice:       map[uint32]int{},
			alternatives: map[uint32][]choiceMatchNode{0: {{label: "leaf"}}},
			value:        &choiceMatchValue{label: "leaf"},
			want:         false,
		},
		{
			name:   "matching cycle",
			root:   0,
			choice: map[uint32]int{0: 0},
			alternatives: map[uint32][]choiceMatchNode{
				0: {{label: "loop", children: []uint32{0}}},
			},
			value: cycle,
			want:  true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := SelectedChoiceMatches(
				test.root,
				test.choice,
				func(class uint32) []choiceMatchNode { return test.alternatives[class] },
				func(node choiceMatchNode) []uint32 { return node.children },
				test.value,
				func(value *choiceMatchValue) []*choiceMatchValue {
					if value == nil {
						return nil
					}
					return value.children
				},
				func(_ uint32, node choiceMatchNode, value *choiceMatchValue) bool {
					return value != nil && node.label == value.label
				},
			)
			if got != test.want {
				t.Fatalf("SelectedChoiceMatches() = %t, want %t", got, test.want)
			}
		})
	}
}
