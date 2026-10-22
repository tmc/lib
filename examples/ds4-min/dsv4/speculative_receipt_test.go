package dsv4

import (
	"fmt"
	"testing"
)

func TestSpeculativeReceipts(t *testing.T) {
	makeCases := func() ([]string, []SpeculativeCaseReceipt) {
		var names []string
		var cases []SpeculativeCaseReceipt
		for i := 0; i < 8; i++ {
			name := fmt.Sprintf("case-%d", i)
			prompt := []int{1, i + 2}
			hash, err := SpeculativeHistorySHA256(prompt)
			if err != nil {
				t.Fatal(err)
			}
			step := SpeculativeStep{Position: 2, Token: 10, HistorySHA256: hash}
			names = append(names, name)
			cases = append(cases, SpeculativeCaseReceipt{Name: name, Prompt: prompt,
				Trace:       SpeculativeTrace{Strict: []int{10}, Speculative: []int{10}, Margins: []float64{1}, AcceptedDrafts: 1, Cycles: 1},
				StrictSteps: []SpeculativeStep{step}, SpeculativeSteps: []SpeculativeStep{step}})
		}
		return names, cases
	}
	names, cases := makeCases()
	contracts := make(map[string]SpeculativeCaseContract)
	for _, c := range cases {
		hash, _ := SpeculativeHistorySHA256(c.Prompt)
		contracts[c.Name] = SpeculativeCaseContract{hash, 1}
	}
	r, err := CheckSpeculativeReceipts(names, contracts, cases, 0.1)
	if err != nil || !r.Pass() {
		t.Fatalf("valid receipt: %+v, %v", r, err)
	}
	for _, tc := range []struct {
		name   string
		mutate func([]SpeculativeCaseReceipt)
	}{
		{"duplicate", func(c []SpeculativeCaseReceipt) { c[1].Name = c[0].Name }},
		{"missing step", func(c []SpeculativeCaseReceipt) { c[0].StrictSteps = nil }},
		{"wrong history", func(c []SpeculativeCaseReceipt) { c[0].Prompt[0]++ }},
		{"rewritten history", func(c []SpeculativeCaseReceipt) {
			c[0].Prompt[0]++
			hash, _ := SpeculativeHistorySHA256(c[0].Prompt)
			c[0].StrictSteps[0].HistorySHA256 = hash
			c[0].SpeculativeSteps[0].HistorySHA256 = hash
		}},
		{"wrong position", func(c []SpeculativeCaseReceipt) { c[0].SpeculativeSteps[0].Position++ }},
		{"wrong token", func(c []SpeculativeCaseReceipt) { c[0].SpeculativeSteps[0].Token++ }},
		{"no acceptance", func(c []SpeculativeCaseReceipt) { c[0].Trace.AcceptedDrafts = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			names, cases := makeCases()
			tc.mutate(cases)
			if _, err := CheckSpeculativeReceipts(names, contracts, cases, 0.1); err == nil {
				t.Fatal("mutation accepted")
			}
		})
	}
}

func ExampleSpeculativeHistorySHA256() {
	hash, err := SpeculativeHistorySHA256([]int{1, 2})
	fmt.Println(len(hash), err)
	// Output: 64 <nil>
}

func ExampleCheckSpeculativeReceipts() {
	_, err := CheckSpeculativeReceipts(nil, nil, nil, 0.001)
	fmt.Println(err)
	// Output: speculative receipts: require eight cases
}

func ExampleRunSpeculativeCase() {
	_, err := RunSpeculativeCase(nil, nil, "unavailable", []int{1}, 1024, 512, 5)
	fmt.Println(err)
	// Output: speculative run: invalid case parameters
}

func TestSpeculativeTopMargin(t *testing.T) {
	i, margin, err := speculativeTopMargin([]float32{1, 3, 3})
	if err != nil || i != 1 || margin != 0 {
		t.Fatalf("tie: %d %g %v", i, margin, err)
	}
}
