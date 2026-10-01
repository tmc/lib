package marker

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func componentMetadata() Metadata {
	zero, roots := 0, 45
	return Metadata{Version: 3, RequestID: "activation", Role: "measured", Phase: "evaluate", Boundary: "submission", ExecutionUnit: "component_batch", Executions: &zero, Invocation: &zero, Batch: &zero, RootCount: &roots, Operation: "gelu", Component: &ComponentContract{Name: "gelu", BankSHA256: strings.Repeat("a", 64), RootSetSHA256: strings.Repeat("b", 64), Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", MathForm: "compiled-unary", ScalarPolicy: "weak-scalar", ExecutionPolicy: "synchronized_eval", RootPolicy: "distinct_banks"}}
}
func TestComponentRoundtrip(t *testing.T) {
	m := componentMetadata()
	text, err := Encode(m)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseRange("component/" + text)
	if err != nil || !reflect.DeepEqual(m, parsed) {
		t.Fatalf("%+v %v", parsed, err)
	}
	if strings.Contains(text, `"token"`) || strings.Contains(text, `"cache_length"`) {
		t.Fatal("invented model fields")
	}
}
func TestComponentValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*Metadata)
	}{
		{"missing contract", func(m *Metadata) { m.Component = nil }},
		{"legacy schema", func(m *Metadata) { m.Version = 2 }},
		{"token", func(m *Metadata) { m.Token = m.Batch }},
		{"cache", func(m *Metadata) { m.CacheLength = m.Batch }},
		{"model phase", func(m *Metadata) { m.Phase = "decode" }},
		{"negative counter", func(m *Metadata) { n := -1; m.Executions = &n }},
		{"root outside set", func(m *Metadata) { m.Root = m.RootCount }},
		{"unknown unit", func(m *Metadata) { m.ExecutionUnit = "logical_forward" }},
		{"bad hash", func(m *Metadata) { m.Component.BankSHA256 = "unknown" }},
		{"missing strides", func(m *Metadata) { m.Component.Strides = nil }},
		{"missing math", func(m *Metadata) { m.Component.MathForm = "" }},
		{"delivered", func(m *Metadata) { m.Event = "delivered" }},
		{"batch start missing counter", func(m *Metadata) { m.Event = "batch_start"; m.Executions = nil }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := componentMetadata()
			tt.change(&m)
			if _, err := Encode(m); err == nil {
				t.Fatalf("accepted %+v", m)
			}
		})
	}
	m := componentMetadata()
	m.Component.Strides = []int{-1}
	if _, err := Encode(m); err != nil {
		t.Fatal("signed stride lost:", err)
	}
}
func TestComponentStrictJSON(t *testing.T) {
	text, err := Encode(componentMetadata())
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		strings.Replace(text, `"name":"gelu"`, `"name":"gelu","name":"other"`, 1),
		strings.Replace(text, `"name":"gelu"`, `"Name":"gelu"`, 1),
		strings.Replace(text, `"name":"gelu"`, `"unknown":"gelu"`, 1),
		strings.Replace(text, `"strides":[1]`, `"strides":[null]`, 1),
		strings.Replace(text, `"shape":[256]`, `"shape":null`, 1),
		strings.Replace(text, prefixV3, prefixV2, 1),
		strings.Replace(text, prefixV3, Prefix, 1),
	} {
		if _, err := Parse(bad); err == nil {
			t.Fatalf("accepted %s", bad)
		}
	}
}
func ExampleComponentContract() {
	c := ComponentContract{Name: "gelu", Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", RootPolicy: "distinct_banks"}
	fmt.Println(c.Name, c.DType, c.RootPolicy)
	// Output: gelu bfloat16 distinct_banks
}
