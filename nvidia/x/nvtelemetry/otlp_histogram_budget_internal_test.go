package nvtelemetry

import (
	"fmt"
	"reflect"
	"testing"
)

// These are arithmetic controls, not valid public closed-profile fixtures.
func TestHistogramAttributeBudgetArithmetic(t *testing.T) {
	for _, tc := range []struct {
		name               string
		ordinary, reserved int
		ok                 bool
	}{
		{"ordinary32", 32, 0, true}, {"ordinary33", 33, 0, false},
		{"mapped45", 32, 13, true}, {"mapped46", 33, 13, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			attrs := map[string]histogramAttribute{}
			for i := 0; i < tc.ordinary; i++ {
				attrs[fmt.Sprintf("ordinary%d", i)] = histogramAttribute{}
			}
			typ := reflect.TypeOf(OTLPSourceTime{})
			for i := 0; i < tc.reserved; i++ {
				attrs["nvtelemetry.source_time."+typ.Field(i).Tag.Get("json")] = histogramAttribute{}
			}
			if got := histogramAttributeBudget(attrs) == nil; got != tc.ok {
				t.Fatalf("accepted=%v want%v", got, tc.ok)
			}
		})
	}
}
func TestHistogramAttributeBudgetExactReservedKeys(t *testing.T) {
	attrs := map[string]histogramAttribute{}
	for i := 0; i < 32; i++ {
		attrs[fmt.Sprintf("ordinary%d", i)] = histogramAttribute{}
	}
	for _, key := range []string{"nvtelemetry.source_time.unknown", "nvtelemetry.source_time.Source_class", "nvtelemetry.source_time.source_class.extra"} {
		attrs[key] = histogramAttribute{}
		if histogramAttributeBudget(attrs) == nil {
			t.Fatalf("exempted %q", key)
		}
		delete(attrs, key)
	}
}
