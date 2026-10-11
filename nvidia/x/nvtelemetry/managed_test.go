package nvtelemetry

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"
)

func ExampleReadExposition() {
	s, err := ReadExposition(context.Background(), strings.NewReader("power_watts{UUID=\"GPU-a\"} 12.5\n"), ExpositionOptions{})
	fmt.Println(s[0].Name, s[0].RawValue, err)
	// Output: power_watts 12.5 <nil>
}

func TestReadExposition(t *testing.T) {
	if _, err := ReadExposition(context.Background(), strings.NewReader("a 1\n"), ExpositionOptions{MaxBytes: math.MaxInt64}); err == nil {
		t.Fatal("overflow limit accepted")
	}
	for _, tc := range []struct {
		name, text, format string
		bad                bool
	}{
		{"integer", "energy 9007199254740993 12\n", "", false},
		{"distinct series", "foo{bar=\"x\"} 1\nfoob{ar=\"x\"} 2\n", "", false},
		{"colon metric", "foo:bar{x=\"y\"} 1\n", "", false},
		{"colon label", "foo{bad:label=\"y\"} 1\n", "", true},
		{"late summary", "metric{quantile=\".9\"} 1\n# TYPE metric summary\n", "", true},
		{"histogram", "# TYPE latency histogram\nlatency_bucket{le=\"1\"} 3\nlatency_sum 2\nlatency_count 3\n", "", false},
		{"openmetrics", "# UNIT power watts\npower 1 1.000000001\n# EOF\n", OpenMetrics100, false},
		{"duplicate", "a{x=\"b\"} 1\na{x=\"b\"} 2\n", "", true},
		{"labels", "a{x=\"b\",x=\"c\"} 1\n", "", true},
		{"exemplar", "a 1 # {trace_id=\"x\"} 1\n", OpenMetrics100, true},
		{"truncated", "a 1\n", OpenMetrics100, true},
		{"overflow", "a 1 9223372036854775807\n", "", true},
		{"exponent bound", "a 1 1e-1000000000\n# EOF\n", OpenMetrics100, true},
		{"escape", "a{x=\"\\t\"} 1\n", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, e := ReadExposition(context.Background(), strings.NewReader(tc.text), ExpositionOptions{Format: tc.format})
			if (e != nil) != tc.bad {
				t.Fatalf("%v", e)
			}
			if tc.name == "integer" && (s[0].RawValue != "9007199254740993" || s[0].TimestampUnix != 12000000) {
				t.Fatal(s)
			}
			if tc.name == "openmetrics" && s[0].TimestampUnix != 1000000001 {
				t.Fatal(s)
			}
		})
	}
	for _, o := range []ExpositionOptions{{MaxBytes: 2}, {MaxSamples: 1}, {MaxLabels: 1}, {MaxLabelBytes: 1}, {MaxLineBytes: 2}} {
		if _, e := ReadExposition(context.Background(), strings.NewReader("a{x=\"b\",y=\"c\"} 1\nb 2\n"), o); e == nil {
			t.Fatalf("limit ignored: %+v", o)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := ReadExposition(ctx, strings.NewReader("a 1\n"), ExpositionOptions{}); e != context.Canceled {
		t.Fatal(e)
	}
}

func FuzzReadExposition(f *testing.F) {
	f.Add("a{x=\"y\"} 1\n")
	f.Fuzz(func(t *testing.T, text string) {
		ReadExposition(context.Background(), strings.NewReader(text), ExpositionOptions{MaxBytes: 4096, MaxSamples: 32, MaxLabels: 8, MaxLabelBytes: 128, MaxLineBytes: 1024})
	})
}
