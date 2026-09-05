package cuptirange

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestWriteMetrics(t *testing.T) {
	for _, metrics := range [][]Metric{nil, {{"", 1}}, {{"m", math.NaN()}}, {{"m", math.Inf(1)}}, {{"m", 1}, {"m", 2}}} {
		var b bytes.Buffer
		if err := WriteMetrics(perfetto.NewWriter(&b), 1, 0, metrics); err == nil || b.Len() != 0 {
			t.Fatal("invalid metrics emitted trace bytes")
		}
	}
	var b bytes.Buffer
	if err := WriteMetrics(perfetto.NewWriter(&b), ^uint64(0), 0, []Metric{{"a", 1}, {"b", 2}}); err == nil || b.Len() != 0 {
		t.Fatal("track overflow accepted")
	}
	if err := WriteMetrics(perfetto.NewWriter(&b), 1, 0, []Metric{{"sm__cycles_elapsed.avg", 7}}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b.Bytes(), []byte("sm__cycles_elapsed.avg")) {
		t.Fatal("lost metric identity")
	}
}

func ExampleWriteMetrics() {
	var b bytes.Buffer
	err := WriteMetrics(perfetto.NewWriter(&b), 1, 0, []Metric{{Name: "example_metric", Value: 7}})
	fmt.Println(err)
	// Output: <nil>
}
