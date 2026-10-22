package dsv4expertprofile

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestReportValuesAndHotlist(t *testing.T) {
	c, _ := New(1, 3, []int{2})
	_ = c.Observe(0, 0, []int{2, 0}, []float32{0.25, 0.75})
	_ = c.Observe(0, 1, []int{0, 1}, []float32{0.5, 0.5})
	var b bytes.Buffer
	m := Metadata{Source: "test", Model: "test", ExpertUsed: 2, HashRouter: []bool{true}}
	if err := c.WriteReport(&b, m); err != nil {
		t.Fatal(err)
	}
	var r struct {
		Records    uint64        `json:"layer_records"`
		Selections uint64        `json:"selections"`
		Cache      []cacheReport `json:"cache_summary"`
		Detail     []layerReport `json:"layers_detail"`
	}
	if err := json.Unmarshal(b.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Records != 2 || r.Selections != 4 || r.Cache[0].Hits != 1 || r.Cache[0].HitRate != 0.25 || r.Cache[0].WeightedHitRate != 0.25 || r.Detail[0].Overlap != 0.5 || r.Detail[0].Jaccard != 1.0/3 || r.Detail[0].Top[0].ID != 0 || r.Detail[0].Top[0].Weight != 1.25 {
		t.Fatalf("wrong value report: %+v", r)
	}
	b.Reset()
	if err := c.WriteHotlist(&b, "test"); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(b.String(), "0 0 2 1.25\n0 1 1 0.5\n0 2 1 0.25\n") {
		t.Fatal(b.String())
	}
	mutant, _ := New(1, 3, []int{2})
	_ = mutant.Observe(0, 0, []int{2, 0}, []float32{0.25, 0.75})
	_ = mutant.Observe(0, 1, []int{1, 2}, []float32{0.5, 0.5})
	if mutant.layers[0].CacheWeights[0] == c.layers[0].CacheWeights[0] {
		t.Fatal("route mutation escaped weighted cache oracle")
	}
	if err := c.WriteReport(shortReportWriter{}, m); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if err := c.WriteHotlist(shortReportWriter{}, "test"); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
}

type shortReportWriter struct{}

func (shortReportWriter) Write(p []byte) (int, error) { return len(p) / 2, nil }
