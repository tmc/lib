package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertprofile"
	"github.com/tmc/lib/examples/ds4-min/dsv4imatrix"
)

func TestProfileCollectorArtifactCoverage(t *testing.T) {
	im, err := dsv4imatrix.New(43, 256, 1, 1, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	ep, err := dsv4expertprofile.New(43, 256, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	p := &profileCollectors{imatrix: im, experts: ep}
	for layer := 0; layer < 42; layer++ {
		if err := p.observe(layer, 0, 1, []int32{0, 1, 2, 3, 4, 5}, []float32{1, 1, 1, 1, 1, 1}, []float32{2}, []float32{1, 2, 3, 4, 5, 6}); err != nil {
			t.Fatal(err)
		}
	}
	dir := t.TempDir()
	o := collectorOptions{imatrix: filepath.Join(dir, "means.dat"), expertProfile: filepath.Join(dir, "experts.json"), dataset: "fixture"}
	if writeImatrixFile(o, p, 1) == nil {
		t.Fatal("missing layer imatrix producer accepted")
	}
	if writeExpertProfile(config{collectors: o}, p) == nil {
		t.Fatal("missing layer expert producer accepted")
	}
	for _, path := range []string{o.imatrix, o.expertProfile} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("incomplete producer created %s: %v", path, err)
		}
	}
	if err := p.observe(42, 0, 1, []int32{0, 1, 2, 3, 4, 5}, []float32{1, 1, 1, 1, 1, 1}, []float32{2}, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if err := writeImatrixFile(o, p, 1); err != nil {
		t.Fatal(err)
	}
	if err := writeExpertProfile(config{collectors: o}, p); err != nil {
		t.Fatal(err)
	}
}

func TestProfileCollectorCoverage(t *testing.T) {
	var rows [43]uint64
	if completeProfileRows(rows, 0) == nil {
		t.Fatal("empty producer accepted")
	}
	for i := range rows {
		rows[i] = 2
	}
	if err := completeProfileRows(rows, 2); err != nil {
		t.Fatal(err)
	}
	for _, got := range []uint64{0, 1, 3} {
		rows[42] = got
		if completeProfileRows(rows, 2) == nil {
			t.Fatalf("partial or duplicate producer accepted: %d", got)
		}
	}
}

func TestProfileCollectorBatchMapping(t *testing.T) {
	im, err := dsv4imatrix.New(43, 256, 2, 2, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	ep, err := dsv4expertprofile.New(43, 256, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	p := &profileCollectors{imatrix: im, experts: ep}
	ids := []int32{9, 2, 7, 4, 6, 1, 3, 8, 5, 10, 11, 12}
	weights := make([]float32, 12)
	for i := range weights {
		weights[i] = 1
	}
	input := []float32{2, 3, 4, 5}
	mid := make([]float32, 24)
	for i := range mid {
		mid[i] = float32(i + 1)
	}
	if err := p.observe(2, 127, 2, ids, weights, input, mid); err != nil {
		t.Fatal(err)
	}
	var data bytes.Buffer
	if err := im.Write(&data, "fixture"); err != nil {
		t.Fatal(err)
	}
	before := append([]byte(nil), data.Bytes()...)
	ids[7] = 3
	if err := p.observe(2, 129, 2, ids, weights, input, mid); err == nil {
		t.Fatal("duplicate route accepted")
	}
	data.Reset()
	_ = im.Write(&data, "fixture")
	if !bytes.Equal(data.Bytes(), before) {
		t.Fatal("route failure changed imatrix")
	}
	if err := (&stateSessionBackend{}).setProfileCollectors(p); err == nil || !strings.Contains(err.Error(), "missing State") {
		t.Fatal("missing State admitted")
	}
	if _, err := newProfileCollectors(true, false, 1); err == nil {
		t.Fatal("sum byte bound ignored")
	}
	if p, err := newProfileCollectors(false, false, 0); err != nil || p != nil {
		t.Fatal("off path allocated collector")
	}
}
