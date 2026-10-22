//go:build linux

package main

import (
	"errors"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"strings"
	"testing"
)

// These guards must return before any device operation. Passing them is
// host-contract evidence, not a CUDA feature-kernel or upload qualification.
func TestDSparkNativeHostGuards(t *testing.T) {
	if _, err := newCUDADSparkWeights(nil, nil); err == nil {
		t.Fatal("unverified weight admission accepted")
	}
	if _, err := newCUDADSparkFeatures(nil, nil); err == nil {
		t.Fatal("missing request accepted")
	}
	if _, err := newCUDADSparkFeatures(&cudaRequestState{runtime: &cudaRuntime{}}, &dsparkModel{}); err == nil {
		t.Fatal("unbound request foundation accepted")
	}
	w := &cudaDSparkWeights{closed: true, tensors: map[string]cudaModelTensor{"x": {name: "x"}}}
	if _, err := w.tensor("x"); err == nil {
		t.Fatal("closed weights accessible")
	}
}

func TestDSparkNativeFeatureIdentity(t *testing.T) {
	makeFeature := func() (*cudaRequestState, *cudaDSparkFeatures) {
		r := &cudaRuntime{pos: 128}
		s := &cudaRequestState{runtime: r, generation: 3}
		f := &cudaDSparkFeatures{r: r, request: s, generation: 3, position: 128, targets: []int{1}, captured: []bool{false}, published: true}
		s.dspark, r.dspark = f, f
		return s, f
	}
	for _, test := range []struct{ kind, want string }{
		{"other-state", "request binding mismatch"},
		{"inactive", "inactive request capture"},
		{"history-generation", "history generation mismatch"},
		{"wrong-frontier", "no pending capture"},
		{"duplicate", "duplicate selected layer"},
		{"shape", "wrong HC shape"},
	} {
		kind := test.kind
		t.Run(kind, func(t *testing.T) {
			s, f := makeFeature()
			f.published = false
			width := Flash.HC * Flash.Embed
			switch kind {
			case "other-state":
				f.request = &cudaRequestState{runtime: f.r, generation: 3}
			case "inactive":
				f.r.dspark = nil
			case "history-generation":
				s.generation++
			case "wrong-frontier":
				f.position++
			case "duplicate":
				f.captured[0] = true
			case "shape":
				width--
			}
			if _, err := f.captureSlot(1, 128, width); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %s", err, test.want)
			}
		})
	}
	_, valid := makeFeature()
	valid.published = false
	if slot, err := valid.captureSlot(1, 128, Flash.HC*Flash.Embed); err != nil || slot != 0 {
		t.Fatalf("valid preflight = %d, %v", slot, err)
	}
	s, f := makeFeature()
	s.invalidateDSpark()
	if s.generation != 4 || f.position != -1 || f.published {
		t.Fatal("request mutation retained published features")
	}
}

func TestDSparkNativeFailedTransitions(t *testing.T) {
	r := &cudaRuntime{pos: 129, ctx: 4096}
	s := &cudaRequestState{runtime: r, generation: 4}
	f := &cudaDSparkFeatures{r: r, request: s, generation: 3, position: 128, published: true, targets: []int{1, 20}, captured: []bool{true, false}, modelSHA: strings.Repeat("a", 64)}
	s.dspark = f
	// A malformed Load must preserve the existing published frontier and
	// generation; no device call is reached by this short input.
	if err := s.Load(strings.NewReader("bad")); err == nil {
		t.Fatal("malformed load accepted")
	}
	if s.generation != 4 || !f.published || f.position != 128 {
		t.Fatal("rejected load mutated feature frontier")
	}
	v := &cudaDSparkFeatureSnapshot{owner: f, captured: []bool{true, true}, modelSHA: f.modelSHA, targets: []int{1, 20}, published: true, position: 128, generation: 3}
	if err := f.validateSnapshot(v, 129, 4); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"model", "layers", "owner", "published-incomplete", "frontier", "generation", "pending", "negative-token", "out-of-vocab-token", "unpublished-captured"} {
		bad := *v
		switch kind {
		case "model":
			bad.modelSHA = strings.Repeat("b", 64)
		case "layers":
			bad.targets = []int{1, 21}
		case "owner":
			bad.owner = &cudaDSparkFeatures{}
		case "published-incomplete":
			bad.captured = []bool{true, false}
		case "frontier":
			bad.position = 4096
		case "generation":
			bad.generation = 4
		case "pending":
			bad.published = false
		case "negative-token":
			bad.token = -1
		case "out-of-vocab-token":
			bad.token = Flash.Vocab
		case "unpublished-captured":
			bad.published, bad.position, bad.token = false, -1, -1
		}
		if err := f.validateSnapshot(&bad, 129, 4); err == nil {
			t.Fatalf("foreign %s admitted", kind)
		}
		if !f.published || f.position != 128 || s.generation != 4 {
			t.Fatal("snapshot rejection changed frontier")
		}
	}
	for _, position := range []int{-1, r.ctx + 1} {
		if err := f.validateSnapshot(v, position, 4); err == nil {
			t.Fatalf("accepted outer frontier %d", position)
		}
	}
	r.dspark = f
	s.generation = 3
	if err := s.finishDSparkEval(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatal("missing layer after target advance did not poison request")
	}
	if f.published || f.position != -1 {
		t.Fatal("failed seal retained published features")
	}
	if s.logits() != nil {
		t.Fatal("poisoned request exposes logits")
	}
	if err := s.Reset(); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatal("poisoned request reset accepted")
	}
	s.generation = ^uint64(0)
	s.invalidateDSpark()
	if s.generation != ^uint64(0) || !errors.Is(r.failed, dsv4.ErrDesynchronized) {
		t.Fatal("generation overflow wrapped")
	}
}
