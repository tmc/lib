// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"context"
	"os"
	"strings"
	"testing"
)

func TestGatherGeneratedSources(t *testing.T) {
	s, err := GatherSource(16)
	if err != nil || !strings.Contains(s, "moe_gather") {
		t.Fatal(err)
	}
	if s, err := PrefillW256Source(); err != nil || !strings.Contains(s, "moe_prefill_w256") {
		t.Fatalf("prefill source: %v", err)
	}
}

func TestMoEGatherReference(t *testing.T) {
	in := make([]float32, 16)
	in[3] = 7
	x, e := Gather16([][]float32{in}, []int{0}, 16)
	if e != nil || x[3] != 7 {
		t.Fatalf("gather=%v err=%v", x, e)
	}
}
func TestGatherAlignmentAndTails(t *testing.T) {
	if _, e := Gather16([][]float32{{1}}, []int{0}, 15); e == nil {
		t.Fatal("accepted unaligned tile width")
	}
	if _, e := Gather16([][]float32{{1, 2}}, []int{2}, 16); e == nil {
		t.Fatal("accepted bad expert")
	}
}

func TestGroupedTopKRouter(t *testing.T) {
	cfg := MoEConfig{
		NumRoutedExperts: 8,
		TopK:             2,
		NumSharedExperts: 1,
	}
	router, err := NewGroupedTopKRouter(cfg)
	if err != nil {
		t.Fatal(err)
	}

	logits := []float32{0.1, 2.5, 0.3, 0.4, 5.0, 1.2, 0.8, 0.2}
	decision, err := router.Route(logits)
	if err != nil {
		t.Fatal(err)
	}

	if len(decision.SelectedExperts) != 2 {
		t.Fatalf("selected experts len = %d, want 2", len(decision.SelectedExperts))
	}
	if decision.SelectedExperts[0] != 4 || decision.SelectedExperts[1] != 1 {
		t.Errorf("selected experts = %v, want [4 1]", decision.SelectedExperts)
	}
	if !decision.HasShared {
		t.Errorf("expected HasShared = true")
	}
}

func TestExpertCacheManager(t *testing.T) {
	tempDir, _ := os.MkdirTemp("", "cudamoe-test-*")
	defer os.RemoveAll(tempDir)

	mgr, err := NewExpertCacheManager(tempDir, 4)
	if err != nil {
		t.Fatalf("NewExpertCacheManager failed: %v", err)
	}
	defer mgr.Close()

	dummyWeights := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	mgr.RegisterExpert(42, dummyWeights)

	if err := mgr.EvictColdExpert(42); err != nil {
		t.Fatalf("EvictColdExpert failed: %v", err)
	}

	restored, err := mgr.GetExpert(context.Background(), 42)
	if err != nil {
		t.Fatalf("GetExpert failed: %v", err)
	}
	if len(restored) != len(dummyWeights) {
		t.Errorf("restored len = %d, want %d", len(restored), len(dummyWeights))
	}
}
