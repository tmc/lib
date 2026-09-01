// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"context"
	"os"
	"testing"
)

func TestGroupedTopKRouter(t *testing.T) {
	cfg := MoEConfig{
		NumRoutedExperts: 8,
		TopK:             2,
		NumSharedExperts: 1,
	}
	router := NewGroupedTopKRouter(cfg)

	logits := []float32{0.1, 2.5, 0.3, 0.4, 5.0, 1.2, 0.8, 0.2}
	decision := router.Route(logits)

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
