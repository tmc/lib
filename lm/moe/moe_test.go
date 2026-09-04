// SPDX-License-Identifier: Apache-2.0

package moe_test

import (
	"context"
	"testing"

	"github.com/tmc/lib/lm/moe"
)

func TestGroupedTopKRouter(t *testing.T) {
	cfg := moe.MoEConfig{
		NumRoutedExperts: 8,
		TopK:             2,
		NumSharedExperts: 1,
	}
	router := moe.NewGroupedTopKRouter(cfg)

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
	tempDir := t.TempDir()

	mgr, err := moe.NewExpertCacheManager(tempDir, 4)
	if err != nil {
		t.Fatalf("NewExpertCacheManager failed: %v", err)
	}
	defer mgr.Close()

	dummyWeights := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if err := mgr.RegisterExpert(42, dummyWeights); err != nil {
		t.Fatalf("RegisterExpert failed: %v", err)
	}

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
