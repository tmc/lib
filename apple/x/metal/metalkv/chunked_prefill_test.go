// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"reflect"
	"testing"
)

func TestChunkedPrefillSchedulerTable(t *testing.T) {
	tests := []struct {
		name       string
		chunkSize  int
		maxBudget  int
		prefills   []*ChunkedPrefillRequest
		decodes    []string
		wantTotal  int
		wantChunks int
		wantDec    int
	}{
		{
			name:      "single prompt smaller than chunk",
			chunkSize: 128,
			maxBudget: 256,
			prefills: []*ChunkedPrefillRequest{
				{
					SeqID:        "seq-1",
					PromptTokens: make([]int32, 64),
					TotalTokens:  64,
				},
			},
			decodes:    []string{},
			wantTotal:  64,
			wantChunks: 1,
			wantDec:    0,
		},
		{
			name:      "single prompt larger than chunk sliced",
			chunkSize: 64,
			maxBudget: 256,
			prefills: []*ChunkedPrefillRequest{
				{
					SeqID:        "seq-1",
					PromptTokens: make([]int32, 160),
					TotalTokens:  160,
				},
			},
			decodes:    []string{},
			wantTotal:  64,
			wantChunks: 1,
			wantDec:    0,
		},
		{
			name:      "prioritizes decodes over prefill chunks",
			chunkSize: 100,
			maxBudget: 50,
			prefills: []*ChunkedPrefillRequest{
				{
					SeqID:        "seq-prefill",
					PromptTokens: make([]int32, 100),
					TotalTokens:  100,
				},
			},
			decodes:    []string{"dec-1", "dec-2", "dec-3"},
			wantTotal:  50, // 3 decodes + 47 prefill tokens
			wantChunks: 1,
			wantDec:    3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewChunkedPrefillScheduler(tt.chunkSize, tt.maxBudget)
			batch := s.Schedule(tt.prefills, tt.decodes)

			if batch.TotalTokens != tt.wantTotal {
				t.Errorf("got TotalTokens %d, want %d", batch.TotalTokens, tt.wantTotal)
			}
			if len(batch.PrefillChunks) != tt.wantChunks {
				t.Errorf("got %d prefill chunks, want %d", len(batch.PrefillChunks), tt.wantChunks)
			}
			if len(batch.DecodeSeqIDs) != tt.wantDec {
				t.Errorf("got %d decode seqs, want %d", len(batch.DecodeSeqIDs), tt.wantDec)
			}
		})
	}
}

func TestChunkedPrefillMultiStepCompletion(t *testing.T) {
	s := NewChunkedPrefillScheduler(100, 250)
	req := &ChunkedPrefillRequest{
		SeqID:        "long-prompt",
		PromptTokens: make([]int32, 230),
		TotalTokens:  230,
	}

	// Step 1: takes 100 tokens (0..100)
	b1 := s.Schedule([]*ChunkedPrefillRequest{req}, nil)
	if len(b1.PrefillChunks) != 1 || len(b1.PrefillChunks[0].Tokens) != 100 {
		t.Fatalf("step 1 expected 100 tokens, got %v", b1.PrefillChunks)
	}
	if b1.PrefillChunks[0].IsLast {
		t.Fatal("step 1 should not be marked IsLast")
	}
	if req.IsPrefillDone {
		t.Fatal("step 1 should not mark request done")
	}

	// Step 2: takes 100 tokens (100..200)
	b2 := s.Schedule([]*ChunkedPrefillRequest{req}, nil)
	if len(b2.PrefillChunks) != 1 || len(b2.PrefillChunks[0].Tokens) != 100 {
		t.Fatalf("step 2 expected 100 tokens, got %v", b2.PrefillChunks)
	}
	if b2.PrefillChunks[0].IsLast {
		t.Fatal("step 2 should not be marked IsLast")
	}
	if req.IsPrefillDone {
		t.Fatal("step 2 should not mark request done")
	}

	// Step 3: takes remaining 30 tokens (200..230)
	b3 := s.Schedule([]*ChunkedPrefillRequest{req}, nil)
	if len(b3.PrefillChunks) != 1 || len(b3.PrefillChunks[0].Tokens) != 30 {
		t.Fatalf("step 3 expected 30 tokens, got %v", b3.PrefillChunks)
	}
	if !b3.PrefillChunks[0].IsLast {
		t.Fatal("step 3 should be marked IsLast")
	}
	if !req.IsPrefillDone {
		t.Fatal("step 3 should mark request done")
	}

	// Step 4: already done, should return 0 chunks
	b4 := s.Schedule([]*ChunkedPrefillRequest{req}, nil)
	if len(b4.PrefillChunks) != 0 {
		t.Fatalf("step 4 expected 0 chunks, got %d", len(b4.PrefillChunks))
	}
}

func TestScheduledBatchDecodeOrder(t *testing.T) {
	s := NewChunkedPrefillScheduler(128, 10)
	decodes := []string{"s1", "s2", "s3"}
	batch := s.Schedule(nil, decodes)

	if !reflect.DeepEqual(batch.DecodeSeqIDs, decodes) {
		t.Errorf("got decodes %v, want %v", batch.DecodeSeqIDs, decodes)
	}
	if batch.TotalTokens != 3 {
		t.Errorf("got TotalTokens %d, want 3", batch.TotalTokens)
	}
}
