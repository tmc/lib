package cudakvrot_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakvrot"
)

func TestSlidingWindowCache(t *testing.T) {
	swc, err := cudakvrot.NewSlidingWindowCache(2, 4, 64, 512)
	if err != nil {
		t.Fatalf("NewSlidingWindowCache failed: %v", err)
	}

	// Pos 100 within window
	start, end := swc.ContextBounds(100)
	if start != 0 || end != 101 {
		t.Errorf("expected bounds [0, 101], got [%d, %d]", start, end)
	}

	// Pos 600 past window 512
	start, end = swc.ContextBounds(600)
	if start != 89 || end != 601 {
		t.Errorf("expected bounds [89, 601], got [%d, %d]", start, end)
	}

	slot := swc.SlotForPosition(600)
	if slot != 600%512 {
		t.Errorf("expected slot %d, got %d", 600%512, slot)
	}
}

func TestChunkPrompt(t *testing.T) {
	prompt := make([]int32, 1000)
	for i := range prompt {
		prompt[i] = int32(i)
	}

	chunks := cudakvrot.ChunkPrompt(prompt, 256)
	if len(chunks) != 4 {
		t.Fatalf("expected 4 chunks for 1000 tokens with chunk size 256, got %d", len(chunks))
	}

	if len(chunks[0].Tokens) != 256 || len(chunks[3].Tokens) != 232 {
		t.Errorf("unexpected chunk lengths: %d, %d", len(chunks[0].Tokens), len(chunks[3].Tokens))
	}
}
