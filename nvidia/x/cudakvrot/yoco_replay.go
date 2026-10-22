package cudakvrot

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// SWAReplayConfig configures constant-window SWA bounded replay.
type SWAReplayConfig struct {
	WindowSize           int
	SpeculativeOvershoot int
	PageBlockSize        int
}

// SWAReplayWindow describes the bounded active token range and circular ring buffer slots for SWA replay.
type SWAReplayWindow struct {
	StartPos      int
	EndPos        int
	ActiveTokens  int
	RequiredPages int
	RingSlots     []int
}

// ResolveSWABoundedReplay computes the bounded rolling window
// [max(0, seqLenK - (WindowSize + SpeculativeOvershoot)), seqLenK) for SWA layers
// and maps logical positions to bounded circular ring buffer slots (pos % ringCapacity).
func ResolveSWABoundedReplay(seqLenK int, queryLen int, cfg SWAReplayConfig) (SWAReplayWindow, error) {
	if cfg.WindowSize == 0 {
		cfg.WindowSize = 128
	}
	if cfg.SpeculativeOvershoot == 0 {
		cfg.SpeculativeOvershoot = 6
	}
	if cfg.PageBlockSize == 0 {
		cfg.PageBlockSize = 64
	}
	if cfg.WindowSize < 0 || cfg.SpeculativeOvershoot < 0 || cfg.PageBlockSize <= 0 {
		return SWAReplayWindow{}, fmt.Errorf("cudakvrot: invalid SWA replay config %+v: %w", cfg, cudaerrors.ErrUnsupported)
	}
	if seqLenK <= 0 || queryLen <= 0 || queryLen > seqLenK {
		return SWAReplayWindow{}, fmt.Errorf("cudakvrot: invalid seqLenK=%d queryLen=%d: %w", seqLenK, queryLen, cudaerrors.ErrUnsupported)
	}

	ringCapacity := cfg.WindowSize + cfg.SpeculativeOvershoot
	startPos := seqLenK - ringCapacity
	if startPos < 0 {
		startPos = 0
	}
	endPos := seqLenK
	activeTokens := endPos - startPos
	requiredPages := (activeTokens + cfg.PageBlockSize - 1) / cfg.PageBlockSize

	ringSlots := make([]int, activeTokens)
	for i := 0; i < activeTokens; i++ {
		pos := startPos + i
		ringSlots[i] = pos % ringCapacity
	}

	return SWAReplayWindow{
		StartPos:      startPos,
		EndPos:        endPos,
		ActiveTokens:  activeTokens,
		RequiredPages: requiredPages,
		RingSlots:     ringSlots,
	}, nil
}
