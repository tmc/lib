package dsv41

import "testing"

func TestYoCoTopologyMatchesSchedule(t *testing.T) {
	topo, err := YoCoTopology()
	if err != nil {
		t.Fatalf("YoCoTopology failed: %v", err)
	}
	producers := topo.ProducerLayers()
	// 2 raw SWA layers (0, 1) + 4 compressedKV owner layers (2, 8, 14, 20) = 6 producers.
	if len(producers) != 6 {
		t.Fatalf("ProducerLayers count = %d, want 6 (%v)", len(producers), producers)
	}
	wantSource := func(layer int) int {
		switch {
		case layer < 2:
			return layer
		case layer < 8:
			return 2
		case layer < 14:
			return 8
		case layer < 20:
			return 14
		default:
			return 20
		}
	}
	for layer := 0; layer < 40; layer++ {
		spec, err := topo.SpecForLayer(layer)
		if err != nil {
			t.Fatalf("SpecForLayer(%d): %v", layer, err)
		}
		if want := wantSource(layer); spec.SourceLayerID != want {
			t.Fatalf("layer %d SourceLayerID = %d, want %d", layer, spec.SourceLayerID, want)
		}
	}
}

func TestSWAReplayWindowAndNVFP4(t *testing.T) {
	win, err := SWAReplayWindow(2048, 1, 4)
	if err != nil {
		t.Fatalf("SWAReplayWindow failed: %v", err)
	}
	if win.ActiveTokens <= 0 || win.ActiveTokens > 512+4 {
		t.Fatalf("unexpected ActiveTokens = %d", win.ActiveTokens)
	}

	cfg := DefaultCandidateBlockConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("DefaultCandidateBlockConfig.Validate: %v", err)
	}

	mults := EngramLayerMultipliers(14, 3, 42)
	if len(mults) != 3 {
		t.Fatalf("EngramLayerMultipliers len = %d, want 3", len(mults))
	}
	for i, m := range mults {
		if m%2 == 0 {
			t.Fatalf("multiplier[%d] = %d is not odd", i, m)
		}
	}

	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(i-16) * 0.25
	}
	packed, scales, err := QuantizeNVFP4(src)
	if err != nil {
		t.Fatalf("QuantizeNVFP4 failed: %v", err)
	}
	if len(packed) != 16 || len(scales) != 2 {
		t.Fatalf("unexpected NVFP4 lengths: packed=%d scales=%d", len(packed), len(scales))
	}
	restored, err := DequantizeNVFP4(packed, scales, len(src))
	if err != nil {
		t.Fatalf("DequantizeNVFP4 failed: %v", err)
	}
	if len(restored) != len(src) {
		t.Fatalf("restored len = %d, want %d", len(restored), len(src))
	}
}
