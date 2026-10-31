package train

import (
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestBuildOnlineLoRAMegakernel_ZeroGradHBM(t *testing.T) {
	cases := []struct {
		name      string
		rank      int
		numLayers int
		arch      hopper.MegakernelArchTarget
		wantGrad  bool
	}{
		{
			name:      "rank8_1layer_sm90a",
			rank:      8,
			numLayers: 1,
			arch:      hopper.TargetSM90A,
			wantGrad:  false,
		},
		{
			name:      "rank16_2layers_sm100a",
			rank:      16,
			numLayers: 2,
			arch:      hopper.TargetSM100A,
			wantGrad:  false,
		},
		{
			name:      "rank32_3layers_sm121a",
			rank:      32,
			numLayers: 3,
			arch:      hopper.TargetSM121A,
			wantGrad:  false,
		},
		{
			name:      "rank64_1layer_materializes_grad",
			rank:      64,
			numLayers: 1,
			arch:      hopper.TargetSM90A,
			wantGrad:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := OnlineLoRAConfig{
				NamePrefix:  "lora_" + tc.name,
				BatchM:      4,
				HiddenDim:   256,
				LoRARank:    tc.rank,
				Alpha:       float32(tc.rank * 2),
				LearnRate:   1e-2,
				WeightDecay: 1e-4,
				NumLayers:   tc.numLayers,
				Arch:        tc.arch,
			}

			plan, mega, err := BuildOnlineLoRAMegakernel(cfg)
			if err != nil {
				t.Fatalf("BuildOnlineLoRAMegakernel(%+v) failed: %v", cfg, err)
			}
			if plan == nil || mega == nil {
				t.Fatalf("BuildOnlineLoRAMegakernel returned nil plan or megakernel")
			}

			if got := HasMaterializedGradTemps(plan.Temps); got != tc.wantGrad {
				t.Errorf("HasMaterializedGradTemps(plan.Temps) = %v, want %v (temps=%+v)", got, tc.wantGrad, plan.Temps)
			}
			if got := HasMaterializedGradTemps(mega.Temps); got != tc.wantGrad {
				t.Errorf("HasMaterializedGradTemps(mega.Temps) = %v, want %v (temps=%+v)", got, tc.wantGrad, mega.Temps)
			}

			stagesPerLayer := 3
			if tc.rank > MaxInRegisterLoRARank {
				stagesPerLayer = 4
			}
			wantStages := tc.numLayers * stagesPerLayer
			if len(mega.Stages) != wantStages {
				t.Fatalf("len(mega.Stages) = %d, want %d", len(mega.Stages), wantStages)
			}

			// Verify inter-stage ready counter producer-consumer dependencies.
			for i, st := range mega.Stages {
				if i == 0 {
					if len(st.WaitSlots) != 0 {
						t.Errorf("stage 0 WaitSlots = %v, want empty", st.WaitSlots)
					}
				} else if len(st.WaitSlots) == 0 {
					t.Errorf("stage %d (%s) has empty WaitSlots", i, st.Name)
				}
				if i < len(mega.Stages)-1 {
					if st.SignalSlot != i {
						t.Errorf("stage %d SignalSlot = %d, want %d", i, st.SignalSlot, i)
					}
				} else if st.SignalSlot != -1 {
					t.Errorf("final stage %d SignalSlot = %d, want -1", i, st.SignalSlot)
				}
			}

			// Verify full CUDA translation unit generation and cross-layer attribution.
			src, err := mega.Source()
			if err != nil {
				t.Fatalf("mega.Source() failed: %v", err)
			}
			if !strings.Contains(src, "eg_ready_wait") || !strings.Contains(src, "eg_ready_signal") {
				t.Errorf("megakernel source missing ready barrier primitives")
			}
			attr := mega.StageAttributionMap()
			if len(attr) != wantStages {
				t.Errorf("len(StageAttributionMap) = %d, want %d", len(attr), wantStages)
			}
		})
	}
}

func TestBuildOnlineLoRAMegakernel_Validation(t *testing.T) {
	bad := []OnlineLoRAConfig{
		{NamePrefix: "123bad", BatchM: 4, HiddenDim: 256, LoRARank: 8, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 0, HiddenDim: 256, LoRARank: 8, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 32, LoRARank: 8, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 0, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 512, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 8, LearnRate: -1, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 8, WeightDecay: -0.1, NumLayers: 1},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 8, NumLayers: 0},
		{NamePrefix: "ok", BatchM: 4, HiddenDim: 256, LoRARank: 8, NumLayers: 1, Arch: "sm_70"},
	}
	for i, cfg := range bad {
		if _, _, err := BuildOnlineLoRAMegakernel(cfg); err == nil {
			t.Errorf("case %d (%+v): expected validation error, got nil", i, cfg)
		}
	}
}

func TestSimulateOnlineLoRAStep_DecreasesMSELoss(t *testing.T) {
	cases := []struct {
		name      string
		batchM    int
		hiddenDim int
		rank      int
		numLayers int
		perLayer  bool
	}{
		{
			name:      "1layer_r8_shared",
			batchM:    4,
			hiddenDim: 64,
			rank:      8,
			numLayers: 1,
			perLayer:  false,
		},
		{
			name:      "2layers_r16_shared",
			batchM:    4,
			hiddenDim: 64,
			rank:      16,
			numLayers: 2,
			perLayer:  false,
		},
		{
			name:      "3layers_r32_per_layer",
			batchM:    4,
			hiddenDim: 64,
			rank:      32,
			numLayers: 3,
			perLayer:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := OnlineLoRAConfig{
				NamePrefix:  "sim_" + tc.name,
				BatchM:      tc.batchM,
				HiddenDim:   tc.hiddenDim,
				LoRARank:    tc.rank,
				Alpha:       float32(tc.rank * 2),
				LearnRate:   5e-2,
				WeightDecay: 1e-4,
				NumLayers:   tc.numLayers,
				Arch:        hopper.TargetSM90A,
			}

			rng := rand.New(rand.NewPCG(101, uint64(tc.rank+tc.numLayers)))
			randSlice := func(n int, scale float32) []float32 {
				out := make([]float32, n)
				for i := range out {
					out[i] = (rng.Float32()*2 - 1) * scale
				}
				return out
			}

			M, D, R, L := tc.batchM, tc.hiddenDim, tc.rank, tc.numLayers
			x := randSlice(M*D, 0.5)
			target := randSlice(M*D, 0.5)

			mult := 1
			if tc.perLayer {
				mult = L
			}
			baseW := randSlice(mult*D*D, 0.15)
			loraA := randSlice(mult*R*D, 0.1)
			loraB := randSlice(mult*D*R, 0.05)

			loss0, loss1, curA, curB, err := SimulateOnlineLoRAStep(cfg, x, target, baseW, loraA, loraB)
			if err != nil {
				t.Fatalf("SimulateOnlineLoRAStep step 1 failed: %v", err)
			}
			if !(loss1 < loss0) {
				t.Fatalf("step 1 did not strictly decrease MSE loss: lossBefore=%g, lossAfter=%g", loss0, loss1)
			}
			if len(curA) != len(loraA) || len(curB) != len(loraB) {
				t.Fatalf("updated adapter lengths (%d, %d) != (%d, %d)", len(curA), len(curB), len(loraA), len(loraB))
			}

			// Run 4 additional online adaptation steps and verify monotonic loss reduction.
			prevLoss := loss1
			for step := 2; step <= 5; step++ {
				lb, la, nextA, nextB, err := SimulateOnlineLoRAStep(cfg, x, target, baseW, curA, curB)
				if err != nil {
					t.Fatalf("SimulateOnlineLoRAStep step %d failed: %v", step, err)
				}
				if !(la < lb) {
					t.Fatalf("step %d did not strictly decrease MSE loss: lossBefore=%g, lossAfter=%g", step, lb, la)
				}
				prevLoss = la
				curA, curB = nextA, nextB
			}
			t.Logf("%s: MSE loss decreased from %g -> %g -> %g over 5 steps", tc.name, loss0, loss1, prevLoss)
		})
	}
}
