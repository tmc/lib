package hopper

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestRubinCrossLayerMegakernelLoweringAndPTX94Preview(t *testing.T) {
	cfg := RubinCrossLayerConfig{
		Name:                    "rubin_nvl144_3layer_mega",
		NumLayers:               3,
		Tokens:                  1,
		HiddenDim:               64,
		QDim:                    64,
		KVDim:                   64,
		IntermediateDim:         64,
		ClusterSize:             32,
		NVL144Ranks:             144,
		UseFP16x2Epilogue:       true,
		RetainInterLayerOutputs: true,
	}

	plan, mega, err := BuildRubinCrossLayerMegakernel(cfg)
	if err != nil {
		t.Fatalf("BuildRubinCrossLayerMegakernel: %v", err)
	}
	if plan == nil || mega == nil {
		t.Fatal("expected non-nil Plan and Megakernel")
	}

	// Each cross-layer boundary has 5 post-attn FFN stages + 1 QKV stage = 6 stages/boundary -> 12 stages for 3 layers (2 boundaries).
	if len(mega.Stages) != 12 {
		t.Fatalf("len(mega.Stages) = %d, want 12", len(mega.Stages))
	}

	cudaSrc, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	if !strings.Contains(cudaSrc, "rubin_nvl144_3layer_mega(") {
		t.Errorf("mega.CUDA() missing kernel entry point:\n%s", cudaSrc)
	}

	ptx := mega.RubinPTX94Preview()
	for _, wantPTX := range []string{
		".version 9.4",
		".target sm_107a",
		".maxclusterrank 32",
		"3.6 TB/s (3600 GB/s) bidirectional",
		"add.rn.f16x2",
		"fma.rn.bf16x2",
		"barrier.cluster.arrive.relaxed.aligned",
		"barrier.cluster.wait.acquire.aligned",
	} {
		if !strings.Contains(ptx, wantPTX) {
			t.Errorf("RubinPTX94Preview missing %q\n%s", wantPTX, ptx)
		}
	}
}

func TestRubinCrossLayerMegakernelHostEmulation(t *testing.T) {
	const (
		m     = 1
		h     = 64
		qDim  = 64
		kvDim = 64
		inter = 64
	)

	fusedPlan, mega, err := BuildRubinCrossLayerMegakernel(RubinCrossLayerConfig{
		Name:                    "rubin_sm107a_emu_mega",
		NumLayers:               2,
		Tokens:                  m,
		HiddenDim:               h,
		QDim:                    qDim,
		KVDim:                   kvDim,
		IntermediateDim:         inter,
		ClusterSize:             32,
		NVL144Ranks:             144,
		UseFP16x2Epilogue:       true,
		RetainInterLayerOutputs: true,
	})
	if err != nil {
		t.Fatalf("BuildRubinCrossLayerMegakernel: %v", err)
	}
	if len(mega.Stages) != 6 {
		t.Fatalf("len(mega.Stages) = %d, want 6", len(mega.Stages))
	}

	rng := rand.New(rand.NewPCG(77, 107))
	envs := []map[string]Value{
		randEnv(rng, fusedPlan.Inputs...),
	}

	outs, _, err := runMegakernelEpochs(t, *mega, envs)
	if err != nil {
		t.Fatalf("runMegakernelEpochs: %v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("len(outs) = %d, want 1", len(outs))
	}

	for ep, env := range envs {
		planOut := runPlan(t, fusedPlan, env)
		agree(t, "rubin_cross_layer", F32, outs[ep], planOut)
		nonZero := 0
		for i, v := range outs[ep] {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("epoch %d outs[%d] = %v", ep, i, v)
			}
			if v != 0 {
				nonZero++
			}
		}
		if nonZero == 0 {
			t.Fatalf("epoch %d: expected non-zero final QKV output from Rubin cross-layer megakernel", ep)
		}
	}
}

func TestRubinCrossLayerMegakernelValidation(t *testing.T) {
	tests := []struct {
		name    string
		cfg     RubinCrossLayerConfig
		wantSub string
	}{
		{
			name:    "bad_name",
			cfg:     RubinCrossLayerConfig{Name: "123-bad"},
			wantSub: "bad name",
		},
		{
			name:    "single_layer_rejected",
			cfg:     RubinCrossLayerConfig{NumLayers: 1},
			wantSub: "NumLayers 1 must be in 2..64",
		},
		{
			name:    "cluster_size_exceeds_32",
			cfg:     RubinCrossLayerConfig{ClusterSize: 64},
			wantSub: "ClusterSize 64 must be a power of two in 1..32",
		},
		{
			name:    "cluster_size_not_power_of_two",
			cfg:     RubinCrossLayerConfig{ClusterSize: 12},
			wantSub: "ClusterSize 12 must be a power of two in 1..32",
		},
		{
			name:    "nvl144_ranks_exceeds_144",
			cfg:     RubinCrossLayerConfig{NVL144Ranks: 288},
			wantSub: "NVL144Ranks 288 must be in 1..144",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := BuildRubinCrossLayerMegakernel(tc.cfg)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSub)
			}
		})
	}
}
