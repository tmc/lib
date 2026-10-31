package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestWarpSpecializedConfigValidation(t *testing.T) {
	post, nextAttnIn := buildTestCrossLayerPlans(t, 1, 256, 256, 64, 256, true, false)
	_, mega, err := BuildCrossLayerMegakernel(post, nextAttnIn, CrossLayerGlue{
		Name: "ws_valid_mega",
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}

	tests := []struct {
		name    string
		mega    *Megakernel
		cfg     WarpSpecializedConfig
		wantErr bool
	}{
		{
			name:    "default_2p_6c_24_232_3stage",
			mega:    &mega,
			cfg:     DefaultWarpSpecializedConfig(),
			wantErr: false,
		},
		{
			name: "valid_4p_4c_40_168_4stage",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:        4,
				ConsumerWarps:        4,
				ProducerMaxRegs:      40,
				ConsumerMaxRegs:      168,
				RingBufferStages:     4,
				EnableGridDepControl: false,
				DynamicStageRegs:     false,
			},
			wantErr: false,
		},
		{
			name:    "nil_megakernel",
			mega:    nil,
			cfg:     DefaultWarpSpecializedConfig(),
			wantErr: true,
		},
		{
			name: "bad_warp_sum",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    2,
				ConsumerWarps:    4,
				ProducerMaxRegs:  24,
				ConsumerMaxRegs:  232,
				RingBufferStages: 3,
			},
			wantErr: true,
		},
		{
			name: "zero_producer_warps",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    0,
				ConsumerWarps:    8,
				ProducerMaxRegs:  24,
				ConsumerMaxRegs:  232,
				RingBufferStages: 3,
			},
			wantErr: true,
		},
		{
			name: "unaligned_producer_regs",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    2,
				ConsumerWarps:    6,
				ProducerMaxRegs:  28,
				ConsumerMaxRegs:  232,
				RingBufferStages: 3,
			},
			wantErr: true,
		},
		{
			name: "unaligned_consumer_regs",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    2,
				ConsumerWarps:    6,
				ProducerMaxRegs:  24,
				ConsumerMaxRegs:  230,
				RingBufferStages: 3,
			},
			wantErr: true,
		},
		{
			name: "exceeds_64k_register_file",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    4,
				ConsumerWarps:    4,
				ProducerMaxRegs:  256,
				ConsumerMaxRegs:  256,
				RingBufferStages: 3,
			},
			wantErr: true,
		},
		{
			name: "ring_buffer_stages_too_small",
			mega: &mega,
			cfg: WarpSpecializedConfig{
				ProducerWarps:    2,
				ConsumerWarps:    6,
				ProducerMaxRegs:  24,
				ConsumerMaxRegs:  232,
				RingBufferStages: 1,
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			wm, err := BuildWarpSpecializedMegakernel(tc.mega, tc.cfg)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if wm == nil {
				t.Fatal("returned nil WarpSpecializedMegakernel")
			}
		})
	}
}

func TestWarpSpecializedMegakernelCodegenAndPTX(t *testing.T) {
	post, nextAttnIn := buildTestCrossLayerPlans(t, 1, 256, 256, 64, 256, true, false)
	_, mega, err := BuildCrossLayerMegakernel(post, nextAttnIn, CrossLayerGlue{
		Name:            "cross_ws_pdl_mega",
		LayerLPrefix:    "l0_",
		LayerNextPrefix: "l1_",
		BridgeInput:     "xn",
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}

	cfg := DefaultWarpSpecializedConfig()
	wm, err := BuildWarpSpecializedMegakernel(&mega, cfg)
	if err != nil {
		t.Fatalf("BuildWarpSpecializedMegakernel: %v", err)
	}

	if len(wm.StageHints) != len(mega.Stages) {
		t.Fatalf("len(StageHints) = %d, want %d", len(wm.StageHints), len(mega.Stages))
	}
	for i, h := range wm.StageHints {
		if h.ProducerRegs%8 != 0 || h.ConsumerRegs%8 != 0 {
			t.Fatalf("stage %d (%s) reg hints (%d, %d) not 8-reg aligned", i, h.StageName, h.ProducerRegs, h.ConsumerRegs)
		}
	}

	cudaSrc := wm.CUDA()
	ptxSrc := wm.PTXPreview()

	wantCUDASubstrings := []string{
		`asm volatile("setmaxnreg.dec.sync.aligned %d;\n" :: "n"(24));`,
		`asm volatile("setmaxnreg.inc.sync.aligned %d;\n" :: "n"(232));`,
		`griddepcontrol.wait;`,
		`griddepcontrol.launch_dependents;`,
		`mbarrier.init.shared.b64`,
		`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`,
		`mbarrier.try_wait.parity.shared.b64`,
		`eg_ws_griddepcontrol_wait();`,
		`eg_ws_griddepcontrol_launch_dependents();`,
		`eg_ws_stage_reg_hint<`,
	}
	for _, sub := range wantCUDASubstrings {
		if !strings.Contains(cudaSrc, sub) {
			t.Errorf("CUDA() missing expected substring %q", sub)
		}
	}

	wantPTXSubstrings := []string{
		`griddepcontrol.wait;`,
		`griddepcontrol.launch_dependents;`,
		`setmaxnreg.dec.sync.aligned 24;`,
		`setmaxnreg.inc.sync.aligned 232;`,
		`mbarrier.init.shared.b64`,
		`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`,
		`mbarrier.try_wait.parity.shared.b64`,
	}
	for _, sub := range wantPTXSubstrings {
		if !strings.Contains(ptxSrc, sub) {
			t.Errorf("PTXPreview() missing expected substring %q", sub)
		}
	}

	// Verify disabling EnableGridDepControl omits PDL wait/launch_dependents from PTXPreview and kernel body.
	cfgNoPDL := cfg
	cfgNoPDL.EnableGridDepControl = false
	wmNoPDL, err := BuildWarpSpecializedMegakernel(&mega, cfgNoPDL)
	if err != nil {
		t.Fatalf("BuildWarpSpecializedMegakernel(no PDL): %v", err)
	}
	if strings.Contains(wmNoPDL.PTXPreview(), "griddepcontrol.") {
		t.Errorf("PTXPreview() with EnableGridDepControl=false should not contain griddepcontrol")
	}
	if strings.Contains(wmNoPDL.CUDA(), "eg_ws_griddepcontrol_wait();") {
		t.Errorf("CUDA() with EnableGridDepControl=false should not call eg_ws_griddepcontrol_wait()")
	}
}

func TestWarpSpecializedMegakernelEmulated(t *testing.T) {
	const (
		tokens = 1
		h      = 256
		qDim   = 256
		kvDim  = 64
		inter  = 256
	)
	post, nextAttnIn := buildTestCrossLayerPlans(t, tokens, h, qDim, kvDim, inter, true, false)
	fusedPlan, mega, err := BuildCrossLayerMegakernel(post, nextAttnIn, CrossLayerGlue{
		Name:               "cross_ws_emu_mega",
		LayerLPrefix:       "l0_",
		LayerNextPrefix:    "l1_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	})
	if err != nil {
		t.Fatalf("BuildCrossLayerMegakernel: %v", err)
	}

	wm, err := BuildWarpSpecializedMegakernel(&mega, DefaultWarpSpecializedConfig())
	if err != nil {
		t.Fatalf("BuildWarpSpecializedMegakernel: %v", err)
	}
	src, err := wm.Source()
	if err != nil {
		t.Fatalf("wm.Source(): %v", err)
	}

	r := rand.New(rand.NewPCG(101, 102))
	envs := make([]map[string]Value, 2)
	for i := range envs {
		envs[i] = randEnv(r, fusedPlan.Inputs...)
	}

	outs, readyWords, err := runMegakernelSourceEpochs(t, wm.Megakernel, src, envs)
	if err != nil {
		t.Fatalf("runMegakernelSourceEpochs: %v", err)
	}
	for ep, env := range envs {
		seqOut := runSequentialCrossLayer(t, post, nextAttnIn, env, "xn")
		agree(t, fmt.Sprintf("ws_epoch_%d_sequential", ep+1), F32, outs[ep], seqOut)
	}
	for i := 0; i < len(wm.Stages)-1; i++ {
		wantWord := PackReadyWord(uint32(len(envs)), uint32(wm.Stages[i].Blocks))
		if readyWords[i] != wantWord {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], wantWord)
		}
	}
}
