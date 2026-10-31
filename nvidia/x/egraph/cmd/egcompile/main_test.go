package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
)

func TestEGCompileCLI(t *testing.T) {
	dir := t.TempDir()
	denseCfgPath := filepath.Join(dir, "dense_config.json")
	if err := os.WriteFile(denseCfgPath, []byte(`{
		"hidden_size": 1024,
		"intermediate_size": 2048,
		"num_hidden_layers": 16,
		"num_attention_heads": 16,
		"num_key_value_heads": 4,
		"rms_norm_eps": 1e-5
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	moeCfgPath := filepath.Join(dir, "moe_config.json")
	if err := os.WriteFile(moeCfgPath, []byte(`{
		"hidden_size": 1024,
		"moe_intermediate_size": 512,
		"num_hidden_layers": 24,
		"num_attention_heads": 16,
		"num_key_value_heads": 2,
		"num_local_experts": 4,
		"num_experts": 16,
		"num_experts_per_tok": 4,
		"attention_gate": true,
		"quantization_config": {"quant_method": "nvfp4"}
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("format_cuda", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := run([]string{"-config", denseCfgPath, "-device", "NVIDIA H200", "-m", "1,16,128", "-seq", "1024", "-format", "cuda"}, &out, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		s := out.String()
		for _, want := range []string{
			"dispatch_layer_m(int M,",
			"dispatch_layer_m_c(int M,",
			"layer_m1_run(",
			"layer_m16_run(",
			"layer_m128_run(",
		} {
			if !strings.Contains(s, want) {
				t.Errorf("cuda output missing %q", want)
			}
		}
	})

	t.Run("format_tvmffi", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := run([]string{"-config", denseCfgPath, "-m", "1,64", "-format", "tvmffi"}, &out, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		s := out.String()
		for _, want := range []string{
			"#include <tvm/ffi/container/tensor.h>",
			"namespace eg {",
			"TensorMatcher(",
			"LaunchKernel(",
		} {
			if !strings.Contains(s, want) {
				t.Errorf("tvmffi output missing %q", want)
			}
		}
		for i, line := range strings.Split(s, "\n") {
			if len(line) > hopper.MaxFFILineLen {
				t.Fatalf("line %d exceeds %d chars (%d)", i+1, hopper.MaxFFILineLen, len(line))
			}
		}
	})

	t.Run("format_json_moe_gb10", func(t *testing.T) {
		var out, errBuf bytes.Buffer
		code := run([]string{"-config", moeCfgPath, "-device", "NVIDIA GB10", "-m", "2,64", "-seq", "512", "-format", "json"}, &out, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		var bundle model.MultiBucketBundle
		if err := json.Unmarshal(out.Bytes(), &bundle); err != nil {
			t.Fatalf("unmarshal json output: %v", err)
		}
		if bundle.DispatchFunc != "dispatch_layer_m" || len(bundle.Buckets) != 2 {
			t.Errorf("bundle = {DispatchFunc:%q Buckets:%v}", bundle.DispatchFunc, bundle.Buckets)
		}
		if bundle.Layers[2].Mode != model.ModePersistentMegakernel {
			t.Errorf("M=2 mode = %q, want %q", bundle.Layers[2].Mode, model.ModePersistentMegakernel)
		}
		if bundle.Layers[64].Mode != model.ModeStreamKWarpSpec {
			t.Errorf("M=64 mode = %q, want %q", bundle.Layers[64].Mode, model.ModeStreamKWarpSpec)
		}
	})

	t.Run("beam_flag", func(t *testing.T) {
		var outJSON, errBuf bytes.Buffer
		code := run([]string{"-config", denseCfgPath, "-device", "NVIDIA H200", "-m", "1,64", "-seq", "512", "-beam", "3", "-format", "json"}, &outJSON, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		var parsed compiledOutput
		if err := json.Unmarshal(outJSON.Bytes(), &parsed); err != nil {
			t.Fatalf("unmarshal beam json output: %v", err)
		}
		for _, m := range []int{1, 64} {
			cands := parsed.Beam[m]
			if len(cands) == 0 || len(cands) > 3 {
				t.Fatalf("M=%d beam len = %d, want 1..3", m, len(cands))
			}
			seenSigs := make(map[string]bool)
			for i, c := range cands {
				if c.PlanSignature == "" {
					t.Errorf("M=%d cand[%d] empty PlanSignature", m, i)
				}
				if seenSigs[c.PlanSignature] {
					t.Errorf("M=%d duplicate PlanSignature %s", m, c.PlanSignature)
				}
				seenSigs[c.PlanSignature] = true
				if c.EstimatedNs <= 0 {
					t.Errorf("M=%d cand[%d] EstimatedNs = %f, want > 0", m, i, c.EstimatedNs)
				}
				if i > 0 && c.EstimatedNs < cands[i-1].EstimatedNs {
					t.Errorf("M=%d beam not sorted by EstimatedNs: %f < %f", m, c.EstimatedNs, cands[i-1].EstimatedNs)
				}
			}
		}

		var outCUDA bytes.Buffer
		errBuf.Reset()
		code = run([]string{"-config", denseCfgPath, "-device", "NVIDIA H200", "-m", "1", "-seq", "512", "-beam", "2", "-format", "cuda"}, &outCUDA, &errBuf)
		if code != 0 {
			t.Fatalf("cuda beam exit %d: %s", code, errBuf.String())
		}
		if !strings.Contains(outCUDA.String(), "Structural beam candidates (M=1") || !strings.Contains(outCUDA.String(), "plan_signature=") {
			t.Errorf("cuda beam output missing structural beam comments:\n%s", outCUDA.String())
		}
	})

	t.Run("ptxas_log_refinement", func(t *testing.T) {
		ptxasLogPath := filepath.Join(dir, "ptxas_loop.log")
		// Step 0 (row_1): clean
		// Step 1 (tile_gemm_2): register spill (24B store, 24B load) -> steps down BN 128 -> 64
		// Step 2 (paged_attention_3): clean
		// Step 3 (tile_gemm_4): low occupancy (220 regs > 192 max) -> steps down BN 128 -> 64
		// Step 4 (row_5): clean
		// Step 5 (tile_gemm_6): WGMMA serialized C7519 -> enables WarpSpecialized / steps down WGMMA
		logContent := `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'step0_row' for 'sm_90a'
ptxas info    : Function properties for step0_row
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function 'step1_q_proj' for 'sm_90a'
ptxas info    : Function properties for step1_q_proj
    32 bytes stack frame, 24 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 168 registers, 16384 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function 'step2_attn' for 'sm_90a'
ptxas info    : Function properties for step2_attn
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, 16384 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function 'step3_o_proj' for 'sm_90a'
ptxas info    : Function properties for step3_o_proj
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 220 registers, 16384 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function 'step4_ffn_norm' for 'sm_90a'
ptxas info    : Function properties for step4_ffn_norm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function 'step5_gate_up' in line 120
ptxas info    : Compiling entry function 'step5_gate_up' for 'sm_90a'
ptxas info    : Function properties for step5_gate_up
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, 16384 bytes smem, 384 bytes cmem[0]
`
		if err := os.WriteFile(ptxasLogPath, []byte(logContent), 0o644); err != nil {
			t.Fatal(err)
		}

		var outJSON, errBuf bytes.Buffer
		code := run([]string{
			"-config", denseCfgPath,
			"-device", "NVIDIA H200",
			"-m", "128",
			"-seq", "512",
			"-ptxas-log", ptxasLogPath,
			"-ptxas-loop",
			"-format", "json",
		}, &outJSON, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		var parsed compiledOutput
		if err := json.Unmarshal(outJSON.Bytes(), &parsed); err != nil {
			t.Fatalf("unmarshal ptxas json output: %v", err)
		}
		ref := parsed.PTXASRefinement[128]
		if ref == nil {
			t.Fatal("missing PTXASRefinement[128]")
		}
		if !ref.Converged {
			t.Errorf("expected PTXASRefinement[128].Converged = true, got false (actions=%v)", ref.Actions)
		}
		if len(ref.Actions) == 0 {
			t.Fatal("expected non-empty refinement actions")
		}
		if ref.InitialSignature == ref.RefinedSignature {
			t.Errorf("expected RefinedSignature != InitialSignature (%s)", ref.InitialSignature)
		}
	})

	t.Run("cross_layer_flag", func(t *testing.T) {
		var outJSON, errBuf bytes.Buffer
		code := run([]string{
			"-config", denseCfgPath,
			"-device", "NVIDIA H200",
			"-m", "1,64",
			"-seq", "512",
			"-cross-layer",
			"-format", "json",
		}, &outJSON, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		var parsed compiledOutput
		if err := json.Unmarshal(outJSON.Bytes(), &parsed); err != nil {
			t.Fatalf("unmarshal cross-layer json output: %v", err)
		}
		for _, m := range []int{1, 64} {
			cl := parsed.CrossLayer[m]
			if cl == nil {
				t.Fatalf("missing CrossLayer[%d]", m)
			}
			if len(cl.Stages) < 6 {
				t.Fatalf("M=%d cross-layer stages = %v (len %d), want >= 6 stages", m, cl.Stages, len(cl.Stages))
			}
			for _, wantStage := range []string{"l0_o_proj", "l0_down", "l0_post_norm", "l1_qkv_proj"} {
				if _, ok := cl.StageAttribution[wantStage]; !ok {
					t.Errorf("M=%d StageAttribution missing %q (got stages %v)", m, wantStage, cl.Stages)
				}
			}
			for _, st := range cl.Stages {
				sp, ok := cl.StageAttribution[st]
				if !ok {
					t.Errorf("M=%d StageAttribution missing stage %q", m, st)
				}
				if sp.StartLine <= 0 || sp.EndLine < sp.StartLine {
					t.Errorf("M=%d stage %q invalid span %+v", m, st, sp)
				}
			}
			if !strings.Contains(cl.CUDASource, "#line 1") {
				t.Errorf("M=%d cross-layer CUDASource missing #line directives", m)
			}
		}

		var outCUDA bytes.Buffer
		errBuf.Reset()
		code = run([]string{"-config", denseCfgPath, "-m", "1", "-cross-layer", "-format", "cuda"}, &outCUDA, &errBuf)
		if code != 0 {
			t.Fatalf("cuda cross-layer exit %d: %s", code, errBuf.String())
		}
		if !strings.Contains(outCUDA.String(), "cross_layer_m1_mega") || !strings.Contains(outCUDA.String(), "#line 1") {
			t.Errorf("cuda output missing cross_layer_m1_mega source")
		}
	})

	t.Run("pgo_cache_dir_flag", func(t *testing.T) {
		cacheDir := filepath.Join(dir, "pgo_cache")
		cache := autotune.NewVerifiedPGOCache(cacheDir)
		cubin := []byte("CUBIN_VERIFIED_M16_PAYLOAD")
		key := autotune.PGOCacheKey{
			Arch:          "sm_90a",
			DriverVersion: "570.124",
			CUDAVersion:   "12.8",
			WeightDigest:  "sha256:dense1024",
			PlanKey:       "layer_m16",
			ShapeBucket:   16,
		}
		proof := autotune.PGOPromotionProof{
			Key:            key,
			CandidateName:  "layer_m16_mega_winner",
			CubinSHA256:    sha256.Sum256(cubin),
			SourceSHA256:   sha256.Sum256([]byte("// source")),
			ScreenBlocks:   3,
			HoldoutBlocks:  8,
			ScreenRatio:    1.08,
			HoldoutRatio:   1.07,
			AANullRatio:    1.001,
			OneSidedPValue: 0.0039,
			TokensMatched:  true,
		}
		if err := cache.Store(key, proof, cubin); err != nil {
			t.Fatalf("cache.Store: %v", err)
		}

		var outJSON, errBuf bytes.Buffer
		code := run([]string{
			"-config", denseCfgPath,
			"-device", "NVIDIA H200",
			"-m", "1,16",
			"-pgo-cache-dir", cacheDir,
			"-format", "json",
		}, &outJSON, &errBuf)
		if code != 0 {
			t.Fatalf("exit %d: %s", code, errBuf.String())
		}
		var parsed compiledOutput
		if err := json.Unmarshal(outJSON.Bytes(), &parsed); err != nil {
			t.Fatalf("unmarshal pgo cache json: %v", err)
		}
		if parsed.PGOCache == nil || len(parsed.PGOCache.Entries) != 1 {
			t.Fatalf("PGOCache = %+v, want 1 verified entry", parsed.PGOCache)
		}
		got := parsed.PGOCache.Entries[0]
		if got.CandidateName != "layer_m16_mega_winner" || !got.TokensMatched || got.CubinBytes != len(cubin) {
			t.Errorf("unexpected PGOCache entry: %+v", got)
		}
	})

	t.Run("errors", func(t *testing.T) {
		for _, args := range [][]string{
			{},
			{"-config", filepath.Join(dir, "missing.json")},
			{"-config", denseCfgPath, "-m", "bad"},
			{"-config", denseCfgPath, "-format", "unknown"},
			{"-config", denseCfgPath, "-beam", "-1"},
			{"-config", denseCfgPath, "-ptxas-loop"},
			{"-config", denseCfgPath, "-ptxas-log", filepath.Join(dir, "missing.log")},
		} {
			var out, errBuf bytes.Buffer
			if code := run(args, &out, &errBuf); code == 0 {
				t.Errorf("args %v succeeded, want non-zero exit code", args)
			}
		}
	})
}
