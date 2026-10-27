package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
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
		var bundle hopper.MultiBucketBundle
		if err := json.Unmarshal(out.Bytes(), &bundle); err != nil {
			t.Fatalf("unmarshal json output: %v", err)
		}
		if bundle.DispatchFunc != "dispatch_layer_m" || len(bundle.Buckets) != 2 {
			t.Errorf("bundle = {DispatchFunc:%q Buckets:%v}", bundle.DispatchFunc, bundle.Buckets)
		}
		if bundle.Layers[2].Mode != hopper.ModePersistentMegakernel {
			t.Errorf("M=2 mode = %q, want %q", bundle.Layers[2].Mode, hopper.ModePersistentMegakernel)
		}
		if bundle.Layers[64].Mode != hopper.ModeStreamKWarpSpec {
			t.Errorf("M=64 mode = %q, want %q", bundle.Layers[64].Mode, hopper.ModeStreamKWarpSpec)
		}
	})

	t.Run("errors", func(t *testing.T) {
		for _, args := range [][]string{
			{},
			{"-config", filepath.Join(dir, "missing.json")},
			{"-config", denseCfgPath, "-m", "bad"},
			{"-config", denseCfgPath, "-format", "unknown"},
		} {
			var out, errBuf bytes.Buffer
			if code := run(args, &out, &errBuf); code == 0 {
				t.Errorf("args %v succeeded, want non-zero exit code", args)
			}
		}
	})
}
