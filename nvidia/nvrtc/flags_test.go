package nvrtc_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvrtc"
)

func TestValidateOption(t *testing.T) {
	tests := []struct {
		name    string
		opt     string
		wantErr string
	}{
		{name: "empty", opt: "", wantErr: "empty option"},
		{name: "nul", opt: "flag\x00bad", wantErr: "contains NUL"},
		{name: "unknown", opt: "--nonexistent-option", wantErr: "unknown option"},
		{name: "fast_math", opt: "--use_fast_math"},
		{name: "fast_math_arg_rejected", opt: "--use_fast_math=true", wantErr: "takes no argument"},
		{name: "std_cxx17", opt: "-std=c++17"},
		{name: "std_invalid", opt: "-std=c++99", wantErr: "invalid value"},
		{name: "arch_compute", opt: "--gpu-architecture=compute_80"},
		{name: "arch_sm", opt: "-arch=sm_90"},
		{name: "arch_invalid", opt: "--gpu-architecture=x86_64", wantErr: "invalid value"},
		{name: "opt_level", opt: "-O3"},
		{name: "define", opt: "-DFEATURE=1"},
		{name: "undefine", opt: "-UFEATURE"},
		{name: "include", opt: "-I/usr/local/cuda/include"},
		{name: "fast_compile", opt: "--Ofast-compile=max"},
		{name: "fast_compile_invalid", opt: "--Ofast-compile=hyper", wantErr: "invalid value"},
		{name: "stack_protector", opt: "--device-stack-protector=true"},
		{name: "no_cache", opt: "--no-cache"},
		{name: "line_info", opt: "--generate-line-info"},
		{name: "device_debug", opt: "-G"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := nvrtc.ValidateOption(tc.opt)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("ValidateOption(%q) = %v, want substring %q", tc.opt, err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateOption(%q) unexpected error: %v", tc.opt, err)
			}
		})
	}
}

func TestArchHelpers(t *testing.T) {
	if got := nvrtc.ComputeArch(8, 0); got != "compute_80" {
		t.Fatalf("ComputeArch(8, 0) = %q, want compute_80", got)
	}
	if got := nvrtc.SmArch(9, 0); got != "sm_90" {
		t.Fatalf("SmArch(9, 0) = %q, want sm_90", got)
	}
}

func TestSupportedOptions(t *testing.T) {
	if len(nvrtc.SupportedOptions) == 0 {
		t.Fatal("SupportedOptions is empty")
	}
}
