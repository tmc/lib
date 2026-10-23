package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

//go:embed loadinline.py
var loadInlinePy []byte

// An inlineUnit is one torch.utils.cpp_extension.load_inline call.
type inlineUnit struct {
	Line              int      `json:"line"`
	Name              string   `json:"name"`
	CUDASources       []string `json:"cuda_sources"`
	ExtraCUDACFlags   []string `json:"extra_cuda_cflags"`
	NoImplicitHeaders bool     `json:"no_implicit_headers"`
	Error             string   `json:"error"`
}

// torchNVCCFlags are the flags load_inline adds to every nvcc command
// (COMMON_NVCC_FLAGS in torch/utils/cpp_extension.py).
var torchNVCCFlags = []string{
	"-D__CUDA_NO_HALF_OPERATORS__",
	"-D__CUDA_NO_HALF_CONVERSIONS__",
	"-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
	"-D__CUDA_NO_HALF2_OPERATORS__",
	"--expt-relaxed-constexpr",
}

// loadInlineUnits extracts the load_inline calls of a Python file, without
// importing it, using python.
func loadInlineUnits(ctx context.Context, python, path string) ([]inlineUnit, error) {
	cmd := exec.CommandContext(ctx, python, "-", path)
	cmd.Stdin = bytes.NewReader(loadInlinePy)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("read load_inline calls with %s: %v\n%s", python, err, stderr.Bytes())
	}
	var units []inlineUnit
	if err := json.Unmarshal(out, &units); err != nil {
		return nil, fmt.Errorf("read load_inline calls: %w", err)
	}
	if len(units) == 0 {
		return nil, errors.New("no load_inline call")
	}
	for _, u := range units {
		if u.Error != "" {
			return nil, fmt.Errorf("load_inline call at line %d: %s", u.Line, u.Error)
		}
	}
	return units, nil
}

var gencodeRE = regexp.MustCompile(`^arch=compute_(\d+[af]?),code=(?:sm_\d+[af]?|\[.*\])$`)

// unitConfig returns the configuration that compiles u as load_inline
// would: torch's flags and defines, the submission's extra_cuda_cflags in
// place of the default -flags (flags given on the command line follow
// them), its -std, and its architectures unless -arch was given.
func unitConfig(c *config, u inlineUnit, archSet, flagsSet bool) (*config, error) {
	cc := *c
	cc.defines = append(listFlag(nil), c.defines...)
	var archs, extra []string
	fl := u.ExtraCUDACFlags
	for i := 0; i < len(fl); i++ {
		f := fl[i]
		switch {
		case strings.HasPrefix(f, "-std="):
			cc.std = strings.TrimPrefix(f, "-std=")
		case f == "-gencode" && i+1 < len(fl), strings.HasPrefix(f, "-gencode="), strings.HasPrefix(f, "--generate-code="):
			var v string
			if f == "-gencode" {
				i++
				v = fl[i]
			} else {
				_, v, _ = strings.Cut(f, "=")
			}
			m := gencodeRE.FindStringSubmatch(v)
			if m == nil {
				return nil, fmt.Errorf("load_inline call at line %d: cannot read %s", u.Line, f)
			}
			archs = append(archs, "sm_"+m[1])
		case strings.HasPrefix(f, "-arch=sm_"):
			archs = append(archs, strings.TrimPrefix(f, "-arch="))
		default:
			extra = append(extra, f)
		}
	}
	if !archSet {
		if len(archs) == 0 {
			// load_inline would compile for the visible GPU; there is none here.
			return nil, fmt.Errorf("load_inline call at line %d: no -gencode or -arch in extra_cuda_cflags; give -arch", u.Line)
		}
		cc.archs = archs
	}
	cc.extra = append(append([]string(nil), torchNVCCFlags...), extra...)
	if flagsSet {
		cc.extra = append(cc.extra, c.extra...)
	}
	cc.defines = append(cc.defines, "TORCH_EXTENSION_NAME="+u.Name, "TORCH_API_INCLUDE_EXTENSION_H")
	return &cc, nil
}

// torchIncludes returns the -isystem flags for torch's headers under the
// site-packages directory site on the compiling host.
func torchIncludes(site string) []string {
	return []string{
		"-isystem", site + "/torch/include",
		"-isystem", site + "/torch/include/torch/csrc/api/include",
	}
}

// writeUnit writes u's CUDA translation unit into dir as load_inline
// would assemble it, and returns its path.
func writeUnit(dir, stem string, u inlineUnit) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	srcs := u.CUDASources
	if !u.NoImplicitHeaders {
		srcs = append([]string{"#include <torch/types.h>", "#include <cuda.h>", "#include <cuda_runtime.h>"}, srcs...)
	}
	path := filepath.Join(dir, stem+".cu")
	return path, os.WriteFile(path, []byte(strings.Join(srcs, "\n")), 0o644)
}

// cudaCMakeMacros stands in for c10/cuda/impl/cuda_cmake_macros.h, which a
// CPU-only torch wheel does not ship but torch/types.h reaches when
// compiled by nvcc.
const cudaCMakeMacros = "#pragma once\n#define C10_CUDA_BUILD_SHARED_LIBS\n"
