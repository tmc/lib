package main

import (
	"bytes"
	"context"
	"debug/elf"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/trace"
	"slices"
	"strings"
	"time"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed main.go capture.go workload.go kernels.cu bundle.README.md nccl_oracle.go nccl_workload.go
var sources embed.FS

type captureMetadata struct {
	Created      string             `json:"created"`
	Device       string             `json:"device"`
	Arch         string             `json:"architecture"`
	GoVersion    string             `json:"go_version"`
	NVRTCVersion string             `json:"nvrtc_version"`
	Elements     int                `json:"elements"`
	Passes       int                `json:"passes"`
	Rounds       int                `json:"fma_rounds"`
	Intervals    map[string][]int64 `json:"cuda_event_nanoseconds"`
	Verified     int                `json:"verified_results"`
	MaxError     float64            `json:"max_absolute_error"`
	NCCL         *ncclResult        `json:"nccl,omitempty"`
}

func capture(dir string, cfg config) (result *captureMetadata, err error) {
	f, err := os.Create(filepath.Join(dir, "go.trace"))
	if err != nil {
		return nil, fmt.Errorf("create trace: %w", err)
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	if err := trace.Start(f); err != nil {
		return nil, fmt.Errorf("start trace: %w", err)
	}
	defer trace.Stop()
	ctx, task := trace.NewTask(context.Background(), "CUDA pipeline")
	defer task.End()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return nil, fmt.Errorf("open device: %w", err)
	}
	defer func() { err = errors.Join(err, dev.Close()) }()
	devName, err := dev.Name()
	if err != nil {
		return nil, err
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, err
	}
	arch := fmt.Sprintf("sm_%d%d", major, minor)
	vm, vn, err := nvrtc.Version()
	if err != nil {
		return nil, err
	}

	meta := &captureMetadata{
		Created:      time.Now().UTC().Format(time.RFC3339),
		Device:       devName,
		Arch:         arch,
		GoVersion:    runtime.Version(),
		NVRTCVersion: fmt.Sprintf("%d.%d", vm, vn),
		Elements:     cfg.elements,
		Passes:       cfg.passes,
		Rounds:       cfg.rounds,
		Intervals:    make(map[string][]int64),
	}
	fmt.Printf("device: %s (%s)\n", devName, arch)
	trace.Log(ctx, "cuda.device", devName+" "+arch)

	captureFiles := make(map[string][]byte)

	entries, err := sources.ReadDir(".")
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		data, err := sources.ReadFile(entry.Name())
		if err != nil {
			return nil, err
		}
		name := entry.Name()
		if name == "bundle.README.md" {
			name = "README.md"
		}
		captureFiles[name] = data
	}

	src := captureFiles["kernels.cu"]
	region := trace.StartRegion(ctx, "NVRTC compile")
	compiled, err := nvrtcx.Compile(string(src), nvrtcx.WithName("kernels.cu"), nvrtcx.WithFlag("--gpu-architecture="+arch), nvrtcx.WithFlag("--generate-line-info"), nvrtcx.WithNoCache())
	region.End()
	if err != nil {
		return nil, fmt.Errorf("compile kernels: %w", err)
	}
	ptx, cubin := compiled.PTX(), compiled.CUBIN()
	if ptx == "" || len(cubin) == 0 {
		return nil, fmt.Errorf("compiler must produce PTX and CUBIN")
	}

	d := cudasass.Disassembler{Path: cfg.nvdisasm, TempDir: dir}
	toolCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	version, err := d.Version(toolCtx)
	if err != nil {
		return nil, fmt.Errorf("nvdisasm version: %w", err)
	}
	assembly, err := d.Disassemble(toolCtx, cubin, cudasass.Options{SourceLines: true})
	if err != nil {
		return nil, fmt.Errorf("disassemble cubin: %w", err)
	}

	captureFiles["kernels.ptx"] = []byte(ptx)
	captureFiles["kernels.cubin"] = cubin
	captureFiles["kernels.sass"] = assembly.Stdout
	captureFiles["nvdisasm.txt"] = version.Stdout

	traceData, err := os.ReadFile(filepath.Join(dir, "go.trace"))
	if err == nil && len(traceData) > 0 {
		captureFiles["go.trace"] = traceData
	}

	cap := &cudacapture.Capture{
		Files: captureFiles,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		return nil, fmt.Errorf("parse code: %w", err)
	}
	virtual, err := cudaptx.ParseString(ptx)
	if err != nil {
		return nil, fmt.Errorf("parse virtual ptx: %w", err)
	}
	discovered, err := discoverKernels(cubin)
	if err != nil {
		return nil, fmt.Errorf("discover kernels from cubin: %w", err)
	}
	for _, name := range discovered {
		k := code.Kernel(name)
		if k == nil {
			continue
		}
		pk := virtual.Kernel(name)
		if pk != nil && len(pk.Instructions) > 0 {
			k.PTXLine = pk.Instructions[0].Line
			for _, inst := range pk.Instructions {
				if isArithmeticOpcode(inst.Opcode) {
					k.PTXLine = inst.Line
					break
				}
			}
		}
	}

	logLines(ctx, "CUDA", string(src))
	logLines(ctx, "PTX", ptx)
	logLines(ctx, "SASS", string(assembly.Stdout))

	mod, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		return nil, fmt.Errorf("load cubin: %w", err)
	}
	defer func() { err = errors.Join(err, mod.Close()) }()

	var profBuf bytes.Buffer
	prof := cudaprof.NewSession(&profBuf)
	prof.SetMappingFile("kernels.cubin", "")
	if err := prof.SetSampleTypes("samples", "stall_mem", "stall_wait", "stall_scoreboard", "gpu_time"); err != nil {
		return nil, fmt.Errorf("set sample types: %w", err)
	}
	var wRes workloadResult
	cudaprof.Do(ctx, cudaprof.Labels("device", devName, "measurement", "CUDA event interval", "attribution", "whole launch at entry; no PC sampling"), func(ctx context.Context) {
		wRes, err = runWorkload(ctx, dev, mod, prof, code, cfg)
	})
	if err != nil {
		return nil, err
	}

	if cfg.ncclMode != "off" && len(cfg.devices) >= 2 {
		ncclRes, ncclErr := runNCCLWorkload(ctx, cfg.devices, cfg.ncclMode, cfg.elements, cfg.passes, prof)
		if ncclErr != nil {
			return nil, fmt.Errorf("nccl workload: %w", ncclErr)
		}
		meta.NCCL = ncclRes
	}

	if err := prof.Close(); err != nil {
		return nil, fmt.Errorf("finalize profile: %w", err)
	}

	meta.Intervals = wRes.Intervals
	meta.Verified = wRes.Verified
	meta.MaxError = wRes.MaxError

	parsedProf, err := profile.ParseData(profBuf.Bytes())
	if err != nil {
		return nil, fmt.Errorf("parse generated profile: %w", err)
	}
	cap.Profile = parsedProf

	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return nil, fmt.Errorf("marshal metadata: %w", err)
	}
	cap.Metadata = metaBytes

	// Write unpacked directory capture
	if err := cudacapture.WriteDir(dir, cap); err != nil {
		return nil, fmt.Errorf("write capture dir: %w", err)
	}

	// Write single self-contained capture.pprof
	captureFile := filepath.Join(dir, "capture.pprof")
	fCap, err := os.Create(captureFile)
	if err != nil {
		return nil, fmt.Errorf("create capture.pprof: %w", err)
	}
	err = cudacapture.Write(fCap, cap)
	err = errors.Join(err, fCap.Close())
	if err != nil {
		os.Remove(captureFile)
		return nil, fmt.Errorf("write capture.pprof: %w", err)
	}

	return meta, nil
}

func logLines(ctx context.Context, category, text string) {
	for i, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) != "" {
			trace.Logf(ctx, category, "%d: %s", i+1, line)
		}
	}
}

func discoverKernels(cubin []byte) ([]string, error) {
	ef, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return nil, fmt.Errorf("read cubin elf: %w", err)
	}
	defer ef.Close()

	var names []string
	seen := make(map[string]bool)

	// Discover function entry points from ELF symbol table
	syms, _ := ef.Symbols()
	for _, s := range syms {
		if elf.ST_TYPE(s.Info) == elf.STT_FUNC && s.Name != "" && !seen[s.Name] {
			seen[s.Name] = true
			names = append(names, s.Name)
		}
	}

	// Also discover from .text.<kernel_name> section headers
	for _, sec := range ef.Sections {
		if strings.HasPrefix(sec.Name, ".text.") {
			kname := strings.TrimPrefix(sec.Name, ".text.")
			if kname != "" && !seen[kname] {
				seen[kname] = true
				names = append(names, kname)
			}
		}
	}
	slices.Sort(names)
	return names, nil
}

func isArithmeticOpcode(op string) bool {
	lower := strings.ToLower(op)
	for _, prefix := range []string{
		"add", "sub", "mul", "fma", "mad", "dp4a", "shfl", "rsqrt", "div", "cvt", "ld", "st",
	} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}
