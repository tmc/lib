package main

import (
	"bytes"
	"context"
	"crypto/sha256"
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
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed main.go capture.go workload.go kernels.cu bundle.README.md nccl_oracle.go nccl_workload.go profileshape.go pcsampling.go
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
	PCSampling   *pcSamplingResult  `json:"pc_sampling,omitempty"`
}

func capture(dir string, cfg config) (result *captureMetadata, err error) {
	// The trace is collected to a scratch file and folded into the capture
	// bundle below; WriteDir refuses to overwrite files already in dir.
	f, err := os.CreateTemp("", "ptx-sass-trace-*.trace")
	if err != nil {
		return nil, fmt.Errorf("create trace: %w", err)
	}
	tracePath := f.Name()
	defer os.Remove(tracePath)
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
	disassemblyInput := cubin
	if sanitized := sanitizeCUDA12CUBIN(cubin); sanitized != nil {
		disassemblyInput = sanitized
		captureFiles["kernels.cubin.original"] = cubin
		captureFiles["disassembly-note.txt"] = []byte("nvrtc CUDA 12 emitted nonstandard ELF identification/version bytes; kernels.cubin is a private repaired copy for offline readers, while kernels.cubin.original retains the original bytes\n")
	}
	assembly, err := d.Disassemble(toolCtx, disassemblyInput, cudasass.Options{
		PrintCode:   true,
		InlineLines: true,
		PTXLines:    true,
	})
	if err != nil {
		return nil, fmt.Errorf("disassemble cubin: %w", err)
	}

	captureFiles["kernels.ptx"] = []byte(ptx)
	captureFiles["kernels.cubin"] = disassemblyInput
	captureFiles["kernels.sass"] = rewriteSASSSourcePath(assembly.Stdout, "kernels.cu")
	captureFiles["nvdisasm.txt"] = version.Stdout

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
	discovered, err := discoverKernels(disassemblyInput)
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
	prof.SetMappingFile("kernels.cubin", fmt.Sprintf("%x", sha256.Sum256(disassemblyInput)))
	if err := prof.SetSampleTypes(sampleTypes(cfg.pcSampling)...); err != nil {
		return nil, fmt.Errorf("set sample types: %w", err)
	}
	// gpu_time is the type that always carries data; a consumer defaulting to
	// index 0 would otherwise render an empty profile.
	if err := prof.SetDefaultSampleType("gpu_time"); err != nil {
		return nil, fmt.Errorf("set default sample type: %w", err)
	}
	prof.SetPeriod("gpu_time", "nanoseconds", 1)
	prof.SetNumericLabels(numericLabelUnits())
	prof.SetDocURL("https://github.com/tmc/lib/tree/main/nvidia/examples/ptx-sass-trace")

	// The PTX text is a second logical image, so PTX frames carry real
	// addresses in the shipped file rather than only when cmd/cudaprof
	// synthesizes the mapping at view time.
	ptxLines := cudaprof.PTXLineOffsets(ptx)
	rec := newLaunchRecorder(prof, ptxLines)
	prof.AddMapping("kernels.ptx", fmt.Sprintf("%x", sha256.Sum256([]byte(ptx))), ptxBase, ptxBase+uint64(len(ptx)))
	var wRes workloadResult
	workloadStart := time.Now()
	runMeasured := func(ctx context.Context) {
		wRes, err = runWorkload(ctx, dev, mod, rec, code, cfg)
	}

	attribution := "whole launch at entry; no PC sampling"
	if cfg.pcSampling {
		attribution = "hardware PC samples attributed to SASS, PTX and source"
	}
	// These three are constant for the whole run, so they are comments rather
	// than labels: a label with one value across every sample splits nothing,
	// and pprof renders it on every tooltip.
	prof.AddComment("device: " + devName)
	prof.AddComment("measurement: CUDA event interval")
	prof.AddComment("attribution: " + attribution)
	if cfg.pcSampling {
		prof.AddComment("label-omissions: hardware PC sample: pass, moe.expert, moe.slot")
	}
	labels := cudaprof.Labels()

	if !cfg.pcSampling {
		cudaprof.Do(ctx, labels, runMeasured)
	} else {
		// The collector needs the context handle, not a bound context: CUPTI
		// takes it explicitly, and runWorkload manages its own thread affinity.
		cuCtx, ctxErr := dev.PrimaryContext()
		if ctxErr != nil {
			return nil, fmt.Errorf("retain primary context for pc sampling: %w", ctxErr)
		}
		defer func() { err = errors.Join(err, cuCtx.Release()) }()

		var sampErr error
		cudaprof.Do(ctx, labels, func(ctx context.Context) {
			meta.PCSampling, sampErr = withPCSampling(ctx, cupti.Context(cuCtx.Raw()), code, cubin, rec, uint32(cfg.pcPeriod), runMeasured)
		})
		// A sampling failure must not discard a workload that ran and verified,
		// so it is reported after the workload's own error.
		if err != nil {
			return nil, err
		}
		if sampErr != nil {
			return nil, fmt.Errorf("pc sampling: %w", sampErr)
		}
		if r := meta.PCSampling; r != nil {
			fmt.Printf("pc sampling: %d samples over %d PCs, %.1f%% resolved to PTX, %d dropped\n",
				r.TotalSamples, r.PCs, 100*r.Resolution(), r.DroppedSamples)
		}
	}
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

	prof.SetTimespan(workloadStart, time.Since(workloadStart))
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

	trace.Stop()
	if traceData, readErr := os.ReadFile(tracePath); readErr == nil && len(traceData) > 0 {
		captureFiles["go.trace"] = traceData
	}

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

// sanitizeCUDA12CUBIN returns a private copy for nvdisasm when CUDA 12 NVRTC
// emits the known nonstandard ELF identification/version bytes. The original
// CUBIN remains in the capture, so the artifact and this diagnostic repair are
// distinguishable. A nil result means the input already has standard bytes.
func sanitizeCUDA12CUBIN(cubin []byte) []byte {
	if len(cubin) < 24 || !bytes.Equal(cubin[:4], []byte{0x7f, 'E', 'L', 'F'}) || cubin[7] != 0x33 || cubin[8] != 0x07 || cubin[20] != 0x81 {
		return nil
	}
	sanitized := append([]byte(nil), cubin...)
	sanitized[7], sanitized[8], sanitized[20] = 0, 0, 1
	return sanitized
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
