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
	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed main.go capture.go workload.go kernels.cu bundle.README.md nccl_oracle.go nccl_workload.go profileshape.go pcsampling.go activity.go kernels.tileir tileiras-remarks.yaml intermediate_ptx.go hotspots.go graph_workload.go cupti_metrics.go
var sources embed.FS

// artifacts names the files this capture bundles its retained code under. The
// capture's declared CodeFiles and the frames the profile carries must agree on
// them, so both are written from here: a frame naming a file the bundle does
// not hold resolves to nothing in every source view.
var artifacts = cudacapture.Artifacts{
	Source: "kernels.cu",
	PTX:    "kernels.ptx",
	SASS:   "kernels.sass",
}

// tileIRArtifact names the bundled tile IR. It is this workload's own level,
// not one cudacapture identifies: the line numbers come from the lowering
// remarks this example generates, so the frames for it are built here.
const tileIRArtifact = "kernels.tileir"

// remarksArtifact names the tile compiler's optimization remarks for the
// kernels in tileIRArtifact.
const remarksArtifact = "tileiras-remarks.yaml"

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
	Hotspots     bool               `json:"instruction_hotspots,omitempty"`
	// Activity is the CUPTI activity tier's account of the same workload: how
	// many kernels the device ran and for how long. The event intervals above
	// are per launch and only where this example recorded one, and a PC sample
	// carries no duration at all, so this is the only field here that can say
	// what fraction of the span no kernel was executing.
	Activity     *activityResult `json:"cupti_activity,omitempty"`
	Divergences  int             `json:"ptx_sass_divergences"`
	PaddingNOPs  int             `json:"sass_padding_nops"`
	InteriorNOPs int             `json:"sass_interior_nops"`
	// SourceFusion is the source-fusion failure this capture was written
	// through, empty on a capture whose PTX and SASS bound to one compilation.
	// It is here as well as in the bundled note because the absence of source
	// attribution is otherwise indistinguishable, in the profile alone, from a
	// compiler that located nothing.
	SourceFusion string             `json:"source_fusion_failure,omitempty"`
	Graph        *graphResult       `json:"cuda_graph,omitempty"`
	SassMetrics  *sassMetricsResult `json:"sass_metrics,omitempty"`
	PMSampling   *pmSamplingResult  `json:"pm_sampling,omitempty"`
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

	cfgOut, err := d.Disassemble(toolCtx, disassemblyInput, cudasass.Options{
		Mode:                       cudasass.ControlFlow,
		PrintInstructionOffsetsCFG: true,
	})
	if err != nil {
		return nil, fmt.Errorf("generate control flow graph: %w", err)
	}
	bbcfgOut, err := d.Disassemble(toolCtx, disassemblyInput, cudasass.Options{
		Mode:                       cudasass.BasicBlockFlow,
		PrintInstructionOffsetsCFG: true,
	})
	if err != nil {
		return nil, fmt.Errorf("generate basic block flow graph: %w", err)
	}

	tileIRData := captureFiles[tileIRArtifact]
	ptx = injectTileIRLocations(ptx, string(tileIRData))

	captureFiles[artifacts.PTX] = []byte(ptx)
	captureFiles["kernels.cubin"] = disassemblyInput
	// The tile IR says what the kernels are; the remarks say what the tile
	// compiler did to them. Shipping the raw YAML leaves a reader to join the
	// two by hand, so the join is done here and fails the capture if a remark
	// names a kernel the bundled tile IR does not define.
	remarkSummary, err := tileIRRemarkSummary(string(captureFiles[remarksArtifact]), string(tileIRData))
	if err != nil {
		return nil, err
	}
	captureFiles["tileiras-remarks.txt"] = []byte(remarkSummary)
	// nvrtc records the path it compiled through, which is a temporary
	// directory this process does not choose, so the old path is read back out
	// of the disassembly rather than assumed. Only kernels.cu is bundled, so
	// every other file nvdisasm names keeps its recorded path.
	sourceMap, err := cudacapture.SourceMapByBase(assembly.Stdout, artifacts.Source)
	if err != nil {
		return nil, fmt.Errorf("map disassembly sources: %w", err)
	}
	captureFiles[artifacts.SASS] = cudacapture.RewriteSASSSources(assembly.Stdout, sourceMap)
	captureFiles["kernels.cfg.dot"] = cfgOut.Stdout
	captureFiles["kernels.bbcfg.dot"] = bbcfgOut.Stdout
	captureFiles["nvdisasm.txt"] = version.Stdout

	cap := &cudacapture.Capture{
		Files: captureFiles,
		Code: cudacapture.CodeFiles{
			Source: artifacts.Source,
			PTX:    artifacts.PTX,
			CUBIN:  "kernels.cubin",
			SASS:   artifacts.SASS,
			CFG:    "kernels.cfg.dot",
			BBCFG:  "kernels.bbcfg.dot",
			TileIR: tileIRArtifact,
		},
	}

	code, fusion, err := loadCaptureCode(cap)
	if err != nil {
		return nil, err
	}
	if fusion != nil {
		// A source-fusion failure invalidates the source and PTX attribution
		// and nothing else: kernel names, CUBIN offsets and sizes, and the
		// disassembly text do not depend on the two line tables agreeing.
		// Discarding the capture over it threw away the profile, the trace and
		// every artifact written below, none of which the failure touches.
		// The note travels in the bundle beside lowering.txt because a reader
		// of the capture never sees this process's stderr.
		fmt.Fprintf(os.Stderr, "ptx-sass-trace: %v\n", fusion)
		fmt.Fprintf(os.Stderr, "ptx-sass-trace: writing the capture without source attribution; see %s in the bundle\n", sourceFusionArtifact)
		captureFiles[sourceFusionArtifact] = []byte(sourceFusionNote(fusion))
		meta.SourceFusion = fusion.Error()
	}
	virtual, err := cudaptx.ParseString(ptx)
	if err != nil {
		return nil, fmt.Errorf("parse virtual ptx: %w", err)
	}
	parsedSASS, err := cudasass.Parse(bytes.NewReader(assembly.Stdout))
	if err != nil {
		return nil, fmt.Errorf("parse sass: %w", err)
	}

	// The PTX says what the compiler was asked for and the SASS says what the
	// hardware was given; shipping both without comparing them leaves the
	// reader to do it by eye. Compare names the places ptxas did something the
	// PTX does not predict: work erased, instructions introduced, registers
	// spilled, and the NOP padding that displaces any instruction mix a
	// prediction was made from.
	lowering, err := cudalowering.Compare(virtual, parsedSASS, nil)
	if err != nil {
		return nil, fmt.Errorf("compare ptx to sass: %w", err)
	}
	//
	// A zero here is reported with the reason it is zero. Every divergence kind
	// has a structural precondition -- Padding needs interior NOPs, Spill needs
	// local memory -- and on a capture that meets none of them, "0 divergences"
	// is produced by the same code path as a Compare that silently stopped
	// working. The preconditions this input actually meets are written into the
	// file, so a reader can tell a clean lowering from an inert check.
	captureFiles["lowering.txt"] = []byte(lowering.String() + loweringCoverage(lowering))
	meta.Divergences = len(lowering.Divergences)
	for _, st := range lowering.Stats {
		meta.PaddingNOPs += st.NOPs
		meta.InteriorNOPs += st.InteriorNOPs
	}
	discovered, err := discoverKernels(disassemblyInput)
	if err != nil {
		return nil, fmt.Errorf("discover kernels from cubin: %w", err)
	}
	alignKernelPTXLines(code, virtual, discovered)

	logLines(ctx, "CUDA", string(src))
	logLines(ctx, "PTX", ptx)
	logLines(ctx, "SASS", string(assembly.Stdout))
	logLines(ctx, "CFG", string(cfgOut.Stdout))

	mod, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		return nil, fmt.Errorf("load cubin: %w", err)
	}
	defer func() { err = errors.Join(err, mod.Close()) }()

	var profBuf bytes.Buffer
	prof := cudaprof.NewSession(&profBuf)
	prof.SetMappingFile("kernels.cubin", fmt.Sprintf("%x", sha256.Sum256(disassemblyInput)))
	if err := prof.SetSampleTypes(sampleTypesWithHotspots(cfg.pcSampling, cfg.hotspots)...); err != nil {
		return nil, fmt.Errorf("set sample types: %w", err)
	}
	// The view a consumer opens, which is the number most readers will ever
	// see. gpu_time is measured in every mode and always carries data, so it is
	// the default -- except under -hotspots, where the per-instruction rows the
	// user asked for carry only modeled_cost and defaulting to gpu_time would
	// open on the launch rows instead. modeled_cost is named for what it is, so
	// opening on it cannot be mistaken for a measurement.
	defaultType := "gpu_time"
	if cfg.hotspots {
		defaultType = "modeled_cost"
	}
	if err := prof.SetDefaultSampleType(defaultType); err != nil {
		return nil, fmt.Errorf("set default sample type: %w", err)
	}
	prof.SetPeriod("gpu_time", "nanoseconds", 1)
	prof.SetNumericLabels(numericLabelUnits())
	prof.SetDocURL("https://github.com/tmc/lib/tree/main/nvidia/x/examples/ptx-sass-trace")

	// The PTX text is a second logical image, so PTX frames carry real
	// addresses in the shipped file rather than only when cmd/cudaprof
	// synthesizes the mapping at view time.
	ptxAddrs := cudacapture.NewPTXAddresses(ptx)
	rec := newLaunchRecorder(prof, ptxAddrs)
	rec.SetHotspots(cfg.hotspots)
	rec.SetSourceAttribution(fusion == nil)
	rec.SetTileIRLines(kernelTileIRLines(string(tileIRData)))
	prof.AddMapping(artifacts.PTX, fmt.Sprintf("%x", sha256.Sum256([]byte(ptx))), cudaprof.PTXMappingStart, cudaprof.PTXMappingStart+uint64(len(ptx)))
	var wRes workloadResult
	workloadStart := time.Now()
	runMeasured := func(ctx context.Context) {
		wRes, err = runWorkload(ctx, dev, mod, rec, code, cfg)
	}

	// Activity tracing brackets the workload itself rather than the profiling
	// path around it, so the span it reports is the interval its own records
	// come from. It is innermost for the same reason: whichever sampler runs
	// outside it, the span stays the workload's.
	var actErr error
	if cfg.activity {
		measured := runMeasured
		runMeasured = func(ctx context.Context) {
			meta.Activity, actErr = withActivity(ctx, measured)
		}
	}

	// This comment is the file's own account of where its numbers came from,
	// and it travels with the capture. The two modes are rejected together in
	// main, so exactly one of these is true of any profile.
	attribution := "whole launch at entry; no PC sampling"
	if cfg.pcSampling {
		attribution = "hardware PC samples attributed to SASS, PTX and source"
	} else if cfg.hotspots {
		attribution = "modeled_cost from a static opcode table across SASS, PTX and TileIR; no value in this profile was sampled"
	}
	// Each sentence above names source or PTX attribution the capture no longer
	// has, so the comment would be false where it is least checkable.
	if fusion != nil {
		attribution += "; SASS only: the PTX and SASS of this capture are not from one compilation, so no instruction carries a source or PTX line (see " + sourceFusionArtifact + ")"
	}
	// These three are constant for the whole run, so they are comments rather
	// than labels: a label with one value across every sample splits nothing,
	// and pprof renders it on every tooltip.
	prof.AddComment("device: " + devName)
	prof.AddComment("measurement: CUDA event interval")
	prof.AddComment("attribution: " + attribution)
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

	// A tier that lost records must not leave a bundle behind for someone to
	// analyse: a capture that reports a transport failure and writes a profile
	// anyway is read as a profile. The counters are named in the error because
	// there is no file to read them out of once the capture is refused.
	if meta.Activity != nil {
		// The profile carries the statement too. cmd/cudaprof computes its
		// coverage denominator from the profile's own gpu_time samples, which
		// are CUDA event intervals; this comment is what tells a reader of that
		// line that a second, device-side account of the same span exists and
		// what it says. It does not change the denominator.
		prof.AddComment("cupti-activity: " + meta.Activity.String())
		data, jsonErr := activityJSON(meta.Activity)
		if jsonErr != nil {
			return nil, jsonErr
		}
		captureFiles["activity.json"] = data
	}
	if actErr != nil {
		return nil, fmt.Errorf("cupti activity tracing: %w", actErr)
	}

	if cfg.ncclMode != "off" && len(cfg.devices) >= 2 {
		ncclRes, ncclErr := runNCCLWorkload(ctx, cfg.devices, cfg.ncclMode, cfg.elements, cfg.passes, prof)
		if ncclErr != nil {
			return nil, fmt.Errorf("nccl workload: %w", ncclErr)
		}
		meta.NCCL = ncclRes
	}

	if cfg.sassMetrics {
		sRes, sData, sErr := collectSassMetrics(dev, parsedSASS)
		if sErr != nil {
			return nil, fmt.Errorf("collect sass metrics: %w", sErr)
		}
		meta.SassMetrics = sRes
		captureFiles["sass_metrics.json"] = sData
	}
	if cfg.pmSampling {
		pmRes, pmData, pmErr := collectPMSampling(dev)
		if pmErr != nil {
			return nil, fmt.Errorf("collect pm sampling: %w", pmErr)
		}
		meta.PMSampling = pmRes
		captureFiles["pm_sampling.json"] = pmData
	}

	// Written after the workload rather than before it: the keys a PC sample
	// cannot carry are whichever ones varied between a kernel's launches, which
	// is a property of the workload and not a list to maintain by hand. A stale
	// list is worse than none, because a label's absence from PC samples is
	// invisible in every pprof surface at once.
	if cfg.pcSampling {
		if omitted := rec.omittedLabels(); len(omitted) > 0 {
			prof.AddComment("label-omissions: hardware PC sample: " + strings.Join(omitted, ", "))
		}
	}

	prof.SetTimespan(workloadStart, time.Since(workloadStart))
	if err := prof.Close(); err != nil {
		return nil, fmt.Errorf("finalize profile: %w", err)
	}

	meta.Intervals = wRes.Intervals
	meta.Verified = wRes.Verified
	meta.MaxError = wRes.MaxError
	meta.Hotspots = cfg.hotspots
	meta.Graph = wRes.Graph

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

// sourceFusionArtifact names the bundled note describing a source-fusion
// failure. It exists only in a capture written through one.
const sourceFusionArtifact = "source-fusion.txt"

// loadCaptureCode parses the capture's code artifacts, separating a
// source-fusion failure from a fatal one. A nonnil fusion comes with a usable
// Code: LoadCode returns the tiers the failure does not invalidate, and the
// caller writes the capture without source attribution rather than not at all.
// Every other error is fatal, because nothing is known about which tiers
// survived it.
func loadCaptureCode(cap *cudacapture.Capture) (*cudacapture.Code, *cudacapture.SourceFusionError, error) {
	code, err := cudacapture.LoadCode(cap)
	if err == nil {
		return code, nil, nil
	}
	var fusion *cudacapture.SourceFusionError
	if errors.As(err, &fusion) && code != nil {
		return code, fusion, nil
	}
	return nil, nil, fmt.Errorf("parse code: %w", err)
}

// sourceFusionNote renders the bundled account of a source-fusion failure. The
// reader of a capture sees the profile and the files, never the run that
// produced them, so what the capture does not contain has to be written down in
// it: a profile whose instructions carry no source line looks exactly like one
// compiled without line information.
func sourceFusionNote(e *cudacapture.SourceFusionError) string {
	var b strings.Builder
	b.WriteString("source attribution unavailable\n\n")
	fmt.Fprintf(&b, "\t%v\n\n", e)
	fmt.Fprintf(&b, `The PTX and the SASS bundled here disagree about the line table of %s, so
no instruction in this capture can be bound to a source statement. Every
instruction therefore carries no source file, no source line, no PTX line and no
inline chain: the binding was refused, not attempted and lost.

The tiers that do not depend on the two agreeing are intact and were written as
usual -- kernel names, CUBIN offsets and sizes, the disassembly text, the control
flow graphs, and the measured launch intervals -- so a kernel-level ranking and
the SASS views of this capture are what they are in any other.

Per-instruction rows, which -hotspots writes, carry their SASS frame alone for
the same reason. The kernel's own source and PTX lines are deliberately not
substituted for the missing ones: putting every instruction of a kernel on one
line renders a refused binding as an attributed profile.
`, e.Source)
	return b.String()
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

// alignKernelPTXLines sets each kernel's PTXLine to the line its entry is
// declared on in the PTX this capture ships.
//
// It used to be set to the line of the first instruction whose opcode "looked
// arithmetic" by prefix -- a line chosen by opcode spelling rather than by the
// PTX's own structure, and one that moves when a prologue instruction is added
// or an opcode is renamed. x/cudacapture removed that guess in favour of the
// parser's StartLine; this restated it on every capture the example wrote.
func alignKernelPTXLines(code *cudacapture.Code, virtual *cudaptx.Module, names []string) {
	for _, name := range names {
		k := code.Kernel(name)
		if k == nil {
			continue
		}
		// A kernel the CUBIN declares but the PTX does not is left alone
		// rather than zeroed: no line is better than a line from elsewhere.
		if pk := virtual.Kernel(name); pk != nil && pk.StartLine > 0 {
			k.PTXLine = pk.StartLine
		}
	}
}

// loweringCoverage reports which divergence kinds the compared input could have
// produced. Compare reporting nothing is the same output whether the lowering
// was clean or the comparison was inert, and the kinds are not symmetric: a
// kernel whose NOPs are all trailing cannot report Padding however broken or
// correct Compare is. Stating the preconditions separates the two readings.
func loweringCoverage(res *cudalowering.Report) string {
	var kernels, interior, nops int
	for _, st := range res.Stats {
		kernels++
		nops += st.NOPs
		interior += st.InteriorNOPs
	}
	var b strings.Builder
	b.WriteString("\ncoverage\n")
	fmt.Fprintf(&b, "\tkernels compared: %d\n", kernels)
	fmt.Fprintf(&b, "\tNOPs: %d, of which interior: %d\n", nops, interior)
	if len(res.Divergences) > 0 {
		return b.String()
	}
	b.WriteString("\tno divergences reported.")
	if interior == 0 {
		b.WriteString(" Every NOP in this capture is trailing alignment\n" +
			"\tpadding, so Padding cannot fire on this input regardless of the\n" +
			"\tcomparison's health; the zero above is not evidence that the check\n" +
			"\truns. The check is exercised against a kernel that does pad interior\n" +
			"\tby TestLoweringDetectsPadding.\n")
	} else {
		b.WriteString(" Interior NOPs are present, so Padding was\n\treachable and did not fire.\n")
	}
	return b.String()
}
