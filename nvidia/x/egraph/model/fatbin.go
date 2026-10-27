package model

import (
	"fmt"
	"strings"
)

// ArchFamily identifies a Hopper or Blackwell hardware capability family.
type ArchFamily string

const (
	// FamilyHopperSM90 is Hopper H100/H200 (sm_90a: WGMMA + TMA + DSMEM clusters).
	FamilyHopperSM90 ArchFamily = "hopper-sm90"

	// FamilyBlackwellDC_SM100 is datacenter Blackwell B200/GB200 (sm_100a / sm_100f:
	// tcgen05 cta_group::1/2 + TMEM + CLC + scale_vec::2X).
	FamilyBlackwellDC_SM100 ArchFamily = "blackwell-dc-sm100"

	// FamilyBlackwellUltra_SM103 is Blackwell Ultra B300/GB300 (sm_103a / sm_103f:
	// tcgen05 scale_vec::4X K=128 + 2x SFU + 288 GiB 12-Hi HBM3e + ConnectX-8 800G).
	FamilyBlackwellUltra_SM103 ArchFamily = "blackwell-ultra-sm103"

	// FamilyBlackwellEdge_SM120 is workstation/consumer/Spark Blackwell RTX PRO 6000,
	// RTX 5090, and DGX Spark GB10 (sm_120a / sm_120f / sm_121a: 5th-gen FP4 Tensor
	// Cores via warp-level mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
	// + CLC + TMA, without TMEM or tcgen05).
	FamilyBlackwellEdge_SM120 ArchFamily = "blackwell-edge-sm120"
)

// Additional TargetArch constants for multi-architecture fatbinary compilation.
const (
	ArchBlackwellSM100F TargetArch = "sm_100f"
	ArchBlackwellSM103  TargetArch = "sm_103a"
	ArchBlackwellSM103F TargetArch = "sm_103f"
	ArchBlackwellSM120  TargetArch = "sm_120a"
	ArchBlackwellSM120F TargetArch = "sm_120f"
)

// ArchTargetSpec describes the compute capability, nvcc target names, and
// hardware instruction features of a Hopper or Blackwell target architecture.
type ArchTargetSpec struct {
	ComputeMajor          int
	ComputeMinor          int
	Target                string
	FamilyTarget          string
	Family                ArchFamily
	HasWGMMA              bool
	HasTCGen05            bool
	HasTMEM               bool
	HasCLC                bool
	HasTMA                bool
	HasScaleVec4X         bool
	HasWarpBlockScaledMMA bool
	MMAInstruction        string
}

// ClassifyComputeCapability returns the [ArchTargetSpec] for a GPU compute
// capability (major, minor). Supported capabilities are (9,0), (10,0), (10,3),
// (12,0), and (12,1).
func ClassifyComputeCapability(major, minor int) (ArchTargetSpec, error) {
	switch {
	case major == 9 && minor == 0:
		return ArchTargetSpec{
			ComputeMajor:   9,
			ComputeMinor:   0,
			Target:         "sm_90a",
			FamilyTarget:   "sm_90a",
			Family:         FamilyHopperSM90,
			HasWGMMA:       true,
			HasTMA:         true,
			MMAInstruction: "wgmma.mma_async",
		}, nil
	case major == 10 && minor == 0:
		return ArchTargetSpec{
			ComputeMajor:   10,
			ComputeMinor:   0,
			Target:         "sm_100a",
			FamilyTarget:   "sm_100f",
			Family:         FamilyBlackwellDC_SM100,
			HasTCGen05:     true,
			HasTMEM:        true,
			HasCLC:         true,
			HasTMA:         true,
			MMAInstruction: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
		}, nil
	case major == 10 && minor == 3:
		return ArchTargetSpec{
			ComputeMajor:   10,
			ComputeMinor:   3,
			Target:         "sm_103a",
			FamilyTarget:   "sm_103f",
			Family:         FamilyBlackwellUltra_SM103,
			HasTCGen05:     true,
			HasTMEM:        true,
			HasCLC:         true,
			HasTMA:         true,
			HasScaleVec4X:  true,
			MMAInstruction: "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
		}, nil
	case major == 12 && minor == 0:
		return ArchTargetSpec{
			ComputeMajor:          12,
			ComputeMinor:          0,
			Target:                "sm_120a",
			FamilyTarget:          "sm_120f",
			Family:                FamilyBlackwellEdge_SM120,
			HasCLC:                true,
			HasTMA:                true,
			HasWarpBlockScaledMMA: true,
			MMAInstruction:        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
		}, nil
	case major == 12 && minor == 1:
		return ArchTargetSpec{
			ComputeMajor:          12,
			ComputeMinor:          1,
			Target:                "sm_121a",
			FamilyTarget:          "sm_120f",
			Family:                FamilyBlackwellEdge_SM120,
			HasCLC:                true,
			HasTMA:                true,
			HasWarpBlockScaledMMA: true,
			MMAInstruction:        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
		}, nil
	default:
		return ArchTargetSpec{}, fmt.Errorf("classify compute capability: unsupported compute capability %d.%d", major, minor)
	}
}

// LookupArchTarget resolves an nvcc target string ("sm_90a", "sm_100a",
// "sm_100f", "sm_103a", "sm_103f", "sm_120a", "sm_120f", "sm_121a") to its
// [ArchTargetSpec].
func LookupArchTarget(target string) (ArchTargetSpec, error) {
	t := strings.ToLower(strings.TrimSpace(target))
	switch t {
	case "sm_90a", "sm_90":
		return ClassifyComputeCapability(9, 0)
	case "sm_100a", "sm_100":
		return ClassifyComputeCapability(10, 0)
	case "sm_100f":
		spec, _ := ClassifyComputeCapability(10, 0)
		spec.Target = "sm_100f"
		return spec, nil
	case "sm_103a", "sm_103":
		return ClassifyComputeCapability(10, 3)
	case "sm_103f":
		spec, _ := ClassifyComputeCapability(10, 3)
		spec.Target = "sm_103f"
		return spec, nil
	case "sm_120a", "sm_120":
		return ClassifyComputeCapability(12, 0)
	case "sm_120f":
		spec, _ := ClassifyComputeCapability(12, 0)
		spec.Target = "sm_120f"
		return spec, nil
	case "sm_121a", "sm_121":
		return ClassifyComputeCapability(12, 1)
	default:
		return ArchTargetSpec{}, fmt.Errorf("lookup arch target: unsupported target %q", target)
	}
}

// MultiArchTargetUnit holds the emitted kernel unit and roofline latency
// estimate for a single target architecture in a [MultiArchBundle].
type MultiArchTargetUnit struct {
	Spec               ArchTargetSpec
	EntrySymbol        string
	CUDASource         string
	EstimatedLatencyUs float64
}

// MultiArchBundle packages per-target specialized CUDA kernel units and a
// unified C++ runtime compute-capability dispatcher for one decoder layer.
type MultiArchBundle struct {
	LayerName      string
	Config         ArchConfig
	Units          []MultiArchTargetUnit
	DispatcherCUDA string
}

func gencodePairForTarget(target string) []string {
	suffix := strings.TrimPrefix(target, "sm_")
	return []string{"-gencode", fmt.Sprintf("arch=compute_%s,code=%s", suffix, target)}
}

// NVCCFlags returns the ordered -gencode arch=compute_...,code=sm_... flag
// pairs for compiling all target units and family-conditional targets in b.
func (b *MultiArchBundle) NVCCFlags() [][]string {
	if b == nil {
		return nil
	}
	var flags [][]string
	seen := make(map[string]bool)
	addTarget := func(t string) {
		if t == "" || seen[t] {
			return
		}
		seen[t] = true
		flags = append(flags, gencodePairForTarget(t))
	}
	for _, u := range b.Units {
		addTarget(u.Spec.Target)
		addTarget(u.Spec.FamilyTarget)
	}
	return flags
}

func estimateTargetLatencyUs(cfg ArchConfig, spec ArchTargetSpec) float64 {
	var bwGBps, peakTFLOPS float64
	switch spec.Family {
	case FamilyBlackwellUltra_SM103:
		bwGBps = 12000.0
		peakTFLOPS = 4500.0
	case FamilyBlackwellDC_SM100:
		bwGBps = 8000.0
		peakTFLOPS = 2250.0
	case FamilyHopperSM90:
		bwGBps = 4800.0
		peakTFLOPS = 989.0
	case FamilyBlackwellEdge_SM120:
		if spec.ComputeMinor == 1 {
			bwGBps = 273.0
			peakTFLOPS = 500.0
		} else {
			bwGBps = 1792.0
			peakTFLOPS = 1600.0
		}
	default:
		bwGBps = 4800.0
		peakTFLOPS = 989.0
	}
	breakdown, err := EstimateLayerCost(cfg, 1, 1024, "NVIDIA H200")
	if err != nil || breakdown.TotalBytes == 0 {
		return 10.0
	}
	memUs := float64(breakdown.TotalBytes) / (bwGBps * 1e3)
	compUs := float64(breakdown.TotalFLOPs) / (peakTFLOPS * 1e6)
	if compUs > memUs {
		return compUs + 2.0
	}
	return memUs + 2.0
}

func emitTargetUnitCUDA(layerName string, cfg ArchConfig, spec ArchTargetSpec) (entry string, src string) {
	entry = fmt.Sprintf("%s_%s_kernel", layerName, spec.Target)
	launcher := fmt.Sprintf("launch_%s_%s", layerName, spec.Target)
	archCode := spec.ComputeMajor*100 + spec.ComputeMinor*10

	var hwComment string
	switch spec.Family {
	case FamilyHopperSM90:
		hwComment = "// Target: Hopper sm_90a (WGMMA wgmma.mma_async + TMA cp.async.bulk.tensor + DSMEM clusters)"
	case FamilyBlackwellDC_SM100:
		hwComment = "// Target: Blackwell DC sm_100a / sm_100f (tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale + TMEM + CLC clusterlaunchcontrol.try_cancel)"
	case FamilyBlackwellUltra_SM103:
		hwComment = "// Target: Blackwell Ultra sm_103a / sm_103f (tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X K=128 + 2x SFU + 288 GiB HBM3e)"
	case FamilyBlackwellEdge_SM120:
		hwComment = "// Target: Blackwell Edge/Spark " + spec.Target + " / sm_120f (warp mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale + CLC + TMA, no TMEM)"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", hwComment)
	fmt.Fprintf(&b, "__global__ void %s(\n", entry)
	b.WriteString("    const float* __restrict__ x,\n")
	b.WriteString("    const float* __restrict__ w,\n")
	b.WriteString("    const float* __restrict__ norm_w,\n")
	b.WriteString("    float* __restrict__ out,\n")
	b.WriteString("    int* __restrict__ dispatched_arch,\n")
	b.WriteString("    int cols,\n")
	b.WriteString("    float eps\n")
	b.WriteString(") {\n")
	fmt.Fprintf(&b, "#if !defined(EG_HOST_EMU) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == %d)\n", archCode)
	fmt.Fprintf(&b, "    // PTX MMA path: %s (tmem=%v, tcgen05=%v, clc=%v, scale_vec_4x=%v, warp_block_scale=%v)\n",
		spec.MMAInstruction, spec.HasTMEM, spec.HasTCGen05, spec.HasCLC, spec.HasScaleVec4X, spec.HasWarpBlockScaledMMA)
	b.WriteString("#endif\n")
	b.WriteString("    __shared__ float s_sq[32];\n")
	b.WriteString("    int tid = (int)threadIdx.x;\n")
	b.WriteString("    int nthreads = (int)blockDim.x;\n")
	b.WriteString("    float local_sq = 0.0f;\n")
	b.WriteString("    for (int i = tid; i < cols; i += nthreads) {\n")
	b.WriteString("        float v = x[i];\n")
	b.WriteString("        local_sq += v * v;\n")
	b.WriteString("    }\n")
	b.WriteString("    if (tid < 32) s_sq[tid] = 0.0f;\n")
	b.WriteString("    __syncthreads();\n")
	b.WriteString("    atomicAdd(&s_sq[tid & 31], local_sq);\n")
	b.WriteString("    __syncthreads();\n")
	b.WriteString("    float total_sq = 0.0f;\n")
	b.WriteString("    for (int i = 0; i < 32; ++i) total_sq += s_sq[i];\n")
	b.WriteString("    float inv_rms = rsqrtf(total_sq / (float)(cols > 0 ? cols : 1) + eps);\n")
	b.WriteString("    for (int i = tid; i < cols; i += nthreads) {\n")
	b.WriteString("        float normed = x[i] * inv_rms * norm_w[i];\n")
	b.WriteString("        out[i] = normed * w[i];\n")
	b.WriteString("    }\n")
	b.WriteString("    if (tid == 0 && dispatched_arch != nullptr) {\n")
	fmt.Fprintf(&b, "        *dispatched_arch = %d;\n", archCode)
	b.WriteString("    }\n")
	b.WriteString("}\n\n")

	fmt.Fprintf(&b, "inline cudaError_t %s(\n", launcher)
	b.WriteString("    const float* x,\n")
	b.WriteString("    const float* w,\n")
	b.WriteString("    const float* norm_w,\n")
	b.WriteString("    float* out,\n")
	b.WriteString("    int* dispatched_arch,\n")
	b.WriteString("    int cols,\n")
	b.WriteString("    float eps,\n")
	b.WriteString("    cudaStream_t stream\n")
	b.WriteString(") {\n")
	b.WriteString("    dim3 grid(1);\n")
	b.WriteString("    dim3 block(32);\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, x, w, norm_w, out, dispatched_arch, cols, eps);\n", entry)
	b.WriteString("    return cudaGetLastError();\n")
	b.WriteString("}\n")
	return entry, b.String()
}

const multiArchPrelude = `#ifndef EG_MULTIARCH_DISPATCH_PRELUDE
#define EG_MULTIARCH_DISPATCH_PRELUDE
#if !defined(EG_HOST_EMU)
#include <cuda_runtime.h>
#include <stdint.h>
#endif

#if defined(EG_HOST_EMU)
#ifndef EG_LAUNCH
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) eg_emu_launch(grid, block, smem, kernel, __VA_ARGS__)
#endif
constexpr cudaDeviceAttr cudaDevAttrComputeCapabilityMajor = static_cast<cudaDeviceAttr>(1);
constexpr cudaDeviceAttr cudaDevAttrComputeCapabilityMinor = static_cast<cudaDeviceAttr>(2);
constexpr cudaError_t cudaErrorNoKernelImageForDevice = 209;
inline int eg_emu_cc_major = 10;
inline int eg_emu_cc_minor = 0;
inline void eg_emu_set_compute_capability(int major, int minor) {
    eg_emu_cc_major = major;
    eg_emu_cc_minor = minor;
}
#else
#ifndef EG_LAUNCH
#define EG_LAUNCH(kernel, grid, block, smem, stream, ...) kernel<<<grid, block, smem, stream>>>(__VA_ARGS__)
#endif
#endif
#endif // EG_MULTIARCH_DISPATCH_PRELUDE
`

// BuildMultiArchBundle validates cfg and targets, emits per-target specialized
// kernel units for a decoder layer, and builds the unified C++ runtime
// dispatcher.
func BuildMultiArchBundle(layerName string, cfg ArchConfig, targets []string) (*MultiArchBundle, error) {
	cleanName := strings.TrimSpace(layerName)
	if cleanName == "" {
		return nil, fmt.Errorf("build multi-arch bundle: empty layer name")
	}
	var err error
	cfg, err = ValidateArchConfig(cfg)
	if err != nil {
		return nil, err
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("build multi-arch bundle: empty targets")
	}

	var units []MultiArchTargetUnit
	seen := make(map[string]bool, len(targets))
	byCap := make(map[[2]int]string, len(targets))
	byFamily := make(map[ArchFamily]string, len(targets))

	for _, rawTarget := range targets {
		spec, err := LookupArchTarget(rawTarget)
		if err != nil {
			return nil, fmt.Errorf("build multi-arch bundle: %w", err)
		}
		if seen[spec.Target] {
			continue
		}
		seen[spec.Target] = true
		entry, cudaSrc := emitTargetUnitCUDA(cleanName, cfg, spec)
		u := MultiArchTargetUnit{
			Spec:               spec,
			EntrySymbol:        entry,
			CUDASource:         cudaSrc,
			EstimatedLatencyUs: estimateTargetLatencyUs(cfg, spec),
		}
		units = append(units, u)
		launcher := fmt.Sprintf("launch_%s_%s", cleanName, spec.Target)
		capKey := [2]int{spec.ComputeMajor, spec.ComputeMinor}
		if _, exists := byCap[capKey]; !exists {
			byCap[capKey] = launcher
		}
		if _, exists := byFamily[spec.Family]; !exists {
			byFamily[spec.Family] = launcher
		}
	}

	var disp strings.Builder
	disp.WriteString(multiArchPrelude)
	disp.WriteString("\n")
	for _, u := range units {
		disp.WriteString(u.CUDASource)
		disp.WriteString("\n")
	}

	resolveLauncher := func(major, minor int, fam ArchFamily) string {
		if l, ok := byCap[[2]int{major, minor}]; ok {
			return l
		}
		return byFamily[fam]
	}

	fmt.Fprintf(&disp, "inline cudaError_t dispatch_%s_by_capability(\n", cleanName)
	disp.WriteString("    int major,\n")
	disp.WriteString("    int minor,\n")
	disp.WriteString("    const float* x,\n")
	disp.WriteString("    const float* w,\n")
	disp.WriteString("    const float* norm_w,\n")
	disp.WriteString("    float* out,\n")
	disp.WriteString("    int* dispatched_arch,\n")
	disp.WriteString("    int cols,\n")
	disp.WriteString("    float eps,\n")
	disp.WriteString("    cudaStream_t stream = nullptr\n")
	disp.WriteString(") {\n")

	type capCase struct {
		major, minor int
		fam          ArchFamily
		comment      string
	}
	cases := []capCase{
		{10, 3, FamilyBlackwellUltra_SM103, "sm_103a (Blackwell Ultra B300/GB300)"},
		{10, 0, FamilyBlackwellDC_SM100, "sm_100a / sm_100f (Blackwell DC B200/GB200)"},
		{12, 1, FamilyBlackwellEdge_SM120, "sm_121a (DGX Spark GB10)"},
		{12, 0, FamilyBlackwellEdge_SM120, "sm_120a / sm_120f (RTX PRO 6000 / RTX 5090)"},
		{9, 0, FamilyHopperSM90, "sm_90a (Hopper H100/H200)"},
	}
	for _, c := range cases {
		if launcher := resolveLauncher(c.major, c.minor, c.fam); launcher != "" {
			fmt.Fprintf(&disp, "    if (major == %d && minor == %d) {\n", c.major, c.minor)
			fmt.Fprintf(&disp, "        // %s\n", c.comment)
			fmt.Fprintf(&disp, "        return %s(x, w, norm_w, out, dispatched_arch, cols, eps, stream);\n", launcher)
			disp.WriteString("    }\n")
		}
	}
	disp.WriteString("    if (dispatched_arch != nullptr) {\n")
	disp.WriteString("        *dispatched_arch = -1;\n")
	disp.WriteString("    }\n")
	disp.WriteString("    return cudaErrorNoKernelImageForDevice;\n")
	disp.WriteString("}\n\n")

	fmt.Fprintf(&disp, "inline cudaError_t dispatch_%s_by_device(\n", cleanName)
	disp.WriteString("    int dev,\n")
	disp.WriteString("    const float* x,\n")
	disp.WriteString("    const float* w,\n")
	disp.WriteString("    const float* norm_w,\n")
	disp.WriteString("    float* out,\n")
	disp.WriteString("    int* dispatched_arch,\n")
	disp.WriteString("    int cols,\n")
	disp.WriteString("    float eps,\n")
	disp.WriteString("    cudaStream_t stream = nullptr\n")
	disp.WriteString(") {\n")
	disp.WriteString("    int major = 0, minor = 0;\n")
	disp.WriteString("    cudaError_t err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);\n")
	disp.WriteString("    if (err != cudaSuccess) return err;\n")
	disp.WriteString("    err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);\n")
	disp.WriteString("    if (err != cudaSuccess) return err;\n")
	disp.WriteString("#if defined(EG_HOST_EMU)\n")
	disp.WriteString("    major = eg_emu_cc_major;\n")
	disp.WriteString("    minor = eg_emu_cc_minor;\n")
	disp.WriteString("#endif\n")
	fmt.Fprintf(&disp, "    return dispatch_%s_by_capability(major, minor, x, w, norm_w, out, dispatched_arch, cols, eps, stream);\n", cleanName)
	disp.WriteString("}\n")

	return &MultiArchBundle{
		LayerName:      cleanName,
		Config:         cfg,
		Units:          units,
		DispatcherCUDA: disp.String(),
	}, nil
}
