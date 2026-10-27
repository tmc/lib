package hopper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

const hopperNativeEnv = "CUDA_EGRAPH_HOPPER_NATIVE"

func hopperNativeArch() string {
	if arch := os.Getenv("CUDA_EGRAPH_HOPPER_NATIVE_ARCH"); arch != "" {
		return arch
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader").Output(); err == nil {
		fields := strings.Fields(string(out))
		if len(fields) > 0 {
			cc := strings.ReplaceAll(strings.TrimSpace(fields[0]), ".", "")
			switch cc {
			case "90", "100", "103", "120", "121":
				return "sm_" + cc + "a"
			default:
				if cc != "" {
					return "sm_" + cc
				}
			}
		}
	}
	return "sm_90a"
}

type hopperNativeGateClosedError struct{}

func (hopperNativeGateClosedError) Error() string {
	return "hopper: native gate is closed: CUDA_EGRAPH_HOPPER_NATIVE=1 unset, no device examined"
}

func (hopperNativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

var errHopperNativeGateClosed error = hopperNativeGateClosedError{}

func checkHopperNativeGate(getenv func(string) string) error {
	if getenv(hopperNativeEnv) != "1" {
		return errHopperNativeGateClosed
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("hopper: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// TestHopperNativeClosedGateControl verifies that when CUDA_EGRAPH_HOPPER_NATIVE
// is not "1", the native qualification gate reports cudaerrors.ErrHardwarePending,
// refuses a presence-only GPU_LOCK_HELD=1, accepts a genuine inherited lock,
// and emits a self-contained native harness for all seven Hopper/Blackwell kernels.
func TestHopperNativeClosedGateControl(t *testing.T) {
	if os.Getenv(hopperNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_EGRAPH_HOPPER_NATIVE != 1")
	}
	err := checkHopperNativeGate(os.Getenv)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkHopperNativeGate(closed) = %v, want errors.Is(..., cudaerrors.ErrHardwarePending)", err)
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("closed gate matched ErrUnsupported: %v", err)
	}

	// Presence-only GPU_LOCK_HELD=1 must be rejected.
	badEnv := func(k string) string {
		switch k {
		case hopperNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return "1"
		default:
			return ""
		}
	}
	if err := checkHopperNativeGate(badEnv); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkHopperNativeGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", err)
	}

	// Genuine inherited lock on host must pass gate check.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("hopper-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	goodEnv := func(k string) string {
		switch k {
		case hopperNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		default:
			return ""
		}
	}
	if err := checkHopperNativeGate(goodEnv); err != nil {
		t.Fatalf("checkHopperNativeGate with inherited lock: %v", err)
	}

	src := buildHopperNativeHarness(t)
	for _, want := range []string{
		"qk_norm_rope_kv_native",
		"mtp_paged_attn_native",
		"mla_paged_attn_native",
		"specdecode_ctrl_native",
		"rmsnorm_dyn_nvfp4_native",
		"swiglu_dyn_nvfp4_native",
		"sm120_paged_attn_native",
		"mutant_rejected",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("buildHopperNativeHarness missing %q", want)
		}
	}
}

// TestHopperNativeQualification compiles and runs the native device
// qualification suite for QKNormRoPEKVAppend, MTPPagedAttention,
// MLAPagedAttention, SpeculativeDecodeController, RMSNormDynamicNVFP4Quant,
// SwiGLUDynamicNVFP4Quant, and SM120PagedAttention, plus a device mutant
// control that perturbs the attention input and verifies the check rejects it.
func TestHopperNativeQualification(t *testing.T) {
	if os.Getenv(hopperNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_HOPPER_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)
	target := hopperNativeArch()

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("user home dir: %v", err)
	}
	tmpBase := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(tmpBase, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", tmpBase, err)
	}
	dir, err := os.MkdirTemp(tmpBase, "hopper-native-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	src := buildHopperNativeHarness(t)
	cuPath := filepath.Join(dir, "hopper_qualify.cu")
	binPath := filepath.Join(dir, "hopper_qualify")
	if err := os.WriteFile(cuPath, []byte(src), 0o600); err != nil {
		t.Fatalf("write %s: %v", cuPath, err)
	}

	buildCtx, buildCancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer buildCancel()
	buildCmd := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O2", "-arch="+target, cuPath, "-o", binPath)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc compile (%s, CUDA %s): %v\n%s", target, cudaVer, err, out)
	}

	runCtx, runCancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer runCancel()
	runCmd := exec.CommandContext(runCtx, binPath)
	out, err := runCmd.CombinedOutput()
	if err != nil {
		exitCode := -1
		if runCmd.ProcessState != nil {
			exitCode = runCmd.ProcessState.ExitCode()
		}
		t.Fatalf("native qualification binary failed (exit=%d, arch=%s, CUDA=%s): %v\n%s",
			exitCode, target, cudaVer, err, out)
	}
	outStr := string(out)
	if !strings.Contains(outStr, "HOPPER_NATIVE_OK") || !strings.Contains(outStr, "mutant_rejected=1") {
		t.Fatalf("native qualification output missing expected markers:\n%s", outStr)
	}
	t.Logf("native qualification passed (arch=%s, CUDA=%s):\n%s", target, cudaVer, strings.TrimSpace(outStr))
}

func buildHopperNativeHarness(t *testing.T) string {
	t.Helper()

	// 1. QKNormRoPEKVAppend fixture (M=1, NumHeads=2, KVHeads=1, HeadDim=16, PageSize=4, MaxPages=2).
	qk := QKNormRoPEKVAppend{
		Name:      "qk_norm_rope_kv_native",
		NumHeads:  2,
		KVHeads:   1,
		HeadDim:   16,
		RotaryDim: 16,
		PageSize:  4,
		MaxPages:  2,
		Eps:       1e-5,
		QKNorm:    true,
		DType:     BF16,
	}
	qkCUDA, err := qk.CUDA()
	if err != nil {
		t.Fatalf("qk.CUDA: %v", err)
	}
	qkLaunch, err := qk.Launcher()
	if err != nil {
		t.Fatalf("qk.Launcher: %v", err)
	}
	const qkStride = (2 + 2*1) * 16 // 64
	qkvIn := make([]float64, qkStride)
	qkvBF16 := make([]uint16, qkStride)
	for i := range qkStride {
		v := float32(0.1 * float64((i%7)-3))
		qkvBF16[i] = BF16Bits(v)
		qkvIn[i] = float64(FromBF16Bits(qkvBF16[i]))
	}
	normW := make([]float64, 16)
	normWBF16 := make([]uint16, 16)
	cosSin := make([]float32, 32)
	for i := range 16 {
		normWBF16[i] = BF16Bits(1.0)
		normW[i] = 1.0
	}
	for i := range 32 {
		angle := float64(i) * 0.05
		if i%2 == 0 {
			cosSin[i] = float32(math.Cos(angle))
		} else {
			cosSin[i] = float32(math.Sin(angle))
		}
	}
	qkInputs := &QKNormRoPEKVInputs{
		QKV:         qkvIn,
		QNormWeight: normW,
		KNormWeight: normW,
		CosSinTable: cosSin,
		Positions:   []int32{1},
		SeqIDs:      []int32{0},
		SeqLens:     []int32{2},
		PageTable:   []int32{0, -1},
		KCache:      make([]float64, 8*16),
		VCache:      make([]float64, 8*16),
	}
	qkRef, err := qk.Reference(1, qkInputs)
	if err != nil {
		t.Fatalf("qk.Reference: %v", err)
	}

	// 2. MTPPagedAttention fixture (batch=1, NumHeads=2, KVHeads=1, HeadDim=16, PageSize=4, MaxPages=2, QTokensPerSeq=2).
	mtp := MTPPagedAttention{
		Name:          "mtp_paged_attn_native",
		NumHeads:      2,
		KVHeads:       1,
		HeadDim:       16,
		PageSize:      4,
		MaxPages:      2,
		QTokensPerSeq: 2,
	}
	mtpCUDA, err := mtp.CUDA()
	if err != nil {
		t.Fatalf("mtp.CUDA: %v", err)
	}
	mtpLaunch, err := mtp.Launcher()
	if err != nil {
		t.Fatalf("mtp.Launcher: %v", err)
	}
	mtpQ := make([]uint16, 2*2*16)
	for i := range mtpQ {
		mtpQ[i] = BF16Bits(float32(0.15 * float64((i%5)-2)))
	}
	mtpK := make([]uint16, 8*16)
	mtpV := make([]uint16, 8*16)
	for i := range mtpK {
		mtpK[i] = BF16Bits(float32(0.12 * float64((i%9)-4)))
		mtpV[i] = BF16Bits(float32(0.2 * float64((i%6)-2)))
	}
	mtpCache := MTPPagedCache{
		KBF16:     mtpK,
		VBF16:     mtpV,
		PageTable: []int32{0, 1},
		SeqLens:   []int32{4},
		DraftMask: CausalDraftMask(2),
	}
	mtpRef, err := mtp.Reference(mtpQ, mtpCache)
	if err != nil {
		t.Fatalf("mtp.Reference: %v", err)
	}

	// 3. MLAPagedAttention fixture (batch=1, NumHeads=2, LatentDim=16, RopeDim=16, PageSize=4, MaxPages=2).
	mla := MLAPagedAttention{
		Name:      "mla_paged_attn_native",
		NumHeads:  2,
		LatentDim: 16,
		RopeDim:   16,
		PageSize:  4,
		MaxPages:  2,
	}
	mlaCUDA, err := mla.CUDA()
	if err != nil {
		t.Fatalf("mla.CUDA: %v", err)
	}
	mlaLaunch, err := mla.Launcher()
	if err != nil {
		t.Fatalf("mla.Launcher: %v", err)
	}
	mlaQNope := make([]uint16, 2*16)
	mlaQPe := make([]uint16, 2*16)
	for i := range mlaQNope {
		mlaQNope[i] = BF16Bits(float32(0.1 * float64((i%5)-2)))
		mlaQPe[i] = BF16Bits(float32(0.08 * float64((i%7)-3)))
	}
	mlaCKV := make([]uint16, 8*16)
	mlaKPE := make([]uint16, 8*16)
	for i := range mlaCKV {
		mlaCKV[i] = BF16Bits(float32(0.14 * float64((i%5)-1)))
		mlaKPE[i] = BF16Bits(float32(0.09 * float64((i%6)-2)))
	}
	mlaCache := MLAPagedCache{
		CKVBF16:   mlaCKV,
		KPEBF16:   mlaKPE,
		PageTable: []int32{0, 1},
		SeqLens:   []int32{4},
	}
	mlaRef, err := mla.Reference(mlaQNope, mlaQPe, mlaCache)
	if err != nil {
		t.Fatalf("mla.Reference: %v", err)
	}

	// 4. SpeculativeDecodeController fixture (DraftTokens=4, MaxSteps=4).
	spec := SpeculativeDecodeController{
		Name:        "specdecode_ctrl_native",
		DraftTokens: 4,
		MaxSteps:    4,
	}
	specCUDA, err := spec.CUDA()
	if err != nil {
		t.Fatalf("spec.CUDA: %v", err)
	}
	specLaunch, err := spec.Launcher()
	if err != nil {
		t.Fatalf("spec.Launcher: %v", err)
	}
	specIn := SpeculativeDecodeInput{
		InitialTokens: []int32{10, 20},
		EOSTokenID:    999,
		MaxSeqLen:     16,
		Steps: []SpeculativeDecodeStep{
			{
				Draft:        []int32{30, 31, 32, 99},
				TargetGreedy: []int32{30, 31, 32, 33, 34},
			},
		},
	}
	specRef, err := spec.Reference(specIn)
	if err != nil {
		t.Fatalf("spec.Reference: %v", err)
	}

	// 5. RMSNormDynamicNVFP4Quant fixture (M=1, Cols=64, Eps=1e-5, In=F32, Residual=true, ScaleLayoutTCGen05Swizzled).
	rmsQuant := RMSNormDynamicNVFP4Quant{
		Name:        "rmsnorm_dyn_nvfp4_native",
		Cols:        64,
		Eps:         1e-5,
		In:          F32,
		Residual:    true,
		ScaleLayout: ScaleLayoutTCGen05Swizzled,
	}
	rmsCUDA, err := rmsQuant.CUDA()
	if err != nil {
		t.Fatalf("rmsQuant.CUDA: %v", err)
	}
	rmsLaunch, err := rmsQuant.Launcher()
	if err != nil {
		t.Fatalf("rmsQuant.Launcher: %v", err)
	}
	rmsX := make([]float32, 64)
	rmsRes := make([]float32, 64)
	rmsW := make([]float32, 64)
	for i := range 64 {
		rmsX[i] = float32(math.Sin(float64(i)*0.17+0.3)) * 2.5
		rmsRes[i] = float32(math.Cos(float64(i)*0.11-0.2)) * 1.4
		rmsW[i] = 0.5 + 0.75*float32(math.Cos(float64(i)*0.09))
	}
	_, rmsWantFP4, rmsWantScales, _, err := rmsQuant.Reference(rmsX, rmsRes, rmsW, 1)
	if err != nil {
		t.Fatalf("rmsQuant.Reference: %v", err)
	}

	// 6. SwiGLUDynamicNVFP4Quant fixture (M=1, Cols=64, In=F32, ScaleLayoutRowMajor).
	swigluQuant := SwiGLUDynamicNVFP4Quant{
		Name:        "swiglu_dyn_nvfp4_native",
		Cols:        64,
		In:          F32,
		ScaleLayout: ScaleLayoutRowMajor,
	}
	swigluCUDA, err := swigluQuant.CUDA()
	if err != nil {
		t.Fatalf("swigluQuant.CUDA: %v", err)
	}
	swigluLaunch, err := swigluQuant.Launcher()
	if err != nil {
		t.Fatalf("swigluQuant.Launcher: %v", err)
	}
	swigluGate := make([]float32, 64)
	swigluUp := make([]float32, 64)
	for i := range 64 {
		swigluGate[i] = float32(math.Sin(float64(i)*0.23+0.1)) * 2.2
		swigluUp[i] = float32(math.Cos(float64(i)*0.19-0.4)) * 1.8
	}
	swigluWantFP4, swigluWantScales, _, err := swigluQuant.Reference(swigluGate, swigluUp, 1)
	if err != nil {
		t.Fatalf("swigluQuant.Reference: %v", err)
	}

	// 7. SM120PagedAttention fixture (batch=1, NumQHeads=4, NumKVHeads=2, HeadDim=16, PageSize=4, MaxPages=2, SeqLen=6, Target="sm_121a", KVFormat=WeightNVFP4).
	sm120Attn := SM120PagedAttention{
		Name:       "sm120_paged_attn_native",
		NumQHeads:  4,
		NumKVHeads: 2,
		HeadDim:    16,
		PageSize:   4,
		MaxPages:   2,
		SeqLen:     6,
		Target:     "sm_121a",
		KVFormat:   WeightNVFP4,
	}
	sm120CUDA, err := sm120Attn.CUDA()
	if err != nil {
		t.Fatalf("sm120Attn.CUDA: %v", err)
	}
	sm120Launch, err := sm120Attn.Launcher()
	if err != nil {
		t.Fatalf("sm120Attn.Launcher: %v", err)
	}
	sm120Q := make([]float32, 4*16)
	for i := range sm120Q {
		sm120Q[i] = float32(0.15 * float64((i%7)-3))
	}
	// 2 pages * 4 tokens/page * 2 KV heads = 16 rows; HeadDim=16 -> 8 packed bytes/row, 1 scale byte/row.
	sm120KPages := make([]uint8, 16*8)
	sm120VPages := make([]uint8, 16*8)
	for i := range sm120KPages {
		c0 := uint8((i + 1) % 7)
		c1 := uint8(((i + 3) % 7) | 8)
		sm120KPages[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
		v0 := uint8((i + 2) % 6)
		v1 := uint8((i + 4) % 6)
		sm120VPages[i] = (v0 & 0xf) | ((v1 & 0xf) << 4)
	}
	sm120KScales := make([]uint8, 16)
	sm120VScales := make([]uint8, 16)
	for i := range 16 {
		sm120KScales[i] = E4M3Bits(0.25)
		sm120VScales[i] = E4M3Bits(0.5)
	}
	sm120PageTable := []int32{1, 0}
	sm120Ref, err := sm120Attn.Reference(sm120Q, sm120KPages, sm120VPages, sm120KScales, sm120VScales, sm120PageTable, 6)
	if err != nil {
		t.Fatalf("sm120Attn.Reference: %v", err)
	}

	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <vector>\n")
	b.WriteString("#define EG_NATIVE_COND_SLOT 1\n")
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(qkCUDA + "\n" + qkLaunch + "\n")
	b.WriteString(mtpCUDA + "\n" + mtpLaunch + "\n")
	b.WriteString(mlaCUDA + "\n" + mlaLaunch + "\n")
	b.WriteString(specCUDA + "\n" + specLaunch + "\n")
	b.WriteString(rmsCUDA + "\n" + rmsLaunch + "\n")
	b.WriteString(swigluCUDA + "\n" + swigluLaunch + "\n")
	b.WriteString(sm120CUDA + "\n" + sm120Launch + "\n")

	fmt.Fprintf(&b, `
static float host_bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static bool check_close(const char* label, const uint16_t* got_bf16, const float* want, int n, float tol) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float g = host_bf16_to_f32(got_bf16[i]);
        float d = std::fabs(g - want[i]);
        if (std::isnan(g) || d > max_diff) max_diff = d;
        if (std::isnan(g) || d > tol) {
            std::printf("FAIL %%s idx=%%d got=%%f want=%%f diff=%%f\n", label, i, g, want[i], d);
            return false;
        }
    }
    std::printf("PASS %%s n=%%d max_abs_diff=%%.6f\n", label, n, max_diff);
    return true;
}

static bool check_close_f32(const char* label, const float* got, const float* want, int n, float tol) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float g = got[i];
        float d = std::fabs(g - want[i]);
        if (std::isnan(g) || d > max_diff) max_diff = d;
        if (std::isnan(g) || d > tol) {
            std::printf("FAIL %%s idx=%%d got=%%f want=%%f diff=%%f\n", label, i, g, want[i], d);
            return false;
        }
    }
    std::printf("PASS %%s n=%%d max_abs_diff=%%.6f\n", label, n, max_diff);
    return true;
}

template <typename T>
static T* dev_dup(const T* host_data, size_t count) {
    T* d = nullptr;
    if (cudaMalloc(&d, count * sizeof(T)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, host_data, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return d;
}

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("cudaGetDeviceProperties failed\n");
        return 2;
    }
    int runtime_ver = 0, driver_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    cudaDriverGetVersion(&driver_ver);
    std::printf("device=%%s sm=%%d.%%d driver=%%d runtime=%%d\n",
                prop.name, prop.minor >= 0 ? prop.major : 0, prop.minor, driver_ver, runtime_ver);

    // 1. QKNormRoPEKVAppend check
    const uint16_t h_qkv[] = {%s};
    const uint16_t h_norm_w[] = {%s};
    const float h_cos_sin[] = {%s};
    const int32_t h_pos[] = {1}, h_seq_ids[] = {0}, h_seqlens_qk[] = {2}, h_pt_qk[] = {0, -1};
    const float want_qk[] = {%s};

    auto* d_qkv = dev_dup(h_qkv, %d);
    auto* d_norm_w = dev_dup(h_norm_w, 16);
    auto* d_cos_sin = dev_dup(h_cos_sin, 32);
    auto* d_pos = dev_dup(h_pos, 1);
    auto* d_seq_ids = dev_dup(h_seq_ids, 1);
    auto* d_seqlens_qk = dev_dup(h_seqlens_qk, 1);
    auto* d_pt_qk = dev_dup(h_pt_qk, 2);
    uint16_t *d_q_out = nullptr, *d_k_cache = nullptr, *d_v_cache = nullptr;
    if (!d_qkv || cudaMalloc(&d_q_out, 32 * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k_cache, 128 * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v_cache, 128 * sizeof(uint16_t)) != cudaSuccess) return 2;

    qk_norm_rope_kv_native_launch(1, d_qkv, d_norm_w, d_norm_w, d_cos_sin, d_pos, d_seq_ids, d_seqlens_qk, nullptr, d_pt_qk, d_q_out, d_k_cache, d_v_cache, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint16_t h_q_out[32] = {};
    if (cudaMemcpy(h_q_out, d_q_out, sizeof(h_q_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (!check_close("QKNormRoPEKVAppend", h_q_out, want_qk, 32, 1e-2f)) return 6;

    // 2. MTPPagedAttention check + Mutant control
    const uint16_t h_mtp_q[] = {%s};
    const uint16_t h_mtp_k[] = {%s};
    const uint16_t h_mtp_v[] = {%s};
    const int32_t h_pt_mtp[] = {0, 1}, h_seqlens_mtp[] = {4};
    const uint64_t h_mask_mtp[] = {1ULL, 3ULL};
    const float want_mtp[] = {%s};

    auto* d_mtp_q = dev_dup(h_mtp_q, 64);
    auto* d_mtp_k = dev_dup(h_mtp_k, 128);
    auto* d_mtp_v = dev_dup(h_mtp_v, 128);
    auto* d_pt_mtp = dev_dup(h_pt_mtp, 2);
    auto* d_seqlens_mtp = dev_dup(h_seqlens_mtp, 1);
    auto* d_mask_mtp = dev_dup(h_mask_mtp, 2);
    uint16_t* d_mtp_out = nullptr;
    if (cudaMalloc(&d_mtp_out, 64 * sizeof(uint16_t)) != cudaSuccess) return 2;

    mtp_paged_attn_native_launch(1, d_mtp_q, d_mtp_k, d_mtp_v, d_pt_mtp, d_seqlens_mtp, d_mask_mtp, d_mtp_out, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint16_t h_mtp_out[64] = {};
    if (cudaMemcpy(h_mtp_out, d_mtp_out, sizeof(h_mtp_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (!check_close("MTPPagedAttention", h_mtp_out, want_mtp, 64, 1e-2f)) return 6;

    // Mutant control on device: perturb V cache values and verify check_close rejects the mutant output.
    uint16_t h_mtp_v_mut[128];
    for (int i = 0; i < 128; ++i) h_mtp_v_mut[i] = 0x40a0u; // +5.0 in BF16
    if (cudaMemcpy(d_mtp_v, h_mtp_v_mut, sizeof(h_mtp_v_mut), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    mtp_paged_attn_native_launch(1, d_mtp_q, d_mtp_k, d_mtp_v, d_pt_mtp, d_seqlens_mtp, d_mask_mtp, d_mtp_out, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint16_t h_mtp_mut_out[64] = {};
    if (cudaMemcpy(h_mtp_mut_out, d_mtp_out, sizeof(h_mtp_mut_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    bool mutant_rejected = false;
    for (int i = 0; i < 64; ++i) {
        float g = host_bf16_to_f32(h_mtp_mut_out[i]);
        if (std::fabs(g - want_mtp[i]) > 1e-2f) {
            mutant_rejected = true;
            break;
        }
    }
    if (!mutant_rejected) {
        std::printf("FAIL mutant control was not rejected\n");
        return 7;
    }

    // 3. MLAPagedAttention check
    const uint16_t h_mla_qn[] = {%s};
    const uint16_t h_mla_qp[] = {%s};
    const uint16_t h_mla_ckv[] = {%s};
    const uint16_t h_mla_kpe[] = {%s};
    const float want_mla[] = {%s};

    auto* d_mla_qn = dev_dup(h_mla_qn, 32);
    auto* d_mla_qp = dev_dup(h_mla_qp, 32);
    auto* d_mla_ckv = dev_dup(h_mla_ckv, 128);
    auto* d_mla_kpe = dev_dup(h_mla_kpe, 128);
    uint16_t* d_mla_out = nullptr;
    if (cudaMalloc(&d_mla_out, 32 * sizeof(uint16_t)) != cudaSuccess) return 2;

    mla_paged_attn_native_launch(1, d_mla_qn, d_mla_qp, d_mla_ckv, d_mla_kpe, d_pt_mtp, d_seqlens_mtp, d_mla_out, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint16_t h_mla_out[32] = {};
    if (cudaMemcpy(h_mla_out, d_mla_out, sizeof(h_mla_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (!check_close("MLAPagedAttention", h_mla_out, want_mla, 32, 1e-2f)) return 6;

    // 4. SpeculativeDecodeController check
    const int32_t h_draft[] = {30, 31, 32, 99};
    const int32_t h_target[] = {30, 31, 32, 33, 34};
    int32_t h_out_toks[16] = {10, 20};
    int32_t h_cur_seq = 2, h_acc_step[4] = {}, h_step_idx = 0;
    unsigned int h_cond_slot = 1u;

    auto* d_draft = dev_dup(h_draft, 4);
    auto* d_target = dev_dup(h_target, 5);
    auto* d_out_toks = dev_dup(h_out_toks, 16);
    auto* d_cur_seq = dev_dup(&h_cur_seq, 1);
    auto* d_acc_step = dev_dup(h_acc_step, 4);
    auto* d_step_idx = dev_dup(&h_step_idx, 1);
    auto* d_cond_slot = dev_dup(&h_cond_slot, 1);

    specdecode_ctrl_native_launch(
        d_draft, d_target, 999, 16, d_out_toks, d_cur_seq, d_acc_step, d_step_idx,
        reinterpret_cast<cudaGraphConditionalHandle>(d_cond_slot), 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (cudaMemcpy(h_out_toks, d_out_toks, sizeof(h_out_toks), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&h_cur_seq, d_cur_seq, sizeof(h_cur_seq), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_acc_step, d_acc_step, sizeof(h_acc_step), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&h_step_idx, d_step_idx, sizeof(h_step_idx), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&h_cond_slot, d_cond_slot, sizeof(h_cond_slot), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;

    const int32_t want_toks[] = {%s};
    if (h_cur_seq != %d || h_acc_step[0] != %d || h_step_idx != 1 || h_cond_slot != 1u) {
        std::printf("FAIL SpeculativeDecodeController state seq=%%d acc=%%d step=%%d cond=%%u\n",
                    h_cur_seq, h_acc_step[0], h_step_idx, h_cond_slot);
        return 8;
    }
    for (int i = 0; i < %d; ++i) {
        if (h_out_toks[i] != want_toks[i]) {
            std::printf("FAIL SpeculativeDecodeController tok[%%d]=%%d want=%%d\n", i, h_out_toks[i], want_toks[i]);
            return 8;
        }
    }
    std::printf("PASS SpeculativeDecodeController seq=%%d accepted=%%d cond=%%u\n", h_cur_seq, h_acc_step[0], h_cond_slot);

    // 5. RMSNormDynamicNVFP4Quant check (0 byte differences against Reference)
    const float h_rms_x[] = {%s};
    const float h_rms_res[] = {%s};
    const float h_rms_w[] = {%s};
    const uint8_t want_rms_fp4[] = {%s};
    const uint8_t want_rms_scales[] = {%s};

    auto* d_rms_x = dev_dup(h_rms_x, 64);
    auto* d_rms_res = dev_dup(h_rms_res, 64);
    auto* d_rms_w = dev_dup(h_rms_w, 64);
    float* d_rms_hout = nullptr;
    uint8_t *d_rms_fp4 = nullptr, *d_rms_scales = nullptr;
    if (!d_rms_x || !d_rms_res || !d_rms_w ||
        cudaMalloc(&d_rms_hout, 64 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_rms_fp4, 32) != cudaSuccess ||
        cudaMalloc(&d_rms_scales, 4) != cudaSuccess) return 2;

    rmsnorm_dyn_nvfp4_native_launch(1, d_rms_x, d_rms_res, d_rms_w, d_rms_hout, d_rms_fp4, d_rms_scales, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint8_t h_rms_fp4[32] = {}, h_rms_scales[4] = {};
    if (cudaMemcpy(h_rms_fp4, d_rms_fp4, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_rms_scales, d_rms_scales, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::memcmp(h_rms_fp4, want_rms_fp4, 32) != 0 || std::memcmp(h_rms_scales, want_rms_scales, 4) != 0) {
        std::printf("FAIL RMSNormDynamicNVFP4Quant byte mismatch\n");
        return 9;
    }
    std::printf("PASS RMSNormDynamicNVFP4Quant fp4_bytes=32 scale_bytes=4 byte_diffs=0\n");

    // 6. SwiGLUDynamicNVFP4Quant check (0 byte differences against Reference)
    const float h_swiglu_gate[] = {%s};
    const float h_swiglu_up[] = {%s};
    const uint8_t want_swiglu_fp4[] = {%s};
    const uint8_t want_swiglu_scales[] = {%s};

    auto* d_swiglu_gate = dev_dup(h_swiglu_gate, 64);
    auto* d_swiglu_up = dev_dup(h_swiglu_up, 64);
    uint8_t *d_swiglu_fp4 = nullptr, *d_swiglu_scales = nullptr;
    if (!d_swiglu_gate || !d_swiglu_up ||
        cudaMalloc(&d_swiglu_fp4, 32) != cudaSuccess ||
        cudaMalloc(&d_swiglu_scales, 4) != cudaSuccess) return 2;

    swiglu_dyn_nvfp4_native_launch(1, d_swiglu_gate, d_swiglu_up, d_swiglu_fp4, d_swiglu_scales, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    uint8_t h_swiglu_fp4[32] = {}, h_swiglu_scales[4] = {};
    if (cudaMemcpy(h_swiglu_fp4, d_swiglu_fp4, 32, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_swiglu_scales, d_swiglu_scales, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::memcmp(h_swiglu_fp4, want_swiglu_fp4, 32) != 0 || std::memcmp(h_swiglu_scales, want_swiglu_scales, 4) != 0) {
        std::printf("FAIL SwiGLUDynamicNVFP4Quant byte mismatch\n");
        return 10;
    }
    std::printf("PASS SwiGLUDynamicNVFP4Quant fp4_bytes=32 scale_bytes=4 byte_diffs=0\n");

    // 7. SM120PagedAttention check (max_abs_diff < 1e-3f against Reference)
    const float h_sm120_q[] = {%s};
    const uint8_t h_sm120_kp[] = {%s};
    const uint8_t h_sm120_vp[] = {%s};
    const uint8_t h_sm120_ks[] = {%s};
    const uint8_t h_sm120_vs[] = {%s};
    const int32_t h_sm120_pt[] = {1, 0};
    const float want_sm120_attn[] = {%s};

    auto* d_sm120_q = dev_dup(h_sm120_q, 64);
    auto* d_sm120_kp = dev_dup(h_sm120_kp, 128);
    auto* d_sm120_vp = dev_dup(h_sm120_vp, 128);
    auto* d_sm120_ks = dev_dup(h_sm120_ks, 16);
    auto* d_sm120_vs = dev_dup(h_sm120_vs, 16);
    auto* d_sm120_pt = dev_dup(h_sm120_pt, 2);
    float* d_sm120_out = nullptr;
    if (!d_sm120_q || !d_sm120_kp || !d_sm120_vp || !d_sm120_ks || !d_sm120_vs || !d_sm120_pt ||
        cudaMalloc(&d_sm120_out, 64 * sizeof(float)) != cudaSuccess) return 2;

    sm120_paged_attn_native_launch(1, d_sm120_q, d_sm120_kp, d_sm120_vp, d_sm120_ks, d_sm120_vs, d_sm120_pt, d_sm120_out, 6, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    float h_sm120_out[64] = {};
    if (cudaMemcpy(h_sm120_out, d_sm120_out, sizeof(h_sm120_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (!check_close_f32("SM120PagedAttention", h_sm120_out, want_sm120_attn, 64, 1e-3f)) return 11;

    std::printf("HOPPER_NATIVE_OK mutant_rejected=1\n");
    return 0;
}
`,
		formatU16List(qkvBF16),
		formatU16List(normWBF16),
		formatF32List(cosSin),
		formatF64AsF32List(qkRef),
		len(qkvBF16),
		formatU16List(mtpQ),
		formatU16List(mtpK),
		formatU16List(mtpV),
		formatF64AsF32List(mtpRef),
		formatU16List(mlaQNope),
		formatU16List(mlaQPe),
		formatU16List(mlaCKV),
		formatU16List(mlaKPE),
		formatF64AsF32List(mlaRef),
		formatI32List(specRef.OutputTokens),
		specRef.FinalSeqLen,
		specRef.AcceptedPerStep[0],
		len(specRef.OutputTokens),
		formatF32List(rmsX),
		formatF32List(rmsRes),
		formatF32List(rmsW),
		formatU8List(rmsWantFP4),
		formatU8List(rmsWantScales),
		formatF32List(swigluGate),
		formatF32List(swigluUp),
		formatU8List(swigluWantFP4),
		formatU8List(swigluWantScales),
		formatF32List(sm120Q),
		formatU8List(sm120KPages),
		formatU8List(sm120VPages),
		formatU8List(sm120KScales),
		formatU8List(sm120VScales),
		formatF32List(sm120Ref),
	)
	return b.String()
}

func formatU8List(v []uint8) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("0x%02xu", x)
	}
	return strings.Join(parts, ", ")
}

func formatU16List(v []uint16) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("0x%04xu", x)
	}
	return strings.Join(parts, ", ")
}

func formatF32List(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = floatLiteral(x)
	}
	return strings.Join(parts, ", ")
}

func formatF64AsF32List(v []float64) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = floatLiteral(float32(x))
	}
	return strings.Join(parts, ", ")
}

func formatI32List(v []int32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = strconv.Itoa(int(x))
	}
	return strings.Join(parts, ", ")
}
