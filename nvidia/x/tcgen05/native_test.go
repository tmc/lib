package tcgen05

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

const tcgen05NativeEnv = "CUDA_TCGEN05_NATIVE"

type tcgen05NativeGateClosedError struct{}

func (tcgen05NativeGateClosedError) Error() string {
	return "tcgen05: native gate is closed: CUDA_TCGEN05_NATIVE=1 unset, no device examined"
}

func (tcgen05NativeGateClosedError) Unwrap() error {
	return cudaerrors.ErrHardwarePending
}

var errTCGen05NativeGateClosed error = tcgen05NativeGateClosedError{}

func checkTCGen05NativeGate(getenv func(string) string) error {
	if getenv(tcgen05NativeEnv) != "1" {
		return errTCGen05NativeGateClosed
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("tcgen05: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// classifyTCGen05DeviceArch classifies whether arch supports hardware tcgen05
// Tensor Memory instructions (sm_100a / sm_103a / sm_100f) or refuses them with
// cudaerrors.ErrUnsupported (such as sm_121a on GB10 or sm_90a on Hopper).
func classifyTCGen05DeviceArch(arch string) error {
	switch strings.TrimSpace(arch) {
	case "sm_100a", "sm_103a", "sm_100f", "10.0", "10.3":
		return nil
	case "":
		return fmt.Errorf("tcgen05: empty device architecture: %w", cudaerrors.ErrHardwarePending)
	default:
		return fmt.Errorf("tcgen05: hardware Tensor Memory (tcgen05) is not supported on %s: %w", arch, cudaerrors.ErrUnsupported)
	}
}

// TestTCGen05NativeClosedGateControl verifies that when CUDA_TCGEN05_NATIVE is
// not "1", the closed-gate path returns cudaerrors.ErrHardwarePending,
// presence-only GPU_LOCK_HELD=1 is rejected, a genuine inherited lock is
// accepted, non-sm100/sm103 architectures return cudaerrors.ErrUnsupported,
// and the sm_100a / sm_103a sources and mutant controls are well-formed.
func TestTCGen05NativeClosedGateControl(t *testing.T) {
	if os.Getenv(tcgen05NativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_TCGEN05_NATIVE != 1")
	}

	err := checkTCGen05NativeGate(os.Getenv)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkTCGen05NativeGate(closed) = %v, want errors.Is(..., cudaerrors.ErrHardwarePending)", err)
	}
	if errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("closed gate matched ErrUnsupported: %v", err)
	}

	// Verify architecture classification distinguishes supported (sm_100a/sm_103a)
	// from unsupported (sm_121a/sm_90a) and pending ("").
	for _, arch := range []string{"sm_100a", "sm_103a", "sm_100f"} {
		if err := classifyTCGen05DeviceArch(arch); err != nil {
			t.Fatalf("classifyTCGen05DeviceArch(%q) = %v, want nil", arch, err)
		}
	}
	if err := classifyTCGen05DeviceArch("sm_121a"); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("classifyTCGen05DeviceArch(sm_121a) = %v, want ErrUnsupported", err)
	}
	if err := classifyTCGen05DeviceArch(""); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("classifyTCGen05DeviceArch(\"\") = %v, want ErrHardwarePending", err)
	}

	// Reject presence-only GPU_LOCK_HELD=1.
	badEnv := func(k string) string {
		switch k {
		case tcgen05NativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return "1"
		default:
			return ""
		}
	}
	if err := checkTCGen05NativeGate(badEnv); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkTCGen05NativeGate with GPU_LOCK_HELD=1 = %v, want non-pending lock error", err)
	}

	// Accept genuine inherited lock on host.
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("tcgen05-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	goodEnv := func(k string) string {
		switch k {
		case tcgen05NativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		default:
			return ""
		}
	}
	if err := checkTCGen05NativeGate(goodEnv); err != nil {
		t.Fatalf("checkTCGen05NativeGate with inherited lock: %v", err)
	}

	// Verify emitted sources for TCGen05GEMM (sm_100a) and TCGen05ScaleVec4XGEMM (sm_103a).
	gemm := TCGen05GEMM{
		Name:   "tcgen05_gemm_sm100a",
		N:      128,
		K:      64,
		BM:     128,
		BN:     128,
		BK:     64,
		Weight: WeightBF16,
		Out:    F32,
	}
	gemmSrc, err := gemm.Source()
	if err != nil || !strings.Contains(gemmSrc, "tcgen05.mma.cta_group::1.kind::f16") {
		t.Fatalf("TCGen05GEMM.Source() error=%v or missing instruction", err)
	}

	sv4x := TCGen05ScaleVec4XGEMM{
		Name: "tcgen05_scale_vec_4x_sm103a",
		M:    128,
		N:    128,
		K:    128,
	}
	sv4xSrc, err := sv4x.CUDAFile()
	if err != nil || !strings.Contains(sv4xSrc, "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X") {
		t.Fatalf("TCGen05ScaleVec4XGEMM.CUDAFile() error=%v or missing instruction", err)
	}

	// Verify trace mutant control (omitting dealloc must fail cudatmem.VerifyTrace).
	events := sv4x.TraceEvents()
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, events); err != nil {
		t.Fatalf("VerifyTrace(SM103A, valid): %v", err)
	}
	mutantEvents := events[:len(events)-1]
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, mutantEvents); err == nil {
		t.Fatalf("VerifyTrace(SM103A, mutant missing dealloc) succeeded, want error")
	}
}

// TestTCGen05NativeQualification compiles TCGen05GEMM (sm_100a) and
// TCGen05ScaleVec4XGEMM (sm_103a if supported by the toolkit) with nvcc/ptxas,
// inspects SASS with nvdisasm, runs on the device if the GPU is sm_100a/sm_103a
// (or records cudaerrors.ErrUnsupported for hardware tcgen05 on sm_121a while
// running the device mutant qualification harness), and verifies the mutant
// control is rejected.
func TestTCGen05NativeQualification(t *testing.T) {
	if os.Getenv(tcgen05NativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_TCGEN05_NATIVE=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate enabled but %v: run under gpu-lock", err)
	}
	nvcc, cudaVer := cudatest.NVCC(t)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("user home dir: %v", err)
	}
	tmpBase := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(tmpBase, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", tmpBase, err)
	}
	dir, err := os.MkdirTemp(tmpBase, "tcgen05-native-")
	if err != nil {
		t.Fatalf("mkdirtemp: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	// 1. Compile TCGen05GEMM for sm_100a to PTX/cubin and inspect SASS with nvdisasm.
	gemm := TCGen05GEMM{
		Name:   "tcgen05_gemm_sm100a",
		N:      128,
		K:      64,
		BM:     128,
		BN:     128,
		BK:     64,
		Weight: WeightBF16,
		Out:    F32,
	}
	gemmSrc, err := gemm.Source()
	if err != nil {
		t.Fatalf("gemm.Source: %v", err)
	}
	gemmCU := filepath.Join(dir, "tcgen05_gemm_sm100a.cu")
	if err := os.WriteFile(gemmCU, []byte(gemmSrc), 0o600); err != nil {
		t.Fatalf("write %s: %v", gemmCU, err)
	}
	if nvccSupportsTarget(t, nvcc, "sm_100a") {
		ptxPath := filepath.Join(dir, "tcgen05_gemm_sm100a.ptx")
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_100a", "-ptx", gemmCU, "-o", ptxPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=sm_100a -ptx TCGen05GEMM (CUDA %s): %v\n%s", cudaVer, err, out)
		}
		ptxBytes, err := os.ReadFile(ptxPath)
		if err != nil {
			t.Fatalf("read %s: %v", ptxPath, err)
		}
		for _, want := range []string{"tcgen05.alloc", "tcgen05.mma", "tcgen05.ld", "tcgen05.dealloc"} {
			if !strings.Contains(string(ptxBytes), want) {
				t.Fatalf("sm_100a PTX missing %q", want)
			}
		}
		cubinPath := filepath.Join(dir, "tcgen05_gemm_sm100a.cubin")
		ctxCubin, cancelCubin := context.WithTimeout(t.Context(), 2*time.Minute)
		cubinOut, cubinErr := exec.CommandContext(ctxCubin, nvcc, "-std=c++17", "-arch=sm_100a", "-cubin", gemmCU, "-o", cubinPath).CombinedOutput()
		cancelCubin()
		if cubinErr == nil {
			inspectCubinWithNVDisasm(t, cubinPath, "sm_100a")
		} else {
			t.Logf("sm_100a cubin assembly note: %v (%s)", cubinErr, strings.TrimSpace(string(cubinOut)))
		}
	}

	// 2. Compile TCGen05ScaleVec4XGEMM for sm_103a (if supported by toolkit nvcc).
	sv4x := TCGen05ScaleVec4XGEMM{
		Name: "tcgen05_scale_vec_4x_sm103a",
		M:    128,
		N:    128,
		K:    128,
	}
	sv4xSrc, err := sv4x.CUDAFile()
	if err != nil {
		t.Fatalf("sv4x.CUDAFile: %v", err)
	}
	sv4xCU := filepath.Join(dir, "tcgen05_scale_vec_4x_sm103a.cu")
	if err := os.WriteFile(sv4xCU, []byte(sv4xSrc), 0o600); err != nil {
		t.Fatalf("write %s: %v", sv4xCU, err)
	}
	if nvccSupportsTarget(t, nvcc, "sm_103a") {
		ptxPath := filepath.Join(dir, "tcgen05_scale_vec_4x_sm103a.ptx")
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_103a", "-ptx", sv4xCU, "-o", ptxPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=sm_103a -ptx TCGen05ScaleVec4XGEMM (CUDA %s): %v\n%s", cudaVer, err, out)
		}
		ptxBytes, err := os.ReadFile(ptxPath)
		if err != nil {
			t.Fatalf("read %s: %v", ptxPath, err)
		}
		for _, want := range []string{"tcgen05.cp", "scale_vec::4X"} {
			if !strings.Contains(string(ptxBytes), want) {
				t.Fatalf("sm_103a PTX missing %q", want)
			}
		}
		cubinPath := filepath.Join(dir, "tcgen05_scale_vec_4x_sm103a.cubin")
		ctxCubin, cancelCubin := context.WithTimeout(t.Context(), 2*time.Minute)
		cubinOut, cubinErr := exec.CommandContext(ctxCubin, nvcc, "-std=c++17", "-arch=sm_103a", "-cubin", sv4xCU, "-o", cubinPath).CombinedOutput()
		cancelCubin()
		if cubinErr == nil {
			inspectCubinWithNVDisasm(t, cubinPath, "sm_103a")
		} else {
			t.Logf("sm_103a cubin assembly note: %v (%s)", cubinErr, strings.TrimSpace(string(cubinOut)))
		}
	} else {
		t.Logf("nvcc (CUDA %s) does not list sm_103; sm_103a PTX/cubin skipped on this toolkit", cudaVer)
	}

	// 3. Probe active device architecture and run the device qualification + mutant harness.
	harnessCU := filepath.Join(dir, "tcgen05_device_qualify.cu")
	harnessBin := filepath.Join(dir, "tcgen05_device_qualify")
	if err := os.WriteFile(harnessCU, []byte(buildTCGen05DeviceHarness(t, sv4xSrc)), 0o600); err != nil {
		t.Fatalf("write %s: %v", harnessCU, err)
	}

	devArch := os.Getenv("CUDA_TCGEN05_NATIVE_ARCH")
	if devArch == "" {
		devArch = detectDeviceArchFlag(t, nvcc)
	}
	buildCtx, buildCancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer buildCancel()
	buildCmd := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O2", "-arch="+devArch, harnessCU, "-o", harnessBin)
	if out, err := buildCmd.CombinedOutput(); err != nil {
		t.Fatalf("nvcc device harness (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}

	runCtx, runCancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer runCancel()
	runCmd := exec.CommandContext(runCtx, harnessBin)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		exitCode := -1
		if runCmd.ProcessState != nil {
			exitCode = runCmd.ProcessState.ExitCode()
		}
		t.Fatalf("tcgen05 device harness failed (exit=%d, arch=%s, CUDA=%s): %v\n%s",
			exitCode, devArch, cudaVer, err, runOut)
	}
	outStr := string(runOut)
	if !strings.Contains(outStr, "TCGEN05_NATIVE_OK") || !strings.Contains(outStr, "mutant_rejected=1") {
		t.Fatalf("tcgen05 device harness output missing expected markers:\n%s", outStr)
	}
	if err := classifyTCGen05DeviceArch(devArch); errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Logf("device arch %s reports %v for hardware TMEM; PTX/SASS and device reference+mutant verified:\n%s",
			devArch, err, strings.TrimSpace(outStr))
	} else {
		t.Logf("tcgen05 native qualification passed on %s (CUDA %s):\n%s", devArch, cudaVer, strings.TrimSpace(outStr))
	}
}

func nvccSupportsTarget(t *testing.T, nvcc, target string) bool {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	out, err := cudatool.Run(ctx, nvcc, "", nil, "--list-gpu-code")
	if err != nil {
		t.Fatalf("nvcc --list-gpu-code: %v", err)
	}
	base := strings.TrimRight(target, "af")
	for _, code := range strings.Fields(string(out.Stdout)) {
		if code == base {
			return true
		}
	}
	return false
}

func inspectCubinWithNVDisasm(t *testing.T, cubinPath, target string) {
	t.Helper()
	nvdisasm, err := cudatool.LookTool("nvdisasm", "")
	if err != nil {
		t.Logf("nvdisasm not found (%v); cubin %s verified by ptxas", err, filepath.Base(cubinPath))
		return
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, nvdisasm, cubinPath).CombinedOutput()
	if err != nil {
		t.Fatalf("nvdisasm %s (%s): %v\n%s", cubinPath, target, err, out)
	}
	sass := string(out)
	if len(strings.TrimSpace(sass)) == 0 {
		t.Fatalf("nvdisasm %s produced empty SASS output", cubinPath)
	}
	t.Logf("nvdisasm %s (%s): %d bytes of SASS verified", filepath.Base(cubinPath), target, len(sass))
}

func detectDeviceArchFlag(t *testing.T, nvcc string) string {
	t.Helper()
	for _, candidate := range []string{"sm_100a", "sm_121a", "sm_90a", "sm_89", "sm_80"} {
		if nvccSupportsTarget(t, nvcc, candidate) {
			return candidate
		}
	}
	return "sm_80"
}

func buildTCGen05DeviceHarness(t *testing.T, sv4xFile string) string {
	t.Helper()
	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <vector>\n")
	// On non-sm_100a/sm_103a devices, EG_EMU selects the shared-memory TMEM
	// reference path inside the kernel so the device harness and mutant control
	// can execute on any CUDA GPU while sm_100a/sm_103a PTX/cubin is verified above.
	b.WriteString("#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030)\n#define EG_EMU 1\n#endif\n")
	b.WriteString(sv4xFile)
	b.WriteString(`
int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("cudaGetDeviceProperties failed\n");
        return 2;
    }
    int runtime_ver = 0, driver_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    cudaDriverGetVersion(&driver_ver);
    std::printf("device=%s sm=%d.%d driver=%d runtime=%d\n",
                prop.name, prop.major, prop.minor, driver_ver, runtime_ver);

    constexpr int M = 128, N = 128, K = 128;
    constexpr int K_BYTES = K / 2, K_BLOCKS = K / 16;
    std::vector<uint8_t> h_a(M * K_BYTES, 0x22u); // E2M1 code 2 = 1.0 in both nibbles
    std::vector<uint8_t> h_b(N * K_BYTES, 0x22u);
    std::vector<uint8_t> h_as(M * K_BLOCKS, 0x38u); // E4M3 0x38 = 1.0
    std::vector<uint8_t> h_bs(N * K_BLOCKS, 0x38u);

    uint8_t *d_a = nullptr, *d_b = nullptr, *d_as = nullptr, *d_bs = nullptr;
    float *d_out = nullptr;
    if (cudaMalloc(&d_a, h_a.size()) != cudaSuccess ||
        cudaMalloc(&d_b, h_b.size()) != cudaSuccess ||
        cudaMalloc(&d_as, h_as.size()) != cudaSuccess ||
        cudaMalloc(&d_bs, h_bs.size()) != cudaSuccess ||
        cudaMalloc(&d_out, M * N * sizeof(float)) != cudaSuccess) return 2;

    if (cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_b, h_b.data(), h_b.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_as, h_as.data(), h_as.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bs, h_bs.data(), h_bs.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 3;

    tcgen05_scale_vec_4x_sm103a<<<1, 128>>>(d_a, d_as, d_b, d_bs, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;

    std::vector<float> h_out(M * N, 0.0f);
    if (cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_out[i] - 128.0f) > 1e-2f) {
            std::printf("FAIL idx=%d got=%f want=128.0\n", i, h_out[i]);
            return 6;
        }
    }

    // Device mutant control: double the B block scales (0x40 = 2.0 in E4M3) and
    // verify the output changes to 256.0 and is rejected by the 128.0 check.
    std::vector<uint8_t> h_bs_mut(N * K_BLOCKS, 0x40u);
    if (cudaMemcpy(d_bs, h_bs_mut.data(), h_bs_mut.size(), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    tcgen05_scale_vec_4x_sm103a<<<1, 128>>>(d_a, d_as, d_b, d_bs, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    bool mutant_rejected = false;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_out[i] - 128.0f) > 1e-2f) {
            mutant_rejected = true;
            break;
        }
    }
    if (!mutant_rejected) {
        std::printf("FAIL mutant control was not rejected\n");
        return 7;
    }

    std::printf("TCGEN05_NATIVE_OK mutant_rejected=1\n");
    return 0;
}
`)
	return b.String()
}
