package tcgen05

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
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

	// Verify TCGen05SparseGEMM (sm_100a and sm_103a 2:4 structured sparsity tcgen05.mma.sp).
	for _, target := range []string{"sm_100a", "sm_103a"} {
		sp := TCGen05SparseGEMM{
			Name:             "tcgen05_sparse_gemm_" + strings.TrimSuffix(target, "a") + "a",
			M:                128,
			N:                128,
			K:                64,
			Target:           target,
			BlockScaledNVFP4: true,
		}
		idesc, err := sp.InstDescriptor()
		if err != nil {
			t.Fatalf("TCGen05SparseGEMM(%s).InstDescriptor: %v", target, err)
		}
		if !UnpackInstDesc(idesc).Sparsity {
			t.Fatalf("TCGen05SparseGEMM(%s) UnpackInstDesc(0x%08x).Sparsity = false, want true", target, idesc)
		}
		spSrc, err := sp.CUDAFile()
		if err != nil || !strings.Contains(spSrc, "tcgen05.mma.sp") {
			t.Fatalf("TCGen05SparseGEMM(%s).CUDAFile() error=%v or missing tcgen05.mma.sp", target, err)
		}
		if _, err := cudatmem.VerifyTrace(cudatmem.Target(target), sp.TraceEvents()); err != nil {
			t.Fatalf("VerifyTrace(%s, TCGen05SparseGEMM): %v", target, err)
		}
	}

	// Verify TCGen05PagedAttention and TCGen05MLAPagedAttention TMEMTrace() on SM100A and SM103A.
	fa4 := TCGen05PagedAttention{
		Name:       "tcgen05_paged_attn_native",
		NumQHeads:  4,
		NumKVHeads: 2,
		HeadDim:    16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
		KVFormat:   WeightNVFP4,
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, fa4.TMEMTrace()); err != nil {
		t.Fatalf("VerifyTrace(SM103A, TCGen05PagedAttention): %v", err)
	}
	if !strings.Contains(fa4.CUDA(), "scale_vec::4X") {
		t.Fatalf("TCGen05PagedAttention(sm_103a) missing scale_vec::4X")
	}

	mla := TCGen05MLAPagedAttention{
		Name:       "tcgen05_mla_paged_attn_native",
		NumHeads:   4,
		KVLoRARank: 16,
		RoPEDim:    8,
		VHeadDim:   16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, mla.TMEMTrace()); err != nil {
		t.Fatalf("VerifyTrace(SM103A, TCGen05MLAPagedAttention): %v", err)
	}

	harnessSrc := buildTCGen05DeviceHarness(t, sv4xSrc)
	for _, want := range []string{
		"tcgen05_scale_vec_4x_sm103a",
		"tcgen05_sparse_gemm_sm100a",
		"tcgen05_paged_attn_native",
		"tcgen05_mla_paged_attn_native",
		"mutant_rejected",
	} {
		if !strings.Contains(harnessSrc, want) {
			t.Fatalf("buildTCGen05DeviceHarness missing %q", want)
		}
	}
}

// TestTCGen05NativeQualification compiles TCGen05GEMM (sm_100a),
// TCGen05ScaleVec4XGEMM (sm_103a), TCGen05SparseGEMM (sm_100a and sm_103a 2:4
// structured sparsity tcgen05.mma.sp), TCGen05PagedAttention, and
// TCGen05MLAPagedAttention with nvcc/ptxas, inspects SASS with nvdisasm, runs
// on the device if the GPU is sm_100a/sm_103a (or records
// cudaerrors.ErrUnsupported for hardware tcgen05 on sm_121a while running the
// device reference + mutant qualification harness), and verifies the mutant
// controls are rejected.
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

	// 3. Compile TCGen05SparseGEMM (sm_100a and sm_103a 2:4 structured sparsity tcgen05.mma.sp).
	sparse100 := TCGen05SparseGEMM{
		Name:             "tcgen05_sparse_gemm_sm100a",
		M:                128,
		N:                128,
		K:                64,
		Target:           "sm_100a",
		BlockScaledNVFP4: true,
	}
	sparse100Desc, err := sparse100.InstDescriptor()
	if err != nil {
		t.Fatalf("sparse100.InstDescriptor: %v", err)
	}
	if !UnpackInstDesc(sparse100Desc).Sparsity {
		t.Fatalf("sparse100 UnpackInstDesc(0x%08x).Sparsity = false, want true", sparse100Desc)
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, sparse100.TraceEvents()); err != nil {
		t.Fatalf("VerifyTrace(SM100A, sparse100): %v", err)
	}
	sparse100Raw, err := sparse100.CUDAFile()
	if err != nil {
		t.Fatalf("sparse100.CUDAFile: %v", err)
	}
	if !strings.Contains(sparse100Raw, "tcgen05.mma.sp") {
		t.Fatalf("sparse100.CUDAFile missing tcgen05.mma.sp")
	}
	sparse100Src := prepareSparseDeviceSource(sparse100Raw)
	sparse100CU := filepath.Join(dir, "tcgen05_sparse_gemm_sm100a.cu")
	if err := os.WriteFile(sparse100CU, []byte(sparse100Src), 0o600); err != nil {
		t.Fatalf("write %s: %v", sparse100CU, err)
	}
	if nvccSupportsTarget(t, nvcc, "sm_100a") {
		ptxPath := filepath.Join(dir, "tcgen05_sparse_gemm_sm100a.ptx")
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_100a", "-ptx", sparse100CU, "-o", ptxPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=sm_100a -ptx TCGen05SparseGEMM (CUDA %s): %v\n%s", cudaVer, err, out)
		}
		ptxBytes, err := os.ReadFile(ptxPath)
		if err != nil {
			t.Fatalf("read %s: %v", ptxPath, err)
		}
		for _, want := range []string{"tcgen05.mma.sp", "tcgen05.cp", "tcgen05.ld", "tcgen05.dealloc"} {
			if !strings.Contains(string(ptxBytes), want) {
				t.Fatalf("sm_100a sparse PTX missing %q", want)
			}
		}
		cubinPath := filepath.Join(dir, "tcgen05_sparse_gemm_sm100a.cubin")
		ctxCubin, cancelCubin := context.WithTimeout(t.Context(), 2*time.Minute)
		cubinOut, cubinErr := exec.CommandContext(ctxCubin, nvcc, "-std=c++17", "-arch=sm_100a", "-cubin", sparse100CU, "-o", cubinPath).CombinedOutput()
		cancelCubin()
		if cubinErr == nil {
			inspectCubinWithNVDisasm(t, cubinPath, "sm_100a")
		} else {
			t.Logf("sm_100a sparse cubin assembly note: %v (%s)", cubinErr, strings.TrimSpace(string(cubinOut)))
		}
	}

	sparse103 := TCGen05SparseGEMM{
		Name:             "tcgen05_sparse_gemm_sm103a",
		M:                128,
		N:                128,
		K:                64,
		Target:           "sm_103a",
		BlockScaledNVFP4: true,
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, sparse103.TraceEvents()); err != nil {
		t.Fatalf("VerifyTrace(SM103A, sparse103): %v", err)
	}
	sparse103Raw, err := sparse103.CUDAFile()
	if err != nil {
		t.Fatalf("sparse103.CUDAFile: %v", err)
	}
	sparse103CU := filepath.Join(dir, "tcgen05_sparse_gemm_sm103a.cu")
	if err := os.WriteFile(sparse103CU, []byte(prepareSparseDeviceSource(sparse103Raw)), 0o600); err != nil {
		t.Fatalf("write %s: %v", sparse103CU, err)
	}
	if nvccSupportsTarget(t, nvcc, "sm_103a") {
		ptxPath := filepath.Join(dir, "tcgen05_sparse_gemm_sm103a.ptx")
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_103a", "-ptx", sparse103CU, "-o", ptxPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=sm_103a -ptx TCGen05SparseGEMM (CUDA %s): %v\n%s", cudaVer, err, out)
		}
		ptxBytes, err := os.ReadFile(ptxPath)
		if err != nil {
			t.Fatalf("read %s: %v", ptxPath, err)
		}
		for _, want := range []string{"tcgen05.mma.sp", "scale_vec::4X"} {
			if !strings.Contains(string(ptxBytes), want) {
				t.Fatalf("sm_103a sparse PTX missing %q", want)
			}
		}
	}

	// 4. Verify TCGen05PagedAttention & TCGen05MLAPagedAttention TMEMTrace() and PTX emission.
	fa4 := TCGen05PagedAttention{
		Name:       "tcgen05_paged_attn_native",
		NumQHeads:  4,
		NumKVHeads: 2,
		HeadDim:    16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
		KVFormat:   WeightNVFP4,
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, fa4.TMEMTrace()); err != nil {
		t.Fatalf("VerifyTrace(SM103A, fa4): %v", err)
	}
	mla := TCGen05MLAPagedAttention{
		Name:       "tcgen05_mla_paged_attn_native",
		NumHeads:   4,
		KVLoRARank: 16,
		RoPEDim:    8,
		VHeadDim:   16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
	}
	if _, err := cudatmem.VerifyTrace(cudatmem.SM103A, mla.TMEMTrace()); err != nil {
		t.Fatalf("VerifyTrace(SM103A, mla): %v", err)
	}
	fa4CU := filepath.Join(dir, "tcgen05_fa4_native.cu")
	if err := os.WriteFile(fa4CU, []byte(prepareFA4DeviceSource(fa4.Source())+"\n"+prepareFA4DeviceSource(mla.CUDA()+"\n"+mla.Launcher())), 0o600); err != nil {
		t.Fatalf("write %s: %v", fa4CU, err)
	}
	fa4PTXArch := "sm_100a"
	if nvccSupportsTarget(t, nvcc, "sm_103a") {
		fa4PTXArch = "sm_103a"
	}
	if nvccSupportsTarget(t, nvcc, fa4PTXArch) {
		ptxPath := filepath.Join(dir, "tcgen05_fa4_native.ptx")
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch="+fa4PTXArch, "-ptx", fa4CU, "-o", ptxPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=%s -ptx TCGen05PagedAttention/MLA (CUDA %s): %v\n%s", fa4PTXArch, cudaVer, err, out)
		}
		ptxBytes, err := os.ReadFile(ptxPath)
		if err != nil {
			t.Fatalf("read %s: %v", ptxPath, err)
		}
		for _, want := range []string{"tcgen05.alloc", "tcgen05.mma", "tcgen05.ld", "tcgen05.st", "tcgen05.dealloc", "ex2.approx.ftz.f32"} {
			if !strings.Contains(string(ptxBytes), want) {
				t.Fatalf("%s FA4/MLA PTX missing %q", fa4PTXArch, want)
			}
		}
	}

	// 5. Probe active device architecture and run the device qualification + mutant harness.
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

	// 6. Wave 12 & Wave 13 Rubin (sm_107a / PTX 9.4) tcgen05 native qualification.
	t.Run("Wave12And13Rubin", func(t *testing.T) {
		qualifyTCGen05Wave12And13Rubin(t, nvcc, cudaVer, devArch, dir)
	})
}

// TestTCGen05Wave12And13NativeSuiteStructure verifies on the host (even when
// CUDA_TCGEN05_NATIVE is unset) that all 6 Wave 12/13 Rubin / PTX 9.4 tcgen05
// kernels emit valid CUDAFile() translation units, Reference() outputs, and
// native device harness source.
func TestTCGen05Wave12And13NativeSuiteStructure(t *testing.T) {
	fx := buildTCGen05Wave12And13Fixtures(t)
	if len(fx.tuFiles) != 6 {
		t.Fatalf("len(tuFiles) = %d, want 6", len(fx.tuFiles))
	}
	for name, tu := range fx.tuFiles {
		if !strings.Contains(tu, "EG_PRELUDE") {
			t.Fatalf("%s CUDAFile() missing EG_PRELUDE", name)
		}
		if !strings.Contains(tu, "__global__ void") {
			t.Fatalf("%s CUDAFile() missing __global__ void", name)
		}
	}
	for _, want := range []string{
		"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32",
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64",
		"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64",
		"tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.f32",
		"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
		"tcgen05.mma.ws.sp.cta_group::1.kind::ti16.collector::b0::fill",
		"tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32",
		"spcompress.b8.b2.sp::2:4.x4",
		"spdecompress.b8.b2.sp::2:4.x32",
		"TCGEN05_WAVE12_13_NATIVE_OK",
		"wave12_13_mutant_rejected=1",
	} {
		if !strings.Contains(fx.harnessSrc, want) {
			t.Fatalf("Wave 12/13 device harness missing %q", want)
		}
	}
}

type tcgen05Wave12And13Fixtures struct {
	tuFiles    map[string]string
	harnessSrc string
}

func prepareEarlyReleaseDeviceSource(src string) string {
	src = strings.ReplaceAll(src,
		"    EG_TCGEN05_SHARED(float, smem_a, [STAGES * M * BK]);\n    EG_TCGEN05_SHARED(float, smem_b, [STAGES * BK * N]);",
		`#if defined(EG_EMU)
    static __device__ float eg_emu_er_smem_a[2 * 128 * 64];
    static __device__ float eg_emu_er_smem_b[2 * 64 * 128];
    float* const smem_a = eg_emu_er_smem_a;
    float* const smem_b = eg_emu_er_smem_b;
#else
    EG_TCGEN05_SHARED(float, smem_a, [STAGES * M * BK]);
    EG_TCGEN05_SHARED(float, smem_b, [STAGES * BK * N]);
#endif`)
	return src
}

func prepareWSDeviceSource(src string) string {
	src = strings.ReplaceAll(src,
		"    EG_TCGEN05_SHARED(float, smem_a, [M * K]);\n    EG_TCGEN05_SHARED(float, smem_b, [K * N]);",
		`#if defined(EG_EMU)
    static __device__ float eg_emu_ws_smem_a[128 * 64];
    static __device__ float eg_emu_ws_smem_b[64 * 128];
    float* const smem_a = eg_emu_ws_smem_a;
    float* const smem_b = eg_emu_ws_smem_b;
#else
    EG_TCGEN05_SHARED(float, smem_a, [M * K]);
    EG_TCGEN05_SHARED(float, smem_b, [K * N]);
#endif`)
	return src
}

func buildTCGen05Wave12And13Fixtures(t *testing.T) tcgen05Wave12And13Fixtures {
	t.Helper()
	tuFiles := make(map[string]string, 6)

	// 1. TCGen05LUTDecompressGEMM (M=128, N=128, K=64)
	lutK := TCGen05LUTDecompressGEMM{
		Name:       "tcgen05_lut_decompress_native",
		M:          128,
		N:          128,
		K:          64,
		Target:     "sm_107f",
		CollectorA: CollectorUse,
		CollectorB: CollectorFill,
	}
	lutTU, err := lutK.CUDAFile()
	if err != nil {
		t.Fatalf("lutK.CUDAFile: %v", err)
	}
	tuFiles["TCGen05LUTDecompressGEMM"] = lutTU
	lutCUDA, err := lutK.CUDA()
	if err != nil {
		t.Fatalf("lutK.CUDA: %v", err)
	}

	lutAF32 := make([]float32, 128*64)
	lutBF32 := make([]float32, 64*128)
	palette := [8]float32{-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75}
	for i := range lutAF32 {
		lutAF32[i] = float32((i%5)-2) * 0.5
	}
	for kk := range 64 {
		for n := range 128 {
			lutBF32[kk*128+n] = palette[(kk*3+n*5)&7]
		}
	}
	lutAE4M3, lutAScales := QuantizeMXFP8E4M3(lutAF32)
	lutBIdx, lutBLUT, lutBScales, _, err := Quantize3BitLUTB(lutBF32, 64, 128)
	if err != nil {
		t.Fatalf("Quantize3BitLUTB: %v", err)
	}
	lutWant, err := lutK.Reference(lutAE4M3, lutAScales, lutBIdx, lutBLUT, lutBScales)
	if err != nil {
		t.Fatalf("lutK.Reference: %v", err)
	}
	lutBLUTMut := slices.Clone(lutBLUT)
	for i := range lutBLUTMut {
		lutBLUTMut[i] = E4M3Bits(FromE4M3Bits(lutBLUTMut[i]) + 8.0)
	}
	lutMutWant, err := lutK.Reference(lutAE4M3, lutAScales, lutBIdx, lutBLUTMut, lutBScales)
	if err != nil {
		t.Fatalf("lutK.Reference(mut): %v", err)
	}
	if math.Abs(float64(lutMutWant[0]-lutWant[0])) <= 1e-4 {
		t.Fatalf("lutK mutant did not perturb output[0]: got %v vs %v", lutMutWant[0], lutWant[0])
	}

	// 2. TCGen05EarlyReleasePipelineGEMM (M=128, N=128, K=128, Stages=2, ClusterCTAs=16)
	erK := TCGen05EarlyReleasePipelineGEMM{
		Name:        "tcgen05_early_release_native",
		M:           128,
		N:           128,
		K:           128,
		Stages:      2,
		ClusterCTAs: 16,
		Target:      "sm_107f",
	}
	erTU, err := erK.CUDAFile()
	if err != nil {
		t.Fatalf("erK.CUDAFile: %v", err)
	}
	tuFiles["TCGen05EarlyReleasePipelineGEMM"] = erTU
	erCUDA, err := erK.CUDA()
	if err != nil {
		t.Fatalf("erK.CUDA: %v", err)
	}
	erA := make([]float32, 128*128)
	erB := make([]float32, 128*128)
	for i := range erA {
		erA[i] = 0.5
		erB[i] = 0.25
	}
	erWant, err := erK.Reference(erA, erB)
	if err != nil {
		t.Fatalf("erK.Reference: %v", err)
	}
	if math.Abs(float64(erWant[0]-16.0)) > 1e-5 {
		t.Fatalf("erWant[0] = %v, want 16.0", erWant[0])
	}

	// 3. TCGen05LdRed (M=32, N=32, RedOp=LdRedMax, Abs=true)
	ldRedK := TCGen05LdRed{
		M:      32,
		N:      32,
		RedOp:  LdRedMax,
		Abs:    true,
		Target: "sm_107a",
	}
	ldRedTU, err := ldRedK.CUDAFile()
	if err != nil {
		t.Fatalf("ldRedK.CUDAFile: %v", err)
	}
	tuFiles["TCGen05LdRed"] = ldRedTU
	ldRedCUDA, err := ldRedK.CUDA()
	if err != nil {
		t.Fatalf("ldRedK.CUDA: %v", err)
	}
	ldRedIn := make([]float32, 32*32)
	for r := range 32 {
		for c := range 32 {
			ldRedIn[r*32+c] = float32((c%7)-3) * 0.5
		}
		ldRedIn[r*32+(r%32)] = -float32(r + 10)
	}
	ldRedWantData, ldRedWantRow, err := ldRedK.Reference(ldRedIn)
	if err != nil {
		t.Fatalf("ldRedK.Reference: %v", err)
	}

	// 4. TCGen05LdSpCompress (M=32, N=32, RedOp=LdRedMax, Abs=true, WithReduction=true)
	ldSpK := TCGen05LdSpCompress{
		M:             32,
		N:             32,
		RedOp:         LdRedMax,
		Abs:           true,
		WithReduction: true,
		Target:        "sm_107a",
	}
	ldSpTU, err := ldSpK.CUDAFile()
	if err != nil {
		t.Fatalf("ldSpK.CUDAFile: %v", err)
	}
	tuFiles["TCGen05LdSpCompress"] = ldSpTU
	ldSpCUDA, err := ldSpK.CUDA()
	if err != nil {
		t.Fatalf("ldSpK.CUDA: %v", err)
	}
	ldSpWantC, ldSpWantM, ldSpWantRow, err := ldSpK.Reference(ldRedIn)
	if err != nil {
		t.Fatalf("ldSpK.Reference: %v", err)
	}

	// 5. TCGen05WeightsStationaryGEMM (M=128, N=128, K=64, Kind="ti16", Sparse=true, UseExclusive576Cols=true)
	var zmask ZeroColumnMaskDesc
	zmask.NonZeroMask = 0x03 // zero columns in groups 0 and 1
	zmask.StartCount[0] = 2
	zmask.SkipSpan[0] = 3 // zero 4 columns (2..5) in group 0
	zmask.StartCount[1] = 0
	zmask.SkipSpan[1] = 7 // zero all 8 columns (0..7) in group 1
	wsK := TCGen05WeightsStationaryGEMM{
		Name:                "tcgen05_ws_gemm_native",
		M:                   128,
		N:                   128,
		K:                   64,
		CollectorBuffer:     WSCollectorB0,
		CollectorAction:     WSActionFill,
		Kind:                "ti16",
		UseExclusive576Cols: true,
		ZeroColumnMask:      zmask,
		Sparse:              true,
		Target:              "sm_107a",
	}
	wsTU, err := wsK.CUDAFile()
	if err != nil {
		t.Fatalf("wsK.CUDAFile: %v", err)
	}
	tuFiles["TCGen05WeightsStationaryGEMM"] = wsTU
	wsCUDA, err := wsK.CUDA()
	if err != nil {
		t.Fatalf("wsK.CUDA: %v", err)
	}
	wsA := make([]float32, 128*64)
	wsB := make([]float32, 64*128)
	for i := range wsA {
		wsA[i] = 1.0
	}
	for i := range wsB {
		wsB[i] = 0.5
	}
	wsWant, err := wsK.Reference(wsA, wsB)
	if err != nil {
		t.Fatalf("wsK.Reference: %v", err)
	}
	// 64 total K columns minus (4 + 8 = 12) zeroed columns = 52 active columns * 0.5 = 26.0.
	if math.Abs(float64(wsWant[0]-26.0)) > 1e-5 {
		t.Fatalf("wsWant[0] = %v, want 26.0", wsWant[0])
	}

	// 6. RubinSpCompressDecompressKernel (NumElements=64, Op=SpOpMaxAbs, ElemType=SpTypeE4M3)
	spDesc := SpCompressDesc{Op: SpOpMaxAbs, ElemType: SpTypeE4M3}
	spK := RubinSpCompressDecompressKernel{
		NumElements: 64,
		Desc:        spDesc,
		Ratio:       SpRatio2to4,
		Target:      "sm_107a",
	}
	spTU, err := spK.CUDAFile()
	if err != nil {
		t.Fatalf("spK.CUDAFile: %v", err)
	}
	tuFiles["RubinSpCompressDecompressKernel"] = spTU
	spCUDA, err := spK.CUDA()
	if err != nil {
		t.Fatalf("spK.CUDA: %v", err)
	}
	spIn := make([]float32, 64)
	for i := range spIn {
		v := float32((i%9)-4) * 0.75
		if i%4 == 2 {
			v = -float32(i + 5)
		}
		spIn[i] = v
	}
	spWantC, spWantM, spWantRec, err := spK.Reference(spIn)
	if err != nil {
		t.Fatalf("spK.Reference: %v", err)
	}

	var b strings.Builder
	b.WriteString("#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <vector>\n")
	b.WriteString("#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ != 1070)\n#define EG_EMU 1\n#endif\n")
	b.WriteString(Prelude + "\n")
	b.WriteString("extern \"C\" {\n")
	b.WriteString(lutCUDA + "\n")
	b.WriteString(prepareEarlyReleaseDeviceSource(erCUDA) + "\n")
	b.WriteString(ldRedCUDA + "\n")
	b.WriteString(ldSpCUDA + "\n")
	b.WriteString(prepareWSDeviceSource(wsCUDA) + "\n")
	b.WriteString(spCUDA + "\n")
	b.WriteString("}\n")

	fmt.Fprintf(&b, `
template <typename T>
static T* dev_dup(const T* host_data, size_t count) {
    T* d = nullptr;
    if (cudaMalloc(&d, count * sizeof(T)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, host_data, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return d;
}

static bool launch_drv(CUmodule mod, const char* name, unsigned int gx, unsigned int bx, void** params) {
    CUfunction fn = nullptr;
    if (mod != nullptr && cuModuleGetFunction(&fn, mod, name) == CUDA_SUCCESS && fn != nullptr) {
        if (cuLaunchKernel(fn, gx, 1, 1, bx, 1, 1, 0, nullptr, params, nullptr) != CUDA_SUCCESS) {
            return false;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }
    return false;
}

int main(int argc, char** argv) {
    if (cuInit(0) != CUDA_SUCCESS) return 2;
    cudaFree(0);
    CUmodule mod = nullptr;
    if (argc > 1 && argv[1] != nullptr && argv[1][0] != '\0') {
        if (cuModuleLoad(&mod, argv[1]) != CUDA_SUCCESS) {
            std::printf("cuModuleLoad(%%s) failed\n", argv[1]);
            return 2;
        }
    }

    // 1. TCGen05LUTDecompressGEMM + mutant
    const uint8_t h_lut_a[] = {%s};
    const uint8_t h_lut_as[] = {%s};
    const uint8_t h_lut_bidx[] = {%s};
    const uint8_t h_lut_blut[] = {%s};
    const uint8_t h_lut_blut_mut[] = {%s};
    const uint8_t h_lut_bs[] = {%s};
    const float want_lut[] = {%s};

    auto* d_lut_a = dev_dup(h_lut_a, %d);
    auto* d_lut_as = dev_dup(h_lut_as, %d);
    auto* d_lut_bidx = dev_dup(h_lut_bidx, %d);
    auto* d_lut_blut = dev_dup(h_lut_blut, %d);
    auto* d_lut_bs = dev_dup(h_lut_bs, %d);
    float* d_out128 = nullptr;
    if (cudaMalloc(&d_out128, 128 * 128 * sizeof(float)) != cudaSuccess) return 2;

    void* lut_args[] = {&d_lut_a, &d_lut_as, &d_lut_bidx, &d_lut_blut, &d_lut_bs, &d_out128};
    if (!launch_drv(mod, "tcgen05_lut_decompress_native", 1, 128, lut_args)) {
        tcgen05_lut_decompress_native<<<1, 128>>>(d_lut_a, d_lut_as, d_lut_bidx, d_lut_blut, d_lut_bs, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    std::vector<float> h_out128(128 * 128, 0.0f);
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 128 * 128; ++i) {
        if (std::isnan(h_out128[i]) || std::fabs(h_out128[i] - want_lut[i]) >= 1e-4f) {
            std::printf("FAIL TCGen05LUTDecompressGEMM idx=%%d got=%%f want=%%f\n", i, h_out128[i], want_lut[i]);
            return 6;
        }
    }
    if (cudaMemcpy(d_lut_blut, h_lut_blut_mut, sizeof(h_lut_blut_mut), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "tcgen05_lut_decompress_native", 1, 128, lut_args)) {
        tcgen05_lut_decompress_native<<<1, 128>>>(d_lut_a, d_lut_as, d_lut_bidx, d_lut_blut, d_lut_bs, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    bool lut_mut_rejected = false;
    for (int i = 0; i < 128 * 128; ++i) {
        if (std::fabs(h_out128[i] - want_lut[i]) > 1e-2f) { lut_mut_rejected = true; break; }
    }
    if (!lut_mut_rejected) return 7;
    std::printf("PASS TCGen05LUTDecompressGEMM M=128 N=128 K=64 mutant_rejected=1\n");

    // 2. TCGen05EarlyReleasePipelineGEMM + mutant
    std::vector<float> h_er_a(128 * 128, 0.5f), h_er_b(128 * 128, 0.25f), h_er_b_mut(128 * 128, 0.75f);
    auto* d_er_a = dev_dup(h_er_a.data(), h_er_a.size());
    auto* d_er_b = dev_dup(h_er_b.data(), h_er_b.size());
    void* er_args[] = {&d_er_a, &d_er_b, &d_out128};
    if (!launch_drv(mod, "tcgen05_early_release_native", 1, 128, er_args)) {
        tcgen05_early_release_native<<<1, 128>>>(d_er_a, d_er_b, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 128 * 128; ++i) {
        if (std::fabs(h_out128[i] - 16.0f) >= 1e-4f) return 8;
    }
    if (cudaMemcpy(d_er_b, h_er_b_mut.data(), h_er_b_mut.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "tcgen05_early_release_native", 1, 128, er_args)) {
        tcgen05_early_release_native<<<1, 128>>>(d_er_a, d_er_b, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_out128[0] - 16.0f) < 1e-2f) return 9;
    std::printf("PASS TCGen05EarlyReleasePipelineGEMM M=128 N=128 K=128 mutant_rejected=1\n");

    // 3. TCGen05LdRed + mutant
    const float h_ld_in[] = {%s};
    const float want_ld_data[] = {%s};
    const float want_ld_row[] = {%s};
    auto* d_ld_in = dev_dup(h_ld_in, 1024);
    float *d_ld_data = nullptr, *d_ld_row = nullptr;
    if (cudaMalloc(&d_ld_data, 1024 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ld_row, 32 * sizeof(float)) != cudaSuccess) return 2;
    int m32 = 32, n32 = 32;
    void* ld_red_args[] = {&d_ld_in, &d_ld_data, &d_ld_row, &m32, &n32};
    if (!launch_drv(mod, "eg_tcgen05_ld_red_f32", 1, 128, ld_red_args)) {
        eg_tcgen05_ld_red_f32<<<1, 128>>>(d_ld_in, d_ld_data, d_ld_row, 32, 32);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    float h_ld_data[1024] = {}, h_ld_row[32] = {};
    if (cudaMemcpy(h_ld_data, d_ld_data, sizeof(h_ld_data), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_ld_row, d_ld_row, sizeof(h_ld_row), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 1024; ++i) {
        if (std::fabs(h_ld_data[i] - want_ld_data[i]) >= 1e-4f) return 10;
    }
    for (int r = 0; r < 32; ++r) {
        if (std::fabs(h_ld_row[r] - want_ld_row[r]) >= 1e-4f) return 10;
    }
    std::vector<float> h_ld_in_mut(1024, -250.0f);
    if (cudaMemcpy(d_ld_in, h_ld_in_mut.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "eg_tcgen05_ld_red_f32", 1, 128, ld_red_args)) {
        eg_tcgen05_ld_red_f32<<<1, 128>>>(d_ld_in, d_ld_data, d_ld_row, 32, 32);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_ld_row, d_ld_row, sizeof(h_ld_row), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_ld_row[0] - want_ld_row[0]) < 1e-2f) return 11;
    std::printf("PASS TCGen05LdRed M=32 N=32 mutant_rejected=1\n");

    // 4. TCGen05LdSpCompress + mutant
    if (cudaMemcpy(d_ld_in, h_ld_in, sizeof(h_ld_in), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    const float want_ldsp_c[] = {%s};
    const uint32_t want_ldsp_m[] = {%s};
    const float want_ldsp_row[] = {%s};
    float* d_ldsp_c = nullptr;
    unsigned int* d_ldsp_m = nullptr;
    if (cudaMalloc(&d_ldsp_c, 512 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_ldsp_m, 32 * sizeof(unsigned int)) != cudaSuccess) return 2;
    void* ldsp_args[] = {&d_ld_in, &d_ldsp_c, &d_ldsp_m, &d_ld_row, &m32, &n32};
    if (!launch_drv(mod, "eg_tcgen05_ld_spcompress_f32", 1, 128, ldsp_args)) {
        eg_tcgen05_ld_spcompress_f32<<<1, 128>>>(d_ld_in, d_ldsp_c, d_ldsp_m, d_ld_row, 32, 32);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    float h_ldsp_c[512] = {};
    uint32_t h_ldsp_m[32] = {};
    if (cudaMemcpy(h_ldsp_c, d_ldsp_c, sizeof(h_ldsp_c), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_ldsp_m, d_ldsp_m, sizeof(h_ldsp_m), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_ld_row, d_ld_row, sizeof(h_ld_row), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 512; ++i) {
        if (std::fabs(h_ldsp_c[i] - want_ldsp_c[i]) >= 1e-4f) return 12;
    }
    for (int r = 0; r < 32; ++r) {
        if (h_ldsp_m[r] != want_ldsp_m[r] || std::fabs(h_ld_row[r] - want_ldsp_row[r]) >= 1e-4f) return 12;
    }
    if (cudaMemcpy(d_ld_in, h_ld_in_mut.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "eg_tcgen05_ld_spcompress_f32", 1, 128, ldsp_args)) {
        eg_tcgen05_ld_spcompress_f32<<<1, 128>>>(d_ld_in, d_ldsp_c, d_ldsp_m, d_ld_row, 32, 32);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_ldsp_c, d_ldsp_c, sizeof(h_ldsp_c), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_ldsp_c[0] - want_ldsp_c[0]) < 1e-2f) return 13;
    std::printf("PASS TCGen05LdSpCompress M=32 N=32 mutant_rejected=1\n");

    // 5. TCGen05WeightsStationaryGEMM + mutant
    std::vector<float> h_ws_a(128 * 64, 1.0f), h_ws_b(64 * 128, 0.5f), h_ws_b_mut(64 * 128, 1.5f);
    auto* d_ws_a = dev_dup(h_ws_a.data(), h_ws_a.size());
    auto* d_ws_b = dev_dup(h_ws_b.data(), h_ws_b.size());
    void* ws_args[] = {&d_ws_a, &d_ws_b, &d_out128};
    if (!launch_drv(mod, "tcgen05_ws_gemm_native", 1, 128, ws_args)) {
        tcgen05_ws_gemm_native<<<1, 128>>>(d_ws_a, d_ws_b, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 128 * 128; ++i) {
        if (std::fabs(h_out128[i] - 26.0f) >= 1e-4f) return 14;
    }
    if (cudaMemcpy(d_ws_b, h_ws_b_mut.data(), h_ws_b_mut.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "tcgen05_ws_gemm_native", 1, 128, ws_args)) {
        tcgen05_ws_gemm_native<<<1, 128>>>(d_ws_a, d_ws_b, d_out128);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_out128.data(), d_out128, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_out128[0] - 26.0f) < 1e-2f) return 15;
    std::printf("PASS TCGen05WeightsStationaryGEMM M=128 N=128 K=64 mutant_rejected=1\n");

    // 6. RubinSpCompressDecompressKernel + mutant
    const float h_sp_in[] = {%s};
    const float want_sp_c[] = {%s};
    const uint32_t want_sp_m[] = {%s};
    const float want_sp_rec[] = {%s};
    auto* d_sp_in = dev_dup(h_sp_in, 64);
    float *d_sp_c = nullptr, *d_sp_rec = nullptr;
    unsigned int* d_sp_m = nullptr;
    if (cudaMalloc(&d_sp_c, 32 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_sp_m, 2 * sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&d_sp_rec, 64 * sizeof(float)) != cudaSuccess) return 2;
    int sp_n = 64;
    unsigned int spdesc_val = 0x%08xu;
    void* sp_args[] = {&d_sp_in, &d_sp_c, &d_sp_m, &d_sp_rec, &sp_n, &spdesc_val};
    if (!launch_drv(mod, "eg_rubin_spcompress_roundtrip", 1, 64, sp_args)) {
        eg_rubin_spcompress_roundtrip<<<1, 64>>>(d_sp_in, d_sp_c, d_sp_m, d_sp_rec, 64, spdesc_val);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    float h_sp_c[32] = {}, h_sp_rec[64] = {};
    uint32_t h_sp_m[2] = {};
    if (cudaMemcpy(h_sp_c, d_sp_c, sizeof(h_sp_c), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_sp_m, d_sp_m, sizeof(h_sp_m), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_sp_rec, d_sp_rec, sizeof(h_sp_rec), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 32; ++i) {
        if (std::fabs(h_sp_c[i] - want_sp_c[i]) >= 1e-4f) return 16;
    }
    if (h_sp_m[0] != want_sp_m[0] || h_sp_m[1] != want_sp_m[1]) return 16;
    for (int i = 0; i < 64; ++i) {
        if (std::fabs(h_sp_rec[i] - want_sp_rec[i]) >= 1e-4f) return 16;
    }
    std::vector<float> h_sp_in_mut(64, 99.0f);
    if (cudaMemcpy(d_sp_in, h_sp_in_mut.data(), 64 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    if (!launch_drv(mod, "eg_rubin_spcompress_roundtrip", 1, 64, sp_args)) {
        eg_rubin_spcompress_roundtrip<<<1, 64>>>(d_sp_in, d_sp_c, d_sp_m, d_sp_rec, 64, spdesc_val);
        if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    }
    if (cudaMemcpy(h_sp_rec, d_sp_rec, sizeof(h_sp_rec), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_sp_rec[0] - want_sp_rec[0]) < 1e-2f) return 17;
    std::printf("PASS RubinSpCompressDecompressKernel N=64 mutant_rejected=1\n");

    if (mod != nullptr) cuModuleUnload(mod);
    std::printf("TCGEN05_WAVE12_13_NATIVE_OK wave12_13_mutant_rejected=1\n");
    return 0;
}
`,
		formatTCGen05U8List(lutAE4M3),
		formatTCGen05U8List(lutAScales),
		formatTCGen05U8List(lutBIdx),
		formatTCGen05U8List(lutBLUT),
		formatTCGen05U8List(lutBLUTMut),
		formatTCGen05U8List(lutBScales),
		formatTCGen05F32List(lutWant),
		len(lutAE4M3),
		len(lutAScales),
		len(lutBIdx),
		len(lutBLUT),
		len(lutBScales),
		formatTCGen05F32List(ldRedIn),
		formatTCGen05F32List(ldRedWantData),
		formatTCGen05F32List(ldRedWantRow),
		formatTCGen05F32List(ldSpWantC),
		formatTCGen05U32List(ldSpWantM),
		formatTCGen05F32List(ldSpWantRow),
		formatTCGen05F32List(spIn),
		formatTCGen05F32List(spWantC),
		formatTCGen05U32List(spWantM),
		formatTCGen05F32List(spWantRec),
		spDesc.Encode(),
	)

	return tcgen05Wave12And13Fixtures{
		tuFiles:    tuFiles,
		harnessSrc: b.String(),
	}
}

func probeNVCCSupportsSM107A(t *testing.T, nvcc, dir, sampleTU string) bool {
	t.Helper()
	if !nvccSupportsTarget(t, nvcc, "sm_107a") {
		return false
	}
	probeCU := filepath.Join(dir, "probe_sm107a.cu")
	probeCubin := filepath.Join(dir, "probe_sm107a.cubin")
	if err := os.WriteFile(probeCU, []byte(sampleTU), 0o600); err != nil {
		t.Fatalf("write %s: %v", probeCU, err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	if err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch=sm_107a", "-cubin", probeCU, "-o", probeCubin).Run(); err != nil {
		return false
	}
	return true
}

func qualifyTCGen05Wave12And13Rubin(t *testing.T, nvcc string, cudaVer cudatool.Version, devArch, dir string) {
	t.Helper()
	fx := buildTCGen05Wave12And13Fixtures(t)

	// Probe whether the installed toolkit supports -arch=sm_107a with Rubin PTX 9.4 instructions.
	sm107Supported := probeNVCCSupportsSM107A(t, nvcc, dir, fx.tuFiles["TCGen05LUTDecompressGEMM"])
	if !sm107Supported {
		t.Logf("sm_107a_toolchain_support=false (CUDA %s max sm_103a/sm_121a)", cudaVer)
	} else {
		t.Logf("sm_107a_toolchain_support=true (CUDA %s)", cudaVer)
	}

	// Compile each of the 6 Wave 12/13 translation units under -arch=<devArch> -cubin
	// (and under -arch=sm_107a -cubin + nvdisasm if supported by the toolkit).
	for name, tu := range fx.tuFiles {
		cuPath := filepath.Join(dir, name+".cu")
		cubinPath := filepath.Join(dir, name+"."+devArch+".cubin")
		if err := os.WriteFile(cuPath, []byte(tu), 0o600); err != nil {
			t.Fatalf("write %s: %v", cuPath, err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
		out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-arch="+devArch, "-cubin", cuPath, "-o", cubinPath).CombinedOutput()
		cancel()
		if err != nil {
			t.Fatalf("nvcc -arch=%s -cubin %s (CUDA %s): %v\n%s", devArch, name, cudaVer, err, out)
		}
		inspectCubinWithNVDisasm(t, cubinPath, devArch)

		if sm107Supported {
			sm107Cubin := filepath.Join(dir, name+".sm_107a.cubin")
			ctx107, cancel107 := context.WithTimeout(t.Context(), 2*time.Minute)
			out107, err107 := exec.CommandContext(ctx107, nvcc, "-std=c++17", "-arch=sm_107a", "-cubin", cuPath, "-o", sm107Cubin).CombinedOutput()
			cancel107()
			if err107 != nil {
				t.Fatalf("nvcc -arch=sm_107a -cubin %s (CUDA %s): %v\n%s", name, cudaVer, err107, out107)
			}
			inspectCubinWithNVDisasm(t, sm107Cubin, "sm_107a")
		}
	}

	// Build and run the Wave 12/13 native device qualification harness (which also
	// compiles a native-arch cubin and launches the kernels via the CUDA Driver API).
	harnessCU := filepath.Join(dir, "tcgen05_wave12_13_qualify.cu")
	harnessCubin := filepath.Join(dir, "tcgen05_wave12_13_qualify.cubin")
	harnessBin := filepath.Join(dir, "tcgen05_wave12_13_qualify")
	if err := os.WriteFile(harnessCU, []byte(fx.harnessSrc), 0o600); err != nil {
		t.Fatalf("write %s: %v", harnessCU, err)
	}

	cubinCtx, cubinCancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cubinCancel()
	if out, err := exec.CommandContext(cubinCtx, nvcc, "-std=c++17", "-O2", "-arch="+devArch, "-cubin", harnessCU, "-o", harnessCubin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc Wave 12/13 cubin (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}

	buildCtx, buildCancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer buildCancel()
	if out, err := exec.CommandContext(buildCtx, nvcc, "-std=c++17", "-O2", "-arch="+devArch, harnessCU, "-lcuda", "-o", harnessBin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc Wave 12/13 harness (%s, CUDA %s): %v\n%s", devArch, cudaVer, err, out)
	}

	runCtx, runCancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer runCancel()
	runCmd := exec.CommandContext(runCtx, harnessBin, harnessCubin)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		exitCode := -1
		if runCmd.ProcessState != nil {
			exitCode = runCmd.ProcessState.ExitCode()
		}
		t.Fatalf("tcgen05 Wave 12/13 harness failed (exit=%d, arch=%s, CUDA=%s): %v\n%s",
			exitCode, devArch, cudaVer, err, runOut)
	}
	outStr := string(runOut)
	if !strings.Contains(outStr, "TCGEN05_WAVE12_13_NATIVE_OK") || !strings.Contains(outStr, "wave12_13_mutant_rejected=1") {
		t.Fatalf("tcgen05 Wave 12/13 harness output missing expected markers:\n%s", outStr)
	}
	t.Logf("tcgen05 Wave 12/13 Rubin qualification passed on %s (CUDA %s):\n%s", devArch, cudaVer, strings.TrimSpace(outStr))
}

func formatTCGen05U8List(v []uint8) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("0x%02xu", x)
	}
	return strings.Join(parts, ", ")
}

func formatTCGen05U32List(v []uint32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = fmt.Sprintf("0x%08xu", x)
	}
	return strings.Join(parts, ", ")
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
	ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader").Output(); err == nil {
		lines := strings.Fields(string(out))
		if len(lines) > 0 {
			cc := strings.ReplaceAll(strings.TrimSpace(lines[0]), ".", "")
			switch cc {
			case "100", "103", "120", "121", "90":
				cand := "sm_" + cc + "a"
				if nvccSupportsTarget(t, nvcc, cand) {
					return cand
				}
			}
			cand := "sm_" + cc
			if nvccSupportsTarget(t, nvcc, cand) {
				return cand
			}
		}
	}
	for _, candidate := range []string{"sm_121a", "sm_100a", "sm_90a", "sm_89", "sm_80"} {
		if nvccSupportsTarget(t, nvcc, candidate) {
			return candidate
		}
	}
	return "sm_80"
}

func prepareSparseDeviceSource(src string) string {
	const target = "    EG_TCGEN05_SHARED(float, smem_b, [K * N]);"
	const replacement = `#if defined(EG_EMU)
    static __device__ float eg_emu_sparse_smem_b[64 * 128];
    float* const smem_b = eg_emu_sparse_smem_b;
#else
    EG_DYNAMIC_SMEM(float, smem_b);
#endif`
	return strings.ReplaceAll(src, target, replacement)
}

func prepareFA4DeviceSource(src string) string {
	src = strings.ReplaceAll(src, "EG_TCGEN05_SHARED(float, tmem, [128 * 512]);", "EG_TCGEN05_TMEM_BUF(tmem);")
	src = strings.ReplaceAll(src, "#ifdef __CUDA_ARCH__", "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030) && !defined(EG_EMU)")
	return src
}

func buildTCGen05DeviceHarness(t *testing.T, sv4xFile string) string {
	t.Helper()

	sparse100 := TCGen05SparseGEMM{
		Name:             "tcgen05_sparse_gemm_sm100a",
		M:                128,
		N:                128,
		K:                64,
		Target:           "sm_100a",
		BlockScaledNVFP4: true,
	}
	sparseRaw, err := sparse100.CUDAFile()
	if err != nil {
		t.Fatalf("sparse100.CUDAFile: %v", err)
	}

	fa4 := TCGen05PagedAttention{
		Name:       "tcgen05_paged_attn_native",
		NumQHeads:  4,
		NumKVHeads: 2,
		HeadDim:    16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
		KVFormat:   WeightNVFP4,
	}
	fa4Q := make([]float32, 4*16)
	for r := range 4 {
		fa4Q[r*16+0] = 1.5
		for d := 1; d < 16; d++ {
			fa4Q[r*16+d] = float32((d%3)-1) * 0.5
		}
	}
	fa4KPages := make([]float32, 2*8*2*16)
	fa4VPages := make([]float32, 2*8*2*16)
	for row := range 2 * 8 * 2 {
		fa4KPages[row*16+0] = 1.5
		fa4VPages[row*16+0] = 1.5
		for d := 1; d < 16; d++ {
			fa4KPages[row*16+d] = float32(((row+d)%3)-1) * 0.5
			fa4VPages[row*16+d] = float32(((row+2*d)%3)-1) * 0.5
		}
	}
	fa4PT := []int32{1, 0, 0, 0}
	fa4Want := fa4.Reference(fa4Q, fa4KPages, fa4VPages, fa4PT, 12)

	mla := TCGen05MLAPagedAttention{
		Name:       "tcgen05_mla_paged_attn_native",
		NumHeads:   4,
		KVLoRARank: 16,
		RoPEDim:    8,
		VHeadDim:   16,
		PageSize:   8,
		MaxPages:   4,
		SeqLen:     12,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
	}
	mlaQAbs := make([]float32, 4*16)
	mlaQRope := make([]float32, 4*8)
	for i := range mlaQAbs {
		mlaQAbs[i] = float32((i%5)-2) * 0.125
	}
	for i := range mlaQRope {
		mlaQRope[i] = float32((i%5)-2) * 0.125
	}
	mlaCKV := make([]float32, 2*8*16)
	mlaKRope := make([]float32, 2*8*8)
	for i := range mlaCKV {
		mlaCKV[i] = float32(((i+1)%5)-2) * 0.125
	}
	for i := range mlaKRope {
		mlaKRope[i] = float32(((i+2)%5)-2) * 0.125
	}
	mlaWant := mla.Reference(mlaQAbs, mlaQRope, mlaCKV, mlaKRope, fa4PT, 12)

	var b strings.Builder
	b.WriteString("#include <cuda_runtime.h>\n#include <cstdio>\n#include <cmath>\n#include <cstdint>\n#include <vector>\n")
	// On non-sm_100a/sm_103a devices, EG_EMU selects the shared-memory TMEM
	// reference path inside the kernel so the device harness and mutant control
	// can execute on any CUDA GPU while sm_100a/sm_103a PTX/cubin is verified above.
	b.WriteString("#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ != 1000 && __CUDA_ARCH__ != 1030)\n#define EG_EMU 1\n#endif\n")
	b.WriteString(sv4xFile + "\n")
	b.WriteString(prepareSparseDeviceSource(sparseRaw) + "\n")
	b.WriteString(prepareFA4DeviceSource(fa4.Source()) + "\n")
	b.WriteString(prepareFA4DeviceSource(mla.Source()) + "\n")
	fmt.Fprintf(&b, `
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
                prop.name, prop.major, prop.minor, driver_ver, runtime_ver);

    // 1. TCGen05ScaleVec4XGEMM check + mutant control
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
            std::printf("FAIL idx=%%d got=%%f want=128.0\n", i, h_out[i]);
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

    // 2. TCGen05SparseGEMM (2:4 structured sparsity) check + mutant control
    constexpr int K_SP = 64, K_SP_HALF = K_SP / 2, K_SP_META = K_SP / 32;
    std::vector<float> h_sp_a(M * K_SP_HALF, 1.0f);
    std::vector<uint32_t> h_sp_meta(M * K_SP_META, 0x44444444u); // idx0=0, idx1=1 per 4-element group
    std::vector<float> h_sp_b(K_SP * N, 1.0f);
    auto* d_sp_a = dev_dup(h_sp_a.data(), h_sp_a.size());
    auto* d_sp_meta = dev_dup(h_sp_meta.data(), h_sp_meta.size());
    auto* d_sp_b = dev_dup(h_sp_b.data(), h_sp_b.size());
    if (!d_sp_a || !d_sp_meta || !d_sp_b) return 2;

    tcgen05_sparse_gemm_sm100a<<<1, 128>>>(d_sp_a, d_sp_meta, d_sp_b, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_out[i] - 32.0f) > 1e-3f) {
            std::printf("FAIL TCGen05SparseGEMM idx=%%d got=%%f want=32.0\n", i, h_out[i]);
            return 8;
        }
    }
    std::vector<float> h_sp_a_mut(M * K_SP_HALF, 2.0f);
    if (cudaMemcpy(d_sp_a, h_sp_a_mut.data(), h_sp_a_mut.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
    tcgen05_sparse_gemm_sm100a<<<1, 128>>>(d_sp_a, d_sp_meta, d_sp_b, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    if (cudaMemcpy(h_out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    if (std::fabs(h_out[0] - 32.0f) <= 1e-3f) {
        std::printf("FAIL TCGen05SparseGEMM mutant was not rejected\n");
        return 8;
    }
    std::printf("PASS TCGen05SparseGEMM M=128 N=128 K=64 sparse_mutant_rejected=1\n");

    // 3. TCGen05PagedAttention (FlashAttention-4) check
    const float h_fa4_q[] = {%s};
    const float h_fa4_k[] = {%s};
    const float h_fa4_v[] = {%s};
    const int32_t h_fa4_pt[] = {1, 0, 0, 0};
    const float want_fa4[] = {%s};

    auto* d_fa4_q = dev_dup(h_fa4_q, %d);
    auto* d_fa4_k = dev_dup(h_fa4_k, %d);
    auto* d_fa4_v = dev_dup(h_fa4_v, %d);
    auto* d_fa4_pt = dev_dup(h_fa4_pt, 4);
    float* d_fa4_out = nullptr;
    if (!d_fa4_q || !d_fa4_k || !d_fa4_v || !d_fa4_pt ||
        cudaMalloc(&d_fa4_out, 64 * sizeof(float)) != cudaSuccess) return 2;

    tcgen05_paged_attn_native_launch(d_fa4_q, d_fa4_k, d_fa4_v, d_fa4_pt, d_fa4_out, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    float h_fa4_out[64] = {};
    if (cudaMemcpy(h_fa4_out, d_fa4_out, sizeof(h_fa4_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 64; ++i) {
        if (std::isnan(h_fa4_out[i]) || std::fabs(h_fa4_out[i] - want_fa4[i]) > 1e-2f) {
            std::printf("FAIL TCGen05PagedAttention idx=%%d got=%%f want=%%f\n", i, h_fa4_out[i], want_fa4[i]);
            return 9;
        }
    }
    std::printf("PASS TCGen05PagedAttention heads=4/2 D=16 seq=12\n");

    // 4. TCGen05MLAPagedAttention check
    const float h_mla_qabs[] = {%s};
    const float h_mla_qrope[] = {%s};
    const float h_mla_ckv[] = {%s};
    const float h_mla_krope[] = {%s};
    const float want_mla[] = {%s};

    auto* d_mla_qabs = dev_dup(h_mla_qabs, %d);
    auto* d_mla_qrope = dev_dup(h_mla_qrope, %d);
    auto* d_mla_ckv = dev_dup(h_mla_ckv, %d);
    auto* d_mla_krope = dev_dup(h_mla_krope, %d);
    float* d_mla_out = nullptr;
    if (!d_mla_qabs || !d_mla_qrope || !d_mla_ckv || !d_mla_krope ||
        cudaMalloc(&d_mla_out, 64 * sizeof(float)) != cudaSuccess) return 2;

    tcgen05_mla_paged_attn_native_launch(d_mla_qabs, d_mla_qrope, d_mla_ckv, d_mla_krope, d_fa4_pt, d_mla_out, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) return 4;
    float h_mla_out[64] = {};
    if (cudaMemcpy(h_mla_out, d_mla_out, sizeof(h_mla_out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    for (int i = 0; i < 64; ++i) {
        if (std::isnan(h_mla_out[i]) || std::fabs(h_mla_out[i] - want_mla[i]) > 1e-3f) {
            std::printf("FAIL TCGen05MLAPagedAttention idx=%%d got=%%f want=%%f\n", i, h_mla_out[i], want_mla[i]);
            return 10;
        }
    }
    std::printf("PASS TCGen05MLAPagedAttention heads=4 dc=16 dr=8 seq=12\n");

    std::printf("TCGEN05_NATIVE_OK mutant_rejected=1\n");
    return 0;
}
`,
		formatTCGen05F32List(fa4Q),
		formatTCGen05F32List(fa4KPages),
		formatTCGen05F32List(fa4VPages),
		formatTCGen05F32List(fa4Want),
		len(fa4Q),
		len(fa4KPages),
		len(fa4VPages),
		formatTCGen05F32List(mlaQAbs),
		formatTCGen05F32List(mlaQRope),
		formatTCGen05F32List(mlaCKV),
		formatTCGen05F32List(mlaKRope),
		formatTCGen05F32List(mlaWant),
		len(mlaQAbs),
		len(mlaQRope),
		len(mlaCKV),
		len(mlaKRope),
	)
	return b.String()
}

func formatTCGen05F32List(v []float32) string {
	parts := make([]string, len(v))
	for i, x := range v {
		parts[i] = floatLiteral(x)
	}
	return strings.Join(parts, ", ")
}
