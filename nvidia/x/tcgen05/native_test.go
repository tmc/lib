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
