package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/driver"
)

// TestGEMMHelper is the target TestCaptureGEMMs captures: one cuBLASLt
// matmul, D = A^T B in float32, bound through purego as laya binds it.
func TestGEMMHelper(t *testing.T) {
	if os.Getenv("CUDAPROF_GEMM_HELPER") != "1" {
		t.Skip("run by TestCaptureGEMMs")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(what string, err error) {
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	check("init", driver.Init(0))
	dev, err := driver.DeviceGet(0)
	check("device", err)
	ctx, err := dev.PrimaryCtxRetain()
	check("retain context", err)
	check("set context", ctx.SetCurrent())

	const m, n, k = 64, 48, 32
	alloc := func(elems int) driver.DevicePtr {
		p, err := driver.MemAlloc[float32](elems)
		check("alloc", err)
		return p
	}
	a, b, d := alloc(k*m), alloc(k*n), alloc(m*n)

	h, err := cublaslt.Create()
	check("cublasLtCreate", err)
	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	check("matmul desc", err)
	op := cublaslt.CublasOpT
	check("transa", desc.SetAttribute(cublaslt.MatmulDescTransa, uintptr(unsafe.Pointer(&op)), uint(unsafe.Sizeof(op))))
	layout := func(rows, cols int) cublaslt.MatrixLayout {
		l, err := cublaslt.NewMatrixLayout(cublaslt.R32F, uint64(rows), uint64(cols), int64(rows))
		check("layout", err)
		return l
	}
	aDesc, bDesc, dDesc := layout(k, m), layout(k, n), layout(m, n)
	alpha, beta := cublaslt.Float32Scalar(1), cublaslt.Float32Scalar(0)
	check("matmul", h.Matmul(desc, alpha, a, aDesc, b, bDesc, beta, d, dDesc, d, dDesc, nil, 0, 0, 0))
	check("synchronize", driver.CtxSynchronize())
}

// TestCaptureGEMMs captures TestGEMMHelper with --gemms and checks that its
// one call was recorded as made and joined to the kernel it launched. Like
// the other GPU validations it runs only with CUDAPROF_GPU_VALIDATE=1, under
// gpu-lock.
func TestCaptureGEMMs(t *testing.T) {
	tools := buildSpinTools(t)
	out := filepath.Join(tools.dir, "gemms.pprof")
	opts, err := parseCaptureArgs([]string{"--inject", tools.agent, "--gemms", "--finalize=exit-group", "-o", out, "--", os.Args[0], "-test.run=^TestGEMMHelper$", "-test.v"})
	if err != nil {
		t.Fatal(err)
	}
	var stdout bytes.Buffer
	opts.Stdout, opts.Stderr = &stdout, &stdout
	opts.Env = append(os.Environ(), "CUDAPROF_GEMM_HELPER=1")
	opts.Timeout = 2 * time.Minute
	result, err := runCapture(context.Background(), opts)
	if err != nil {
		t.Fatalf("capture: %v\n%s", err, stdout.Bytes())
	}
	if result.TargetExitCode != 0 || !result.CaptureSuccess {
		t.Fatalf("capture: exit %d, capture_success=%v: %s\n%s", result.TargetExitCode, result.CaptureSuccess, result.CaptureError, stdout.Bytes())
	}
	// The logger must not write to the target's stdout.
	if strings.Contains(stdout.String(), "Adesc=") {
		t.Errorf("cuBLASLt log reached the target's output:\n%s", stdout.Bytes())
	}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	c, err := readGEMMCensus(out, compareOptions{phase: -1})
	if err != nil {
		t.Fatal(err)
	}
	if c.Calls != 1 || c.Joined != 1 || len(c.Rows) != 1 {
		t.Fatalf("census: %d calls, %d joined, %d rows; want 1, 1, 1: %+v", c.Calls, c.Joined, len(c.Rows), c.Rows)
	}
	r := c.Rows[0]
	if r.Shape() != "64x48x32" || r.Op != "TN" || r.Types != "R_32F,R_32F->R_32F" || r.Beta != "0" || r.API != "cublasLtMatmul" {
		t.Errorf("row = %s %s %s beta=%s %s; want 64x48x32 TN R_32F,R_32F->R_32F beta=0 cublasLtMatmul", r.Shape(), r.Op, r.Types, r.Beta, r.API)
	}
	if r.KernelNanos <= 0 || r.Kernel() == "-" {
		t.Errorf("row has no kernel: %+v", r)
	}
	t.Logf("%s %s %s algo %s: %s, %d ns", r.Shape(), r.Op, r.Types, r.Algo, r.Kernel(), r.KernelNanos)
}
