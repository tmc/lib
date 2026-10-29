package train

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const trainNativeEnv = "CUDA_EGRAPH_TRAIN_NATIVE"

type trainNativeGateClosedError struct{}

func (trainNativeGateClosedError) Error() string {
	return "train: native gate is closed: CUDA_EGRAPH_TRAIN_NATIVE=1 unset, no device examined"
}

func (trainNativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkTrainNativeGate(getenv func(string) string) error {
	if getenv(trainNativeEnv) != "1" {
		return trainNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("train: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// grpoLSELine is the pass-2 softmax line of FusedGRPOChunk. The mutant
// drops its log-sum-exp subtraction, so dZ uses unnormalized exp(z).
const (
	grpoLSELine   = "const float p = expf(z * INV_T - lse);"
	grpoNoLSELine = "const float p = expf(z * INV_T);"
)

func grpoMutant(src string) (string, error) {
	if strings.Count(src, grpoLSELine) != 1 {
		return "", fmt.Errorf("grpo mutant: %q not found exactly once", grpoLSELine)
	}
	return strings.Replace(src, grpoLSELine, grpoNoLSELine, 1), nil
}

// trainSpinSource is the compute kernel that runs while the C2C overlap
// probe copies device memory to pinned host memory.
const trainSpinSource = `__global__ void train_c2c_spin(float* out, int iters) {
    float x = (float)threadIdx.x * 1e-3f;
    for (int i = 0; i < iters; ++i) x = fmaf(x, 0.999999f, 1e-6f);
    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
`

const muonNativeRows, muonNativeCols = 12, 20

// trainNativeSources returns the NVRTC sources ([Prelude] plus kernels, no
// host launchers) and the kernel names each defines.
func trainNativeSources() (map[string]string, map[string][]string, error) {
	f := newGRPOKernelFixture()
	grpo, err := f.kernel("grpo_native").CUDA()
	if err != nil {
		return nil, nil, err
	}
	mutant, err := grpoMutant(grpo)
	if err != nil {
		return nil, nil, err
	}
	muon, err := MuonNewtonSchulz{Name: "muon_native", Rows: muonNativeRows, Cols: muonNativeCols}.CUDA()
	if err != nil {
		return nil, nil, err
	}
	src := map[string]string{
		"grpo":        Prelude + grpo,
		"grpo_mutant": Prelude + mutant,
		"muon":        Prelude + muon,
		"spin":        trainSpinSource,
	}
	names := map[string][]string{
		"grpo":        {"grpo_native"},
		"grpo_mutant": {"grpo_native"},
		"muon":        {"muon_native_normalize", "muon_native_gram", "muon_native_poly", "muon_native_update", "muon_native_copy"},
		"spin":        {"train_c2c_spin"},
	}
	return src, names, nil
}

// TestTrainNativeClosedGateControl checks the gate on the host, that the
// native sources build, and that the GRPO mutant the native test must
// reject is rejected by the same check under cudahostemu.
func TestTrainNativeClosedGateControl(t *testing.T) {
	if os.Getenv(trainNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_EGRAPH_TRAIN_NATIVE != 1")
	}
	if err := checkTrainNativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("closed gate = %v, want ErrHardwarePending", err)
	}
	presenceOnly := func(k string) string {
		switch k {
		case trainNativeEnv, "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := checkTrainNativeGate(presenceOnly); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("GPU_LOCK_HELD=1 gate = %v, want lock error", err)
	}
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("train-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	held := func(k string) string {
		switch k {
		case trainNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		}
		return ""
	}
	if err := checkTrainNativeGate(held); err != nil {
		t.Fatalf("gate with inherited lock: %v", err)
	}

	src, names, err := trainNativeSources()
	if err != nil {
		t.Fatal(err)
	}
	for key, ns := range names {
		for _, n := range ns {
			if !strings.Contains(src[key], " "+n+"(") {
				t.Errorf("%s source does not define %s", key, n)
			}
		}
		if strings.Contains(strings.TrimPrefix(src[key], Prelude), "_launch(") {
			t.Errorf("%s source includes a host launcher", key)
		}
	}
	if src["grpo"] == src["grpo_mutant"] {
		t.Fatal("mutant source equals baseline")
	}

	f := newGRPOKernelFixture()
	full, err := f.kernel("grpo_mut").Emit()
	if err != nil {
		t.Fatal(err)
	}
	full, err = grpoMutant(full)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.check(f.runEmu(t, "grpo_mut", full)); err == nil {
		t.Fatal("emulated LSE-dropped mutant passed the GRPO check")
	} else {
		t.Logf("emulated mutant rejected: %v", err)
	}
}

// TestTrainNativeQualification NVRTC-compiles FusedGRPOChunk, its
// LSE-dropped mutant and MuonNewtonSchulz for the device, runs them and
// compares with the host references at the emulator tests' tolerances. The
// mutant must fail. It then measures how much a device-to-host copy into
// pinned memory overlaps a compute kernel on another stream, checks the
// copied data, and logs the overlap without asserting a threshold.
func TestTrainNativeQualification(t *testing.T) {
	if os.Getenv(trainNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_TRAIN_NATIVE=1 required")
	}
	if err := checkTrainNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	target := os.Getenv("CUDA_EGRAPH_TRAIN_NATIVE_ARCH")
	if target == "" {
		target = fmt.Sprintf("sm_%d%da", major, minor)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	rtcMajor, rtcMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device sm_%d%d target=%s driver=%d nvrtc=%d.%d", major, minor, target, driverVersion, rtcMajor, rtcMinor)

	src, names, err := trainNativeSources()
	if err != nil {
		t.Fatal(err)
	}
	stream := newTrainStream(t, dev)

	t.Run("grpo", func(t *testing.T) {
		k := compileTrainNative(t, dev, target, src["grpo"], names["grpo"])
		if err := runGRPONative(t, dev, stream, k["grpo_native"]); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("grpo_mutant_rejected", func(t *testing.T) {
		k := compileTrainNative(t, dev, target, src["grpo_mutant"], names["grpo_mutant"])
		err := runGRPONative(t, dev, stream, k["grpo_native"])
		if err == nil {
			t.Fatal("LSE-dropped mutant passed the GRPO check")
		}
		t.Logf("mutant_rejected=1: %v", err)
	})
	t.Run("muon", func(t *testing.T) {
		k := compileTrainNative(t, dev, target, src["muon"], names["muon"])
		for _, steps := range []int{4, 5} {
			if err := runMuonNative(t, dev, stream, k, steps); err != nil {
				t.Errorf("steps=%d: %v", steps, err)
			}
		}
	})
	t.Run("c2c_overlap", func(t *testing.T) {
		k := compileTrainNative(t, dev, target, src["spin"], names["spin"])
		runOverlapNative(t, dev, k["train_c2c_spin"])
	})
}

func newTrainStream(t *testing.T, dev *cuda.Device) *cuda.Stream {
	t.Helper()
	s, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		// A failed drain leaves the stream and its buffers retained.
		if err := s.Synchronize(ctx); err != nil {
			t.Errorf("drain stream: %v", err)
			return
		}
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	return s
}

func compileTrainNative(t *testing.T, dev *cuda.Device, target, src string, names []string) map[string]*cuda.Kernel {
	t.Helper()
	cm, err := nvrtcx.Compile(src, nvrtcx.WithName("train_native.cu"), nvrtcx.WithPhysicalArchTarget(target),
		nvrtcx.WithNameExpressions(names...), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtc compile for %s: %v", target, err)
	}
	mod, err := cm.LoadModule(dev)
	if err != nil {
		t.Fatalf("load module: %v", err)
	}
	t.Cleanup(func() {
		if err := mod.Close(); err != nil {
			t.Error(err)
		}
	})
	kernels := make(map[string]*cuda.Kernel, len(names))
	for _, n := range names {
		lowered, ok := cm.LoweredName(n)
		if !ok {
			t.Fatalf("no lowered name for %s", n)
		}
		k, err := mod.Function(lowered)
		if err != nil {
			t.Fatalf("function %s: %v", n, err)
		}
		kernels[n] = k
	}
	return kernels
}

// devBuf allocates a device copy of host, freed when the test ends.
func devBuf[T any](t *testing.T, dev *cuda.Device, host []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(host))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := s.CopyFromHost(host); err != nil {
		t.Fatal(err)
	}
	return s
}

func syncStream(s *cuda.Stream) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	return s.Synchronize(ctx)
}

// kernelArgs returns the kernel parameter array for values, each of which must
// be a pointer to the argument.
func kernelArgs(values ...any) []unsafe.Pointer {
	p := make([]unsafe.Pointer, len(values))
	for i, v := range values {
		switch v := v.(type) {
		case *driver.DevicePtr:
			p[i] = unsafe.Pointer(v)
		case *int32:
			p[i] = unsafe.Pointer(v)
		case *float32:
			p[i] = unsafe.Pointer(v)
		default:
			panic(fmt.Sprintf("kernelArgs: unsupported %T", v))
		}
	}
	return p
}

func runGRPONative(t *testing.T, dev *cuda.Device, s *cuda.Stream, k *cuda.Kernel) error {
	t.Helper()
	f := newGRPOKernelFixture()
	M, V, D := len(f.h), f.cfg.VocabSize, f.cfg.HiddenDim
	poisonF := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = math.Float32frombits(0x7fc00000)
		}
		return v
	}
	loss := devBuf(t, dev, poisonF(M))
	dH := devBuf(t, dev, poisonF(M*D))
	dW := devBuf(t, dev, make([]float32, V*D))
	H := devBuf(t, dev, flat32(f.h))
	W := devBuf(t, dev, flat32(f.w))
	tgt := devBuf(t, dev, f.target)
	old := devBuf(t, dev, f.old)
	ref := devBuf(t, dev, f.ref)
	adv := devBuf(t, dev, f.adv)
	ptrs := []driver.DevicePtr{loss.Ptr(), dH.Ptr(), dW.Ptr(), H.Ptr(), W.Ptr(), tgt.Ptr(), old.Ptr(), ref.Ptr(), adv.Ptr()}
	rows, scale := int32(M), 1/float32(M)
	a := kernelArgs(&ptrs[0], &ptrs[1], &ptrs[2], &ptrs[3], &ptrs[4], &ptrs[5], &ptrs[6], &ptrs[7], &ptrs[8], &rows, &scale)
	if err := k.Launch([3]uint32{uint32(M), 1, 1}, [3]uint32{64, 1, 1}, 0, s.Raw(), a); err != nil {
		t.Fatal(err)
	}
	if err := syncStream(s); err != nil {
		t.Fatal(err)
	}
	gotLoss, gotDH, gotDW := make([]float32, M), make([]float32, M*D), make([]float32, V*D)
	if err := errors.Join(loss.CopyToHost(gotLoss), dH.CopyToHost(gotDH), dW.CopyToHost(gotDW)); err != nil {
		t.Fatal(err)
	}
	return f.check(gotLoss, gotDH, gotDW)
}

// runMuonNative repeats MuonNewtonSchulz.Launcher's launch sequence.
func runMuonNative(t *testing.T, dev *cuda.Device, s *cuda.Stream, k map[string]*cuda.Kernel, steps int) error {
	t.Helper()
	const R, C = muonNativeRows, muonNativeCols
	cfg := MuonConfig{Steps: steps}
	g := illConditioned(3, R, C, 10)
	X := devBuf(t, dev, make([]float32, R*C))
	Xtmp := devBuf(t, dev, make([]float32, R*C))
	A := devBuf(t, dev, make([]float32, R*R))
	B := devBuf(t, dev, make([]float32, R*R))
	G := devBuf(t, dev, flat32(g))
	x, xt, pa, pb, pg := X.Ptr(), Xtmp.Ptr(), A.Ptr(), B.Ptr(), G.Ptr()
	launch := func(name string, grid, block uint32, a []unsafe.Pointer) {
		t.Helper()
		if err := k[name].Launch([3]uint32{grid, 1, 1}, [3]uint32{block, 1, 1}, 0, s.Raw(), a); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	gram, upd := uint32((R*R+127)/128), uint32((R*C+127)/128)
	launch("muon_native_normalize", 1, 256, kernelArgs(&x, &pg))
	cur, nxt := &x, &xt
	for range steps {
		launch("muon_native_gram", gram, 128, kernelArgs(&pa, cur))
		launch("muon_native_poly", gram, 128, kernelArgs(&pb, &pa))
		launch("muon_native_update", upd, 128, kernelArgs(nxt, cur, &pb))
		cur, nxt = nxt, cur
	}
	if cur != &x {
		launch("muon_native_copy", upd, 128, kernelArgs(&x, cur))
	}
	if err := syncStream(s); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, R*C)
	if err := X.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	return checkMuonOutput(got, g, cfg)
}

// runOverlapNative times a compute kernel on one stream and a
// device-to-host copy into pinned memory on another, alone and together,
// and logs overlap = (compute + copy − both) / min(compute, copy). Times
// are host wall clock around enqueue and synchronize, so they include
// launch latency. Every copy is checked against the uploaded pattern.
func runOverlapNative(t *testing.T, dev *cuda.Device, spin *cuda.Kernel) {
	const n = 32 << 20 // 128 MiB of uint32
	const blocks, threads = 48, 256
	pattern := make([]uint32, n)
	for i := range pattern {
		pattern[i] = uint32(i)*2654435761 ^ 0x5bd1e995
	}
	src := devBuf(t, dev, pattern)
	out := devBuf(t, dev, make([]float32, blocks*threads))
	var pinned cudamem.HostAllocation[uint32]
	if err := dev.WithContext(func() error {
		var err error
		pinned, err = cudamem.AllocHost[uint32](n)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.WithContext(pinned.Free); err != nil {
			t.Error(err)
		}
	})
	compute, copier := newTrainStream(t, dev), newTrainStream(t, dev)

	outPtr := out.Ptr()
	var iters int32
	startSpin := func() {
		t.Helper()
		if err := spin.Launch([3]uint32{blocks, 1, 1}, [3]uint32{threads, 1, 1}, 0, compute.Raw(), kernelArgs(&outPtr, &iters)); err != nil {
			t.Fatal(err)
		}
	}
	host := pinned.Slice()
	startCopy := func() {
		t.Helper()
		for i := range host {
			host[i] = 0xffffffff
		}
		if err := src.CopyToHostAsync(host, copier); err != nil {
			t.Fatal(err)
		}
	}
	checkCopy := func() {
		t.Helper()
		for i, v := range host {
			if v != pattern[i] {
				t.Fatalf("pinned copy word %d = %#x, want %#x", i, v, pattern[i])
			}
		}
	}
	timed := func(fn func()) time.Duration {
		t.Helper()
		start := time.Now()
		fn()
		if err := errors.Join(syncStream(compute), syncStream(copier)); err != nil {
			t.Fatal(err)
		}
		return time.Since(start)
	}

	copyTime := timed(startCopy)
	checkCopy()
	// Size the compute kernel to take about as long as the copy.
	iters = 1 << 14
	spinTime := timed(startSpin)
	scaled := int64(iters) * int64(copyTime) / max(1, int64(spinTime))
	iters = int32(min(max(scaled, 1<<10), 1<<26))

	best := [3]time.Duration{1 << 62, 1 << 62, 1 << 62}
	for range 3 {
		best[0] = min(best[0], timed(startSpin))
		best[1] = min(best[1], timed(startCopy))
		checkCopy()
		best[2] = min(best[2], timed(func() { startSpin(); startCopy() }))
		checkCopy()
	}
	tc, tx, both := best[0].Seconds(), best[1].Seconds(), best[2].Seconds()
	overlap := (tc + tx - both) / min(tc, tx)
	t.Logf("c2c_overlap: bytes=%d iters=%d compute=%.3fms copy=%.3fms (%.1f GB/s) both=%.3fms overlap_fraction=%.3f",
		4*n, iters, 1e3*tc, 1e3*tx, 4*float64(n)/tx/1e9, 1e3*both, overlap)
}
