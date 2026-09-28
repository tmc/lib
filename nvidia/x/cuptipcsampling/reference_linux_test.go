//go:build linux

package cuptipcsampling_test

import (
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestNativeStartStopMatchesReferenceSample(t *testing.T) {
	if os.Getenv("CUPTI_PC_REFERENCE_TEST") != "1" {
		t.Skip("CUPTI_PC_REFERENCE_TEST=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Skipf("native qualification pending: inherited gpu-lock required: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	root, cudaRoot := pcReferenceRoots(t)
	sample := buildPCReferenceSample(t, root, cudaRoot)
	if err := checkPCReferenceBinary(sample); err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0o700); err != nil {
		t.Fatal(err)
	}
	work, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cuptipcsampling-reference-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(work) })
	cmd := exec.Command(sample)
	cmd.Dir = work
	cmd.Env = pcWithLibraryPath(os.Environ(), filepath.Join(cudaRoot, "lib64"))
	refOutput, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reference sample %s: %v\n%s", sample, err, refOutput)
	}
	refTotal, err := parsePCReferenceTotal(string(refOutput))
	if err != nil {
		t.Fatalf("parse reference output: %v\n%s", err, refOutput)
	}
	sampleLib, err := pcSampleCUPTILibrary(sample)
	if err != nil {
		t.Fatal(err)
	}
	sampleVersion, err := pcCUPTIVersionAt(sampleLib)
	if err != nil {
		t.Fatal(err)
	}
	packageVersion, err := cupti.GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	if sampleVersion != packageVersion {
		t.Fatalf("CUPTI version mismatch: sample=%d package=%d (%s)", sampleVersion, packageVersion, sampleLib)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	got, err := captureTutorialPCWorkload(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("reference build command: make -B -C %s CUDA_INSTALL_PATH=%s", filepath.Join(root, "pc_sampling_start_stop"), cudaRoot)
	t.Logf("reference total samples=%d package total samples=%d sample-CUPTI=%d package-CUPTI=%d library=%s", refTotal, got, sampleVersion, packageVersion, sampleLib)
	if got != refTotal {
		t.Fatalf("PC sampling result disagrees with independent start/stop sample: package=%d reference=%d", got, refTotal)
	}
}

func captureTutorialPCWorkload(dev *cuda.Device) (uint64, error) {
	const source = `
extern "C" __global__ void VectorAdd(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
extern "C" __global__ void VectorSubtract(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}
extern "C" __global__ void VectorMultiply(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}`
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		return 0, err
	}
	module, err := cm.Load(dev)
	if err != nil {
		return 0, err
	}
	defer func() { _ = dev.WithContext(module.Close) }()
	var add, sub, mul driver.Function
	if err := dev.WithContext(func() error {
		var err error
		if add, err = module.Function("VectorAdd"); err != nil {
			return err
		}
		if sub, err = module.Function("VectorSubtract"); err != nil {
			return err
		}
		mul, err = module.Function("VectorMultiply")
		return err
	}); err != nil {
		return 0, err
	}
	const n = 32000
	a, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, err
	}
	defer a.Free()
	b, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, err
	}
	defer b.Free()
	c, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, err
	}
	defer c.Free()
	input := make([]int32, n)
	for i := range input {
		input[i] = int32(i)
	}
	if err := a.CopyFromHost(input); err != nil {
		return 0, err
	}
	if err := b.CopyFromHost(input); err != nil {
		return 0, err
	}
	var total uint64
	err = dev.WithContext(func() error {
		ctx, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		inv, err := referenceReasonInventory(ctx)
		if err != nil {
			return err
		}
		config := cuptipcsampling.Config{
			Period: 10, WorkerSleepMS: 10,
			MaxReasons:   len(inv.Reasons),
			ScratchBytes: 8 << 20, HardwareBytes: 8 << 20, MaxNativeBytes: 64 << 20,
			PCsPerBuffer: 512, MaxOutputBytes: 8 << 20, MaxReads: 16,
			MaxNames: 4096, MaxNameBytes: 256, MaxTotalNameBytes: 4096 * 256,
		}
		for _, reason := range inv.Reasons {
			config.ReasonIDs = append(config.ReasonIDs, reason.Index)
		}
		session, err := cuptipcsampling.Start(ctx, config)
		if err != nil {
			if session != nil {
				err = errors.Join(err, session.Close())
			}
			return err
		}
		closed := false
		defer func() {
			if !closed {
				_ = session.Close()
			}
		}()
		launch := func(fn driver.Function) error {
			return cudalaunch.Kernel(fn).Launch(a, b, c, n)
		}
		for _, fn := range []driver.Function{add, sub, mul, add, sub} {
			if err := launch(fn); err != nil {
				return err
			}
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		if err := session.Stop(); err != nil {
			return err
		}
		batch, readErr := session.Read()
		closeErr := session.Close()
		closed = closeErr == nil
		if err := errors.Join(readErr, closeErr); err != nil {
			return err
		}
		if !batch.Valid {
			return fmt.Errorf("package returned an invalid sampling batch")
		}
		final, ok := session.Final()
		if !ok || !final.Valid {
			return fmt.Errorf("package final sampling batch unavailable or invalid")
		}
		total = batch.TotalSamples + final.TotalSamples
		if total == 0 {
			return fmt.Errorf("package recorded no PC samples")
		}
		return nil
	})
	return total, err
}

func referenceReasonInventory(ctx driver.Context) (cuptipcsampling.ReasonInventory, error) {
	disable, err := cuptipcsampling.Enable(ctx)
	if err != nil {
		return cuptipcsampling.ReasonInventory{}, err
	}
	inv, queryErr := cuptipcsampling.StallReasons(ctx, 1<<10)
	return inv, errors.Join(queryErr, disable())
}

var pcTotalLine = regexp.MustCompile(`total samples:\s*([0-9]+)`)

func parsePCReferenceTotal(output string) (uint64, error) {
	matches := pcTotalLine.FindAllStringSubmatch(output, -1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("reference output has no total samples lines")
	}
	var total uint64
	for _, match := range matches {
		value, err := strconv.ParseUint(match[1], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("parse total samples %q: %w", match[1], err)
		}
		if ^uint64(0)-total < value {
			return 0, fmt.Errorf("reference total samples overflow")
		}
		total += value
	}
	if total == 0 {
		return 0, fmt.Errorf("reference total samples is zero")
	}
	return total, nil
}

func TestParsePCReferenceTotal(t *testing.T) {
	got, err := parsePCReferenceTotal("total samples: 17, dropped samples: 0\ntotal samples: 29, dropped samples: 1\n")
	if err != nil || got != 46 {
		t.Fatalf("parsePCReferenceTotal = %d, %v; want 46", got, err)
	}
}

func checkPCReferenceBinary(path string) error {
	f, err := elf.Open(path)
	if err != nil {
		return fmt.Errorf("open reference ELF: %w", err)
	}
	defer f.Close()
	deps, err := f.ImportedLibraries()
	if err != nil {
		return fmt.Errorf("read reference dependencies: %w", err)
	}
	ldd := exec.Command("ldd", path)
	resolved, err := ldd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ldd reference binary: %w: %s", err, resolved)
	}
	for _, line := range strings.Split(string(resolved), "\n") {
		fields := strings.Fields(line)
		for i, field := range fields {
			if field == "=>" && i+1 < len(fields) && filepath.IsAbs(fields[i+1]) {
				deps = append(deps, fields[i+1])
			}
		}
		if len(fields) > 0 && filepath.IsAbs(fields[0]) {
			deps = append(deps, fields[0])
		}
	}
	hasGoBuildID := false
	for _, section := range f.Sections {
		if section.Name == ".note.go.buildid" {
			hasGoBuildID = true
			break
		}
	}
	return checkPCReferenceDependencies(deps, hasGoBuildID)
}

func pcReferenceRoots(t *testing.T) (string, string) {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	root := os.Getenv("CUPTI_TUTORIAL_ROOT")
	if root == "" {
		root = filepath.Join(home, "go", "src", "github.com", "eunomia-bpf", "cupti-tutorial")
	}
	cudaRoot := os.Getenv("CUPTI_CUDA_ROOT")
	if cudaRoot == "" {
		cudaRoot = os.Getenv("CUDA_HOME")
	}
	if cudaRoot == "" {
		cudaRoot = os.Getenv("CUDA")
	}
	if cudaRoot == "" {
		cudaRoot = "/usr/local/cuda"
	}
	return root, cudaRoot
}

func buildPCReferenceSample(t *testing.T, root, cudaRoot string) string {
	t.Helper()
	dir := filepath.Join(root, "pc_sampling_start_stop")
	if _, err := os.Stat(filepath.Join(dir, "Makefile")); err != nil {
		t.Skipf("reference tutorial sample unavailable at %s: %v", dir, err)
	}
	if _, err := os.Stat(filepath.Join(cudaRoot, "bin", "nvcc")); err != nil {
		t.Skipf("CUDA toolkit nvcc unavailable under %s: %v", cudaRoot, err)
	}
	cmd := exec.Command("make", "-B", "-C", dir, "CUDA_INSTALL_PATH="+cudaRoot)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reference build command make -B -C %s CUDA_INSTALL_PATH=%s: %v\n%s", dir, cudaRoot, err, output)
	}
	path := filepath.Join(dir, "pc_sampling_start_stop")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("reference build did not produce %s: %v", path, err)
	}
	t.Logf("reference build command: make -B -C %s CUDA_INSTALL_PATH=%s", dir, cudaRoot)
	t.Logf("reference build output:\n%s", output)
	return path
}

func pcSampleCUPTILibrary(binary string) (string, error) {
	cmd := exec.Command("ldd", binary)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("ldd %s: %w: %s", binary, err, out)
	}
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.Contains(line, "libcupti") {
			continue
		}
		fields := strings.Fields(line)
		for i := range fields {
			if fields[i] == "=>" && i+1 < len(fields) && filepath.IsAbs(fields[i+1]) {
				return fields[i+1], nil
			}
		}
		if len(fields) > 0 && filepath.IsAbs(fields[0]) {
			return fields[0], nil
		}
	}
	return "", fmt.Errorf("ldd output has no resolved libcupti dependency for %s: %s", binary, out)
}

func pcCUPTIVersionAt(path string) (uint32, error) {
	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return 0, fmt.Errorf("dlopen %s: %w", path, err)
	}
	defer purego.Dlclose(h)
	symbol, err := purego.Dlsym(h, "cuptiGetVersion")
	if err != nil {
		return 0, fmt.Errorf("dlsym cuptiGetVersion in %s: %w", path, err)
	}
	var getVersion func(*uint32) int32
	purego.RegisterFunc(&getVersion, symbol)
	var version uint32
	if result := getVersion(&version); result != 0 {
		return 0, fmt.Errorf("cuptiGetVersion in %s returned %d", path, result)
	}
	return version, nil
}

func pcWithLibraryPath(env []string, lib string) []string {
	out := append([]string(nil), env...)
	for i := range out {
		if strings.HasPrefix(out[i], "LD_LIBRARY_PATH=") {
			out[i] = "LD_LIBRARY_PATH=" + lib + ":" + strings.TrimPrefix(out[i], "LD_LIBRARY_PATH=")
			return out
		}
	}
	return append(out, "LD_LIBRARY_PATH="+lib)
}
