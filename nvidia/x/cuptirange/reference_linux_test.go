//go:build linux

package cuptirange_test

import (
	"debug/elf"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const rangeReferenceMetric = "smsp__warps_launched.avg"

func TestNativeRangeMatchesReferenceSample(t *testing.T) {
	if os.Getenv("CUPTI_RANGE_REFERENCE_TEST") != "1" {
		t.Skip("CUPTI_RANGE_REFERENCE_TEST=1 required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("CUPTI_RANGE_REFERENCE_TEST=1 requires an inherited gpu-lock: %v", err)
	}
	chip := os.Getenv("CUPTI_CHIP")
	if chip == "" {
		t.Fatal("CUPTI_CHIP must identify the qualified device chip")
	}
	root, cudaRoot := referenceRoots(t)
	sample := buildReferenceSample(t, root, cudaRoot, "userrange_profiling", "user_range_profiling")
	if err := checkReferenceBinary(sample); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	cmd := exec.Command(sample, "0", rangeReferenceMetric)
	cmd.Dir = work
	cmd.Env = withLibraryPath(os.Environ(), filepath.Join(cudaRoot, "lib64"))
	refOutput, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reference sample %s: %v\n%s", sample, err, refOutput)
	}
	refValue, err := parseMetricValue(string(refOutput), rangeReferenceMetric)
	if err != nil {
		t.Fatalf("parse reference output: %v\n%s", err, refOutput)
	}
	packageVersion, err := cupti.GetVersion()
	if err != nil {
		t.Fatal(err)
	}
	sampleCUPTI, err := sampleCUPTILibrary(sample)
	if err != nil {
		t.Fatal(err)
	}
	sampleVersion, err := cupiVersionAt(sampleCUPTI)
	if err != nil {
		t.Fatal(err)
	}
	if packageVersion != sampleVersion {
		t.Fatalf("CUPTI version mismatch: package=%d sample=%d (%s)", packageVersion, sampleVersion, sampleCUPTI)
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
	got, err := captureTutorialVectorWorkload(dev, chip)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%s sm_%d%d reference sample=%g package=%g metric=%s reference-CUPTI=%d package-CUPTI=%d library=%s", chip, major, minor, refValue, got, rangeReferenceMetric, sampleVersion, packageVersion, sampleCUPTI)
	if referenceMetricRound(got) != refValue {
		t.Fatalf("CUPTI range result disagrees at sample precision: package_raw=%g package_rounded=%g reference_raw=%g", got, referenceMetricRound(got), refValue)
	}
}

func sampleCUPTILibrary(binary string) (string, error) {
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

func cupiVersionAt(path string) (uint32, error) {
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

func captureTutorialVectorWorkload(dev *cuda.Device, chip string) (float64, error) {
	const source = `
extern "C" __global__ void VectorAdd(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
extern "C" __global__ void VectorSubtract(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
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
	var add, sub driver.Function
	if err := dev.WithContext(func() error {
		var err error
		add, err = module.Function("VectorAdd")
		if err != nil {
			return err
		}
		sub, err = module.Function("VectorSubtract")
		return err
	}); err != nil {
		return 0, err
	}
	const n = 50000
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
	plan, err := cuptirange.NewPlan(chip, nil, []string{rangeReferenceMetric}, 16<<20)
	if err != nil {
		return 0, err
	}
	defer plan.Close()
	capture, err := plan.CaptureReplay(dev, cuptirange.Limits{MaxRanges: 4, MaxRangeTreeNodes: 8, MaxCounterBytes: 16 << 20, MaxPasses: 16}, cuptirange.CaptureOptions{Mode: cuptirange.Regions, MaxDepth: 2}, cuptirange.Replay{
		Synchronize: driver.CtxSynchronize,
		Run: func(_ uint, scope *cuptirange.Scope) error {
			return scope.Region("vector-workload", func() error {
				if err := cudalaunch.Kernel(add).Grid((n+255)/256, 1, 1).Block(256, 1, 1).Launch(a, b, c, n); err != nil {
					return err
				}
				return cudalaunch.Kernel(sub).Grid((n+255)/256, 1, 1).Block(256, 1, 1).Launch(a, b, c, n)
			})
		},
	})
	if capture != nil {
		defer capture.Close()
	}
	if err != nil {
		return 0, err
	}
	for _, r := range capture.RangeInfo {
		if r.Name == "vector-workload" {
			// The tutorial's NVPW evaluator accepts isolated=true. This package
			// uses CUPTI's host evaluator, whose API has no equivalent field, so
			// this comparison does not claim parity for that evaluator mode.
			values, err := plan.Evaluate(capture.Image, r.Index)
			if err != nil {
				return 0, err
			}
			if len(values) != 1 {
				return 0, fmt.Errorf("metric returned %d values", len(values))
			}
			return values[0], nil
		}
	}
	return 0, fmt.Errorf("vector-workload range missing")
}

func checkReferenceBinary(path string) error {
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
	return checkReferenceDependencies(deps, hasGoBuildID)
}

func referenceRoots(t *testing.T) (string, string) {
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

func buildReferenceSample(t *testing.T, root, cudaRoot, dir, binary string) string {
	t.Helper()
	sampleDir := filepath.Join(root, dir)
	if _, err := os.Stat(filepath.Join(sampleDir, "Makefile")); err != nil {
		t.Skipf("reference tutorial sample unavailable at %s: %v", sampleDir, err)
	}
	if _, err := os.Stat(filepath.Join(cudaRoot, "bin", "nvcc")); err != nil {
		t.Skipf("CUDA toolkit nvcc unavailable under %s: %v", cudaRoot, err)
	}
	buildDir, err := copyReferenceSampleTree(t, root, sampleDir)
	if err != nil {
		t.Fatalf("copy reference tutorial to temporary build directory: %v", err)
	}
	sampleDir = buildDir
	cmd := exec.Command("make", "-B", "-C", sampleDir, "CUDA_INSTALL_PATH="+cudaRoot)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("reference build command make -B -C %s CUDA_INSTALL_PATH=%s: %v\n%s", sampleDir, cudaRoot, err, output)
	}
	path := filepath.Join(sampleDir, binary)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("reference build did not produce %s: %v\n%s", path, err, output)
	}
	t.Logf("reference build command: make -B -C %s CUDA_INSTALL_PATH=%s", sampleDir, cudaRoot)
	t.Logf("reference build output:\n%s", output)
	return path
}

func withLibraryPath(env []string, lib string) []string {
	out := append([]string(nil), env...)
	for i := range out {
		if strings.HasPrefix(out[i], "LD_LIBRARY_PATH=") {
			out[i] = "LD_LIBRARY_PATH=" + lib + ":" + strings.TrimPrefix(out[i], "LD_LIBRARY_PATH=")
			return out
		}
	}
	return append(out, "LD_LIBRARY_PATH="+lib)
}
