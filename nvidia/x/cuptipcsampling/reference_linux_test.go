//go:build linux

package cuptipcsampling_test

import (
	"debug/elf"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"unsafe"

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
		t.Fatalf("CUPTI_PC_REFERENCE_TEST=1 requires an inherited gpu-lock: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	root, cudaRoot := pcReferenceRoots(t)
	sample := buildPCReferenceSample(t, root, cudaRoot)
	if err := checkPCReferenceBinary(sample); err != nil {
		t.Fatal(err)
	}
	work := t.TempDir()
	var referenceRuns []pcReferenceReport
	for i := 0; i < 3; i++ {
		cmd := exec.Command(sample)
		cmd.Dir = work
		cmd.Env = pcWithLibraryPath(os.Environ(), filepath.Join(cudaRoot, "lib64"))
		out, runErr := cmd.CombinedOutput()
		if runErr != nil {
			t.Fatalf("reference sample run %d %s: %v\n%s", i+1, sample, runErr, out)
		}
		report, parseErr := parsePCReferenceOutput(string(out))
		if parseErr != nil {
			t.Fatalf("parse reference run %d: %v\n%s", i+1, parseErr, out)
		}
		if report.Total == 0 {
			t.Fatalf("reference run %d recorded no user samples (offered=%d dropped=%d non-user=%d)", i+1, report.Offered, report.Dropped, report.NonUser)
		}
		referenceRuns = append(referenceRuns, report)
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
	period, err := pcSamplingDefaultPeriod(dev)
	if err != nil {
		t.Fatal(err)
	}
	got, packageKernels, packageReasons, err := captureTutorialPCWorkload(dev, period)
	if err != nil {
		t.Fatal(err)
	}
	var minRef, maxRef uint64 = ^uint64(0), 0
	var sumRef uint64
	for i, reference := range referenceRuns {
		t.Logf("reference run %d user samples=%d range samples=%v kernels=%v reasons=%v", i+1, reference.Total, reference.Counts, reference.Kernels, reference.Reasons)
		if reference.Total < minRef {
			minRef = reference.Total
		}
		if reference.Total > maxRef {
			maxRef = reference.Total
		}
		sumRef += reference.Total
		if i > 0 && (!equalStrings(reference.Kernels, referenceRuns[0].Kernels) || !equalStrings(reference.Reasons, referenceRuns[0].Reasons)) {
			t.Fatalf("reference deterministic inventory changed: first=(%v,%v) run%d=(%v,%v)", referenceRuns[0].Kernels, referenceRuns[0].Reasons, i+1, reference.Kernels, reference.Reasons)
		}
	}
	mean := float64(sumRef) / float64(len(referenceRuns))
	spread := maxRef - minRef
	// PC samples are discrete stochastic counts. Use the measured three-run
	// envelope when wider, with a 3-sigma Poisson counting bound as a floor.
	tolerance := max(float64(spread), 3*math.Sqrt(mean))
	t.Logf("package user samples=%d kernels=%v reasons=%v period=2^%d cycles; reference period=CUPTI default=2^%d cycles; reference mean=%g spread=%d comparison_tolerance=%g CUPTI sample=%d package=%d library=%s", got, packageKernels, packageReasons, period, period, mean, spread, tolerance, sampleVersion, packageVersion, sampleLib)
	if !equalStrings(packageKernels, referenceRuns[0].Kernels) || !equalStrings(packageReasons, referenceRuns[0].Reasons) {
		t.Fatalf("deterministic inventory mismatch: package kernels=%v reasons=%v reference kernels=%v reasons=%v", packageKernels, packageReasons, referenceRuns[0].Kernels, referenceRuns[0].Reasons)
	}
	if math.Abs(float64(got)-mean) > tolerance {
		t.Fatalf("package user total %d outside reference spread tolerance around mean %g (spread=%d tolerance=%g)", got, mean, spread, tolerance)
	}
}

func captureTutorialPCWorkload(dev *cuda.Device, period uint32) (uint64, []string, []string, error) {
	const source = `
__global__ void VectorAdd(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
__global__ void VectorSubtract(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] - b[i];
}
__global__ void VectorMultiply(const int *a, const int *b, int *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}`
	cm, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		return 0, nil, nil, err
	}
	module, err := cm.Load(dev)
	if err != nil {
		return 0, nil, nil, err
	}
	defer func() { _ = dev.WithContext(module.Close) }()
	var add, sub, mul driver.Function
	if err := dev.WithContext(func() error {
		var err error
		if add, err = module.Function(tutorialVectorAddName); err != nil {
			return err
		}
		if sub, err = module.Function(tutorialVectorSubtractName); err != nil {
			return err
		}
		mul, err = module.Function(tutorialVectorMultiplyName)
		return err
	}); err != nil {
		return 0, nil, nil, err
	}
	const n = 32000
	a, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, nil, nil, err
	}
	defer a.Free()
	b, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, nil, nil, err
	}
	defer b.Free()
	c, err := cudamem.Alloc[int32](dev, n)
	if err != nil {
		return 0, nil, nil, err
	}
	defer c.Free()
	input := make([]int32, n)
	for i := range input {
		input[i] = int32(i)
	}
	if err := a.CopyFromHost(input); err != nil {
		return 0, nil, nil, err
	}
	if err := b.CopyFromHost(input); err != nil {
		return 0, nil, nil, err
	}
	var total uint64
	kernels, reasons := map[string]bool{}, map[string]bool{}
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
			Period: period, WorkerSleepMS: 10,
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
			blocks := uint32((n + 255) / 256)
			return cudalaunch.Kernel(fn).Grid(blocks, 1, 1).Block(256, 1, 1).Launch(a, b, c, n)
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
		user, err := pcReferenceUserSamples(batch.TotalSamples, batch.DroppedSamples, batch.NonUserSamples)
		if err != nil {
			return err
		}
		finalUser, err := pcReferenceUserSamples(final.TotalSamples, final.DroppedSamples, final.NonUserSamples)
		if err != nil {
			return err
		}
		if ^uint64(0)-user < finalUser {
			return fmt.Errorf("package PC user sample sum overflow")
		}
		total = user + finalUser
		for _, sampledBatch := range []cuptipcsampling.Batch{batch, final} {
			for _, pc := range sampledBatch.PCs {
				if pc.Name == "" {
					continue
				}
				kernels[pc.Name] = true
				for _, reason := range pc.Reasons {
					if reason.Samples > 0 {
						reasons[inv.Lookup(reason.Index)] = true
					}
				}
			}
		}
		if total == 0 {
			return fmt.Errorf("package recorded no PC samples")
		}
		return nil
	})
	return total, sortedKeys(kernels), sortedKeys(reasons), err
}

func sortedKeys(set map[string]bool) []string {
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func referenceReasonInventory(ctx driver.Context) (cuptipcsampling.ReasonInventory, error) {
	disable, err := cuptipcsampling.Enable(ctx)
	if err != nil {
		return cuptipcsampling.ReasonInventory{}, err
	}
	inv, queryErr := cuptipcsampling.StallReasons(ctx, 1<<10)
	return inv, errors.Join(queryErr, disable())
}

func pcSamplingDefaultPeriod(dev *cuda.Device) (uint32, error) {
	var period uint32
	err := dev.WithContext(func() error {
		ctx, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		disable, err := cuptipcsampling.Enable(ctx)
		if err != nil {
			return err
		}
		defer disable()
		info := cupti.PCSamplingConfigurationInfo{
			AttributeType:   cupti.PcSamplingConfigurationAttrTypeSamplingPeriod,
			AttributeStatus: cupti.Result(-1),
		}
		params := cupti.PCSamplingConfigurationInfoParams{
			Size: uint(unsafe.Sizeof(cupti.PCSamplingConfigurationInfoParams{})),
			Ctx:  ctx, NumAttributes: 1,
			PPCSamplingConfigurationInfo: uintptr(unsafe.Pointer(&info)),
		}
		if err := cupti.PCSamplingGetConfigurationAttribute(&params); err != nil {
			return err
		}
		if info.AttributeStatus != 0 || info.AttributeData[0] < 5 || info.AttributeData[0] > 31 {
			return fmt.Errorf("CUPTI default sampling period is invalid: status=%v data=%v", info.AttributeStatus, info.AttributeData)
		}
		period = uint32(info.AttributeData[0])
		return nil
	})
	return period, err
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
	buildDir, err := copyReferenceSampleTree(t, root, dir)
	if err != nil {
		t.Fatalf("copy reference tutorial to temporary build directory: %v", err)
	}
	dir = buildDir
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
