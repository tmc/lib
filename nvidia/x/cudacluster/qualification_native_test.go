//go:build native

package cudacluster

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// This is a bounded producer for the DSM matrix row. Existing owner fixtures
// retain the actual C++ commands/output and verify every launch and sync.
// Compilation is offline: this gate requires source-bound prebuilt fixtures.
func TestClusterQualification_RankAndLocalWindowControl(t *testing.T) {
	// This file builds only with the native tag, so a closed gate here is a
	// vacuous native run. It fails wrapping ErrHardwarePending; it never skips.
	if err := requireNativeLock(os.Getenv); errors.Is(err, errNativeGateClosed) {
		reportNativeGate(t, true, err)
	}
	if err := requireNativeLock(os.Getenv); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"CUDA_CLUSTER_PREBUILT_DIR", "CUDA_CLUSTER_RUN_DIR", "DSM_QUALIFICATION_OUTPUT"} {
		if !filepath.IsAbs(os.Getenv(name)) {
			t.Fatalf("absolute %s required", name)
		}
	}
	dir := os.Getenv("DSM_QUALIFICATION_OUTPUT")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	name, err := dev.Name()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]qualificationReduction, 0, 8)
	var facts cudatest.QualificationFacts
	for _, size := range []int{2, 4} {
		got, err := runReduction(t, size, MutantNone)
		if err != nil {
			t.Fatal(err)
		}
		// Each successful call runs one emitted kernel and one separate
		// cooperative-groups oracle, and checks both synchronization results.
		rows = append(rows, qualificationReduction{"healthy", size, got})
		facts.PositiveRuns += 2
		facts.Launches += 2
		facts.CompletedLaunches += 2
		facts.Synchronizations += 2
		facts.Expected += 2
		facts.Checked += 2
	}
	got, err := runReduction(t, 4, MutantLocalWindow)
	if err != nil {
		t.Fatal(err)
	}
	rows = append(rows, qualificationReduction{"local-control", 4, got})
	facts.NegativeRuns++
	facts.Launches++
	facts.CompletedLaunches++
	facts.Synchronizations++
	// Five fixed attempts are declared before execution. A launch failure or
	// timeout is a harness failure here, never a numerical mutant detection.
	for range 5 {
		got, err := runReduction(t, 4, MutantNoSync)
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, qualificationReduction{"publication-control", 4, got})
		facts.NegativeRuns++
		facts.Launches++
		facts.CompletedLaunches++
		facts.Synchronizations++
	}
	if err := checkQualificationReductions(rows); err != nil {
		t.Fatal(err)
	}
	if err := runRankGeometry(t, 4); err != nil {
		t.Fatal(err)
	}
	// The geometry program checks all four observed ranks against its 2x2
	// declaration and launch dimensions before it exits successfully.
	facts.PositiveRuns++
	facts.Launches++
	facts.CompletedLaunches++
	facts.Synchronizations++
	facts.Expected += 4
	facts.Checked += 4
	facts.RankGeometryMatched = true
	facts.PostReadOrdered = true // Bound to the owner fixture's post-read barrier.
	facts.CooperativeOracleChecked = true
	facts.LocalWindowControlDetected = true
	write := func(path string, value any) {
		data, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, path), append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("reductions.json", rows)
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() || info.Size() > 16<<20 {
		t.Fatal("executing binary exceeds qualification artifact limit")
	}
	binary, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	if err != nil {
		t.Fatal(err)
	}
	if len(binary) > 16<<20 {
		t.Fatal("executing binary grew beyond qualification artifact limit")
	}
	if err := os.WriteFile(filepath.Join(dir, "executed.test"), binary, 0600); err != nil {
		t.Fatal(err)
	}
	write("producer.json", struct {
		GoVersion, Platform, DriverLibrary, DeviceName, DeviceUUID, ComputeCapability string
		DriverVersion                                                                 int
		ExecutableSHA256                                                              string
		Facts                                                                         cudatest.QualificationFacts
	}{runtime.Version(), runtime.GOOS + "/" + runtime.GOARCH, driver.Path(), name,
		fmt.Sprintf("%x", uuid.Bytes), fmt.Sprintf("%d.%d", major, minor), version,
		fmt.Sprintf("%x", sha256.Sum256(binary)), facts})
}
