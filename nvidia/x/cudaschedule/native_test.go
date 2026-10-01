//go:build native

package cudaschedule_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// TestCampaignNativeHarness is the checked-in child-process harness. It loads
// and launches the exact PTX path appended by RunCampaign.
func TestCampaignNativeHarness(t *testing.T) {
	ptxPath := os.Args[len(os.Args)-1]
	if !strings.HasSuffix(ptxPath, ".ptx") {
		return
	}
	image, err := os.ReadFile(ptxPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := driver.Load(""); err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	var raw driver.Module
	if err := dev.WithContext(func() error {
		var err error
		raw, err = driver.ModuleLoadData(append(image, 0))
		return err
	}); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(image)
	fmt.Fprintf(os.Stdout, "CUDASCHEDULE_LOADED_PTX_SHA256:%s\n", hex.EncodeToString(digest[:]))
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		_ = dev.WithContext(raw.Close)
		t.Fatal(err)
	}
	defer mod.Close()
	kernel, err := mod.Function("schedule_race")
	if err != nil {
		t.Fatal(err)
	}
	output, err := cudamem.Alloc[uint64](dev, 2)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint64, 2)
	if err := dev.WithContext(func() (err error) {
		defer func() { err = errors.Join(err, output.Free()) }()
		if err := output.CopyFromHost([]uint64{0, 0}); err != nil {
			return fmt.Errorf("initialize output: %w", err)
		}
		if err := cudalaunch.OwnedKernel(kernel).Block(64, 1, 1).Launch(output); err != nil {
			return fmt.Errorf("launch: %w", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fmt.Errorf("synchronize: %w", err)
		}
		if err := output.CopyToHost(got); err != nil {
			return fmt.Errorf("read output: %w", err)
		}
		return nil
	}); err != nil {
		t.Fatalf("native child: %v", err)
	}
	if err := mod.Close(); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintf(os.Stdout, "READER_DELAY_NS: %d\n", got[1])
	if got[1] < scheduleReaderDelayThreshold {
		t.Fatalf("reader delay %d ns below threshold %d ns", got[1], scheduleReaderDelayThreshold)
	}
	if got[0] != 1 {
		fmt.Fprintf(os.Stdout, "MISMATCH: shared handoff produced %d, want 1\n", got[0])
		return
	}
	fmt.Fprintln(os.Stdout, "PASS: shared handoff produced 1")
}

// schedule_race.ptx gives the reader a bounded loop so the pristine writer
// usually publishes first. A delay before the writer's shared store can
// reverse that order; the sentinel then makes a lost write fail the output check.
func TestCampaign_FindingReproducesFromSeed(t *testing.T) {
	if _, err := os.Stat("/dev/nvidiactl"); err != nil {
		t.Skip("no NVIDIA device")
	}
	if os.Getenv("CUDA_SCHEDULE_NATIVE") != "1" {
		t.Skip("set CUDA_SCHEDULE_NATIVE=1 to run the native schedule campaign")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native campaign requires GPU_LOCK_HELD; run under gpu-lock")
	}
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	readPTX := func(name string) string {
		t.Helper()
		ptx, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return string(ptx)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	build := func(images map[string][]byte) func(context.Context, uint64, bool, string) (string, []string, error) {
		return func(_ context.Context, seed uint64, pristine bool, ptxPath string) (string, []string, error) {
			image, err := os.ReadFile(ptxPath)
			if err != nil {
				return "", nil, err
			}
			images[strconv.FormatUint(seed, 10)+"/"+strconv.FormatBool(pristine)] = append([]byte(nil), image...)
			return executable, []string{"-test.run=^TestCampaignNativeHarness$"}, nil
		}
	}
	options := func(source string, start, end uint64, images map[string][]byte) cudaschedule.CampaignOptions {
		return cudaschedule.CampaignOptions{
			PTXSource: source, SeedStart: start, SeedEnd: end, Intensity: 2,
			LaunchGeometry: "grid=1x1x1 block=64x1x1",
			InputReference: "schedule_race zero-initialized uint64 output",
			MaxSleepNs:     cudaschedule.DefaultMaxSleepNs,
			ConfirmRuns:    1, Timeout: 3 * time.Second, PristineTimeout: 8 * time.Second,
			BuildVariant: build(images),
		}
	}
	const seedStart, seedEnd = uint64(17), uint64(49)
	cleanImages := make(map[string][]byte)
	clean, err := cudaschedule.RunCampaign(context.Background(), options(readPTX("schedule_race_clean.ptx"), seedStart, seedEnd, cleanImages))
	if err != nil {
		t.Fatal(err)
	}
	checkBaseline := func(report *cudaschedule.CampaignReport) {
		t.Helper()
		if err := checkScheduleBaseline(report); err != nil {
			t.Fatal(err)
		}
		t.Logf("baseline output: %s", report.Baseline.Stdout)
	}
	checkBaseline(clean)
	if clean.FindingCount() != 0 {
		t.Fatalf("synchronized control reported findings: %+v", clean.Seeds)
	}
	firstImages := make(map[string][]byte)
	first, err := cudaschedule.RunCampaign(context.Background(), options(readPTX("schedule_race.ptx"), seedStart, seedEnd, firstImages))
	if err != nil {
		t.Fatal(err)
	}
	checkBaseline(first)
	var seed uint64
	found := false
	for _, result := range first.Seeds {
		if result.Verdict == cudaschedule.VerdictMismatch {
			seed, found = result.Seed, true
			break
		}
	}
	if !found {
		t.Fatalf("sweep found no shared-memory race mismatch: %+v", first.Seeds)
	}
	sites, err := cudaschedule.Analyze(readPTX("schedule_race.ptx"))
	if err != nil {
		t.Fatal(err)
	}
	// This bounds a single visit to each perturbation site, not the entire
	// clock-based loop. The watchdog also bounds repeated loop visits.
	const threadsPerBlock = 64
	worstInjected := uint64(threadsPerBlock) * uint64(len(sites)) * uint64(cudaschedule.DefaultMaxSleepNs)
	if worstInjected >= uint64((3 * time.Second).Nanoseconds()) {
		t.Fatalf("worst-case injected sleep %d ns reaches the 3 s run timeout", worstInjected)
	}
	t.Logf("race fixture worst-case injected sleep=%d ns (%d threads x %d sites x %d ns), below 3 s run timeout", worstInjected, threadsPerBlock, len(sites), cudaschedule.DefaultMaxSleepNs)
	firstPTX := append([]byte(nil), firstImages[strconv.FormatUint(seed, 10)+"/false"]...)
	secondImages := make(map[string][]byte)
	second, err := cudaschedule.RunCampaign(context.Background(), options(readPTX("schedule_race.ptx"), seed, seed+1, secondImages))
	if err != nil {
		t.Fatal(err)
	}
	checkBaseline(second)
	if len(second.Seeds) != 1 || second.Seeds[0].Verdict != cudaschedule.VerdictMismatch {
		t.Fatalf("seed %d did not reproduce its mismatch: second=%+v", seed, second.Seeds)
	}
	if len(firstPTX) == 0 || !bytes.Equal(firstPTX, secondImages[strconv.FormatUint(seed, 10)+"/false"]) {
		t.Fatal("replayed seed produced different PTX bytes")
	}
}
