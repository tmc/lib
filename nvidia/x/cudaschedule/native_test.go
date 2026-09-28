//go:build native

package cudaschedule_test

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
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
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		_ = dev.WithContext(raw.Close)
		t.Fatal(err)
	}
	defer mod.Close()
	kernel, err := mod.Function("schedule_hang")
	if err != nil {
		t.Fatal(err)
	}
	if err := kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil); err != nil {
		t.Fatal(err)
	}
	if err := mod.Close(); err != nil {
		t.Fatal(err)
	}
	fmt.Fprintln(os.Stdout, "PASS: exact PTX path loaded and launched")
}

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
	ptx, err := os.ReadFile(filepath.Join("testdata", "schedule_hang.ptx"))
	if err != nil {
		t.Fatal(err)
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
	options := func(start, end uint64, images map[string][]byte) cudaschedule.CampaignOptions {
		return cudaschedule.CampaignOptions{
			PTXSource: string(ptx), SeedStart: start, SeedEnd: end, Intensity: 2,
			ConfirmRuns: 1, Timeout: 3 * time.Second,
			BuildVariant: build(images),
		}
	}
	firstImages := make(map[string][]byte)
	first, err := cudaschedule.RunCampaign(context.Background(), options(17, 21, firstImages))
	if err != nil {
		t.Fatal(err)
	}
	var seed uint64
	found := false
	for _, result := range first.Seeds {
		if result.Verdict.IsFinding() {
			seed, found = result.Seed, true
			break
		}
	}
	if !found {
		t.Fatalf("sweep found no schedule-dependent timeout: %+v", first.Seeds)
	}
	firstPTX := append([]byte(nil), firstImages[strconv.FormatUint(seed, 10)+"/false"]...)
	secondImages := make(map[string][]byte)
	second, err := cudaschedule.RunCampaign(context.Background(), options(seed, seed+1, secondImages))
	if err != nil {
		t.Fatal(err)
	}
	if len(second.Seeds) != 1 || second.Seeds[0].Verdict != first.Seeds[seed-17].Verdict {
		t.Fatalf("seed %d did not reproduce: first=%+v second=%+v", seed, first.Seeds[seed-17], second.Seeds)
	}
	if len(firstPTX) == 0 || !bytes.Equal(firstPTX, secondImages[strconv.FormatUint(seed, 10)+"/false"]) {
		t.Fatal("replayed seed produced different PTX bytes")
	}
}
