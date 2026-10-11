//go:build linux && native

package ncclras_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/ncclras"
	"github.com/tmc/lib/nvidia/x/nccltest"
)

// TestNativeMultiGPUDiagnostics is a finite preparation fixture, not a default
// hardware test. It requires two explicitly selected devices, a pinned Inspector
// plugin and a fresh retained directory. CUDA/NCCL calls need an outer process
// watchdog; a Go context alone cannot interrupt those calls.
func TestNativeMultiGPUDiagnostics(t *testing.T) {
	if os.Getenv("NCCLDIAG_NATIVE") != "1" {
		t.Skip("multi-GPU NCCL qualification pending: explicit native admission required")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	parts := strings.Split(os.Getenv("NCCLDIAG_DEVICES"), ",")
	if len(parts) != 2 {
		t.Fatal("NCCLDIAG_DEVICES must explicitly name two distinct device ordinals")
	}
	devices := make([]int32, 2)
	for i, s := range parts {
		n, err := strconv.ParseInt(s, 10, 32)
		if err != nil || n < 0 {
			t.Fatal("invalid device ordinal")
		}
		devices[i] = int32(n)
	}
	if devices[0] == devices[1] {
		t.Fatal("multi-GPU control cannot reuse a device")
	}
	address := os.Getenv("NCCLDIAG_RAS_ADDR")
	host, _, err := net.SplitHostPort(address)
	if err != nil || host != "127.0.0.1" {
		t.Fatal("NCCLDIAG_RAS_ADDR must explicitly name an IPv4 loopback port")
	}
	plugin := os.Getenv("NCCLDIAG_INSPECTOR_PLUGIN")
	want := os.Getenv("NCCLDIAG_INSPECTOR_SHA256")
	wantBytes, err := hex.DecodeString(want)
	if err != nil || len(wantBytes) != sha256.Size || !filepath.IsAbs(plugin) {
		t.Fatal("absolute pinned Inspector plugin required")
	}
	info, err := os.Lstat(plugin)
	if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 || info.Size() > 128<<20 {
		t.Fatal("bounded regular Inspector plugin required")
	}
	f, err := diagnosticFile(plugin, 128<<20)
	if err != nil {
		t.Fatal(err)
	}
	opened, err := f.Stat()
	if err != nil || !opened.Mode().IsRegular() || !os.SameFile(info, opened) {
		f.Close()
		t.Fatal("Inspector plugin descriptor differs")
	}
	hash := sha256.New()
	n, readErr := io.Copy(hash, io.LimitReader(f, 128<<20+1))
	closeErr := f.Close()
	if readErr != nil || closeErr != nil || n != info.Size() || !bytes.Equal(hash.Sum(nil), wantBytes) {
		t.Fatal("Inspector plugin identity differs")
	}
	out := os.Getenv("NCCLDIAG_OUTPUT")
	if !filepath.IsAbs(out) || filepath.Clean(out) != out {
		t.Fatal("absolute fresh retained directory required")
	}
	if err := os.Mkdir(out, 0700); err != nil {
		t.Fatal(err)
	}
	inspectorDir := filepath.Join(out, "inspector")
	if err := os.Mkdir(inspectorDir, 0700); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{
		"NCCL_RAS_ENABLE": "1", "NCCL_RAS_ADDR": address,
		"NCCL_PROFILER_PLUGIN": plugin, "NCCL_INSPECTOR_ENABLE": "1",
		"NCCL_INSPECTOR_DUMP_DIR": inspectorDir, "NCCL_INSPECTOR_PROM_DUMP": "0",
		"NCCL_INSPECTOR_OTEL_EXPORT": "0", "NCCL_INSPECTOR_DUMP_VERBOSE": "0",
		"NCCL_INSPECTOR_DUMP_THREAD_ENABLE": "1", "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "100000",
		"NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES": "0",
	} {
		t.Setenv(k, v)
	}
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil || count < 2 {
		t.Fatalf("two actual CUDA devices required: %d %v", count, err)
	}
	for _, ordinal := range devices {
		if int(ordinal) >= count {
			t.Fatal("device ordinal unavailable")
		}
		d, err := driver.DeviceGet(int(ordinal))
		if err != nil {
			t.Fatal(err)
		}
		uuid, err := d.Uuid()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("actual device ordinal=%d UUID=%v", ordinal, uuid)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	client := ncclras.Client{Address: address, Timeout: 5 * time.Second}
	hostname, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(inspectorDir, fmt.Sprintf("%s-pid%d.log", hostname, os.Getpid()))
	readCapture := func() ([]byte, error) {
		f, err := diagnosticFile(path, diagnosticCaptureBytes)
		if os.IsNotExist(err) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		b, readErr := io.ReadAll(io.LimitReader(f, diagnosticCaptureBytes+1))
		closeErr := f.Close()
		if readErr != nil {
			return nil, readErr
		}
		return b, closeErr
	}
	queries := 0
	commHash := ""
	err = nccltest.RunCycles(nccltest.Case{Operation: nccltest.AllReduce, DataType: nccltest.Float32, Bytes: 65536}, nccltest.Options{Devices: devices, Checks: 1, Iterations: 1}, 2, func(result nccltest.Result) error {
		if result.Ranks != 2 || result.Comparisons <= 0 || result.Wrong != 0 {
			return fmt.Errorf("multi-GPU numerical control failed: %+v", result)
		}
		s, err := client.Status(ctx)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(out, fmt.Sprintf("status-%d.json", queries)), s.Raw, 0600); err != nil {
			return err
		}
		if s.Collection.Timeouts != 0 || len(s.Communicators) != 1 {
			return fmt.Errorf("incomplete RAS collection")
		}
		comm := s.Communicators[0]
		version := fmt.Sprintf("%d.%d.%d", result.NCCLVersion/10000, (result.NCCLVersion%10000)/100, result.NCCLVersion%100)
		if s.Version != version || (commHash != "" && comm.Hash != commHash) {
			return fmt.Errorf("RAS compiler version or communicator identity differs")
		}
		commHash = comm.Hash
		if comm.Size != 2 || comm.RanksCount != 2 || comm.MissingRanksCount != 0 {
			return fmt.Errorf("RAS membership differs from actual two-device job")
		}
		for _, rank := range comm.Ranks {
			if rank.PID != os.Getpid() || rank.CUDADevice != int(devices[rank.Rank]) || rank.Counts["AllReduce"] != uint64((queries+1)*2) {
				return fmt.Errorf("RAS rank/device/process/collective binding differs")
			}
		}
		b, err := json.Marshal(result)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(out, fmt.Sprintf("result-%d.json", queries)), b, 0600); err != nil {
			return err
		}
		queries++
		if queries == 2 {
			// Observe periodic output while the clique is alive. Finalization does
			// not guarantee a dump in the pinned Inspector implementation.
			captureCtx, stop := context.WithTimeout(ctx, 8*time.Second)
			defer stop()
			capture, err := waitDiagnosticCapture(captureCtx, readCapture, hostname, os.Getpid(), commHash, 100*time.Millisecond)
			if err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(out, "inspector-capture.jsonl"), capture, 0600)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if queries != 2 {
		t.Fatal("missing live RAS observations")
	}
	// No additional collectives are submitted during cleanup. Validate the
	// retained final file against the same oracle after the writer stops.
	finalCtx, stop := context.WithTimeout(context.Background(), time.Second)
	defer stop()
	if _, err := waitDiagnosticCapture(finalCtx, readCapture, hostname, os.Getpid(), commHash, 100*time.Millisecond); err != nil {
		t.Fatal(err)
	}

}

func diagnosticFile(path string, max int64) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), path)
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 0 || info.Size() > max {
		f.Close()
		return nil, fmt.Errorf("bounded regular diagnostic file required: %s", path)
	}
	return f, nil
}
