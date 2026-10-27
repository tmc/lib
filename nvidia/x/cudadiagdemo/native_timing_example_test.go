package cudadiagdemo

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func ExampleNativeTiming() {
	// Host fakes stand in for the device streams and a logical clock for
	// the monotonic clock; the executor and export are RunNative's. This is
	// not native execution, and no observation here is CUDA API entry.
	dir, err := os.MkdirTemp("", "cudadiagdemo-timing-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	var clock atomic.Int64
	e := (&fakeNative{}).executor()
	e.now = func() int64 { return clock.Add(1) }
	attempts, err := e.run(context.Background(), 1, planNativeRequests(1, 2), &nativeCleanup{})
	if err != nil {
		fmt.Println(err)
		return
	}
	result := &NativeResult{HostClockDomain: "process_monotonic"}
	result.recordAttempts(attempts)
	if err := exportNative(dir, result, nil); err != nil {
		fmt.Println(err)
		return
	}

	// Verify the timing bytes from the receipt alone.
	data, err := os.ReadFile(filepath.Join(dir, "native-receipt.json"))
	if err != nil {
		fmt.Println(err)
		return
	}
	var receipt struct {
		Timing *cudaregress.ArtifactRef `json:"timing"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil || receipt.Timing == nil {
		fmt.Println("no timing reference", err)
		return
	}
	b, err := os.ReadFile(receipt.Timing.Path)
	if err != nil || fmt.Sprintf("%x", sha256.Sum256(b)) != receipt.Timing.Digest || int64(len(b)) != receipt.Timing.SizeBytes {
		fmt.Println("unverified timing", err)
		return
	}
	// Place each attempt's ready observation among its wrappers by host time.
	s := bufio.NewScanner(bytes.NewReader(b))
	for s.Scan() {
		var line struct {
			RequestID string       `json:"request_id"`
			Attempt   string       `json:"attempt"`
			Timing    NativeTiming `json:"timing"`
			Calls     []NativeCall `json:"calls"`
		}
		if err := json.Unmarshal(s.Bytes(), &line); err != nil || line.Timing.Ready == nil {
			fmt.Println("unverified line", err)
			return
		}
		var events []string
		ready := false
		for _, c := range line.Calls {
			if !ready && line.Timing.Ready.HostNS < c.StartNS {
				events, ready = append(events, "ready"), true
			}
			events = append(events, c.Name)
		}
		fmt.Printf("%s/%s %s: %s\n", line.RequestID, line.Attempt, line.Timing.Recipe, strings.Join(events, " "))
	}
	// Output:
	// request-1/baseline launch-ready-v1: cuMemcpyHtoDAsync_v2 ready cuLaunchKernel cuStreamSynchronize cuMemcpyDtoH_v2
	// request-1/candidate launch-ready-v1: cuMemcpyHtoDAsync_v2 ready cuLaunchKernel cuStreamSynchronize cuMemcpyDtoH_v2
	// request-1/device launch-ready-v1: cuMemcpyHtoDAsync_v2 ready cuLaunchKernel cuLaunchKernel cuStreamSynchronize cuMemcpyDtoH_v2
}
