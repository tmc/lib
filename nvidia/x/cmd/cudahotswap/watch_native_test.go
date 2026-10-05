//go:build linux && native

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/nativereceipt"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// nativeWatchOutput advances the disk source only after observing the actual
// checked report. Watch invokes the writer serially; no timer guesses readiness.
type nativeWatchOutput struct {
	bytes.Buffer
	path, artifacts string
	cancel          context.CancelFunc
	index           int
	receipt         *nativereceipt.Session
}

func nativeWatchSource(index int) string {
	switch index {
	case 0:
		return "extern \"C\" __global__ void replay(float*x,float*y,const float*w,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]-w[i];}"
	case 1:
		return "extern \"C\" __global__ void replay(float*x){x[threadIdx.x]=0.0f;}"
	default:
		return "extern \"C\" __global__ void replay(float*x,float*y,const float*w,unsigned n){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=w[i]+x[i];}"
	}
}

func (w *nativeWatchOutput) Write(data []byte) (int, error) {
	var event struct {
		Kernel string
		Result *cudahotswap.PushResult
		Error  string
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return 0, err
	}
	want := []cudahotswap.Outcome{cudahotswap.OutcomeWrongAnswer, cudahotswap.OutcomeABIMismatch, cudahotswap.OutcomeOK}
	if w.index >= len(want) || event.Kernel != "kernel" || event.Result == nil || event.Result.Report.Outcome != want[w.index] {
		return 0, fmt.Errorf("watch report %d: %+v", w.index, event)
	}
	if (event.Error != "") != (w.index < 2) {
		return 0, errors.New("watch error disagrees with classified outcome")
	}
	if w.index != 1 && (event.Result.Report.Replay == nil || !event.Result.Report.Replay.Native) {
		return 0, errors.New("watch report lacks native replay evidence")
	}
	if event.Result.Published != (w.index == 2) {
		return 0, errors.New("watch publication disagrees with checked outcome")
	}
	sourceSum := fmt.Sprintf("%x", sha256.Sum256([]byte(nativeWatchSource(w.index))))
	artifact := ""
	if event.Result.Report.Compile != nil {
		if event.Result.Report.Compile.SourceSHA256 != sourceSum {
			return 0, errors.New("watch compilation source disagrees with actual candidate")
		}
		artifact = event.Result.Report.Compile.PTXSHA256
	}
	if err := w.receipt.Capture(fmt.Sprintf("watch-report-%d", w.index), sourceSum, artifact); err != nil {
		return 0, err
	}
	if err := os.WriteFile(filepath.Join(w.artifacts, fmt.Sprintf("report-%d.json", w.index)), data, 0600); err != nil {
		return 0, err
	}
	w.Buffer.Write(data)
	w.index++
	if w.index == len(want) {
		w.cancel()
		return len(data), nil
	}
	code := nativeWatchSource(w.index)
	if err := os.WriteFile(filepath.Join(w.artifacts, fmt.Sprintf("candidate-%d.cu", w.index)), []byte(code), 0600); err != nil {
		return 0, err
	}
	// Exercise an editor's rename-save rather than an in-place mtime update.
	temp := w.path + ".save"
	if err := os.WriteFile(temp, []byte(code), 0600); err != nil {
		return 0, err
	}
	if err := os.Rename(temp, w.path); err != nil {
		return 0, err
	}
	return len(data), nil
}

func TestNativeWatchCLIRefusalThenRepair(t *testing.T) {
	if os.Getenv("HOTSWAP_CLI_NATIVE") != "1" {
		t.Skip("HOTSWAP_CLI_NATIVE=1 requires reviewed native dispatch")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	dir := os.Getenv("HOTSWAP_CLI_ARTIFACTS")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute HOTSWAP_CLI_ARTIFACTS required")
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	// Keep this process's compiler artifacts with its reports. Refuse reuse:
	// an existing cache would weaken attribution to this native invocation.
	cache := filepath.Join(dir, "compiler-cache")
	if err := os.Mkdir(cache, 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("XDG_CACHE_HOME", cache)
	path := filepath.Join(dir, "watch.cu")
	code := []byte(nativeWatchSource(0))
	if err := os.WriteFile(path, code, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "candidate-0.cu"), code, 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	proof, err := nativereceipt.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := errors.Join(proof.Write(filepath.Join(dir, "native-libraries.json")), proof.Close()); err != nil {
			t.Errorf("native library receipt: %v", err)
		}
	})
	if err := proof.Capture("watch-before-run", "", ""); err != nil {
		t.Fatal(err)
	}
	output := &nativeWatchOutput{path: path, artifacts: dir, cancel: cancel, receipt: proof}
	var stderr bytes.Buffer
	err = run(ctx, []string{"-watch", path, "-n", "32"}, output, &stderr, newSession)
	if err != nil && !onlyCancellation(err) {
		t.Fatalf("actual CLI: %v; stderr=%s", err, &stderr)
	}
	if output.index != 3 {
		t.Fatalf("observed %d reports, want wrong/ABI/repair; %v", output.index, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "stdout.jsonl"), output.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "stderr.log"), stderr.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
}
