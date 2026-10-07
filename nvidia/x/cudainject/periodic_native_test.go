//go:build linux && native

package cudainject_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudainject"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

// TestAgentPeriodicDrainNative is an explicitly enabled finite gate. The
// enclosing operation must pin compiler/toolkit/runtime inputs and retire the
// inherited admitted group on any refusal. It never acquires its own lease.
// Optional CUDAINJECT_CUPTI_INCLUDE and CUDAINJECT_CUPTI_LIBRARY must be
// supplied together as absolute paths for a separate CUPTI installation.
func TestAgentPeriodicDrainNative(t *testing.T) {
	if os.Getenv("CUDAINJECT_NATIVE") != "1" {
		t.Skip("CUDAINJECT_NATIVE=1 required")
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	compiler, home := os.Getenv("CUDAINJECT_CC"), os.Getenv("CUDAINJECT_CUDA_HOME")
	dir := os.Getenv("CUDAINJECT_RESULTS")
	if !filepath.IsAbs(dir) {
		t.Fatal("absolute fresh CUDAINJECT_RESULTS required")
	}
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	agent, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Compiler: compiler, CUDAHome: home, Directory: filepath.Join(dir, "agent"), CUPTIInclude: os.Getenv("CUDAINJECT_CUPTI_INCLUDE"), CUPTILibrary: os.Getenv("CUDAINJECT_CUPTI_LIBRARY")})
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(dir, "worker.c")
	if err := os.WriteFile(source, periodicWorkerSource, 0600); err != nil {
		t.Fatal(err)
	}
	worker := filepath.Join(dir, "worker")
	args := []string{"-std=c11", "-Wall", "-Wextra", "-Werror", source, "-I" + filepath.Join(home, "include"), "-L" + filepath.Join(home, "lib64/stubs"), "-lcuda", "-o", worker}
	compileOutput := &periodicOutput{remaining: 64 << 10}
	compile := exec.CommandContext(ctx, compiler, args...)
	compile.Stdout, compile.Stderr = compileOutput, compileOutput
	compile.WaitDelay = time.Second
	compileErr := compile.Run()
	retainErr := os.WriteFile(filepath.Join(dir, "worker-compile.log"), compileOutput.Bytes(), 0600)
	if err := errors.Join(compileErr, retainErr); err != nil {
		t.Fatalf("worker compile: %v", err)
	}
	transport := filepath.Join(dir, "transport")
	if err := os.Mkdir(transport, 0700); err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(ctx, worker)
	runtimeInputs, err := periodicRuntimeInputs(worker, agent.Path, os.Getenv("CUDAINJECT_DRIVER_LIBRARY"), os.Getenv("CUDAINJECT_CUPTI_LIBRARY_FILE"))
	if err != nil {
		t.Fatal(err)
	}
	command.Env = append(os.Environ(), "CUDA_INJECTION64_PATH="+agent.Path, "CUDAPROF_TRANSPORT="+transport, "CUDAPROF_SESSION=01010101010101010101010101010101", "CUDAPROF_FLUSH_INTERVAL_MS=5", "CUDAPROF_API_ACTIVITY=1")
	command.WaitDelay = time.Second
	stdout, stderr := &periodicOutput{remaining: 64 << 10}, &periodicOutput{remaining: 64 << 10}
	command.Stdout, command.Stderr = stdout, stderr
	input, err := command.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	workerPID := command.Process.Pid
	// The bounded output's snapshots are safe while the worker is waiting.
	var before cudainject.Recovery
	var journal string
	deadline := time.Now().Add(2 * time.Second)
	found := false
	for time.Now().Before(deadline) {
		paths, _ := filepath.Glob(filepath.Join(transport, "process-*", "capture.journal"))
		if len(paths) == 1 {
			journal = paths[0]
			before, err = cudainject.ReadJournal(journal, cudainject.Limits{})
			if err == nil {
				_, outputErr := periodicWorkerOutput(stdout.Bytes(), workerPID, false)
				if outputErr == nil && periodicEvidence(before, workerPID, false) == nil {
					found = true
					break
				}
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	var runtime periodicRuntime
	var runtimeErr error
	if found {
		runtime, runtimeErr = observePeriodicRuntime(workerPID, runtimeInputs)
		if runtimeErr != nil {
			cancel()
		}
	}
	workerIdentity, workerErr := finishPeriodicWorker(found, workerPID, input, cancel, command.Wait, stdout, stderr, func(name string, body []byte) error { return os.WriteFile(filepath.Join(dir, name), body, 0600) })
	workerErr = errors.Join(workerErr, runtimeErr)
	if workerErr != nil {
		t.Fatalf("periodic completion: %v stdout=%s stderr=%s", workerErr, stdout.Bytes(), stderr.Bytes())
	}
	final, err := cudainject.ReadJournal(journal, cudainject.Limits{})
	if err != nil || periodicEvidence(final, workerPID, true) != nil {
		t.Fatalf("final %+v %v", final.Stats, err)
	}
	identities := map[string]string{}
	for _, path := range []string{compiler, source, agent.Path, worker} {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		identities[path] = fmt.Sprintf("%x", sha256.Sum256(body))
	}
	receipt := struct {
		Owner         cudalock.Owner
		Agent         *cudainject.Agent
		Inputs        map[string]string
		Before, Final cudainject.Stats
		Worker        periodicWorkerIdentity
		Runtime       periodicRuntime
	}{owner, agent, identities, before.Stats, final.Stats, workerIdentity, runtime}
	b, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "receipt.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}
