//go:build native && cuda_tma_multicast_qualification

package cudatma

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func qualificationFile(path, digest string, limit int64) ([]byte, error) {
	if !filepath.IsAbs(path) {
		return nil, fmt.Errorf("absolute pinned artifact path required")
	}
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		return nil, e
	}
	if !info.Mode().IsRegular() || info.Size() > limit {
		return nil, fmt.Errorf("artifact is not regular or exceeds read bound")
	}
	b, e := io.ReadAll(io.LimitReader(f, limit+1))
	if e != nil {
		return nil, e
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("artifact grew beyond read bound")
	}
	h := sha256.Sum256(b)
	if hex.EncodeToString(h[:]) != digest {
		return nil, fmt.Errorf("artifact digest differs from admission: %s", path)
	}
	return b, nil
}
func admittedQualification() (qualificationAdmission, error) {
	var a qualificationAdmission
	if os.Getenv("CUDA_TMA_MULTICAST_NATIVE") != "1" {
		return a, fmt.Errorf("native packet dispatch requires explicit admission; gate closed")
	}
	path, pin := os.Getenv("CUDA_TMA_MULTICAST_ADMISSION"), os.Getenv("CUDA_TMA_MULTICAST_ADMISSION_SHA256")
	if !digestString(pin, 32) {
		return a, fmt.Errorf("exact native admission digest required before any driver load")
	}
	b, e := qualificationFile(path, pin, 1<<20)
	if e != nil {
		return a, e
	}
	a, e = decodeQualificationAdmission(b)
	if e != nil {
		return a, e
	}
	if a.AuthorizationID != os.Getenv("CUDA_TMA_MULTICAST_AUTHORIZATION_ID") {
		return a, fmt.Errorf("specific native approval identity differs from manifest")
	}
	host, e := os.Hostname()
	if e != nil {
		return a, e
	}
	if host != a.Host {
		return a, fmt.Errorf("actual host differs from admission")
	}
	owner, e := cudalock.Inherited(os.Getenv)
	if e != nil {
		return a, e
	}
	if owner.PID != a.LockPID || owner.Epoch != a.LockEpoch || owner.Host != strings.Split(a.Host, ".")[0] || os.Getenv("GPU_LOCKDIR") != a.LockDir {
		return a, fmt.Errorf("actual inherited lock owner/directory differs from exact binding")
	}
	exe, e := os.Executable()
	if e != nil {
		return a, e
	}
	if _, e = qualificationFile(exe, a.ExecutableSHA256, 128<<20); e != nil {
		return a, e
	}
	if _, e = qualificationFile(a.DriverFile, a.DriverSHA256, 128<<20); e != nil {
		return a, e
	}
	if _, e = qualificationFile(a.NVCCFile, a.NVCCSHA256, 128<<20); e != nil {
		return a, e
	} // Read/hash only; no compiler discovery or execution.
	if _, e = qualificationFile(a.QuarantineContractFile, a.QuarantineContractSHA256, 1<<20); e != nil {
		return a, e
	}
	for _, artifact := range a.Cases {
		if _, e = qualificationFile(artifact.SourceFile, artifact.SourceSHA256, 1<<20); e != nil {
			return a, e
		}
		compiler, e := qualificationFile(artifact.CompilerReceiptFile, artifact.CompilerReceiptSHA256, 1<<20)
		if e != nil {
			return a, e
		}
		var provenance qualificationCompilerReceipt
		if e = json.Unmarshal(compiler, &provenance); e != nil {
			return a, e
		}
		if e = provenance.validate(a, artifact); e != nil {
			return a, e
		}
		if _, e = qualificationFile(artifact.PTXFile, artifact.PTXSHA256, 16<<20); e != nil {
			return a, e
		}
		cubin, e := qualificationFile(artifact.CubinFile, artifact.CubinSHA256, 128<<20)
		if e != nil {
			return a, e
		}
		if e = qualificationCubin(cubin); e != nil {
			return a, e
		}
	}
	return a, nil
}

type qualificationTrace struct {
	mu        sync.Mutex
	file      *os.File
	remaining uint64
}

func (t *qualificationTrace) write(operation string, value any) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	b, e := json.Marshal(struct {
		At        time.Time
		Operation string
		Value     any
	}{time.Now().UTC(), operation, value})
	if e != nil {
		return e
	}
	b = append(b, '\n')
	if uint64(len(b)) > t.remaining {
		return fmt.Errorf("explicit receipt budget exhausted")
	}
	if _, e = t.file.Write(b); e != nil {
		return e
	}
	t.remaining -= uint64(len(b))
	return t.file.Sync()
}
func qualificationArtifactFor(a qualificationAdmission, id string) (qualificationArtifact, qualificationCase, error) {
	for _, artifact := range a.Cases {
		if artifact.CaseID == id {
			for _, c := range qualificationCases() {
				if c.ID == id {
					return artifact, c, nil
				}
			}
		}
	}
	return qualificationArtifact{}, qualificationCase{}, fmt.Errorf("case not in exact reviewed admission")
}

// This supervisor never declares process termination to be GPU completion.
// It has no lock release, target reset or retry operation. Unknown completion
// writes a quarantine receipt and stops. The manifest's external owner must
// enforce the separately reviewed persistent-lock quarantine contract.
func TestNativeClusterMulticastReference(t *testing.T) {
	a, e := admittedQualification()
	if e != nil {
		t.Fatal(e)
	}
	if os.Getenv("CUDA_TMA_MULTICAST_CHILD") != "" {
		t.Fatal("supervisor cannot run as a child")
	}
	if e = os.Mkdir(a.OutputDir, 0700); e != nil {
		t.Fatalf("new exclusive receipt directory required: %v", e)
	}
	exe, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Duration(a.OverallSeconds)*time.Second)
	defer cancel()
	for _, artifact := range a.Cases {
		if e = ctx.Err(); e != nil {
			t.Fatal(e)
		}
		dir := filepath.Join(a.OutputDir, artifact.CaseID)
		if e = os.Mkdir(dir, 0700); e != nil {
			t.Fatal(e)
		}
		child, c := context.WithTimeout(ctx, time.Duration(a.ChildSeconds)*time.Second)
		cmd := exec.CommandContext(child, exe, "-test.run=^TestTMAQualificationChild$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), "CUDA_TMA_MULTICAST_CHILD="+artifact.CaseID, "CUDA_TMA_MULTICAST_CHILD_OUTPUT="+dir)
		log, e := os.OpenFile(filepath.Join(dir, "child.log"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if e != nil {
			c()
			t.Fatal(e)
		}
		boundedLog := &qualificationLogWriter{writer: log, remaining: a.ReceiptBytes}
		cmd.Stdout, cmd.Stderr = boundedLog, boundedLog
		runErr := cmd.Run()
		deadlineErr := child.Err()
		c()
		closeErr := log.Close()
		receipt := struct {
			CaseID                           string
			Error, DeadlineError, CloseError string
			ProcessExitProvesCUDACompletion  bool
		}{artifact.CaseID, errorText(runErr), errorText(deadlineErr), errorText(closeErr), false}
		e = qualificationFinishChild(artifact.CaseID, artifact.SourceSHA256, qualificationLaunches(artifact.CaseID),
			errors.Join(runErr, deadlineErr, closeErr),
			func() error {
				data, e := json.MarshalIndent(receipt, "", "  ")
				if e != nil {
					return e
				}
				return os.WriteFile(filepath.Join(dir, "supervisor.json"), append(data, '\n'), 0600)
			},
			func() ([]byte, error) { return os.ReadFile(filepath.Join(dir, "completed.json")) },
			func(reason error) error {
				data, e := json.MarshalIndent(struct {
					OwnerPID                                 int
					DeviceUUID, ContractSHA256, Error, State string
				}{a.LockPID, a.DeviceUUID, a.QuarantineContractSHA256, errorText(reason), "retained/unknown; stop all further cases; external lock owner must confirm teardown before release"}, "", "  ")
				if e != nil {
					return e
				}
				return os.WriteFile(filepath.Join(a.OutputDir, "QUARANTINE.json"), append(data, '\n'), 0600)
			})
		if e != nil {
			t.Fatalf("single attempt uncertain; native completion/reclamation is not inferred from child exit: %v", e)
		}
	}
}

func TestTMAQualificationChild(t *testing.T) {
	id := os.Getenv("CUDA_TMA_MULTICAST_CHILD")
	if id == "" {
		t.Fatal("child invocation requires exact admitted case")
	}
	a, e := admittedQualification()
	if e != nil {
		t.Fatal(e)
	}
	artifact, c, e := qualificationArtifactFor(a, id)
	if e != nil {
		t.Fatal(e)
	}
	dir := os.Getenv("CUDA_TMA_MULTICAST_CHILD_OUTPUT")
	if dir != filepath.Join(a.OutputDir, id) {
		t.Fatal("child receipt path differs from admission")
	}
	f, e := os.OpenFile(filepath.Join(dir, "operations.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	trace := &qualificationTrace{file: f, remaining: a.ReceiptBytes}
	if e = trace.write("admission-checked-before-load", a); e != nil {
		t.Fatal(e)
	}
	ptx, e := qualificationFile(artifact.PTXFile, artifact.PTXSHA256, 16<<20)
	if e != nil {
		t.Fatal(e)
	}
	if e = qualificationPTX(string(ptx), c); e != nil {
		t.Fatal(e)
	}
	cubin, e := qualificationFile(artifact.CubinFile, artifact.CubinSHA256, 128<<20)
	if e != nil {
		t.Fatal(e)
	}
	if e = qualificationCubin(cubin); e != nil {
		t.Fatal(e)
	}
	owner, e := openQualificationOwner(a, c, cubin, trace.write)
	if e != nil {
		if owner != nil {
			cleanup := owner.close()
			trace.write("preparation-error", struct{ Error, Cleanup string }{errorText(e), errorText(cleanup)})
		}
		t.Fatal(e)
	}
	launches := 0
	run := func(name string, epoch uint32, coords [5]int32, parity bool) error {
		result, err := owner.run(name, epoch, coords, parity)
		if write := trace.write("actual-run", result); write != nil {
			err = errors.Join(err, write)
		}
		if err != nil {
			return err
		}
		if !result.LaunchReturned || !result.Drained || !result.OwnershipRefusals || !result.CancelledWaitRetained {
			return fmt.Errorf("actual launch/drain/ownership evidence missing")
		}
		launches++
		if c.Control == "wrong-swizzle" {
			if !strings.Contains(result.Comparison, "numerical/canary mismatch") || result.SuccessfulRetirement {
				return fmt.Errorf("completed wrong-layout control was not numerically detected")
			}
			return nil
		}
		if c.Control == "single-destination" {
			p, _ := NewMulticastPlan(c.config())
			r, _ := p.Requirements()
			out := result.ActualOutput[64 : 64+int(r.OutputBytes)]
			s0, _ := decodeQualificationStatus(result.ActualStatus[64:96])
			s1, _ := decodeQualificationStatus(result.ActualStatus[96:128])
			if result.Comparison == "" || result.SuccessfulRetirement || s0.Selected != 1 || s0.Bytes != r.TileBytes || s1.Selected != 0 || s1.Bytes != 0 || s0.Retired != 1 || s1.Retired != 1 || !bytes.Equal(out[:r.TileBytes], result.Expected) || !bytes.Equal(out[r.TileBytes:], bytesOf(0xa5, int(r.TileBytes))) {
				return fmt.Errorf("truthful ordinary single-destination control not detected")
			}
			return nil
		}
		if result.Comparison != "" || !result.SuccessfulRetirement {
			return fmt.Errorf("healthy numerical/retirement check: %s", result.Comparison)
		}
		return nil
	}
	if c.Control != "" {
		e = run("tma_multicast", 0, c.coordinates(0), false)
	} else if c.Parity {
		// First run has generations1/2 on distinct held slots/maps. References
		// subsequently initialize independent fresh phase0 barriers per launch.
		e = run("tma_parity", 0, c.coordinates(0), true)
		for epoch := uint32(0); epoch < 2 && e == nil; epoch++ {
			e = run("tma_ordinary", epoch, c.coordinates(0), false)
		}
	} else {
		for scenario := 0; scenario < 3 && e == nil; scenario++ {
			for epoch := uint32(0); epoch < 2 && e == nil; epoch++ {
				for _, name := range []string{"tma_ordinary", "tma_multicast"} {
					e = run(name, epoch, c.coordinates(scenario), false)
					if e != nil {
						break
					}
				}
			}
		}
	}
	cleanup := owner.close()
	if log := trace.write("case-terminal", struct {
		Error, CleanupError string
		CompletedLaunches   int
	}{errorText(e), errorText(cleanup), launches}); log != nil {
		e = errors.Join(e, log)
	}
	if e != nil || cleanup != nil {
		t.Fatal(errors.Join(e, cleanup))
	}
	if launches != qualificationLaunches(id) {
		t.Fatal("completed launch count differs from declared case")
	}
	completed := struct {
		CaseID           string
		Launches         int
		CleanupCompleted bool
		SourceSHA256     string
	}{id, launches, true, artifact.SourceSHA256}
	data, e := json.MarshalIndent(completed, "", "  ")
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(dir, "completed.json"), append(data, '\n'), 0600); e != nil {
		t.Fatal(e)
	}
}

type qualificationLogWriter struct {
	writer    io.Writer
	remaining uint64
}

func (w *qualificationLogWriter) Write(b []byte) (int, error) {
	if uint64(len(b)) > w.remaining {
		return 0, fmt.Errorf("child log exceeds explicit receipt budget")
	}
	n, e := w.writer.Write(b)
	w.remaining -= uint64(n)
	return n, e
}
