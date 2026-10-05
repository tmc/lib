package cudaschedule

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// These are CPU-authored readbacks, not device observations.
func reexecOutput(kind string) *cudaoutputcoverage.RecordedOutput {
	ref := []uint32{math.Float32bits(1), math.Float32bits(2), math.Float32bits(3)}
	a, b := append([]uint32(nil), ref...), append([]uint32(nil), ref...)
	fillA, fillB := math.Float32bits(-101), math.Float32bits(-202)
	if kind == "residue" {
		a[2], b[2] = fillA, fillB
	}
	ordinary := append([]uint32(nil), ref...)
	if kind == "wrong" {
		ordinary[1] = math.Float32bits(999)
	}
	source, input := strings.Repeat("a", 64), strings.Repeat("b", 64)
	return &cudaoutputcoverage.RecordedOutput{
		PTXDigest: strings.Repeat("c", 64), Ordinary: ordinary,
		TwoFill: cudaoutputcoverage.TwoFill{
			Contract: cudaoutputcoverage.Contract{Buffer: "out", Elements: 3, Domain: []cudaoutputcoverage.Range{{Start: 0, End: 3}}, Kind: cudaoutputcoverage.AtLeastOnce, Deterministic: true},
			FillA:    fillA, FillB: fillB, Reference: ref, ReadbackA: a, ReadbackB: b,
			Geometry:     cudaoutputcoverage.Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{3, 1, 1}},
			SourceDigest: source, InputDigest: input, Target: "sm_121", MathMode: "precise"},
		Policy: cudatest.CUDAOutputPolicy{Target: "sm_121", DType: "fp32", MathMode: "precise", SourceDigest: source, InputDigest: input, Elements: 3, NaN: "reject", Infinity: "reject", Zero: "bits"},
	}
}

func TestIndependentOutputWorker(t *testing.T) {
	mode := os.Getenv("CUDASCHEDULE_CPU_RECEIPT_WORKER")
	if mode == "" {
		return
	}
	var buf bytes.Buffer
	if err := WriteOutputReceipt(&buf, reexecOutput(mode)); err != nil {
		t.Fatal(err)
	}
	wire := buf.String()
	switch mode {
	case "wrong_pid":
		var r outputReceipt
		if err := json.Unmarshal([]byte(strings.TrimSuffix(strings.TrimPrefix(wire, outputReceiptPrefix), "\n")), &r); err != nil {
			t.Fatal(err)
		}
		r.PID++
		body, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		wire = outputReceiptPrefix + string(body) + "\n"
	case "wrong_ptx":
		wire = strings.ReplaceAll(wire, strings.Repeat("c", 64), strings.Repeat("d", 64))
	case "duplicate":
		wire += wire
	case "torn":
		wire = strings.TrimSuffix(wire, "\n")
	}
	if _, err := os.Stdout.WriteString(wire); err != nil {
		t.Fatal(err)
	}
	os.Exit(0)
}

func TestIndependentOutputReexec(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"healthy", "residue", "wrong", "wrong_pid", "wrong_ptx", "duplicate", "torn"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, exe, "-test.run=^TestIndependentOutputWorker$")
			cmd.Env = append(os.Environ(), "CUDASCHEDULE_CPU_RECEIPT_WORKER="+mode)
			var stdout, stderr bytes.Buffer
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			pid := cmd.Process.Pid // Retain before Wait releases os.Process.
			if pid <= 1 || pid == os.Getpid() {
				t.Fatalf("invalid child PID %d", pid)
			}
			if err := cmd.Wait(); err != nil {
				t.Fatalf("worker: %v: %s", err, stderr.String())
			}
			clean, out, parseErr := takeOutputReceipt(stdout.String(), pid, strings.Repeat("c", 64))
			if mode == "wrong_pid" || mode == "wrong_ptx" || mode == "duplicate" || mode == "torn" {
				if parseErr == nil {
					t.Fatal("invalid child receipt accepted")
				}
				return
			}
			if parseErr != nil || clean != "" || out == nil {
				t.Fatalf("receipt: clean=%q out=%v err=%v", clean, out, parseErr)
			}
			run := RunRecord{Exited: true, PTXDigest: out.PTXDigest, ExpectedPTXDigest: strings.Repeat("c", 64), PTXDigestSource: "runner_receipt", Output: out}
			want := VerdictMismatch
			if mode == "healthy" {
				want = VerdictPass
			}
			if got := Classify(run); got != want {
				t.Fatalf("classify=%v want=%v", got, want)
			}
			t.Logf("CPU child PID=%d output bytes=%d verdict=%s", pid, stdout.Len(), want)
		})
	}
}
