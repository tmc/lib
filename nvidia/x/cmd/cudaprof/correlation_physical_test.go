//go:build linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

type correlationContract struct {
	Kernels      int               `json:"kernels"`
	Copies       int               `json:"copies"`
	GraphReplays int               `json:"graph_replays"`
	Hashes       map[string]string `json:"hashes"`
	SourceHashes map[string]string `json:"source_hashes"`
}

// TestPhysicalCorrelation requires the toolkit and an exclusive GPU hold. It
// never substitutes a synthetic record for work missing from the real capture.
func TestPhysicalCorrelation(t *testing.T) {
	if os.Getenv("CUDAPROF_CORRELATION_TEST") != "1" {
		t.Skip("set CUDAPROF_CORRELATION_TEST=1 under an exclusive GPU hold")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	dir := t.TempDir()
	run := func(name string, args ...string) {
		t.Helper()
		cmd := exec.CommandContext(ctx, name, args...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%s: %v\n%s", name, err, out)
		}
	}
	agent := os.Getenv("CUDAPROF_CORRELATION_AGENT")
	if agent == "" {
		run("sh", "../../../internal/cudainject/native/build_agent.sh", dir)
		agent = filepath.Join(dir, "cupti_agent.so")
	}
	binary := filepath.Join(dir, "fixture")
	run("go", "build", "-o", binary, "./testdata/correlation")
	var stdout, stderr bytes.Buffer
	result, err := runCapture(ctx, captureOptions{command: []string{binary, dir}, Inject: agent, GraphTrace: graphTraceNode, RequireComplete: true, Stdout: &stdout, Stderr: &stderr, DrainTimeout: time.Second})
	if err != nil || !result.CaptureSuccess || result.TargetExitCode != 0 {
		t.Fatalf("capture: %v, %+v\nstderr: %s", err, result, &stderr)
	}
	t.Cleanup(func() { os.RemoveAll(result.ScratchDir) })
	var contract correlationContract
	if err := json.Unmarshal(stdout.Bytes(), &contract); err != nil {
		t.Fatalf("contract: %v: %s", err, &stdout)
	}
	images := map[uint32][]byte{}
	for _, m := range result.Modules {
		if !m.Stored {
			continue
		}
		// Stored paths are relative to the transport root's worker directory.
		matches, err := filepath.Glob(filepath.Join(result.ScratchDir, "process-*", "modules", filepath.Base(m.File)))
		if err != nil {
			t.Fatal(err)
		}
		if len(matches) != 1 {
			t.Fatalf("module %d has %d image paths, want exactly one", m.ID, len(matches))
		}
		for _, path := range matches {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			images[m.ID] = data
		}
	}
	sources := map[string][]byte{}
	for _, hash := range contract.Hashes {
		data, err := os.ReadFile(filepath.Join(dir, hash+".cu"))
		if err != nil {
			t.Fatal(err)
		}
		sources[hash] = data
	}
	if out := os.Getenv("CUDAPROF_CORRELATION_OUTPUT"); out != "" {
		if err := os.MkdirAll(out, 0700); err != nil {
			t.Fatal(err)
		}
		out, err = os.MkdirTemp(out, "run-")
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("retained evidence: %s", out)
		if err := writeCaptureStatus(filepath.Join(out, "capture.pprof"), result, captureArtifactBudget{}); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, "contract.json"), stdout.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
		for hash, data := range sources {
			if err := os.WriteFile(filepath.Join(out, hash+".cu"), data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := qualifyCorrelation(result, contract, images, sources); err != nil {
		t.Fatalf("%v\nmodules=%+v\nactivities=%+v\napis=%+v\nscratch=%s", err, result.Modules, result.Activities, result.APIRecords, result.ScratchDir)
	}
	t.Logf("qualified %d kernels, %d copies, %d graph replays; exact loaded image and NVTX operation for every execution", contract.Kernels, contract.Copies, contract.GraphReplays)
	t.Run("missing journal record", func(t *testing.T) {
		paths, err := filepath.Glob(filepath.Join(result.ScratchDir, "process-*", "capture.journal"))
		if err != nil || len(paths) != 1 {
			t.Fatalf("journal paths %v: %v", paths, err)
		}
		data, err := os.ReadFile(paths[0])
		if err != nil {
			t.Fatal(err)
		}
		d := wire.NewDecoder(bytes.NewReader(data), wire.Limits{})
		var broken bytes.Buffer
		removed := false
		for {
			record, err := d.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			var a captureActivity
			if record.Kind == recordActivity {
				if err := json.Unmarshal(record.Payload, &a); err != nil {
					t.Fatal(err)
				}
			}
			if !removed && a.Subtype == "kernel" {
				removed = true
				continue
			}
			if err := wire.Write(&broken, record, wire.Limits{}); err != nil {
				t.Fatal(err)
			}
		}
		if !removed {
			t.Fatal("no real kernel frame removed")
		}
		path := filepath.Join(t.TempDir(), "capture.journal")
		if err := os.WriteFile(path, broken.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
		_, _, _, _, _, _, err = observeJournal(path, result.Session, result.TargetPID, controllerMaxJournalRecords)
		if err == nil || !strings.Contains(err.Error(), "sequence") {
			t.Fatalf("missing journal frame accepted or unexpected rejection: %v", err)
		}
		t.Logf("incomplete journal rejected before timeline analysis: %v", err)
		cap := testCapture(map[string][]byte{"capture.journal": data}, completeStatus())
		report, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords})
		if err != nil || report.Activities != contract.Kernels+contract.Copies {
			t.Fatalf("complete timeline: report=%+v error=%v", report, err)
		}
		cap.Files["capture.journal"] = broken.Bytes()
		if _, err := analyzeGaps(cap, gapOptions{maxRecords: controllerMaxJournalRecords}); err == nil || !strings.Contains(err.Error(), "sequence gap") {
			t.Fatalf("incomplete timeline: %v", err)
		} else {
			t.Logf("gaps report refused missing device work: %v", err)
		}

	})
	t.Run("missing activity", func(t *testing.T) {
		broken := result
		broken.Activities = append([]captureActivity(nil), result.Activities...)
		for i, a := range broken.Activities {
			if a.Subtype == "kernel" {
				broken.Activities = append(broken.Activities[:i], broken.Activities[i+1:]...)
				break
			}
		}
		if err := qualifyCorrelation(broken, contract, images, sources); err == nil {
			t.Fatal("missing work qualified as complete")
		} else {
			t.Log(err)
		}
	})
	t.Run("missing artifact", func(t *testing.T) {
		broken := map[uint32][]byte{}
		for id, data := range images {
			broken[id] = data
		}
		for _, a := range result.Activities {
			if a.Subtype == "kernel" {
				delete(broken, a.ModuleID)
				break
			}
		}
		if err := qualifyCorrelation(result, contract, broken, sources); err == nil {
			t.Fatal("missing image borrowed another module's same-named kernel")
		} else {
			t.Log(err)
		}
	})
	t.Run("missing source", func(t *testing.T) {
		broken := map[string][]byte{}
		for hash, data := range sources {
			broken[hash] = data
		}
		delete(broken, contract.Hashes["advance"])
		if err := qualifyCorrelation(result, contract, images, broken); err == nil {
			t.Fatal("missing source borrowed another same-named kernel")
		} else {
			t.Log(err)
		}
	})
	t.Run("wrong source", func(t *testing.T) {
		broken := map[string][]byte{}
		for hash, data := range sources {
			broken[hash] = data
		}
		broken[contract.Hashes["advance"]] = sources[contract.Hashes["prepare"]]
		if err := qualifyCorrelation(result, contract, images, broken); err == nil {
			t.Fatal("same-named source substituted for another image")
		} else {
			t.Log(err)
		}
	})

}

// qualifyCorrelation is a workload invariant, independent of the producer's
// own drop counter. It intentionally refuses both partial and unresolved work.
func qualifyCorrelation(r captureResult, c correlationContract, images map[uint32][]byte, sources map[string][]byte) error {
	kernels, copies, graphs, syncs := 0, 0, 0, 0
	streams := map[uint32]bool{}
	p, index := cudacapture.BuildActivityProfileIndex(r.Session, uint64(r.TargetPID), r.Activities, r.APIRecords)
	labels := map[int][]string{}
	for sample, i := range index {
		if i >= 0 {
			labels[i] = p.Sample[sample].Label["nvtx"]
			a := r.Activities[i]
			v := p.Sample[sample].Value
			if a.Subtype == "kernel" && (v[0] <= 0 || v[1] != 0 || v[2] != 0) {
				return fmt.Errorf("kernel mixed with memory time")
			}
			if a.Subtype == "memcpy" && (v[0] != 0 || v[1] <= 0 || v[2] != 128) {
				return fmt.Errorf("copy mixed with kernel time or wrong byte count")
			}
			if a.Subtype != "kernel" && a.Subtype != "memcpy" {
				return fmt.Errorf("host wait included in device samples")
			}

		}
	}
	counts := map[string]int{}
	replayIDs := map[uint32]bool{}
	graphIDs := map[uint32]bool{}
	nodeIDs := map[uint64]bool{}
	var prepareEnd uint64
	var advanceStarts []uint64
	var eventWait bool
	for i, a := range r.Activities {
		switch a.Subtype {
		case "memcpy":
			names := labels[i]
			want := "prepare"
			if a.CopyKind == 2 {
				want = "advance"
			} else if a.CopyKind != 1 {
				return fmt.Errorf("unexpected copy kind %d", a.CopyKind)
			}
			if len(names) != 1 || names[0] != want {
				return fmt.Errorf("copy operation labels %v, want %s", names, want)
			}
			copies++
		case "sync":
			syncs++
			if a.SyncType == 2 {
				eventWait = true
			}
		case "kernel":
			kernels++
			streams[a.Stream] = true
			if !a.Valid || a.End <= a.Start {
				return fmt.Errorf("invalid device execution")
			}
			if a.GraphID != 0 {
				graphs++
				graphIDs[a.GraphID] = true
				nodeIDs[a.GraphNodeID] = true
				if a.GraphNodeID == 0 {
					return fmt.Errorf("unresolved graph node")
				}
				replayIDs[a.Correlation] = true
			}
			names := labels[i]
			if len(names) != 1 {
				return fmt.Errorf("kernel %d has operation labels %v", i, names)
			}
			want, ok := c.Hashes[names[0]]
			if !ok {
				return fmt.Errorf("unexpected operation %q", names[0])
			}
			image, ok := images[a.ModuleID]
			if !ok {
				return fmt.Errorf("unresolved loaded module %d", a.ModuleID)
			}
			if got := fmt.Sprintf("%x", sha256.Sum256(image)); got != want {
				return fmt.Errorf("operation %s: loaded hash %s, want %s", names[0], got, want)
			}

			source, ok := sources[want]
			if !ok {
				return fmt.Errorf("unresolved source for operation %s image %s", names[0], want)
			}
			if fmt.Sprintf("%x", sha256.Sum256(source)) != c.SourceHashes[names[0]] {
				return fmt.Errorf("source hash mismatch for operation %s", names[0])
			}
			counts[names[0]]++
			if names[0] == "prepare" {
				prepareEnd = a.End
			} else {
				advanceStarts = append(advanceStarts, a.Start)
			}
		}
	}
	if kernels != c.Kernels || copies != c.Copies || graphs != c.GraphReplays || len(replayIDs) != c.GraphReplays {
		return fmt.Errorf("partial work: kernels %d/%d copies %d/%d graph nodes %d/%d replays %d/%d", kernels, c.Kernels, copies, c.Copies, graphs, c.GraphReplays, len(replayIDs), c.GraphReplays)
	}
	if len(graphIDs) != 1 || len(nodeIDs) != 1 || !eventWait {
		return fmt.Errorf("unresolved dependency: graphs=%d nodes=%d event_wait=%v", len(graphIDs), len(nodeIDs), eventWait)
	}
	for _, start := range advanceStarts {
		if start < prepareEnd {
			return fmt.Errorf("event dependency violated: advance started before prepare completed")
		}
	}
	if len(streams) != 2 || syncs == 0 || len(r.APIRecords) == 0 || counts["prepare"] != 1 || counts["advance"] != 3 {
		return fmt.Errorf("unresolved operations: streams=%d syncs=%d host calls=%d executions=%v", len(streams), syncs, len(r.APIRecords), counts)
	}
	return nil
}
