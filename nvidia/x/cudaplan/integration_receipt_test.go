package cudaplan_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func checkIntegrationReceipt(planned []cudaplan.StepPlan, actual []cudaplan.ExecutedStep, completed bool, graphReplay bool) error {
	if !completed {
		return errors.New("integration: native completion not observed")
	}
	if len(planned) == 0 || len(actual) != len(planned) {
		return errors.New("integration: missing executed steps")
	}
	for i, step := range planned {
		want := step.Backend
		if graphReplay {
			if step.Backend != cudaplan.BackendLaunch {
				return fmt.Errorf("integration: step %d is not an admitted graph kernel", i)
			}
			want = cudaplan.BackendGraph
		}
		if actual[i].Index != i || actual[i].NodeName != step.NodeName || actual[i].Backend != want {
			return fmt.Errorf("integration: executed step %d differs from plan", i)
		}
	}
	return nil
}

// This reuses the numerical launch/replay harness instead of treating compiled
// source or an opened CUDA stream as execution. Native evidence exists only
// when CUDAPLAN_GRAPH_NATIVE is enabled under the coordinator's inherited lease.
func TestEndToEndSlicesRecordTheirExecutedBackend(t *testing.T) {
	if os.Getenv("CUDAPLAN_GRAPH_NATIVE") == "1" {
		if _, err := cudalock.Inherited(os.Getenv); err != nil {
			t.Fatal(err)
		}
	}
	runNativePlanReferenceChecked(t, "kernel", false, func(p *cudaplan.ExecutablePlan, c *cudaplan.Completion, _ *cuda.Device, _ map[string][]float32, graphReplay bool) {
		if err := checkIntegrationReceipt(p.Steps(), c.ExecutedSteps(), c.Done(), graphReplay); err != nil {
			t.Fatal(err)
		}
	})
}

// Profile data is supplied independently, never fabricated by this test. A
// successful identity match does not prove capture completeness, timing joins
// or physical metric extraction. Those remain separate producer gates.
func TestNativePlanProfileSidecarIdentity(t *testing.T) {
	path := os.Getenv("CUDAPLAN_PROFILE_SIDECAR")
	if path == "" {
		t.Skip("requires an independently captured profile sidecar")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, readErr := io.ReadAll(io.LimitReader(f, 1<<20+1))
	closeErr := f.Close()
	if err := errors.Join(readErr, closeErr); err != nil {
		t.Fatal(err)
	}
	if len(raw) > 1<<20 {
		t.Fatal("profile sidecar exceeds 1 MiB")
	}
	var snapshot cudaprof.ProfileSnapshot
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&snapshot); err != nil {
		t.Fatal(err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatal("trailing sidecar JSON", err)
	}
	checked := false
	runNativePlanReferenceChecked(t, "kernel", false, func(p *cudaplan.ExecutablePlan, c *cudaplan.Completion, dev *cuda.Device, data map[string][]float32, graphReplay bool) {
		if err := checkIntegrationReceipt(p.Steps(), c.ExecutedSteps(), c.Done(), graphReplay); err != nil {
			t.Fatal(err)
		}
		if checked {
			return
		}
		steps := p.Steps()
		if steps[0].Artifact == nil {
			t.Fatal("executed step has no artifact")
		}
		uuid, err := dev.Raw().Uuid()
		if err != nil {
			t.Fatal(err)
		}
		device := fmt.Sprintf("GPU-%x-%x-%x-%x-%x", uuid.Bytes[:4], uuid.Bytes[4:6], uuid.Bytes[6:8], uuid.Bytes[8:10], uuid.Bytes[10:])
		target := cudaprof.ProfileTarget{CodeDigest: steps[0].Artifact.Digest, Region: steps[0].NodeName, Workload: "cudaplan-native-kernel:inputs=" + integrationInputsHash(data), Device: device, CollectorMode: "pc_sampling"}
		t.Logf("expected_sidecar_target=%+v", target)
		if _, err := snapshot.Guidance(target); err != nil {
			t.Fatal(err)
		}
		stale := target
		stale.CodeDigest = hex.EncodeToString(make([]byte, sha256.Size))
		if _, err := snapshot.Guidance(stale); !errors.Is(err, cudaprof.ErrProfileTargetMismatch) {
			t.Fatal("stale artifact accepted", err)
		}
		checked = true
		t.Logf("sidecar_sha256=%x executed_target=%+v", sha256.Sum256(raw), target)
	})
	if !checked {
		t.Fatal("no completed numerical execution consumed the sidecar")
	}
}

func integrationInputsHash(data map[string][]float32) string {
	names := make([]string, 0, len(data))
	for name := range data {
		names = append(names, name)
	}
	slices.Sort(names)
	h := sha256.New()
	var word [8]byte
	for _, name := range names {
		binary.LittleEndian.PutUint64(word[:], uint64(len(name)))
		h.Write(word[:])
		h.Write([]byte(name))
		binary.LittleEndian.PutUint64(word[:], uint64(len(data[name])))
		h.Write(word[:])
		for _, value := range data[name] {
			binary.LittleEndian.PutUint32(word[:4], math.Float32bits(value))
			h.Write(word[:4])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// These synthetic controls exercise receipt admission, not GPU execution.
func TestIntegrationReceiptRefusesContradictions(t *testing.T) {
	planned := []cudaplan.StepPlan{{NodeName: "multiply", Backend: cudaplan.BackendLaunch}, {NodeName: "add", Backend: cudaplan.BackendLaunch}}
	good := []cudaplan.ExecutedStep{{Index: 0, NodeName: "multiply", Backend: cudaplan.BackendLaunch}, {Index: 1, NodeName: "add", Backend: cudaplan.BackendLaunch}}
	if err := checkIntegrationReceipt(planned, good, true, false); err != nil {
		t.Fatal(err)
	}
	graph := append([]cudaplan.ExecutedStep(nil), good...)
	for i := range graph {
		graph[i].Backend = cudaplan.BackendGraph
	}
	if err := checkIntegrationReceipt(planned, graph, true, true); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name  string
		steps []cudaplan.ExecutedStep
		done  bool
	}{
		{"unrequested graph", graph, true},
		{"late completion", good, false}, {"missing launch", good[:1], true},
		{"wrong node", []cudaplan.ExecutedStep{{Index: 0, NodeName: "other", Backend: cudaplan.BackendLaunch}, good[1]}, true},
		{"wrong index", []cudaplan.ExecutedStep{{Index: 1, NodeName: "multiply", Backend: cudaplan.BackendLaunch}, good[1]}, true},
		{"wrong backend", []cudaplan.ExecutedStep{{Index: 0, NodeName: "multiply", Backend: cudaplan.BackendCuBLASLt}, good[1]}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := checkIntegrationReceipt(planned, tt.steps, tt.done, false); err == nil {
				t.Fatal("contradictory execution receipt accepted")
			}
		})
	}
}

func TestIntegrationSidecarUsesCompiledArtifactIdentity(t *testing.T) {
	snapshot, registry, output := graphFixture(t, "sm_121a", false)
	plan, err := cudaplan.Compile(snapshot, cudaplan.WithKernelRegistry(registry))
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	step := plan.Steps()[0]
	if step.Artifact == nil {
		t.Fatal("no compiler artifact")
	}
	if err := step.Artifact.Validate(); err != nil {
		t.Fatal(err)
	}
	data := replayInputs(0)
	target := cudaprof.ProfileTarget{CodeDigest: step.Artifact.Digest, Region: step.NodeName, Workload: output + ":inputs=" + integrationInputsHash(data), Device: "synthetic-device", CollectorMode: "synthetic_fixture"}
	sidecar, err := cudaprof.NewProfileSnapshot(cudaprof.ProfileSnapshot{Version: 1, CodeDigest: target.CodeDigest, Region: target.Region, Workload: target.Workload, TokensHash: target.TokensHash, Device: target.Device, CollectorMode: target.CollectorMode, Coverage: 1, Metrics: map[string]int64{"stall": 1}, MetricUnits: map[string]string{"stall": "samples"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sidecar.Guidance(target); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*cudaprof.ProfileTarget){func(p *cudaprof.ProfileTarget) { p.CodeDigest = hex.EncodeToString(make([]byte, 32)) }, func(p *cudaprof.ProfileTarget) { p.Region += "-stale" }, func(p *cudaprof.ProfileTarget) { p.Device = "wrong-device" }, func(p *cudaprof.ProfileTarget) {
		p.Workload = output + ":inputs=" + integrationInputsHash(replayInputs(1))
	}} {
		wrong := target
		mutate(&wrong)
		if _, err := sidecar.Guidance(wrong); err == nil {
			t.Fatal("mismatched sidecar accepted")
		}
	}
}
