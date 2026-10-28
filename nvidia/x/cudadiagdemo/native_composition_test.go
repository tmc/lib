package cudadiagdemo

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// Real public portable service execution and its retained source observations;
// this is not a native capture or an independently qualified NVIDIA receipt.
func compositionFixture(t *testing.T) (*ServiceFixture, *NativeResult, NativeEvidence, []byte) {
	t.Helper()
	sf := newTestFixture(t)
	baseline, baselineErr := sf.Process(context.Background(), Request{ID: "ordinary", WorkloadName: "same-name", Data: []float64{-3, 0, 2.5}, Mode: ModeBaseline})
	if baselineErr != nil || !baseline.Observation.Complete || !reflect.DeepEqual(baseline.Output, []float64{-5, 1, 6}) {
		t.Fatal("ordinary independent baseline", baseline, baselineErr)
	}
	responses, err := sf.BatchProcess(context.Background(), []Request{{ID: "same", WorkloadName: "same-name", Data: []float64{-3, 0, 2.5}, Mode: ModeHostSubmissionDelay, Delay: 5 * time.Millisecond}, {ID: "same", WorkloadName: "same-name", Data: []float64{-3, 0, 2.5}, Mode: ModeDeviceBottleneck, Delay: 5 * time.Millisecond}}, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, response := range responses {
		if !reflect.DeepEqual(response.Output, []float64{-5, 1, 6}) {
			t.Fatal("independent output reference", response.Output)
		}
	}
	raw, expect, err := sf.captureTiming(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	expect.oraclePassed, expect.baselinePassed = true, baseline.Observation.Complete
	r := &NativeResult{Synthetic: true, Qualification: "synthetic service host control"}
	packet := NativeEvidence{Version: 1, Synthetic: true, Records: []NativeEvidenceRecord{}}
	thresholds := timingThresholds()
	thresholds.BaselinePreEntryDuration = time.Millisecond
	for i, response := range responses {
		attempt := fmt.Sprintf("attempt-%d", i)
		r.Requests = append(r.Requests, NativeRequestLog{Attempt: attempt, StartedNS: response.Observation.Service.Start, FinishedNS: response.Observation.Service.End, Complete: response.Observation.Complete, GroundTruth: response.GroundTruth})
		input, err := decodeTiming(raw, expect, response, thresholds)
		if err != nil {
			t.Fatal(err)
		}
		packet.Records = append(packet.Records, NativeEvidenceRecord{RequestID: "same", Attempt: attempt, OpID: response.OpID, Phase: "pre_entry", Input: input})
	}
	return sf, r, packet, raw
}

func compositionRaw(t *testing.T, packet NativeEvidence) []byte {
	t.Helper()
	raw, err := json.Marshal(packet)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestNativeCompositionPublicConcurrentIdentity(t *testing.T) {
	_, r, packet, _ := compositionFixture(t)
	out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(compositionRaw(t, packet))})
	if err != nil {
		t.Fatal(err)
	}
	if !out.Synthetic || len(out.Requests) != 2 || out.Requests[0].OpID == out.Requests[1].OpID || out.Requests[0].RequestID != out.Requests[1].RequestID || out.Requests[0].Attempt == out.Requests[1].Attempt {
		t.Fatal(out)
	}
	if !out.Requests[0].Diagnosis.Admitted || out.Requests[0].Diagnosis.Mechanism != cudadiag.MechanismHostSubmissionDelay || !out.Requests[1].Diagnosis.Admitted || out.Requests[1].Diagnosis.Mechanism != cudadiag.MechanismDeviceBottleneck {
		t.Fatal(out.Requests)
	}
	r.Requests[0].GroundTruth.Mode = ModeDeviceBottleneck
	r.Requests[0].GroundTruth.HostPrepDelay = time.Hour
	other, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(compositionRaw(t, packet))})
	if err != nil || !reflect.DeepEqual(out.Requests, other.Requests) {
		t.Fatal("ground truth manufactured diagnosis", other, err)
	}
}

func TestNativeCompositionEvidenceRefusals(t *testing.T) {
	for _, mode := range []string{"clock", "loss", "wrong phase", "wrong op", "wrong attempt", "wrong thread", "duplicate", "missing", "incomplete", "native scope", "cancel", "byte bound", "request bound", "scheduler thread", "operation alias"} {
		t.Run(mode, func(t *testing.T) {
			_, r, packet, _ := compositionFixture(t)
			options := NativeCompositionOptions{}
			wantErr := false
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch mode {
			case "clock":
				packet.Records[0].Input.Receipt.ClockDomain = "other"
			case "loss":
				loss := packet.Records[0].Input.Receipt.LostRecords["activity"]
				loss.Unknown = true
				packet.Records[0].Input.Receipt.LostRecords["activity"] = loss
			case "wrong phase":
				packet.Records[0].Phase = "api_scheduler"
				wantErr = true
			case "wrong op":
				packet.Records[0].OpID++
				wantErr = true
			case "wrong attempt":
				packet.Records[0].Attempt = "other"
				wantErr = true
			case "wrong thread":
				packet.Records[0].Input.PreEntries[0].Submission.TID++
				wantErr = true
			case "duplicate":
				packet.Records = append(packet.Records, packet.Records[0])
				wantErr = true
			case "missing":
				packet.Records = packet.Records[1:]
			case "incomplete":
				r.Requests[0].Complete = false
			case "native scope":
				r.Synthetic = false
				wantErr = true
			case "cancel":
				cancel()
				wantErr = true
			case "byte bound":
				options.MaxBytes = 1
				wantErr = true
			case "request bound":
				options.MaxRequests = 1
				wantErr = true
			case "operation alias":
				r.Requests[1].GroundTruth.OpID = r.Requests[0].GroundTruth.OpID
				packet.Records[1].OpID = packet.Records[0].OpID
				packet.Records[1].Input.PreEntries[0].Submission.OpID = packet.Records[0].OpID
				wantErr = true
			case "scheduler thread":
				packet.Records[0].Phase = "api_scheduler"
				packet.Records[0].Input.PreEntries = nil
				packet.Records[0].Input.Sched.TID = r.Requests[0].GroundTruth.TID + 1
				wantErr = true
			}
			options.Evidence = bytes.NewReader(compositionRaw(t, packet))
			out, err := ComposeNativeEvidence(ctx, r, options)
			if wantErr {
				if err == nil {
					t.Fatal("invalid evidence admitted", out)
				}
				if mode == "operation alias" && !strings.Contains(err.Error(), "operation assigned to multiple") {
					t.Fatal("alias not refused at population identity gate", err)
				}
				if mode == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
				return
			}
			if err != nil || len(out.Requests) != 2 || out.Requests[0].Diagnosis.Admitted {
				t.Fatal(out, err)
			}
		})
	}
}

func TestNativeCompositionPartialAndJSON(t *testing.T) {
	_, r, packet, _ := compositionFixture(t)
	r.Synthetic = false
	r.HostClockDomain = "process_monotonic"
	r.HostClockEpochID = "unqualified-host-control"
	r.Trials = []NativeTrial{{RequestID: r.Requests[0].GroundTruth.RequestID, Attempt: r.Requests[0].Attempt, Calls: []NativeCall{{Name: "cuLaunchKernel", StartNS: r.Requests[0].StartedNS, EndNS: r.Requests[0].StartedNS + 1}}}}
	out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{})
	if err != nil || out.Synthetic || len(out.Requests) != 2 || out.Requests[0].Diagnosis.Admitted || out.Requests[1].Diagnosis.Admitted || out.Requests[0].Refusal == "" {
		t.Fatal(out, err)
	}
	if len(out.Requests[0].Observations.HostCalls) != 1 || !out.Requests[0].Observations.Receipt.Partial || len(out.Requests[0].Observations.Kernels) != 0 || out.Requests[0].Observations.Receipt.ClockUncertaintyNS != nil {
		t.Fatal("partial native wrapper facts not retained or clock fabricated", out)
	}
	r.Synthetic = true
	raw := compositionRaw(t, packet)
	for _, data := range []string{strings.Replace(string(raw), `"version":1`, `"version":2`, 1), strings.Replace(string(raw), `"version":1`, `"version":1,"version":1`, 1), string(raw) + " null", string(raw[:len(raw)-1])} {
		if _, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: strings.NewReader(data)}); err == nil {
			t.Fatal("invalid version/duplicate/truncated evidence admitted")
		}
	}
}

func TestNativeCompositionExistingReplayChain(t *testing.T) {
	sf, r, packet, source := compositionFixture(t)
	raw := compositionRaw(t, packet)
	dir := t.TempDir()
	ref := func(name string, body []byte, role string) cudaregress.ArtifactRef {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, body, 0600); err != nil {
			t.Fatal(err)
		}
		return cudaregress.ArtifactRef{Name: name, Path: path, Digest: digestBytes(body), SizeBytes: int64(len(body)), Role: role}
	}
	capture := ref("observations.json", raw, "trace")
	sourceRef := ref("service-capture.tar", source, "source")
	input := ref("input.json", []byte(`[-3,0,2.5]`), "input")
	reference := ref("reference.json", []byte(`[-5,1,6]`), "reference")
	environment := cudaregress.Environment{Model: "portable synthetic service", Clock: portableClock}
	counter := 0
	runner := cudaregress.ReplayFunc(func(ctx context.Context, arm string, round int) (cudatrial.Receipt, []float64, error) {
		counter++
		request := Request{ID: fmt.Sprintf("executed-%d", counter), Data: []float64{-3, 0, 2.5}, Mode: ModeHostSubmissionDelay, Delay: 5 * time.Millisecond}
		if arm == "candidate" {
			request.Mode = ModeBaseline
			request.Delay = 0
		}
		start := time.Now()
		response, err := sf.Process(ctx, request)
		finish := time.Now()
		if err != nil {
			return cudatrial.Receipt{}, nil, err
		}
		receipt := cudatrial.Receipt{Complete: true, Started: start, Finished: finish, WallElapsed: finish.Sub(start), WallNs: finish.Sub(start).Nanoseconds(), TimingKind: cudatrial.TimingWallElapsed}
		if err := receipt.RecordTiming(cudatrial.TimingWallElapsed, finish.Sub(start)); err != nil {
			return receipt, nil, err
		}
		return receipt, response.Output, nil
	})
	replay := &NativeReplay{RequestID: "same", Attempt: r.Requests[0].Attempt, Capture: capture, Environment: environment, Runner: runner, PlanOptions: cudaregress.PlanOptions{WarmupRuns: 1, Rounds: 3, Order: "ABBA", ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingWallElapsed, Interval: "portable_service"}, Sources: []cudaregress.ArtifactRef{capture, sourceRef}, Inputs: []cudaregress.InputRef{{Name: input.Name, Path: input.Path, Digest: input.Digest, ShapeDims: []int64{3}, NumericalMode: "float64_cpu"}}, Oracle: cudaregress.CorrectnessOracle{Method: "numerical-tolerance", Tolerance: 1e-6, ReferencePath: reference.Path, ReferenceDigest: reference.Digest}, NullControls: cudaregress.NullControlConfig{Enabled: true, BasketRuns: 3, MaxNoiseFloorNs: 100 * time.Millisecond}, StoppingRules: cudaregress.StoppingRules{MaxTrials: 10, RelativeThreshold: .1, MinSamples: 3, HaltOnDrift: false}}, ReplayOptions: cudaregress.ReplayOptions{CurrentEnvironment: environment, ReferenceValues: []float64{-5, 1, 6}}}
	out, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(raw), Replay: replay})
	if err != nil || out.Incident == nil || out.Plan == nil || out.PlanReceipt == nil || !out.PlanReceipt.ArtifactClosureValid || out.Execution == nil || !out.Execution.OutputsVerified || out.Execution.RoundsCompleted != 3 || out.Comparison == nil || counter == 0 {
		t.Fatal(out, err, counter)
	}
	// Device-only evidence never starts an intervention for host delay.
	replay.Attempt = r.Requests[1].Attempt
	before := counter
	refused, err := ComposeNativeEvidence(context.Background(), r, NativeCompositionOptions{Evidence: bytes.NewReader(raw), Replay: replay})
	if err != nil || refused.ReplayRefusal == "" || refused.Plan != nil || counter != before {
		t.Fatal(refused, err)
	}
}
