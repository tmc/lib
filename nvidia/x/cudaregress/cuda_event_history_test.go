package cudaregress_test

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudaregress"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestCUDAEventReplayHistoryCompare(t *testing.T) {
	inc := validAdmittedIncident()
	inc.ObservedEnvironment.Domain = "test-domain"
	const digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	plan, _, err := cudaregress.NewPlanFromIncident(inc, cudaregress.PlanOptions{
		Oracle:         cudaregress.CorrectnessOracle{Method: "digest-exact", ExpectedDigest: digest},
		Rounds:         3,
		ExecutionScope: cudaregress.TimingScope{Channel: cudatrial.TimingCUDAEvent},
		StoppingRules:  cudaregress.StoppingRules{RelativeThreshold: 0.1, MinSamples: 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_121a\n.address_size 64\n.visible .entry matmul() { ret; }\n"), "matmul", cudakernel.Schema{}, "sm_121a")
	if err != nil {
		t.Fatal(err)
	}
	link := cudaregress.HistoryLinkOptions{CompilerInputKey: "test-source", Artifact: artifact, Compiler: cudacache.CompilerIdentity{NVRTCVersion: "12.8", DriverSONAME: "libcuda.so.1", TargetArch: "sm_121a"}, Workload: cudacache.WorkloadIdentity{ProgramID: "gemm_repro", NumericalMode: "fp32", ShapeDims: []int{128, 128}}, Entry: "matmul", Qualified: true}
	makeRecord := func(candidate time.Duration) cudaprofilehistory.Record {
		t.Helper()
		runner := cudaregress.ReplayFunc(func(ctx context.Context, name string, round int) (cudatrial.Receipt, []float64, error) {
			d := 100 * time.Nanosecond
			if name == "candidate" {
				d = candidate
			}
			r := cudatrial.Receipt{Name: name, Round: round, Complete: true, Digest: digest}
			if err := r.RecordTiming(cudatrial.TimingCUDAEvent, d); err != nil {
				return r, nil, err
			}
			return r, nil, nil
		})
		exec, comparison, err := cudaregress.Replay(context.Background(), plan, runner, cudaregress.ReplayOptions{CurrentEnvironment: inc.ObservedEnvironment})
		if err != nil {
			t.Fatal(err)
		}
		record, err := cudaregress.LinkToProfileHistory(plan, exec, comparison, link)
		if err != nil {
			t.Fatal(err)
		}
		if len(record.Facts) != 1 || record.Facts[0].Name != string(cudatrial.TimingCUDAEvent) {
			t.Fatalf("facts = %+v", record.Facts)
		}
		return record
	}
	before, after := makeRecord(100*time.Nanosecond), makeRecord(130*time.Nanosecond)
	policy := cudaprofilehistory.Policy{Timing: string(cudatrial.TimingCUDAEvent), RelativeThreshold: 0.1, MinSamples: 3}
	comparison, err := cudaprofilehistory.Compare(before, after, policy)
	if err != nil || comparison.Performance != cudaprofilehistory.Fail {
		t.Fatalf("event history compare = %+v, %v", comparison, err)
	}
	policy.Timing = string(cudatrial.TimingGPUBusy)
	comparison, err = cudaprofilehistory.Compare(before, after, policy)
	if err != nil || comparison.Performance != cudaprofilehistory.Unknown {
		t.Fatalf("missing busy compare = %+v, %v", comparison, err)
	}
	policy.Timing = "unknown"
	if _, err := cudaprofilehistory.Compare(before, after, policy); err == nil {
		t.Fatal("unknown channel accepted")
	}
}
