package cudaregress

import (
	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"testing"
	"time"
)

func historyFixture(t *testing.T) (*ExperimentPlan, *ExecutionReceipt, *ComparisonReceipt, HistoryLinkOptions) {
	t.Helper()
	p := &ExperimentPlan{Version: Version, PlanID: "p", IncidentID: "i", Rounds: 1, ExecutionOrder: "ABBA", StoppingRules: StoppingRules{MaxTrials: 2}, TimingScope: TimingScope{Channel: cudatrial.TimingWallElapsed}}
	var err error
	p.PlanDigest, err = ComputePlanDigest(p)
	if err != nil {
		t.Fatal(err)
	}
	e := &ExecutionReceipt{ExecutionID: "e", PlanID: p.PlanID, PlanDigest: p.PlanDigest, OutputsVerified: true, EnvironmentObserved: Environment{Model: "test-device", Driver: "test-driver", CUDA: "test-cuda", Clock: "test-clock", Domain: "test-domain"}, ArmReceipts: []cudatrial.Receipt{{Name: "candidate", Complete: true, WallElapsed: time.Millisecond}}}
	c := &ComparisonReceipt{ExecutionID: e.ExecutionID, PlanID: p.PlanID, CandidateArm: "candidate", TimingChannel: cudatrial.TimingWallElapsed, CandidateDuration: time.Millisecond, Verdict: cudaprofilehistory.Pass}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry f() { ret; }\n"), "f", cudakernel.Schema{}, "sm_90")
	if err != nil {
		t.Fatal(err)
	}
	o := HistoryLinkOptions{CompilerInputKey: "test-input", Compiler: cudacache.CompilerIdentity{TargetArch: "sm_90", NVRTCVersion: "test-version", DriverSONAME: "test-library"}, Workload: cudacache.WorkloadIdentity{ProgramID: "test-workload", NumericalMode: "fp32", ShapeDims: []int{1}}, Entry: "f", Artifact: a}
	return p, e, c, o
}

func TestHistoryRejectsUnprovenClosure(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ExperimentPlan, *ExecutionReceipt, *ComparisonReceipt, *HistoryLinkOptions)
	}{
		{"artifact", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			o.Artifact = nil
		}},
		{"compiler", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			o.Compiler.NVRTCVersion = ""
		}},
		{"clock", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			e.EnvironmentObserved.Clock = ""
		}},
		{"workload", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			o.Workload.NumericalMode = ""
		}},
		{"plan", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			e.PlanDigest = "other"
		}},
		{"execution", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			c.ExecutionID = "other"
		}},
		{"channel", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			c.TimingChannel = cudatrial.TimingGPUBusy
		}},
		{"partial", func(p *ExperimentPlan, e *ExecutionReceipt, c *ComparisonReceipt, o *HistoryLinkOptions) {
			e.Partial = true
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, e, c, o := historyFixture(t)
			tt.mutate(p, e, c, &o)
			if _, err := LinkToProfileHistory(p, e, c, o); err == nil {
				t.Fatal("unproven closure accepted")
			}
		})
	}
	p, e, c, o := historyFixture(t)
	r, err := LinkToProfileHistory(p, e, c, o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Environment.Clock != "test-clock" || r.Artifact.Digest != o.Artifact.Digest || r.Facts[0].Samples != 1 {
		t.Fatalf("changed supplied closure: %+v", r)
	}
}

func TestComparisonWithholdsCandidateOnlyFamily(t *testing.T) {
	p, e, _, _ := historyFixture(t)
	p.StoppingRules.MinSamples = 1
	e.ArmReceipts = append(e.ArmReceipts, cudatrial.Receipt{Name: "baseline", Complete: true, WallElapsed: time.Millisecond})
	c, err := CompareExecution(p, e, ReplayOptions{BaselineFamilies: []string{"activity"}, CandidateFamilies: []string{"activity", "pc"}})
	if err != nil {
		t.Fatal(err)
	}
	if c.Verdict != cudaprofilehistory.Unknown || len(c.MissingFamilies) != 1 || c.MissingFamilies[0] != "pc" {
		t.Fatalf("changed capture family admitted: %+v", c)
	}
}
