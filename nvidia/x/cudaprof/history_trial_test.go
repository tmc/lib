package cudaprof

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func historyTrialFixture(t *testing.T) cudaprofilehistory.Record {
	t.Helper()
	schema, e := cudakernel.NewSchema()
	if e != nil {
		t.Fatal(e)
	}
	a, e := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_90")
	if e != nil {
		t.Fatal(e)
	}
	compiler := cudacache.CompilerIdentity{NVRTCVersion: "13", DriverSONAME: "libcuda", TargetArch: "sm_90", OutputKind: nvrtcx.OutputPTX}
	workload := cudacache.WorkloadIdentity{ProgramID: "test", ShapeDims: []int{1}, NumericalMode: "fp32"}
	key := cudacache.CompilerCacheKey("source-input", compiler)
	d := strings.Repeat("1", 64)
	return cudaprofilehistory.Record{Version: 1, SourceDigest: d, InputDigest: d, PlanDigest: d, CompilerInputKey: "source-input", Compiler: compiler, CompilerKey: key, Workload: workload, WorkloadKey: workload.Key(), NumericalMode: "fp32", Entry: "kernel", Target: "sm_90", Artifact: a, Identity: cudaevidence.Identity{ArtifactDigest: a.Digest, Entry: "kernel", Target: "sm_90", CompilerKey: key, WorkloadKey: workload.Key()}}
}

func TestHistoryTrialKeepsExactInt64TimingsAndRefusesDetachedCorrectness(t *testing.T) {
	for _, ns := range []int64{1 << 53, (1 << 53) + 1, 1 << 60, math.MaxInt64} {
		trial := cudatrial.Receipt{Complete: true, Timings: map[cudatrial.TimingKind]time.Duration{cudatrial.TimingGPUBusy: time.Duration(ns)}}
		q := HistoryTrialQualification{Native: true, Basis: "exact supplied timing", Producer: "producer", Locator: "run", Samples: 1, Loss: &cudaevidence.Loss{Total: 1}}
		r, err := HistoryTrial(historyTrialFixture(t), trial, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, q)
		if err != nil {
			t.Fatal(err)
		}
		v := r.Facts[0].Value.Record()
		exact := ns == 1<<53 || ns == 1<<60
		if exact != (v.Number != nil) || (!exact && r.Facts[0].Qualified) {
			t.Fatalf("rounded timing promoted ns=%d fact=%+v", ns, r.Facts[0])
		}
	}
	base := historyTrialFixture(t)
	d := strings.Repeat("1", 64)
	c := cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Pass, Method: "digest-exact", Oracle: "oracle", ExpectedDigest: d, ActualDigest: d}
	if _, err := HistoryTrial(base, cudatrial.Receipt{Complete: true, Digest: strings.Repeat("2", 64)}, c, HistoryTrialQualification{}); err == nil {
		t.Fatal("detached correctness accepted")
	}
}

func TestHistoryTrialRetainsFailuresAndDoesNotInventTiming(t *testing.T) {
	for _, kind := range []string{"legacy duration", "failed", "explicit unqualified", "explicit qualified"} {
		t.Run(kind, func(t *testing.T) {
			trial := cudatrial.Receipt{Name: "arm", Complete: true, DeviceDuration: 100 * time.Nanosecond, WallElapsed: 20 * time.Nanosecond, Started: time.Unix(1, 0), Finished: time.Unix(2, 0)}
			q := HistoryTrialQualification{Basis: "actual explicit channel", Producer: "native-fixture", Locator: "run", Samples: 4, Loss: &cudaevidence.Loss{Total: 4}}
			if kind == "failed" {
				trial.Complete = false
				trial.Err = errors.New("rejected numerical output")
			}
			if strings.HasPrefix(kind, "explicit") {
				trial.Timings = map[cudatrial.TimingKind]time.Duration{cudatrial.TimingGPUBusy: 0}
			}
			if kind == "explicit qualified" {
				q.Native = true
			}
			got, err := HistoryTrial(historyTrialFixture(t), trial, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, q)
			if err != nil {
				t.Fatal(err)
			}
			if got.Correctness.Verdict != cudaprofilehistory.Unknown {
				t.Fatal("Complete became correctness pass")
			}
			if kind == "legacy duration" && len(got.Facts) != 0 {
				t.Fatal("duration fallback invented timing")
			}
			if kind == "failed" && (got.Status != "execution_failed" || !strings.Contains(string(got.Receipt), "rejected numerical output")) {
				t.Fatal("rejected trial omitted")
			}
			if strings.HasPrefix(kind, "explicit") {
				if len(got.Facts) != 1 || got.Facts[0].Value.Record().Number == nil || *got.Facts[0].Value.Record().Number != 0 || got.Facts[0].Qualified != q.Native {
					t.Fatal("explicit measured zero lost")
				}
			}
			h := sha256.Sum256(got.Receipt)
			if got.ReceiptDigest != hex.EncodeToString(h[:]) {
				t.Fatal("adapter receipt digest mismatch")
			}
		})
	}
}

func TestHistoryTrialMissingImageCannotCarryPass(t *testing.T) {
	r := historyTrialFixture(t)
	r.Artifact = nil
	r.Identity = cudaevidence.Identity{}
	d := strings.Repeat("1", 64)
	c := cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Pass, Method: "digest-exact", Oracle: "oracle", ActualDigest: d, ExpectedDigest: d}
	if _, err := HistoryTrial(r, cudatrial.Receipt{}, c, HistoryTrialQualification{}); err == nil {
		t.Fatal("missing image has correctness pass")
	}
}

type historyUnencodableError struct{ Callback func() }

func (historyUnencodableError) Error() string { return "actual failure" }
func TestHistoryTrialUnencodableErrorRetained(t *testing.T) {
	trial := cudatrial.Receipt{Name: "actual attempt", Err: historyUnencodableError{Callback: func() {}}}
	got, err := HistoryTrial(historyTrialFixture(t), trial, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, HistoryTrialQualification{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != "execution_failed" || !strings.Contains(string(got.Receipt), "actual failure") || !strings.Contains(string(got.Receipt), "actual attempt") {
		t.Fatalf("failed attempt lost: %+v", got)
	}
}
func TestHistoryTrialReceiptBoundBeforeEncoding(t *testing.T) {
	trial := cudatrial.Receipt{Provenance: map[string]string{"oversized": strings.Repeat("x", 1<<20)}}
	if _, err := HistoryTrial(historyTrialFixture(t), trial, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, HistoryTrialQualification{}); err == nil {
		t.Fatal("oversized receipt admitted")
	}
}
