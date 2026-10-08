package cudasanitizer_test

import (
	"bytes"
	"crypto/sha256"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasanitizer"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// This exercises host state and parsed report association. The events and log
// are test inputs, with no device producer or live sanitizer verdict implied.
func TestHostCoveragePersistenceNoveltyAndCrashAssociation(t *testing.T) {
	p, err := cudatrial.NewPersistentCoverage(32, 2)
	if err != nil {
		t.Fatal(err)
	}
	g, err := cudasanitizer.NewTraceNovelty(cudasanitizer.GatingCoverageIncrease, 32, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []struct {
		blocks []uint64
		want   bool
	}{
		{[]uint64{5, 9}, true},
		{[]uint64{5, 9}, false},
		{[]uint64{5, 7}, true},
	} {
		if err := p.Notify(cudatrial.SignalStart); err != nil {
			t.Fatal(err)
		}
		for _, block := range step.blocks {
			if _, err := p.Record(0, block); err != nil {
				t.Fatal(err)
			}
		}
		if err := p.Notify(cudatrial.SignalEnd); err != nil {
			t.Fatal(err)
		}
		if got, err := g.Evaluate(p.LastComplete().Bytes()); err != nil || got != step.want {
			t.Fatalf("evaluate %v = %v, %v; want %v", step.blocks, got, err, step.want)
		}
	}
	last := p.LastComplete().Bytes()
	merged := p.Merged().Bytes()
	if last[21] != 2 || last[27] != 0 || merged[27] != 1 {
		t.Fatal("completed trace and campaign union were conflated")
	}
	if err := p.Notify(cudatrial.SignalStart); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Record(1, 15); err != nil {
		t.Fatal(err)
	}
	p.Fault()
	if !bytes.Equal(last, p.LastComplete().Bytes()) || !bytes.Equal(merged, p.Merged().Bytes()) {
		t.Fatal("fault altered completed evidence")
	}
	if err := p.Notify(cudatrial.SignalStart); err == nil {
		t.Fatal("faulted owner resumed")
	}
	identity := cudasanitizer.CrashIdentity{
		Kernel: "host-test", Image: sha256.Sum256([]byte("image")),
		Epoch: "epoch", Trial: "preceding", Input: sha256.Sum256([]byte("preceding")), Ordinal: 2,
	}
	failure := identity
	failure.Trial, failure.Input, failure.Ordinal = "failing", sha256.Sum256([]byte("failing")), 3
	trace := cudasanitizer.CompletedCrashTrace{Identity: identity, Complete: true, MapSize: 32, Bitmap: p.LastComplete()}
	result := &cudasanitizer.Result{
		Config: cudasanitizer.Config{Tool: cudasanitizer.Memcheck}, Outcome: cudasanitizer.OutcomeSanitizerErrors,
		Log: cudasanitizer.ParseLog("========= Invalid __global__ write of size 4 bytes\n=========\n========= ERROR SUMMARY: 1 error\n"),
	}
	observation, err := cudasanitizer.CrashSignature(result, trace, failure)
	if err != nil {
		t.Fatal(err)
	}
	if observation.TraceDigest != sha256.Sum256(last) || observation.Preceding != identity || observation.Failure != failure {
		t.Fatal("crash lost preceding trace or distinct failing identity")
	}
	trace.Bitmap = p.Merged()
	unionObservation, err := cudasanitizer.CrashSignature(result, trace, failure)
	if err != nil || unionObservation.Signature == observation.Signature {
		t.Fatal("campaign union and preceding trace had the same signature", err)
	}
}
