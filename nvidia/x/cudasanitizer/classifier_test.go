package cudasanitizer

import (
	"crypto/sha256"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestSanitizerClassifier_DeduplicatesByTraceHash(t *testing.T) {
	classifier := NewSanitizerClassifier()

	m1, err := cudatrial.NewThreadEdgeMap(32, 1)
	if err != nil {
		t.Fatal(err)
	}
	m1.Record(0, 3)

	p1 := CrashIdentity{
		Kernel:  "test_kernel",
		Image:   sha256.Sum256([]byte("img")),
		Epoch:   "ep1",
		Trial:   "t0",
		Input:   sha256.Sum256([]byte("inp0")),
		Ordinal: 0,
	}
	f1 := p1
	f1.Trial, f1.Input, f1.Ordinal = "t1", sha256.Sum256([]byte("inp1")), 1

	res1 := &Result{
		Config:  Config{Tool: Memcheck},
		Outcome: OutcomeSanitizerErrors,
		Log:     ParseLog("========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n"),
	}

	trace1 := CompletedCrashTrace{
		Identity: p1,
		Complete: true,
		MapSize:  32,
		Bitmap:   m1.Snapshot(),
	}

	// First defect classification: must be novel
	obs1, isNovel1, err := classifier.Classify(res1, trace1, f1)
	if err != nil {
		t.Fatal(err)
	}
	if !isNovel1 {
		t.Fatal("first crash should be novel")
	}
	if res1.CrashObservation == nil || res1.CrashObservation.Signature != obs1.Signature {
		t.Fatal("result did not retain crash observation")
	}

	// Second defect from another trial but with identical trace hash & finding:
	// must be recognized as duplicate (isNovel == false) with matching signature.
	f2 := f1
	f2.Trial, f2.Input = "t2", sha256.Sum256([]byte("inp2"))
	p2 := p1
	p2.Trial, p2.Input = "t1_prev", sha256.Sum256([]byte("inp1_prev"))
	trace2 := CompletedCrashTrace{
		Identity: p2,
		Complete: true,
		MapSize:  32,
		Bitmap:   m1.Snapshot(), // identical trace
	}

	res2 := &Result{
		Config:  Config{Tool: Memcheck},
		Outcome: OutcomeSanitizerErrors,
		Log:     ParseLog("========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n"),
	}

	obs2, isNovel2, err := classifier.Classify(res2, trace2, f2)
	if err != nil {
		t.Fatal(err)
	}
	if isNovel2 {
		t.Fatal("duplicate crash trace & finding was incorrectly marked novel")
	}
	if obs2.Signature != obs1.Signature {
		t.Fatalf("signatures should match: %x != %x", obs2.Signature, obs1.Signature)
	}
	if classifier.UniqueCount() != 1 {
		t.Fatalf("expected 1 unique crash, got %d", classifier.UniqueCount())
	}

	// Third defect with a DIFFERENT trace (different edge executed): must be novel!
	m2, _ := cudatrial.NewThreadEdgeMap(32, 1)
	m2.Record(0, 5) // different block
	trace3 := CompletedCrashTrace{
		Identity: p1,
		Complete: true,
		MapSize:  32,
		Bitmap:   m2.Snapshot(),
	}
	res3 := &Result{
		Config:  Config{Tool: Memcheck},
		Outcome: OutcomeSanitizerErrors,
		Log:     ParseLog("========= Invalid __global__ write of size 4 bytes\n========= ERROR SUMMARY: 1 error\n"),
	}

	obs3, isNovel3, err := classifier.Classify(res3, trace3, f1)
	if err != nil {
		t.Fatal(err)
	}
	if !isNovel3 {
		t.Fatal("distinct trace crash should be novel")
	}
	if obs3.Signature == obs1.Signature {
		t.Fatal("distinct trace should produce distinct signature")
	}
	if classifier.UniqueCount() != 2 {
		t.Fatalf("expected 2 unique crashes, got %d", classifier.UniqueCount())
	}
}
