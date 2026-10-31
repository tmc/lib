package cudaprof

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprofilehistory"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func TestHistoryTrialCUDAEvent(t *testing.T) {
	q := HistoryTrialQualification{Basis: "event interval", Producer: "test", Locator: "offline", Samples: 1, Loss: &cudaevidence.Loss{Total: 1}}
	trial := cudatrial.Receipt{Complete: true}
	if err := trial.RecordTiming(cudatrial.TimingCUDAEvent, 123*time.Nanosecond); err != nil {
		t.Fatal(err)
	}
	r, err := HistoryTrial(historyTrialFixture(t), trial, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, q)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Facts) != 1 || r.Facts[0].Name != string(cudatrial.TimingCUDAEvent) || r.Facts[0].Qualified {
		t.Fatalf("facts = %+v", r.Facts)
	}
	v := r.Facts[0].Value.Record()
	if v.Number == nil || *v.Number != 123 || v.Unit != "ns" {
		t.Fatalf("event value = %+v", v)
	}
	unknown := cudatrial.Receipt{Complete: true, Timings: map[cudatrial.TimingKind]time.Duration{"unknown": time.Nanosecond}}
	if _, err := HistoryTrial(historyTrialFixture(t), unknown, cudaprofilehistory.Correctness{Verdict: cudaprofilehistory.Unknown}, q); err == nil {
		t.Fatal("unknown timing accepted")
	}
}
