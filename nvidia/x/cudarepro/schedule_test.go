package cudarepro

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func scheduledCase(t *testing.T) Case {
	t.Helper()
	src, err := os.ReadFile("testdata/lane59-pristine.ptx")
	if err != nil {
		t.Fatal(err)
	}
	r, err := cudaschedule.Perturb(string(src), cudaschedule.Options{Seed: 17, Intensity: 1})
	if err != nil || len(r.Report.Decisions) < 2 {
		t.Fatalf("fixture decisions: %+v, %v", r.Report, err)
	}
	return Case{Source: string(src), Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{1, 1, 1}, Schedule: &Schedule{Options: cudaschedule.Options{Seed: 17, Intensity: 1}, Decisions: r.Report.Decisions}}
}

func scheduledRecord(t *testing.T, c Case) cudaschedule.RunRecord {
	t.Helper()
	ptx, err := c.Artifact()
	if err != nil {
		t.Fatal(err)
	}
	r := mismatchRecord(0)
	r.ExpectedPTXDigest = fmt.Sprintf("%x", sha256.Sum256([]byte(ptx)))
	r.PTXDigest = r.ExpectedPTXDigest
	return r
}

func TestReduceRecordedSites(t *testing.T) {
	original := scheduledCase(t)
	frozen := cloneCase(original)
	essential := original.Schedule.Decisions[1].Site.Ordinal
	emptyExecuted := false
	r, err := Reduce(context.Background(), original, scheduledRecord(t, original), ReduceOptions{Repeats: 3, MaxCandidates: 100}, func(_ context.Context, c Case) (cudaschedule.RunRecord, error) {
		run := scheduledRecord(t, c)
		run.Stdout = "PASS"
		if len(c.Schedule.Decisions) == 0 {
			emptyExecuted = true
		}
		for _, d := range c.Schedule.Decisions {
			if d.Site.Ordinal == essential {
				run.Stdout = "MISMATCH"
			}
		}
		// Caller mutation must not alter the case retained by Reduce.
		c.Schedule.Decisions = nil
		return run, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !emptyExecuted || len(r.Reduced.Schedule.Decisions) != 1 || r.Reduced.Schedule.Decisions[0].Site.Ordinal != essential {
		t.Fatalf("did not retain only essential site: %+v, empty=%v", r.Reduced.Schedule, emptyExecuted)
	}
	if r.Reduced.Source != original.Source || !reflect.DeepEqual(original, frozen) || r.Reduced.Schedule.Options != original.Schedule.Options {
		t.Fatal("edited source, options, or caller decisions")
	}
}

func TestScheduleRefusesUnboundRuns(t *testing.T) {
	for _, phase := range []string{"finding", "baseline", "candidate"} {
		t.Run(phase, func(t *testing.T) {
			c := scheduledCase(t)
			finding := scheduledRecord(t, c)
			if phase == "finding" {
				finding.PTXDigest = strings.Repeat("a", 64)
			}
			calls := 0
			_, err := Reduce(context.Background(), c, finding, ReduceOptions{Repeats: 2, MaxCandidates: 10}, func(_ context.Context, c Case) (cudaschedule.RunRecord, error) {
				calls++
				run := scheduledRecord(t, c)
				if phase == "baseline" || phase == "candidate" && calls > 2 {
					run.ExpectedPTXDigest = strings.Repeat("a", 64)
					run.PTXDigest = run.ExpectedPTXDigest
				}
				return run, nil
			})
			if err == nil || phase == "finding" && calls != 0 {
				t.Fatalf("accepted unbound %s: calls=%d, err=%v", phase, calls, err)
			}
		})
	}
}

func TestScheduleDeepCopyAndValidation(t *testing.T) {
	c := scheduledCase(t)
	v := 7
	c.Schedule.Decisions[0].Site.BlockIndex = &v
	copy := cloneCase(c)
	*copy.Schedule.Decisions[0].Site.BlockIndex = 9
	if *c.Schedule.Decisions[0].Site.BlockIndex != 7 {
		t.Fatal("shared site block pointer")
	}
	c.Schedule.Decisions[0].Site.Ordinal = -1
	calls := 0
	_, err := Reduce(context.Background(), c, mismatchRecord(0), ReduceOptions{Repeats: 2, MaxCandidates: 10}, func(context.Context, Case) (cudaschedule.RunRecord, error) {
		calls++
		return mismatchRecord(0), nil
	})
	if err == nil || calls != 0 {
		t.Fatal("invalid recorded site executed")
	}
}

func TestScheduleArtifactUsesProducerReplay(t *testing.T) {
	c := scheduledCase(t)
	c.Schedule.Decisions = c.Schedule.Decisions[:2]
	expected, err := cudaschedule.Replay(c.Source, c.Schedule.Options, c.Schedule.Decisions)
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Artifact()
	if err != nil || got != expected.PTX || got == c.Source {
		t.Fatalf("artifact did not replay the recorded subset: %v", err)
	}
	// A different pristine body must not silently bind the same decisions.
	c.Source = strings.Replace(c.Source, c.Schedule.Decisions[0].Site.Text, "ret", 1)
	if _, err := c.Artifact(); err == nil {
		t.Fatal("accepted stale source sites")
	}
}

func ExampleCase_Artifact() {
	c := Case{Source: ".version 8.7\n.target sm_70\n.address_size 64\n.visible .entry k() { ret; }\n", Schedule: &Schedule{Options: cudaschedule.Options{Seed: 17, Intensity: 1}}}
	ptx, err := c.Artifact()
	fmt.Println(ptx == c.Source, err)
	// Output: true <nil>
}

func ExampleSchedule() {
	s := Schedule{Options: cudaschedule.Options{Seed: 17, Intensity: 1}}
	fmt.Println(s.Options.Seed, len(s.Decisions))
	// Output: 17 0
}
