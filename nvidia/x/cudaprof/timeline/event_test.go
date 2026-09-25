package timeline

import (
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

// TestEventsReadTheJournalTheyWereHanded pins where the timestamps come from.
// The pprof projection is an aggregate with no timestamps by construction; a
// reader that stopped there reported that a capture with its journal attached
// carried no timestamps, and advise recommended a re-capture -- a run under
// GPU lock to reproduce a file already on disk.
func TestEventsReadTheJournalTheyWereHanded(t *testing.T) {
	b := journaltest.New()
	b.KernelOn(0, 1, 1_000, 3_000, 1)
	b.KernelOn(0, 1, 6_000, 7_000, 2)
	cap := journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus())

	// The projection is present and useless: samples with names and no
	// timestamps, which is what a real capture of this shape holds. A guard on
	// "did the projection produce anything" never fires here.
	cap.Profile = &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample: []*profile.Sample{
			{Value: []int64{2_000}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"gemm"}}},
			{Value: []int64{1_000}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"softmax"}}},
		},
	}

	streams, opaque := Events(cap)
	if opaque {
		t.Error("graph replays reported opaque in a journal with no graph work")
	}
	var got []Event
	for _, evs := range streams {
		got = append(got, evs...)
	}
	if len(got) != 2 {
		t.Fatalf("stream events = %d, want the journal's 2: the untimed projection was kept, or the two sources were merged", len(got))
	}
	for _, ev := range got {
		if ev.Start == 0 || ev.End == 0 {
			t.Errorf("event %+v carries no timestamps: the journal has them", ev)
		}
	}
}

// TestAnyEventTimed asserts the predicate directly on a zero-origin activity
// because no journal can supply one: the decoder treats start == 0 as unset
// and drops the record, so the only way to pin which field is the sentinel is
// here. Testing Start would read a device timeline that begins at zero as
// having no timestamps, and send the reader back to the projection.
func TestAnyEventTimed(t *testing.T) {
	if !anyEventTimed([]Event{{Start: 0, End: 500}}) {
		t.Error("an activity starting at 0 read as untimed: End is the sentinel, not Start")
	}
	if anyEventTimed([]Event{{Start: 0, End: 0}, {Start: 100, End: 100}}) {
		t.Error("an event with no interval read as timed")
	}
}

// TestAPIRecordsComeFromTheJournal covers the sibling of the timeline defect,
// one layer in. A capture written through compactCaptureStatus omits the
// per-record arrays, so a reader looking only at capture-status.json finds no
// host launch records and reports that the capture has none -- while its
// journal held 124834, all correlated.
func TestAPIRecordsComeFromTheJournal(t *testing.T) {
	b := journaltest.New()
	b.KernelOn(0, 1, 5_000, 6_000, 1)
	b.API(1_000, 2_000, 1)
	cap := journaltest.Capture(map[string][]byte{"capture.journal": b.Finalize()}, journaltest.CompleteStatus())

	apis := APIRecords(cap)
	if len(apis) != 1 {
		t.Fatalf("api records = %d, want the journal's 1", len(apis))
	}
	if got := apis[1]; got.Start != 1_000 || got.End != 2_000 {
		t.Errorf("api record = %+v, want start 1000 end 2000", got)
	}

	// A capture with no journal and no per-record array still reports none:
	// the point is to stop the false claim, not every claim.
	if got := APIRecords(&cudacapture.Capture{}); len(got) != 0 {
		t.Errorf("api records = %d for an empty capture, want 0", len(got))
	}
}
