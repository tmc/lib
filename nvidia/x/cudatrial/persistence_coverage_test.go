package cudatrial

import "testing"

func TestPersistentSignal_ResetAndMergeCoverage(t *testing.T) {
	p, _ := NewPersistentCoverage(32, 2)
	if err := p.Notify(SignalEnd); err == nil {
		t.Fatal("end without start accepted")
	}
	if err := p.Notify(0); err == nil {
		t.Fatal("unknown signal accepted")
	}
	for _, block := range []uint64{5, 7} {
		if err := p.Notify(SignalStart); err != nil {
			t.Fatal(err)
		}
		if err := p.Notify(SignalStart); err == nil {
			t.Fatal("nested start accepted")
		}
		if _, err := p.Record(0, block); err != nil {
			t.Fatal(err)
		}
		if err := p.Notify(SignalEnd); err != nil {
			t.Fatal(err)
		}
	}
	if p.Merged().counts[21] != 1 || p.Merged().counts[23] != 1 || p.Merged().counts[18] != 0 {
		t.Fatal("iteration predecessors were not reset")
	}
	last := p.LastComplete()
	p.Notify(SignalStart)
	p.Record(0, 9)
	p.Fault()
	if err := p.Notify(SignalStart); err == nil {
		t.Fatal("faulted owner recovered")
	}
	if p.Merged().counts[25] != 0 || p.LastComplete().counts[23] != 1 || last.counts[23] != 1 {
		t.Fatal("fault assigned coverage or erased preceding evidence")
	}
	if _, err := p.Record(0, 1); err == nil {
		t.Fatal("faulted record accepted")
	}
}
