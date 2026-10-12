package cudatma_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatma"
)

func slotDescriptor() cudatma.Descriptor {
	return cudatma.Descriptor{
		Target:       cudatma.TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [cudatma.MaxRank]uint64{64, 32},
		Strides:      [cudatma.MaxRank]uint64{128},
		Box:          [cudatma.MaxRank]uint32{64, 1},
		ElementBytes: 2,
	}
}

func load(d cudatma.Descriptor) cudatma.Transfer {
	return cudatma.Transfer{Descriptor: d, Direction: cudatma.GlobalToShared, Bytes: 128}
}

// issue issues a transfer of the slot's current descriptor, taking the
// descriptor and generation from one call to Current.
func issue(s *cudatma.DescriptorSlot) (cudatma.TransferHandle, error) {
	d, gen := s.Current()
	return s.Issue(gen, load(d))
}

func newSlot(t *testing.T) *cudatma.DescriptorSlot {
	t.Helper()
	s, err := cudatma.NewDescriptorSlot(slotDescriptor())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestDescriptorSlotLifecycle(t *testing.T) {
	s := newSlot(t)
	if g := s.Generation(); g != 1 {
		t.Fatalf("Generation() = %d, want 1", g)
	}
	h, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	if h.Generation() != 1 || s.InFlight() != 1 {
		t.Fatalf("handle generation %d, in flight %d; want 1, 1", h.Generation(), s.InFlight())
	}
	if err := s.Complete(h); err != nil {
		t.Fatal(err)
	}
	if n := s.InFlight(); n != 0 {
		t.Fatalf("InFlight() = %d after Complete, want 0", n)
	}
	next := slotDescriptor()
	next.Base = 0x2000
	if err := s.Update(next); err != nil {
		t.Fatal(err)
	}
	if g := s.Generation(); g != 2 {
		t.Fatalf("Generation() = %d after Update, want 2", g)
	}
	if s.Descriptor() != next {
		t.Fatalf("Descriptor() = %+v, want %+v", s.Descriptor(), next)
	}
	h2, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	if h2.Generation() != 2 {
		t.Fatalf("handle generation %d, want 2", h2.Generation())
	}
	if err := s.Complete(h2); err != nil {
		t.Fatal(err)
	}
}

func TestDescriptorSlotUpdateInFlight(t *testing.T) {
	s := newSlot(t)
	before, gen := s.Current()
	h, err := s.Issue(gen, load(before))
	if err != nil {
		t.Fatal(err)
	}
	next := slotDescriptor()
	next.Base = 0x2000
	if err := s.Update(next); !errors.Is(err, cudatma.ErrInFlight) {
		t.Fatalf("Update with a transfer in flight = %v, want ErrInFlight", err)
	}
	if s.Descriptor() != before || s.Generation() != 1 {
		t.Fatalf("refused Update changed the slot: generation %d, descriptor %+v", s.Generation(), s.Descriptor())
	}
	if err := s.Complete(h); err != nil {
		t.Fatal(err)
	}
	if err := s.Update(next); err != nil {
		t.Fatalf("Update after Complete = %v, want nil", err)
	}
}

// TestDescriptorSlotStaleGeneration completes a handle from an earlier
// generation while a later transfer with the same position in its own
// generation is in flight. Accepting it would retire the later transfer.
func TestDescriptorSlotStaleGeneration(t *testing.T) {
	s := newSlot(t)
	old, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(old); err != nil {
		t.Fatal(err)
	}
	next := slotDescriptor()
	next.Base = 0x2000
	if err := s.Update(next); err != nil {
		t.Fatal(err)
	}
	cur, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(old); !errors.Is(err, cudatma.ErrStaleGeneration) {
		t.Fatalf("Complete(stale handle) = %v, want ErrStaleGeneration", err)
	}
	if n := s.InFlight(); n != 1 {
		t.Fatalf("InFlight() = %d after refused stale Complete, want 1", n)
	}
	if err := s.Complete(cur); err != nil {
		t.Fatal(err)
	}
}

// TestDescriptorSlotForeignHandle completes a handle issued by one slot on
// another slot that has a transfer at the same position in flight.
// Accepting it would retire the other slot's transfer.
func TestDescriptorSlotForeignHandle(t *testing.T) {
	a := newSlot(t)
	b := newSlot(t)
	ha, err := issue(a)
	if err != nil {
		t.Fatal(err)
	}
	hb, err := issue(b)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Complete(ha); !errors.Is(err, cudatma.ErrForeignHandle) {
		t.Fatalf("b.Complete(handle from a) = %v, want ErrForeignHandle", err)
	}
	if err := b.Complete(cudatma.TransferHandle{}); !errors.Is(err, cudatma.ErrForeignHandle) {
		t.Fatalf("Complete(zero handle) = %v, want ErrForeignHandle", err)
	}
	if a.InFlight() != 1 || b.InFlight() != 1 {
		t.Fatalf("in flight a=%d b=%d after refused completions, want 1, 1", a.InFlight(), b.InFlight())
	}
	if err := a.Complete(ha); err != nil {
		t.Fatal(err)
	}
	if err := b.Complete(hb); err != nil {
		t.Fatal(err)
	}
}

// TestDescriptorSlotEqualUpdateStaleIssue issues under a generation the
// slot replaced with an equal descriptor. The values match; the generation
// does not, so the request is stale.
func TestDescriptorSlotEqualUpdateStaleIssue(t *testing.T) {
	s := newSlot(t)
	d, gen := s.Current()
	if err := s.Update(d); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Issue(gen, load(d)); !errors.Is(err, cudatma.ErrStaleGeneration) {
		t.Fatalf("Issue(generation %d) after Update to an equal descriptor = %v, want ErrStaleGeneration", gen, err)
	}
	if n := s.InFlight(); n != 0 {
		t.Fatalf("InFlight() = %d after refused Issue, want 0", n)
	}
}

// TestDescriptorSlotCurrentUnderUpdate issues from Current while another
// goroutine updates the slot between two different descriptors. A
// descriptor and generation read together always match, so an issue may be
// refused as stale but never as a descriptor mismatch.
//
// The test can fail only when an Update lands inside the window it probes,
// so it detects a torn read in Current by chance, not on every run, and it
// is blind with one processor, where the updater rarely interleaves. Under
// -race it also catches a Current that reads without the lock. The stale
// count it logs is how often an Update interleaved at all; zero means the
// run exercised nothing. Issue's check order is pinned deterministically by
// [TestDescriptorSlotIssueChecksGenerationFirst].
func TestDescriptorSlotCurrentUnderUpdate(t *testing.T) {
	const n = 5000
	s := newSlot(t)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-done:
				return
			default:
			}
			d := slotDescriptor()
			d.Base = 0x1000 * uint64(1+i%2)
			// Refused with ErrInFlight while the issuer holds a transfer.
			s.Update(d)
		}
	}()
	var accepted, stale int
	for i := 0; i < n; i++ {
		h, err := issue(s)
		if errors.Is(err, cudatma.ErrStaleGeneration) {
			stale++
			continue
		}
		if err != nil {
			t.Fatalf("issue %d: %v", i, err)
		}
		accepted++
		if err := s.Complete(h); err != nil {
			t.Fatalf("complete %d: %v", i, err)
		}
	}
	close(done)
	wg.Wait()
	if accepted+stale != n || accepted == 0 {
		t.Fatalf("accepted %d, stale %d, of %d", accepted, stale, n)
	}
	t.Logf("accepted %d, stale %d, of %d", accepted, stale, n)
}

// TestDescriptorSlotIssueChecksGenerationFirst issues with a generation and
// descriptor saved before an Update to a different descriptor. Both are out
// of date; the generation is the cause, so the refusal must be stale rather
// than a descriptor mismatch.
func TestDescriptorSlotIssueChecksGenerationFirst(t *testing.T) {
	s := newSlot(t)
	old, gen := s.Current()
	next := old
	next.Base = 0x2000
	if err := s.Update(next); err != nil {
		t.Fatal(err)
	}
	_, err := s.Issue(gen, load(old))
	if !errors.Is(err, cudatma.ErrStaleGeneration) || errors.Is(err, cudatma.ErrDescriptorMismatch) {
		t.Fatalf("Issue(old generation, old descriptor) = %v, want ErrStaleGeneration only", err)
	}
	if n := s.InFlight(); n != 0 {
		t.Fatalf("InFlight() = %d after refused Issue, want 0", n)
	}
}

func TestDescriptorSlotDoubleComplete(t *testing.T) {
	s := newSlot(t)
	h, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := issue(s)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(h); err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(h); !errors.Is(err, cudatma.ErrNotInFlight) {
		t.Fatalf("second Complete = %v, want ErrNotInFlight", err)
	}
	if n := s.InFlight(); n != 1 {
		t.Fatalf("InFlight() = %d after refused double Complete, want 1", n)
	}
	if err := s.Complete(h2); err != nil {
		t.Fatal(err)
	}
}

func TestDescriptorSlotRefusals(t *testing.T) {
	stale := slotDescriptor()
	stale.Base = 0x3000
	bad := slotDescriptor()
	bad.Base = 8
	tests := []struct {
		name string
		op   func(*cudatma.DescriptorSlot) error
		want error // nil means any error
	}{
		{"issue descriptor not held by the slot", func(s *cudatma.DescriptorSlot) error {
			_, gen := s.Current()
			_, err := s.Issue(gen, load(stale))
			return err
		}, cudatma.ErrDescriptorMismatch},
		{"issue under generation 0", func(s *cudatma.DescriptorSlot) error {
			_, err := s.Issue(0, load(s.Descriptor()))
			return err
		}, cudatma.ErrStaleGeneration},
		{"issue under a future generation", func(s *cudatma.DescriptorSlot) error {
			_, err := s.Issue(2, load(s.Descriptor()))
			return err
		}, cudatma.ErrFutureGeneration},
		{"issue invalid transfer", func(s *cudatma.DescriptorSlot) error {
			d, gen := s.Current()
			tr := load(d)
			tr.Bytes = 0
			_, err := s.Issue(gen, tr)
			return err
		}, nil},
		{"issue direct tmem on sm_121a", func(s *cudatma.DescriptorSlot) error {
			d, gen := s.Current()
			tr := load(d)
			tr.Direction = cudatma.GlobalToTMEM
			_, err := s.Issue(gen, tr)
			return err
		}, nil},
		{"update invalid descriptor", func(s *cudatma.DescriptorSlot) error {
			return s.Update(bad)
		}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := newSlot(t)
			before := s.Descriptor()
			err := tt.op(s)
			if err == nil {
				t.Fatal("got nil, want an error")
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Fatalf("got %v, want %v", err, tt.want)
			}
			if s.InFlight() != 0 || s.Generation() != 1 || s.Descriptor() != before {
				t.Fatalf("refusal changed the slot: in flight %d, generation %d", s.InFlight(), s.Generation())
			}
		})
	}
	if _, err := cudatma.NewDescriptorSlot(bad); err == nil {
		t.Fatal("NewDescriptorSlot(invalid) = nil error")
	}
}

func TestDescriptorSlotConcurrent(t *testing.T) {
	s := newSlot(t)
	d, gen := s.Current()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				h, err := s.Issue(gen, load(d))
				if err != nil {
					t.Error(err)
					return
				}
				if err := s.Complete(h); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	if n := s.InFlight(); n != 0 {
		t.Fatalf("InFlight() = %d, want 0", n)
	}
}

func ExampleDescriptorSlot() {
	d := cudatma.Descriptor{
		Target:       cudatma.TargetSM121a,
		Rank:         2,
		Base:         0x1000,
		Dims:         [cudatma.MaxRank]uint64{64, 32},
		Strides:      [cudatma.MaxRank]uint64{128},
		Box:          [cudatma.MaxRank]uint32{64, 1},
		ElementBytes: 2,
	}
	s, err := cudatma.NewDescriptorSlot(d)
	if err != nil {
		fmt.Println(err)
		return
	}
	cur, gen := s.Current()
	h, _ := s.Issue(gen, cudatma.Transfer{Descriptor: cur, Direction: cudatma.GlobalToShared, Bytes: 128})

	next := d
	next.Base = 0x2000
	fmt.Println(errors.Is(s.Update(next), cudatma.ErrInFlight))
	fmt.Println(s.Complete(h))
	fmt.Println(errors.Is(s.Complete(h), cudatma.ErrNotInFlight))
	fmt.Println(s.Update(next), s.Generation())
	fmt.Println(errors.Is(s.Complete(h), cudatma.ErrStaleGeneration))
	_, err = s.Issue(gen, cudatma.Transfer{Descriptor: next, Direction: cudatma.GlobalToShared, Bytes: 128})
	fmt.Println(errors.Is(err, cudatma.ErrStaleGeneration))
	// Output:
	// true
	// <nil>
	// true
	// <nil> 2
	// true
	// true
}
