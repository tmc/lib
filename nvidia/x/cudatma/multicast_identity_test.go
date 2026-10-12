package cudatma_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatma"
)

func TestMulticastCopiedHandlesShareCompletion(t *testing.T) {
	original, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil {
		t.Fatal(err)
	}
	copyHandle := original

	if err := copyHandle.Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	if original.CompleteAll() || copyHandle.CompleteAll() {
		t.Fatal("one destination completed the multicast")
	}
	if err := original.Complete(1, 64); !errors.Is(err, cudatma.ErrDuplicateCompletion) {
		t.Fatalf("second completion through another handle = %v, want ErrDuplicateCompletion", err)
	}
	if err := original.Complete(3, 64); err != nil {
		t.Fatal(err)
	}
	if !original.CompleteAll() || !copyHandle.CompleteAll() {
		t.Fatal("copied handles disagree after all destinations complete")
	}
}

func TestMulticastConstructorsHaveIndependentIdentity(t *testing.T) {
	first, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil {
		t.Fatal(err)
	}
	second, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil {
		t.Fatal(err)
	}

	for _, rank := range []uint32{1, 3} {
		if err := first.Complete(rank, 64); err != nil {
			t.Fatal(err)
		}
	}
	if !first.CompleteAll() {
		t.Fatal("first transaction did not complete")
	}
	if second.CompleteAll() {
		t.Fatal("separate constructor shared completion state")
	}
	if err := second.Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	if second.CompleteAll() {
		t.Fatal("second transaction completed with a destination pending")
	}
	if err := second.Complete(3, 64); err != nil || !second.CompleteAll() {
		t.Fatalf("second completion = %v, CompleteAll = %v", err, second.CompleteAll())
	}
}

func TestMulticastPublicFieldsCannotReconfigureTransaction(t *testing.T) {
	input := []uint32{3, 1}
	m, err := cudatma.NewMulticast(0, input, 64)
	if err != nil {
		t.Fatal(err)
	}
	input[0] = 2
	if len(m.Destinations) != 2 || m.Destinations[0] != 1 || m.Destinations[1] != 3 {
		t.Fatalf("constructor did not copy and sort destinations: %v", m.Destinations)
	}

	copyHandle := m
	copyHandle.SourceRank = 4
	copyHandle.Mask = uint64(1) << 2
	copyHandle.Bytes = 1
	copyHandle.Destinations[0] = 2
	copyHandle.Destinations = copyHandle.Destinations[:1]
	copyHandle.Destinations = []uint32{2}

	if !m.Participating(1) || !m.Participating(3) || m.Participating(2) || copyHandle.Participating(2) {
		t.Fatal("public snapshot mutation changed canonical participation")
	}
	if err := copyHandle.Complete(2, 64); err == nil {
		t.Fatal("mutated public mask admitted a non-destination")
	}
	if err := copyHandle.Complete(1, 1); err == nil {
		t.Fatal("mutated public byte count admitted a short completion")
	}
	if m.CompleteAll() || copyHandle.CompleteAll() {
		t.Fatal("invalid completions changed transaction state")
	}
	if err := copyHandle.Complete(1, 64); err != nil {
		t.Fatalf("valid completion rejected after snapshot mutation: %v", err)
	}
	if m.CompleteAll() {
		t.Fatal("public destination bookkeeping made transaction complete early")
	}
	if err := m.Complete(3, 64); err != nil || !copyHandle.CompleteAll() {
		t.Fatalf("final canonical destination completion = %v, CompleteAll = %v", err, copyHandle.CompleteAll())
	}
}

func TestMulticastRejectsNonparticipantsAndPartialCompletions(t *testing.T) {
	m, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil {
		t.Fatal(err)
	}
	copyHandle := m
	if err := copyHandle.Complete(2, 64); err == nil {
		t.Fatal("accepted nonparticipant")
	}
	if err := copyHandle.Complete(1, 32); err == nil {
		t.Fatal("accepted partial byte count")
	}
	if m.CompleteAll() || copyHandle.CompleteAll() {
		t.Fatal("refused completion changed shared state")
	}
	if err := m.Complete(1, 64); err != nil {
		t.Fatalf("valid completion after refusal = %v", err)
	}
	if err := copyHandle.Complete(1, 1); !errors.Is(err, cudatma.ErrDuplicateCompletion) {
		t.Fatalf("duplicate with wrong byte count = %v, want duplicate first", err)
	}
	if m.CompleteAll() {
		t.Fatal("duplicate refusal completed a pending destination")
	}
	if err := copyHandle.Complete(3, 64); err != nil || !m.CompleteAll() {
		t.Fatalf("final completion = %v, CompleteAll = %v", err, m.CompleteAll())
	}
}

func TestMulticastZeroAndNilHandlesRefuse(t *testing.T) {
	var zero cudatma.Multicast
	if err := zero.Complete(1, 64); err == nil {
		t.Fatal("zero value accepted completion")
	}
	if zero.CompleteAll() || zero.Participating(1) {
		t.Fatal("zero value reported participation or completion")
	}

	var nilHandle *cudatma.Multicast
	if err := nilHandle.Complete(1, 64); err == nil {
		t.Fatal("nil handle accepted completion")
	}

	bad, err := cudatma.NewMulticast(16, []uint32{1}, 64)
	if err == nil {
		t.Fatal("constructor accepted invalid source rank")
	}
	if bad.CompleteAll() || bad.Participating(1) {
		t.Fatal("constructor error returned an active transaction")
	}
}

func TestMulticastConcurrentCopiedHandles(t *testing.T) {
	t.Run("same rank completes once", func(t *testing.T) {
		m, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
		if err != nil {
			t.Fatal(err)
		}
		copyHandle := m
		start := make(chan struct{})
		results := make(chan error, 2)
		var wg sync.WaitGroup
		for _, handle := range []*cudatma.Multicast{&m, &copyHandle} {
			wg.Add(1)
			go func(handle *cudatma.Multicast) {
				defer wg.Done()
				<-start
				results <- handle.Complete(1, 64)
			}(handle)
		}
		close(start)
		wg.Wait()
		close(results)
		successes, duplicates := 0, 0
		for err := range results {
			switch {
			case err == nil:
				successes++
			case errors.Is(err, cudatma.ErrDuplicateCompletion):
				duplicates++
			default:
				t.Fatalf("concurrent completion = %v", err)
			}
		}
		if successes != 1 || duplicates != 1 || m.CompleteAll() {
			t.Fatalf("successes=%d duplicates=%d complete=%v", successes, duplicates, m.CompleteAll())
		}
	})

	t.Run("distinct ranks and completion reads", func(t *testing.T) {
		m, err := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
		if err != nil {
			t.Fatal(err)
		}
		copyHandle := m
		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(4)
		go func() {
			defer wg.Done()
			<-start
			if err := m.Complete(1, 64); err != nil {
				t.Errorf("Complete(1) = %v", err)
			}
		}()
		go func() {
			defer wg.Done()
			<-start
			if err := copyHandle.Complete(3, 64); err != nil {
				t.Errorf("Complete(3) = %v", err)
			}
		}()
		for i := 0; i < 2; i++ {
			go func() {
				defer wg.Done()
				<-start
				for j := 0; j < 100; j++ {
					_ = m.CompleteAll()
					_ = copyHandle.Participating(1)
				}
			}()
		}
		close(start)
		wg.Wait()
		if !m.CompleteAll() || !copyHandle.CompleteAll() {
			t.Fatal("concurrent destination completions were not both recorded")
		}
	})
}

func ExampleMulticast() {
	m, _ := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	copyHandle := m
	_ = copyHandle.Complete(1, 64)
	fmt.Println(m.CompleteAll())
	_ = m.Complete(3, 64)
	fmt.Println(copyHandle.CompleteAll())

	separate, _ := cudatma.NewMulticast(0, []uint32{1, 3}, 64)
	fmt.Println(separate.CompleteAll())
	// Output:
	// false
	// true
	// false
}
