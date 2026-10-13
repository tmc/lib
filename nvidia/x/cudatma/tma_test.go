package cudatma

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func descriptor() Descriptor {
	return Descriptor{Target: TargetSM121a, Rank: 2, Base: 0x1000, Dims: [MaxRank]uint64{64, 32, 1}, Strides: [MaxRank]uint64{128}, Box: [MaxRank]uint32{64, 1, 1}, ElementBytes: 2}
}
func TestTensorMapLayout(t *testing.T) {
	if err := descriptor().Validate(); err != nil {
		t.Fatal(err)
	}
	d := descriptor()
	d.Base = 8
	if err := d.Validate(); err == nil {
		t.Fatal("accepted unaligned base")
	}
}

// TestTensorMapLifetime checks that a descriptor owned by a DescriptorSlot
// cannot be replaced while a transfer issued from it is in flight, and that
// the transfer's handle is retired by the generation it was issued under.
func TestTensorMapLifetime(t *testing.T) {
	d := descriptor()
	tr := Transfer{Descriptor: d, Direction: GlobalToShared, Bytes: 64}
	if err := tr.Validate(); err != nil {
		t.Fatal(err)
	}
	s, err := NewDescriptorSlot(d)
	if err != nil {
		t.Fatal(err)
	}
	_, gen := s.Current()
	h, err := s.Issue(gen, tr)
	if err != nil {
		t.Fatal(err)
	}
	moved := d
	moved.Base = 0x2000
	if err := s.Update(moved); !errors.Is(err, ErrInFlight) {
		t.Fatalf("Update of an in-flight descriptor = %v, want ErrInFlight", err)
	}
	if s.Descriptor() != d {
		t.Fatal("refused Update replaced the in-flight descriptor")
	}
	if err := s.Complete(h); err != nil {
		t.Fatal(err)
	}
	if err := s.Update(moved); err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(h); !errors.Is(err, ErrStaleGeneration) {
		t.Fatalf("Complete after Update = %v, want ErrStaleGeneration", err)
	}
}
func TestTransferTargetSupport(t *testing.T) {
	d := descriptor()
	if err := (Transfer{Descriptor: d, Direction: GlobalToTMEM, Bytes: 64}).Validate(); err == nil {
		t.Fatal("accepted direct sm_121a global-to-tmem")
	}
	d.Target = TargetSM100a
	if err := (Transfer{Descriptor: d, Direction: GlobalToTMEM, Bytes: 64, Staged: true}).Validate(); err != nil {
		t.Fatal(err)
	}
}
func TestMulticastRankMask(t *testing.T) {
	m, err := NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil || !m.Participating(3) || m.Participating(2) {
		t.Fatalf("multicast=%#v err=%v", m, err)
	}
	if _, err := NewMulticast(0, []uint32{1, 1}, 64); err == nil {
		t.Fatal("accepted duplicate rank")
	}
}
func TestMulticastTransactionAccounting(t *testing.T) {
	m, _ := NewMulticast(0, []uint32{1, 3}, 64)
	if err := m.Complete(1, 32); err == nil {
		t.Fatal("accepted incomplete transaction")
	}
	if err := m.Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	if m.CompleteAll() {
		t.Fatal("completed with missing destination")
	}
	if err := m.Complete(3, 64); err != nil || !m.CompleteAll() {
		t.Fatal(err)
	}
}
func TestClusterAddressNotMulticast(t *testing.T) {
	m, _ := NewMulticast(0, []uint32{2}, 8)
	if m.Participating(0) {
		t.Fatal("source became destination")
	}
}
func TestRejectUnverifiedTMEMTransfer(t *testing.T) {
	d := descriptor()
	d.Target = "sm_90a"
	if err := (Transfer{Descriptor: d, Direction: GlobalToTMEM, Bytes: 1, Staged: true}).Validate(); err == nil {
		t.Fatal("accepted unsupported target")
	}
}

// TestGeneratedTMAProtocol checks that the fixture names the four
// instructions the protocol is built from. It is a presence check on text,
// not a check that the text assembles: it stayed green for as long as the
// fixture named mbarrier.try_wait.parity with a destination ptxas rejects.
// TestCompileTMAInstructionFixture is the gate; this only catches an
// instruction being dropped from the fixture entirely.
func TestGeneratedTMAProtocol(t *testing.T) {
	s, err := TMAFixtureSource(TargetSM121a)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"cp.async.bulk.tensor.2d", "mbarrier.init", "mbarrier.arrive.expect_tx", "mbarrier.try_wait"} {
		if !strings.Contains(s, want) {
			t.Fatalf("fixture missing %q", want)
		}
	}
}

func TestMulticastSourceSupported(t *testing.T) {
	s, err := MulticastSource(TargetSM100a)
	if err != nil || !strings.Contains(s, ".multicast::cluster") {
		t.Fatalf("source: %v", err)
	}
	if _, err := MulticastSource(TargetSM121a); err == nil {
		t.Fatal("accepted unsupported SM121 multicast")
	}
}

func FuzzDescriptorValidation(f *testing.F) {
	f.Add(uint64(0x1000), uint32(64), uint32(2))
	f.Fuzz(func(t *testing.T, base uint64, width, elem uint32) {
		d := descriptor()
		d.Base = base
		d.Box[0] = width
		d.ElementBytes = elem
		_ = d.Validate()
	})
}

// higherRank returns a descriptor of the given rank with every dimension,
// stride and box extent the validator requires.
func higherRank(rank int) Descriptor {
	d := Descriptor{Target: TargetSM121a, Rank: rank, Base: 0x1000, ElementBytes: 2}
	// Fill every slot the arrays have, so a rank outside the admitted range
	// is rejected by Validate rather than by indexing here.
	for i := 0; i < MaxRank; i++ {
		d.Dims[i] = 64
		d.Box[i] = 1
		d.Strides[i] = 128
	}
	d.Box[0] = 64
	// The innermost dimension carries no stride; Validate requires the unused
	// slot to be zero. A valid rank leaves it zero, an out-of-range rank is
	// still rejected by Validate rather than by indexing here.
	if rank >= 1 && rank <= MaxRank {
		d.Strides[rank-1] = 0
	}
	return d
}

func TestValidateRejectsUnusedStride(t *testing.T) {
	for rank := 2; rank <= MaxRank; rank++ {
		d := higherRank(rank)
		if err := d.Validate(); err != nil {
			t.Fatalf("rank %d: Validate() = %v, want nil", rank, err)
		}
		d.Strides[rank-1] = 128
		if err := d.Validate(); err == nil {
			t.Errorf("rank %d: Validate() with a nonzero unused stride = nil, want an error", rank)
		}
	}
}

func TestValidateRank(t *testing.T) {
	tests := []struct {
		rank int
		ok   bool
	}{
		{1, true},
		{2, true},
		{3, true},
		{4, true},
		{5, true},
		{6, false},
	}
	for _, tt := range tests {
		err := higherRank(tt.rank).Validate()
		if tt.ok && err != nil {
			t.Errorf("rank %d: Validate() = %v, want nil", tt.rank, err)
		}
		if !tt.ok && err == nil {
			t.Errorf("rank %d: Validate() = nil, want an error", tt.rank)
		}
	}
}

func TestValidateStridesAcrossRank(t *testing.T) {
	// A rank-5 tensor needs four global strides. Zeroing any one of them
	// must be rejected; the innermost has no stride and is not read.
	for i := 0; i < 4; i++ {
		d := higherRank(5)
		d.Strides[i] = 0
		if err := d.Validate(); err == nil {
			t.Errorf("stride %d zero: Validate() = nil, want an error", i)
		}
	}
	d := higherRank(5)
	d.Strides[4] = 0
	if err := d.Validate(); err != nil {
		t.Errorf("unused stride 4 zero: Validate() = %v, want nil", err)
	}
	d = higherRank(5)
	d.Strides[2] = 24
	if err := d.Validate(); err == nil {
		t.Error("stride 2 not a 16-byte multiple: Validate() = nil, want an error")
	}
}

func TestValidateTensorMapOptions(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*Descriptor)
		wantErr bool
	}{
		{"defaults are none", func(*Descriptor) {}, false},
		{"L2 promotion 128B", func(d *Descriptor) { d.L2Promotion = driver.TensorMapL2PromotionL2128B }, false},
		{"L2 promotion 256B", func(d *Descriptor) { d.L2Promotion = driver.TensorMapL2PromotionL2256B }, false},
		{"L2 promotion unknown", func(d *Descriptor) { d.L2Promotion = 9 }, true},
		{"OOB fill NaN", func(d *Descriptor) { d.OOBFill = driver.TensorMapFloatOobFillNanRequestZeroFma }, false},
		{"OOB fill unknown", func(d *Descriptor) { d.OOBFill = 9 }, true},
		{"interleave 16B at rank 4", func(d *Descriptor) { d.Interleave = driver.TensorMapInterleave16B }, false},
		{"interleave unknown", func(d *Descriptor) { d.Interleave = 9 }, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := higherRank(4)
			tt.mutate(&d)
			err := d.Validate()
			if tt.wantErr && err == nil {
				t.Error("Validate() = nil, want an error")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}

func TestValidateInterleaveNeedsRank(t *testing.T) {
	d := higherRank(2)
	d.Interleave = driver.TensorMapInterleave32B
	if err := d.Validate(); err == nil {
		t.Error("interleave at rank 2: Validate() = nil, want an error")
	}
}

// TestMulticastDuplicateCompletion checks that a destination rank completes
// at most once. A second completion from the same rank is a protocol error,
// not an update: accepting it would let one rank's barrier stand in for two.
func TestMulticastDuplicateCompletion(t *testing.T) {
	m, err := NewMulticast(0, []uint32{1, 3}, 64)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	if err := m.Complete(1, 64); err == nil {
		t.Fatal("Complete(1) twice = nil, want an error")
	} else if !errors.Is(err, ErrDuplicateCompletion) {
		t.Fatalf("Complete(1) twice = %v, want ErrDuplicateCompletion", err)
	}
	if m.CompleteAll() {
		t.Fatal("CompleteAll() = true with rank 3 pending")
	}
	if err := m.Complete(3, 64); err != nil || !m.CompleteAll() {
		t.Fatalf("Complete(3) = %v, CompleteAll() = %v", err, m.CompleteAll())
	}
}

// TestTransferHandleNotReissued forges a handle by editing the generation of
// a retired one. Sequence numbers continue across generations, so the forged
// handle names no transfer in flight and cannot retire the transfer issued
// after the update.
func TestTransferHandleNotReissued(t *testing.T) {
	d := descriptor()
	tr := Transfer{Descriptor: d, Direction: GlobalToShared, Bytes: 64}
	s, err := NewDescriptorSlot(d)
	if err != nil {
		t.Fatal(err)
	}
	_, gen := s.Current()
	old, err := s.Issue(gen, tr)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Complete(old); err != nil {
		t.Fatal(err)
	}
	if err := s.Update(d); err != nil {
		t.Fatal(err)
	}
	_, gen = s.Current()
	cur, err := s.Issue(gen, tr)
	if err != nil {
		t.Fatal(err)
	}
	if cur.seq == old.seq {
		t.Fatalf("handle sequence %d reissued after Update", cur.seq)
	}
	forged := old
	forged.gen = s.Generation()
	if err := s.Complete(forged); !errors.Is(err, ErrNotInFlight) {
		t.Fatalf("Complete(retired handle with current generation) = %v, want ErrNotInFlight", err)
	}
	if n := s.InFlight(); n != 1 {
		t.Fatalf("InFlight() = %d after refused forged Complete, want 1", n)
	}
	if err := s.Complete(cur); err != nil {
		t.Fatal(err)
	}
}

// wantMulticastErr reports whether err is exactly the refusal want, and
// whether it wraps ErrDuplicateCompletion exactly when dup is set. Matching
// the full text keeps one refusal from standing in for another: a
// nonparticipant, a byte mismatch, a duplicate and an uninitialized handle
// are all errors, and only the reason says which check refused.
func wantMulticastErr(t *testing.T, what string, err error, want string, dup bool) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s = nil, want %q", what, want)
	}
	if err.Error() != want {
		t.Fatalf("%s = %q, want %q", what, err.Error(), want)
	}
	if errors.Is(err, ErrDuplicateCompletion) != dup {
		t.Fatalf("%s: errors.Is(err, ErrDuplicateCompletion) = %v, want %v", what, !dup, dup)
	}
}

func newMulticast13(t *testing.T) Multicast {
	t.Helper()
	m, err := NewMulticast(0, []uint32{3, 1}, 64)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestMulticastCopiesShareTerminalCompletion checks that every value copy of
// a Multicast, however it was copied, names the one transaction its
// constructor created: a completion through any copy is visible through all
// of them, and once every destination completes the transaction is terminal
// for every copy, including one made afterwards.
func TestMulticastCopiesShareTerminalCompletion(t *testing.T) {
	m := newMulticast13(t)
	assigned := m
	slice := []Multicast{m}
	holder := struct{ h Multicast }{m}
	viaPointer := &assigned
	byValue := func(c Multicast) Multicast { return c }(m)
	copies := map[string]*Multicast{
		"original": &m,
		"assigned": &assigned,
		"slice":    &slice[0],
		"field":    &holder.h,
		"pointer":  viaPointer,
		"by value": &byValue,
	}
	agree := func(step string, want bool) {
		t.Helper()
		for name, c := range copies {
			if got := c.CompleteAll(); got != want {
				t.Fatalf("%s: %s.CompleteAll() = %v, want %v", step, name, got, want)
			}
		}
	}

	agree("before completion", false)
	if err := slice[0].Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	agree("after rank 1", false)
	for name, c := range copies {
		wantMulticastErr(t, name+".Complete(1, 64) after rank 1", c.Complete(1, 64), "rank 1: duplicate completion", true)
	}
	agree("after refused duplicates", false)
	if err := holder.h.Complete(3, 64); err != nil {
		t.Fatal(err)
	}
	agree("after rank 3", true)

	late := m
	copies["late"] = &late
	for name, c := range copies {
		for _, r := range []uint32{1, 3} {
			wantMulticastErr(t, name+".Complete after terminal", c.Complete(r, 64),
				fmt.Sprintf("rank %d: duplicate completion", r), true)
		}
	}
	agree("after terminal re-completion", true)
}

// TestMulticastConstructorsDoNotAlias checks that two constructions from the
// same arguments are separate transactions, and that a handle rebound to
// another constructor's value follows that transaction, not its old one.
func TestMulticastConstructorsDoNotAlias(t *testing.T) {
	a := newMulticast13(t)
	b := newMulticast13(t)
	aCopy := a
	for _, r := range []uint32{1, 3} {
		if err := aCopy.Complete(r, 64); err != nil {
			t.Fatal(err)
		}
	}
	if !a.CompleteAll() {
		t.Fatal("a.CompleteAll() = false after both destinations")
	}
	if b.CompleteAll() {
		t.Fatal("b.CompleteAll() = true: separate constructors share completion")
	}
	if err := b.Complete(1, 64); err != nil {
		t.Fatalf("b.Complete(1, 64) = %v after a completed rank 1, want nil", err)
	}
	if b.CompleteAll() {
		t.Fatal("b.CompleteAll() = true with rank 3 pending")
	}

	rebound := a
	if !rebound.CompleteAll() {
		t.Fatal("copy of a does not report a's completion")
	}
	rebound = b
	wantMulticastErr(t, "rebound.Complete(1, 64)", rebound.Complete(1, 64), "rank 1: duplicate completion", true)
	if err := rebound.Complete(3, 64); err != nil {
		t.Fatal(err)
	}
	if !b.CompleteAll() || !rebound.CompleteAll() {
		t.Fatal("rebound handle did not complete b")
	}
}

// TestMulticastStaleHandleRefusals checks the handles a caller may still hold
// after the transaction moved on: a copy taken before terminal completion
// refuses every rank as a duplicate, and a copy kept across rebinding the
// variable to a new transaction neither reaches nor is reached by it.
func TestMulticastStaleHandleRefusals(t *testing.T) {
	m := newMulticast13(t)
	stale := m
	for _, r := range []uint32{1, 3} {
		if err := m.Complete(r, 64); err != nil {
			t.Fatal(err)
		}
	}
	if !stale.CompleteAll() {
		t.Fatal("stale.CompleteAll() = false after terminal completion through another copy")
	}
	wantMulticastErr(t, "stale.Complete(1, 64)", stale.Complete(1, 64), "rank 1: duplicate completion", true)
	wantMulticastErr(t, "stale.Complete(3, 64)", stale.Complete(3, 64), "rank 3: duplicate completion", true)
	wantMulticastErr(t, "stale.Complete(2, 64)", stale.Complete(2, 64), "rank 2 is not a destination", false)

	old := m
	m = newMulticast13(t)
	if m.CompleteAll() {
		t.Fatal("new transaction inherited the old one's completion")
	}
	if !old.CompleteAll() {
		t.Fatal("old handle lost its transaction when the variable was rebound")
	}
	wantMulticastErr(t, "old.Complete(1, 64)", old.Complete(1, 64), "rank 1: duplicate completion", true)
	if err := m.Complete(1, 64); err != nil {
		t.Fatalf("new.Complete(1, 64) = %v, want nil", err)
	}
	if m.CompleteAll() {
		t.Fatal("new transaction completed with rank 3 pending")
	}
}

// TestMulticastConfigTamperingRefusals checks that the exported fields are
// snapshots: rewriting them on a copy changes neither which ranks the
// transaction admits, nor the byte count it requires, nor when it is
// terminal. Each refusal leaves the shared state as it was.
func TestMulticastConfigTamperingRefusals(t *testing.T) {
	tests := []struct {
		name   string
		tamper func(*Multicast)
		rank   uint32
		bytes  uint64
		want   string
	}{
		{"mask admits rank 2", func(c *Multicast) { c.Mask |= 1 << 2 }, 2, 64, "rank 2 is not a destination"},
		{"mask of all ranks", func(c *Multicast) { c.Mask = 1<<16 - 1 }, 0, 64, "rank 0 is not a destination"},
		{"destinations name rank 2", func(c *Multicast) { c.Destinations = []uint32{2} }, 2, 64, "rank 2 is not a destination"},
		{"source rank 2", func(c *Multicast) { c.SourceRank = 2 }, 2, 64, "rank 2 is not a destination"},
		{"bytes lowered", func(c *Multicast) { c.Bytes = 32 }, 1, 32, "rank 1 completed 32 bytes, want 64"},
		{"bytes raised", func(c *Multicast) { c.Bytes = 128 }, 3, 128, "rank 3 completed 128 bytes, want 64"},
		{"rank past mask width", func(c *Multicast) { c.Mask = ^uint64(0) }, 16, 64, "rank 16 is not a destination"},
		{"rank past shift width", func(c *Multicast) { c.Mask = ^uint64(0) }, 64, 64, "rank 64 is not a destination"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := newMulticast13(t)
			c := m
			tt.tamper(&c)
			wantMulticastErr(t, "tampered Complete", c.Complete(tt.rank, tt.bytes), tt.want, false)
			if c.Participating(2) || !c.Participating(1) || !c.Participating(3) {
				t.Fatal("tampered fields changed Participating")
			}
			if m.CompleteAll() || c.CompleteAll() {
				t.Fatal("refused completion changed shared state")
			}
			for _, r := range []uint32{1, 3} {
				if err := c.Complete(r, 64); err != nil {
					t.Fatalf("Complete(%d, 64) on tampered copy = %v, want nil", r, err)
				}
			}
			if !m.CompleteAll() || !c.CompleteAll() {
				t.Fatal("constructor destinations did not make the transaction terminal")
			}
		})
	}

	// A narrowed mask must not make the transaction terminal early.
	m := newMulticast13(t)
	c := m
	c.Mask = 1 << 1
	c.Destinations = []uint32{1}
	if err := c.Complete(1, 64); err != nil {
		t.Fatal(err)
	}
	if c.CompleteAll() || m.CompleteAll() {
		t.Fatal("narrowed mask completed the transaction with rank 3 pending")
	}
}

// TestMulticastUninitializedRefusals checks that a Multicast that did not come
// from a successful NewMulticast is refused for that reason, including a
// literal whose exported fields describe a valid transaction.
func TestMulticastUninitializedRefusals(t *testing.T) {
	failed := func(source uint32, dest []uint32, bytes uint64) Multicast {
		m, err := NewMulticast(source, dest, bytes)
		if err == nil {
			t.Fatalf("NewMulticast(%d, %v, %d) = nil error", source, dest, bytes)
		}
		return m
	}
	var zero Multicast
	tests := []struct {
		name string
		m    Multicast
	}{
		{"zero value", Multicast{}},
		{"copy of zero value", zero},
		{"mask literal", Multicast{Mask: 1<<1 | 1<<3}},
		{"full literal", Multicast{SourceRank: 0, Destinations: []uint32{1, 3}, Mask: 1<<1 | 1<<3, Bytes: 64}},
		{"bad source", failed(16, []uint32{1, 3}, 64)},
		{"no destinations", failed(0, nil, 64)},
		{"zero bytes", failed(0, []uint32{1, 3}, 0)},
		{"bad destination", failed(0, []uint32{1, 16}, 64)},
		{"duplicate destination", failed(0, []uint32{1, 1}, 64)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.m
			for _, r := range []uint32{1, 3} {
				wantMulticastErr(t, "Complete", m.Complete(r, 64), "uninitialized multicast", false)
				if m.Participating(r) {
					t.Fatalf("Participating(%d) = true", r)
				}
			}
			if m.CompleteAll() {
				t.Fatal("CompleteAll() = true")
			}
		})
	}

	var nilHandle *Multicast
	wantMulticastErr(t, "nil.Complete", nilHandle.Complete(1, 64), "nil multicast", false)
}

// TestMulticastConcurrentCopiesCompleteOnce races two completions of every
// rank through separate copies. Each rank must succeed exactly once and refuse
// exactly once as a duplicate, and every copy must then report completion.
func TestMulticastConcurrentCopiesCompleteOnce(t *testing.T) {
	var dest []uint32
	for r := uint32(0); r < 16; r++ {
		dest = append(dest, r)
	}
	m, err := NewMulticast(0, dest, 64)
	if err != nil {
		t.Fatal(err)
	}
	handles := make([]Multicast, 2*len(dest))
	for i := range handles {
		handles[i] = m
	}
	errs := make([]error, len(handles))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range handles {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			errs[i] = handles[i].Complete(uint32(i/2), 64)
		}(i)
	}
	close(start)
	wg.Wait()
	for r := range dest {
		a, b := errs[2*r], errs[2*r+1]
		if a != nil && b != nil || a == nil && b == nil {
			t.Fatalf("rank %d: completions = %v, %v; want exactly one nil", r, a, b)
		}
		dup := a
		if dup == nil {
			dup = b
		}
		wantMulticastErr(t, "racing duplicate", dup, fmt.Sprintf("rank %d: duplicate completion", r), true)
	}
	for i := range handles {
		if !handles[i].CompleteAll() {
			t.Fatalf("handle %d: CompleteAll() = false", i)
		}
	}
}

// TestMulticastDuplicateBeforeBytes pins the order of Complete's refusals
// when a call is both a duplicate and carries the wrong byte count: the
// duplicate check runs first, so the error names the duplicate, wraps
// [ErrDuplicateCompletion] and never reports the byte mismatch. Each bad
// count is first shown to be refused for its bytes on a pending rank, so the
// duplicate result cannot come from a byte check that never fires.
func TestMulticastDuplicateBeforeBytes(t *testing.T) {
	for _, bytes := range []uint64{0, 1, 63, 65, 1 << 63, ^uint64(0)} {
		t.Run(fmt.Sprint(bytes), func(t *testing.T) {
			m := newMulticast13(t)
			copyHandle := m
			wantMulticastErr(t, fmt.Sprintf("pending Complete(1, %d)", bytes), m.Complete(1, bytes),
				fmt.Sprintf("rank 1 completed %d bytes, want 64", bytes), false)
			if err := m.Complete(1, 64); err != nil {
				t.Fatal(err)
			}
			for name, h := range map[string]*Multicast{"original": &m, "copy": &copyHandle} {
				wantMulticastErr(t, fmt.Sprintf("duplicate %s.Complete(1, %d)", name, bytes), h.Complete(1, bytes),
					"rank 1: duplicate completion", true)
			}
			if m.CompleteAll() {
				t.Fatal("CompleteAll() = true with rank 3 pending")
			}
			if err := copyHandle.Complete(3, 64); err != nil {
				t.Fatal(err)
			}
			for _, r := range []uint32{1, 3} {
				wantMulticastErr(t, fmt.Sprintf("terminal Complete(%d, %d)", r, bytes), m.Complete(r, bytes),
					fmt.Sprintf("rank %d: duplicate completion", r), true)
			}
			if !m.CompleteAll() || !copyHandle.CompleteAll() {
				t.Fatal("CompleteAll() = false after both destinations")
			}
		})
	}
}
