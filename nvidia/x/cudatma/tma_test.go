package cudatma

import (
	"errors"
	"strings"
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

func TestMulticastSourceUnsupported(t *testing.T) {
	if _, err := MulticastSource(TargetSM100a); err == nil {
		t.Fatal("accepted unqualified multicast source")
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
		{1, false},
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
