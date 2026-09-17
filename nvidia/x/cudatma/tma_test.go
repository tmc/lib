package cudatma

import (
	"strings"
	"testing"
)

func descriptor() Descriptor {
	return Descriptor{Target: TargetSM121a, Rank: 2, Base: 0x1000, Dims: [3]uint64{64, 32, 1}, Strides: [3]uint64{128, 0, 0}, Box: [3]uint32{64, 1, 1}, ElementBytes: 2}
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
func TestTensorMapLifetime(t *testing.T) {
	d := descriptor()
	if err := (Transfer{Descriptor: d, Direction: GlobalToShared, Bytes: 64}).Validate(); err != nil {
		t.Fatal(err)
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
