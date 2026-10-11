package ncclinspector

import "testing"

func TestStallInvalidIntervals(t *testing.T) {
	base := CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, HostPID: 1, Operation: "AllReduce", HasDeviceActivity: true,
		DeviceStart: BoundedTimestamp{Nanoseconds: 100, Domain: ClockDomainHostMonotonic}, DeviceEnd: BoundedTimestamp{Nanoseconds: 200, Domain: ClockDomainHostMonotonic}}
	for _, tc := range []struct {
		name   string
		change func(*CollectiveObservation)
	}{
		{"negative uncertainty", func(o *CollectiveObservation) { o.DeviceStart.UncertaintyNS = -1 }},
		{"reversed", func(o *CollectiveObservation) { o.DeviceEnd.Nanoseconds = 50 }},
		{"unknown domain", func(o *CollectiveObservation) { o.DeviceStart.Domain = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := base
			tc.change(&o)
			if err := NewTwoRankAttributor().AddObservation(o); err == nil {
				t.Fatal("invalid interval accepted")
			}
		})
	}
}
func TestStallMissingEvidenceNeverNormal(t *testing.T) {
	a := NewTwoRankAttributor()
	for rank := 0; rank < 2; rank++ {
		if err := a.AddObservation(CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, Rank: rank, HostPID: rank + 1, Operation: "AllReduce"}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := a.Attribute("0x1234567890abcdef", 0)
	if err != nil || got.Admitted || got.Diagnosis == DiagnosisNormal {
		t.Fatalf("empty evidence admitted: %+v %v", got, err)
	}
}
func TestAlignedRejectsNegativeUncertainty(t *testing.T) {
	_, err := (BoundedTimestamp{Domain: ClockDomainHostMonotonic, UncertaintyNS: -1}).Aligned(ClockMapping{Domain: ClockDomainHostMonotonic})
	if err == nil {
		t.Fatal("negative uncertainty accepted")
	}
}

func TestCrossHostRequiresMappedEpoch(t *testing.T) {
	a := NewTwoRankAttributor()
	for rank := 0; rank < 2; rank++ {
		host := "host-a"
		if rank == 1 {
			host = "host-b"
		}
		if err := a.AddObservation(CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", TotalRanks: 2, Rank: rank, HostPID: rank + 1, Hostname: host, Operation: "AllReduce", HasHostEnqueue: true,
			HostEnqueueStart: BoundedTimestamp{Nanoseconds: int64(rank+1) * 100, Domain: ClockDomainHostMonotonic}, HostEnqueueEnd: BoundedTimestamp{Nanoseconds: int64(rank+1) * 100, Domain: ClockDomainHostMonotonic}}); err != nil {
			t.Fatal(err)
		}
	}
	got, err := a.Attribute("0x1234567890abcdef", 0)
	if err != nil || got.Admitted || got.Diagnosis != DiagnosisIncompatibleClocks {
		t.Fatalf("unmapped cross-host clocks admitted: %+v %v", got, err)
	}
}
