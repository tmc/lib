package ncclinspector_test

import (
	"bytes"
	"errors"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

const (
	validCommHashA = "0x91541b8a24505d53"
	validCommHashB = "0xa2b3c4d5e6f70819"
	truncatedHash  = "0x91541b8a24505d" // 16 chars: 14 hex digits (missing 2 hex digits)
)

func TestValidateCommunicatorHash(t *testing.T) {
	for _, tc := range []struct {
		name    string
		hash    string
		wantErr error
	}{
		{"valid_hash", validCommHashA, nil},
		{"valid_hash_b", validCommHashB, nil},
		{"truncated_hash_16_chars", truncatedHash, ncclinspector.ErrTruncatedHash},
		{"short_hash", "0x1234", ncclinspector.ErrTruncatedHash},
		{"long_hash", "0x91541b8a24505d5399", ncclinspector.ErrInvalidHash},
		{"missing_prefix", "91541b8a24505d5312", ncclinspector.ErrInvalidHash},
		{"non_hex_chars", "0x91541b8a24505dzz", ncclinspector.ErrInvalidHash},
		{"empty", "", ncclinspector.ErrTruncatedHash},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ncclinspector.ValidateCommunicatorHash(tc.hash)
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("expected valid hash %q, got error: %v", tc.hash, err)
				}
			} else {
				if err == nil || !errors.Is(err, tc.wantErr) {
					t.Fatalf("hash %q error = %v, want %v", tc.hash, err, tc.wantErr)
				}
			}
		})
	}
}

func TestObservationFromRecord_Verbose(t *testing.T) {
	b, err := os.ReadFile("testdata/verbose.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	records, err := ncclinspector.Read(bytes.NewReader(b), ncclinspector.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}

	clock := ncclinspector.ClockMapping{
		Domain:        ncclinspector.ClockDomainHostMonotonic,
		OffsetNS:      0,
		UncertaintyNS: 50,
	}

	// verbose.jsonl header has a short hash "0xe62dedaa97644a" from NVIDIA's documentation
	// Update header to valid full 18-char hash for attribution validation
	records[0].Header.ID = validCommHashA
	records[0].Header.Rank = 0
	records[0].Header.Ranks = 2

	obs, err := ncclinspector.ObservationFromRecord(records[0], clock)
	if err != nil {
		t.Fatal(err)
	}

	if !obs.HasHostEnqueue || !obs.HasDeviceActivity || obs.HasCompletion {
		t.Fatalf("stages missing: host=%v dev=%v comp=%v", obs.HasHostEnqueue, obs.HasDeviceActivity, obs.HasCompletion)
	}
	if obs.HostEnqueueStart.Nanoseconds <= 0 || obs.HostEnqueueEnd.Nanoseconds < obs.HostEnqueueStart.Nanoseconds {
		t.Fatalf("invalid host enqueue: %+v", obs.HostEnqueueStart)
	}
	if obs.DeviceStart.Nanoseconds < obs.HostEnqueueStart.Nanoseconds {
		t.Fatalf("device start %d before host enqueue %d", obs.DeviceStart.Nanoseconds, obs.HostEnqueueStart.Nanoseconds)
	}
	if obs.HasCompletion {
		t.Fatal("record publication was promoted to completion")
	}
	if obs.TransportObserved {
		t.Fatal("transport observation should remain unobserved (false)")
	}
}

func TestAttributeTwoRankDelay_HostEnqueueSkew(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Rank 0 enqueues at t=1,000,000 ns
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              0,
		TotalRanks:        2,
		Sequence:          1,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1001,
		Hostname:          "node-0",
		ClockEpoch:        "synthetic-known-clock",
		ClockEvidenceRef:  "synthetic-mapping",
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 1_020_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 1_050_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	// Rank 1 enqueues late at t=10,000,000 ns (9ms later, well beyond +/- 1000ns uncertainty)
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              1,
		TotalRanks:        2,
		Sequence:          1,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1002,
		Hostname:          "node-1",
		ClockEpoch:        "synthetic-known-clock",
		ClockEvidenceRef:  "synthetic-mapping",
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 10_000_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 10_010_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 10_020_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 10_050_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Admitted {
		t.Fatalf("expected admitted report, got reason: %s", report.Reason)
	}
	if report.Diagnosis != ncclinspector.DiagnosisHostEnqueueSkew {
		t.Fatalf("diagnosis = %v, want %v", report.Diagnosis, ncclinspector.DiagnosisHostEnqueueSkew)
	}
	if report.DelayedRank != 1 {
		t.Fatalf("delayed rank = %d, want 1", report.DelayedRank)
	}
	if report.DelayNS != 9_000_000 {
		t.Fatalf("delay ns = %d, want 9000000", report.DelayNS)
	}
}

func TestAttributeTwoRankDelay_DeviceStartDelay(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Both ranks enqueue at approximately same time (t=1,000,000 ns vs t=1,000,200 ns with 500ns uncertainty)
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              0,
		TotalRanks:        2,
		Sequence:          2,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1001,
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 1_020_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 1_050_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	// Rank 1 device kernel start delayed to t=5,000,000 ns (preceding stream work on GPU)
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              1,
		TotalRanks:        2,
		Sequence:          2,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1002,
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_100, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_100, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 5_000_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 5_030_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Admitted {
		t.Fatalf("expected admitted report, got reason: %s", report.Reason)
	}
	if report.Diagnosis != ncclinspector.DiagnosisDeviceStartDelay {
		t.Fatalf("diagnosis = %v, want %v", report.Diagnosis, ncclinspector.DiagnosisDeviceStartDelay)
	}
	if report.DelayedRank != 1 {
		t.Fatalf("delayed rank = %d, want 1", report.DelayedRank)
	}
}

func TestAttributeTwoRankDelay_ExecutionImbalance(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Both ranks enqueue and start concurrently
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              0,
		TotalRanks:        2,
		Sequence:          3,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1001,
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 1_005_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 1_030_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic}, // 20µs
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	// Rank 1 takes 100µs on device (80µs longer)
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash:  validCommHashA,
		Rank:              1,
		TotalRanks:        2,
		Sequence:          3,
		Operation:         "AllReduce",
		Bytes:             65536,
		HostPID:           1002,
		HostEnqueueStart:  ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:    ncclinspector.BoundedTimestamp{Nanoseconds: 1_005_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic},
		DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 1_110_000, UncertaintyNS: 100, Domain: ncclinspector.ClockDomainHostMonotonic}, // 100µs
		HasHostEnqueue:    true,
		HasDeviceActivity: true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Admitted {
		t.Fatalf("expected admitted report, got reason: %s", report.Reason)
	}
	if report.Diagnosis != ncclinspector.DiagnosisExecutionImbalance {
		t.Fatalf("diagnosis = %v, want %v", report.Diagnosis, ncclinspector.DiagnosisExecutionImbalance)
	}
	if report.DelayedRank != 1 {
		t.Fatalf("delayed rank = %d, want 1", report.DelayedRank)
	}
}

func TestWithholdOrderWhenUncertaintyExceedsInterval(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Rank 0 enqueues at 1,000,000 ns with +/- 500ns uncertainty
	// Rank 1 enqueues at 1,000,400 ns with +/- 500ns uncertainty
	// Combined uncertainty is 1,000 ns; measured delta is 400 ns (<= 1,000 ns)
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash: validCommHashA,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         4,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          1001,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash: validCommHashA,
		Rank:             1,
		TotalRanks:       2,
		Sequence:         4,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          1002,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_400, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_400, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 4)
	if err != nil {
		t.Fatal(err)
	}
	if report.Admitted {
		t.Fatal("expected order and causality to be withheld when uncertainty exceeds interval")
	}
	if report.Diagnosis != ncclinspector.DiagnosisOrderIndeterminate {
		t.Fatalf("diagnosis = %v, want %v", report.Diagnosis, ncclinspector.DiagnosisOrderIndeterminate)
	}
	if report.DelayedRank != -1 {
		t.Fatalf("delayed rank = %d, want -1", report.DelayedRank)
	}
}

func TestPartialRankReporting(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Only rank 0 reports
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash: validCommHashA,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         5,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          1001,
		HasHostEnqueue:   false,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 5)
	if err != nil {
		t.Fatal(err)
	}
	if report.Admitted {
		t.Fatal("expected partial rank report to not be admitted as a full attribution")
	}
	if report.Diagnosis != ncclinspector.DiagnosisPartialRank {
		t.Fatalf("diagnosis = %v, want %v", report.Diagnosis, ncclinspector.DiagnosisPartialRank)
	}
	if len(report.ReportedRanks) != 1 || report.ReportedRanks[0] != 0 {
		t.Fatalf("reported ranks = %v, want [0]", report.ReportedRanks)
	}
	if len(report.MissingRanks) != 1 || report.MissingRanks[0] != 1 {
		t.Fatalf("missing ranks = %v, want [1]", report.MissingRanks)
	}
}

func TestIncompatibleClockDomains(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash: validCommHashA,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         6,
		Operation:        "AllReduce",
		HostPID:          1001,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash: validCommHashA,
		Rank:             1,
		TotalRanks:       2,
		Sequence:         6,
		Operation:        "AllReduce",
		HostPID:          1002,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, Domain: ncclinspector.ClockDomainHostRealtime},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, Domain: ncclinspector.ClockDomainHostRealtime},
		HasHostEnqueue:   true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(validCommHashA, 6)
	if err != nil {
		t.Fatal(err)
	}
	if report.Admitted || report.Diagnosis != ncclinspector.DiagnosisIncompatibleClocks {
		t.Fatalf("expected IncompatibleClocks refusal, got: %+v", report)
	}
}

func TestNegativeFixtures(t *testing.T) {
	t.Run("truncated_communicator_hash", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: truncatedHash,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         1,
			HostPID:          1001,
		})
		if err == nil || !errors.Is(err, ncclinspector.ErrTruncatedHash) {
			t.Fatalf("expected ErrTruncatedHash, got: %v", err)
		}
	})

	t.Run("swapped_rank_out_of_bounds", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             2, // out of bounds for 2-rank clique
			TotalRanks:       2,
			Sequence:         1,
			HostPID:          1001,
		})
		if err == nil || !errors.Is(err, ncclinspector.ErrRankMismatch) {
			t.Fatalf("expected ErrRankMismatch, got: %v", err)
		}
	})

	t.Run("duplicate_sequence", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		obs := ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         10,
			HostPID:          1001,
		}
		if err := attr.AddObservation(obs); err != nil {
			t.Fatal(err)
		}
		err := attr.AddObservation(obs)
		if err == nil || !errors.Is(err, ncclinspector.ErrDuplicateSequence) {
			t.Fatalf("expected ErrDuplicateSequence, got: %v", err)
		}
	})

	t.Run("rank_restart_pid_change", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		if err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         10,
			HostPID:          1001,
		}); err != nil {
			t.Fatal(err)
		}
		// PID changes from 1001 to 2002 on same rank & comm -> rank restart!
		err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         11,
			HostPID:          2002,
		})
		if err == nil || !errors.Is(err, ncclinspector.ErrRankRestarted) {
			t.Fatalf("expected ErrRankRestarted for PID change, got: %v", err)
		}
	})

	t.Run("rank_restart_sequence_rewind", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		if err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         50,
			HostPID:          1001,
		}); err != nil {
			t.Fatal(err)
		}
		// Sequence rewinds to 10 -> restart / reset!
		err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash: validCommHashA,
			Rank:             0,
			TotalRanks:       2,
			Sequence:         10,
			HostPID:          1001,
		})
		if err == nil || !errors.Is(err, ncclinspector.ErrRankRestarted) {
			t.Fatalf("expected ErrRankRestarted for sequence rewind, got: %v", err)
		}
	})
}

func TestMultiCommunicatorAndReuse(t *testing.T) {
	attr := ncclinspector.NewTwoRankAttributor()

	// Communicator A with sequences 1, 2
	for seq := uint64(1); seq <= 2; seq++ {
		for rank := 0; rank < 2; rank++ {
			if err := attr.AddObservation(ncclinspector.CollectiveObservation{
				CommunicatorHash:  validCommHashA,
				Rank:              rank,
				TotalRanks:        2,
				Sequence:          seq,
				Operation:         "AllReduce",
				HostPID:           1000 + rank,
				HasDeviceActivity: true,
				DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: int64(seq * 1000), Domain: ncclinspector.ClockDomainHostMonotonic},
				DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: int64(seq*1000 + 50), Domain: ncclinspector.ClockDomainHostMonotonic},
			}); err != nil {
				t.Fatalf("commA seq %d rank %d: %v", seq, rank, err)
			}
		}
		rep, err := attr.Attribute(validCommHashA, seq)
		if err != nil || rep.Diagnosis != ncclinspector.DiagnosisPartialRank {
			t.Fatalf("commA seq %d: err=%v rep=%+v", seq, err, rep)
		}
	}

	// Communicator B with sequence 1 (same sequence number as A, must not collide)
	for rank := 0; rank < 2; rank++ {
		if err := attr.AddObservation(ncclinspector.CollectiveObservation{
			CommunicatorHash:  validCommHashB,
			Rank:              rank,
			TotalRanks:        2,
			Sequence:          1,
			Operation:         "Broadcast",
			HostPID:           3000 + rank,
			HasDeviceActivity: true,
			DeviceStart:       ncclinspector.BoundedTimestamp{Nanoseconds: 5000, Domain: ncclinspector.ClockDomainHostMonotonic},
			DeviceEnd:         ncclinspector.BoundedTimestamp{Nanoseconds: 5080, Domain: ncclinspector.ClockDomainHostMonotonic},
		}); err != nil {
			t.Fatalf("commB rank %d: %v", rank, err)
		}
	}

	repB, err := attr.Attribute(validCommHashB, 1)
	if err != nil || repB.Operation != "Broadcast" {
		t.Fatalf("commB attribution failed: err=%v rep=%+v", err, repB)
	}
}
