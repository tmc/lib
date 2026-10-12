package cuptipcsampling

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
)

// A batch returned by a successful Read has left the native buffer. When the
// phase batch limit is reached, the final drain must not read such a batch
// only to discard it and report its samples as still remaining in hardware.
func TestPhaseCaptureLimitKeepsDrainedBatch(t *testing.T) {
	var reads []uint64 // TotalSamples handed out by each successful native read
	starter := func(ctx driver.Context, cfg Config) (*Session, error) {
		query := func(p *cupti.PCSamplingGetDataParams) error {
			if err := testPublicOnePC(p); err != nil {
				return err
			}
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			d.TotalSamples = 10 * uint64(len(reads)+1)
			d.RemainingNumPcs = 1
			reads = append(reads, d.TotalSamples)
			return nil
		}
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, query)
	}
	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches = 1
	cfg.Config.MaxReads = 2

	report, _ := capturePhase(context.Background(), 5, cfg, starter, func() error { return nil }, func(context.Context) error { return nil })

	var drained uint64
	for _, n := range reads {
		drained += n
	}
	var kept uint64
	for _, b := range report.Batches {
		kept += b.TotalSamples
	}
	if kept != drained {
		t.Errorf("native reads returned samples %v, report batches kept %d of %d", reads, kept, drained)
	}
	if report.TotalSamples != drained {
		t.Errorf("TotalSamples = %d, want %d drained from native buffers", report.TotalSamples, drained)
	}
	if report.RemainingPCs != 1 || !report.DrainLimited || report.FinalDrained {
		t.Errorf("RemainingPCs %d DrainLimited %v FinalDrained %v; want 1 from the read after Stop, limited, not drained",
			report.RemainingPCs, report.DrainLimited, report.FinalDrained)
	}
}

// stalePhaseStarter returns a starter whose first native read, taken by the
// poll goroutine before Stop, finds the buffer momentarily empty, and whose
// later reads find 4 PC records the workload produced afterwards.
func stalePhaseStarter(reads *atomic.Int32, polled chan struct{}) func(driver.Context, Config) (*Session, error) {
	return func(ctx driver.Context, cfg Config) (*Session, error) {
		query := func(p *cupti.PCSamplingGetDataParams) error {
			if err := testPublicOnePC(p); err != nil {
				return err
			}
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			if reads.Add(1) == 1 {
				d.TotalSamples, d.RemainingNumPcs = 5, 0
				close(polled)
			} else {
				d.TotalSamples, d.RemainingNumPcs = 7, 4
			}
			return nil
		}
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, query)
	}
}

// When the poll goroutine spends MaxBatches before Stop, the final drain
// reads nothing. A remaining count from that pre-Stop read says nothing
// about the buffer at the end of the phase, so the report must show loss.
func TestPhaseCaptureLimitBeforeStopIsLoss(t *testing.T) {
	var reads atomic.Int32
	polled := make(chan struct{})
	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches = 1
	cfg.Limits.PollInterval = time.Millisecond
	cfg.Config.MaxReads = 2
	report, err := capturePhase(context.Background(), 5, cfg, stalePhaseStarter(&reads, polled), func() error { return nil },
		func(context.Context) error { <-polled; return nil })
	if err != nil {
		t.Fatalf("capturePhase: %v", err)
	}
	if !report.Loss() {
		t.Errorf("Loss() = false with remainingPCs=%d after %d reads; the final drain was cut by MaxBatches", report.RemainingPCs, reads.Load())
	}
}

// flushPhaseStarter returns a starter whose single drain read reports one PC
// remaining and whose disable flushes that PC into the configured output,
// which the session returns as its final batch.
func flushPhaseStarter() func(driver.Context, Config) (*Session, error) {
	return func(ctx driver.Context, cfg Config) (*Session, error) {
		api := testPublicControlAPI()
		set := api.set
		var output uintptr
		api.set = func(p *cupti.PCSamplingConfigurationInfoParams) error {
			for _, a := range unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes)) {
				if a.AttributeType == cupti.PcSamplingConfigurationAttrTypeSamplingDataBuffer {
					output = uintptr(a.AttributeData[0])
				}
			}
			return set(p)
		}
		api.disable = func(*cupti.PCSamplingDisableParams) error {
			if output == 0 {
				return errors.New("no output buffer configured")
			}
			p := cupti.PCSamplingGetDataParams{PcSamplingData: output}
			if err := testPublicOnePC(&p); err != nil {
				return err
			}
			d := (*cupti.PCSamplingData)(nativePointer(output))
			d.TotalSamples, d.RemainingNumPcs = 20, 0
			return nil
		}
		query := func(p *cupti.PCSamplingGetDataParams) error {
			if err := testPublicOnePC(p); err != nil {
				return err
			}
			d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
			d.TotalSamples, d.RemainingNumPcs = 10, 1
			return nil
		}
		return startSession(ctx, cfg, api, testPublicInventory, testPublicNames, query)
	}
}

// A disable that flushes the remaining PC must not lose the flushed batch,
// and the report must still not claim a completed final drain.
func TestPhaseCaptureLimitWithFlushingDisable(t *testing.T) {
	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches = 1
	cfg.Config.MaxReads = 2
	report, err := capturePhase(context.Background(), 5, cfg, flushPhaseStarter(), func() error { return nil }, func(context.Context) error { return nil })
	if err != nil {
		t.Fatalf("capturePhase: %v", err)
	}
	if report.TotalSamples != 30 || len(report.Batches) != 2 || report.Batches[1].RemainingPCs != 0 {
		t.Fatalf("total %d batches %d, want 30 in 2 with the flushed final batch", report.TotalSamples, len(report.Batches))
	}
	if !report.Loss() {
		t.Error("Loss() = false; the final drain stopped at MaxBatches")
	}
}

// TestPhaseCaptureDrainStates checks the final-drain fields that SealPhase
// derives completeness from.
func TestPhaseCaptureDrainStates(t *testing.T) {
	for _, tt := range []struct {
		name                    string
		run                     func() PhaseReport
		drained, limited, final bool
		remaining               uint64
	}{
		{"limit before stop", func() PhaseReport {
			var reads atomic.Int32
			polled := make(chan struct{})
			cfg := testPhaseConfig()
			cfg.Limits.MaxBatches, cfg.Limits.PollInterval, cfg.Config.MaxReads = 1, time.Millisecond, 2
			r, _ := capturePhase(context.Background(), 5, cfg, stalePhaseStarter(&reads, polled), func() error { return nil },
				func(context.Context) error { <-polled; return nil })
			return r
		}, false, true, true, 0},
		{"limit with flushing disable", func() PhaseReport {
			cfg := testPhaseConfig()
			cfg.Limits.MaxBatches, cfg.Config.MaxReads = 1, 2
			r, _ := capturePhase(context.Background(), 5, cfg, flushPhaseStarter(), func() error { return nil }, func(context.Context) error { return nil })
			return r
		}, false, true, true, 1},
		{"drained", func() PhaseReport {
			starter := func(ctx driver.Context, cfg Config) (*Session, error) {
				return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicOnePC)
			}
			r, _ := capturePhase(context.Background(), 5, testPhaseConfig(), starter, func() error { return nil }, func(context.Context) error { return nil })
			return r
		}, true, false, true, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := tt.run()
			if r.FinalDrained != tt.drained || r.DrainLimited != tt.limited || r.FinalBatch != tt.final || r.RemainingPCs != tt.remaining || r.Loss() != tt.limited {
				t.Errorf("FinalDrained %v DrainLimited %v FinalBatch %v RemainingPCs %d Loss %v; want %v %v %v %d %v",
					r.FinalDrained, r.DrainLimited, r.FinalBatch, r.RemainingPCs, r.Loss(), tt.drained, tt.limited, tt.final, tt.remaining, tt.limited)
			}
			if uint64(len(r.Batches)) != uint64(r.Info.Reads)+1 {
				t.Errorf("%d batches from %d reads plus the final batch", len(r.Batches), r.Info.Reads)
			}
		})
	}
}

// manifestInventory names a base reason, its _not_issued companion and a
// second base reason, sorted by index as Lookup requires.
var manifestInventory = ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{
	{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
	{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"},
	{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
}}

var manifestStart = time.Date(2026, 10, 11, 12, 0, 0, 0, time.UTC)

// manifestBatches returns one complete batch: 20 offered samples, 3 non-user,
// 1 dropped, so 16 eligible. A named PC holds 10 long_scoreboard plus 4
// companion samples and 4 selected; an unnamed PC holds 2.
func manifestBatches() []Batch {
	return []Batch{{
		CUPTIVersion: 130001, CollectionMode: cupti.PcSamplingCollectionModeContinuous, Valid: true,
		TotalSamples: 20, DroppedSamples: 1, NonUserSamples: 3, RangeID: 1,
		PCs: []PC{
			{CRC: 0x1234, Offset: 0x40, Name: "kernel", Reasons: []StallSamples{{1, 10}, {2, 4}, {3, 4}}},
			{CRC: 0x1234, Offset: 0x80, Reasons: []StallSamples{{1, 2}}},
		},
	}}
}

func manifestInfo(period uint64) Info {
	return Info{
		CUPTIVersion: 130001, Inventory: manifestInventory, Verified: true, Closed: true,
		Config:    Config{Period: 10, ReasonIDs: []uint32{1, 2, 3}},
		Effective: []cupti.PCSamplingConfigurationInfo{{AttributeType: cupti.PcSamplingConfigurationAttrTypeSamplingPeriod, AttributeData: [3]uint64{period}}},
	}
}

func manifestReport(batches []Batch) PhaseReport {
	r := PhaseReport{Info: manifestInfo(10), Batches: batches, BatchesDrained: uint(len(batches)), FinalDrained: true}
	r.Info.Reads = uint(len(batches))
	for _, b := range batches {
		if b.Valid {
			r.TotalSamples += b.TotalSamples
		}
		r.DroppedSamples += b.DroppedSamples
		r.RemainingPCs += b.RemainingPCs
	}
	return r
}

func manifestRaw(t testing.TB, batches []Batch) []byte {
	t.Helper()
	var raw []byte
	for _, b := range batches {
		var err error
		if raw, err = b.AppendBinary(raw); err != nil {
			t.Fatal(err)
		}
	}
	return raw
}

func manifestObservation(t testing.TB, batches []Batch) PhaseObservation {
	return PhaseObservation{Start: manifestStart, End: manifestStart.Add(40 * time.Millisecond), Raw: manifestRaw(t, batches)}
}

var manifestRequest = PhaseRequest{Period: 10, MaxBatches: 4, Window: 50 * time.Millisecond}

func TestSealPhaseCounts(t *testing.T) {
	batches := manifestBatches()
	obs := manifestObservation(t, batches)
	m, err := SealPhase(manifestRequest, manifestReport(batches), obs)
	if err != nil {
		t.Fatal(err)
	}
	// Recompute the identity from the raw reasons, independently of SealPhase.
	var nonCompanion uint64
	for _, pc := range batches[0].PCs {
		for _, r := range pc.Reasons {
			if !strings.HasSuffix(manifestInventory.Lookup(r.Index), "_not_issued") {
				nonCompanion += uint64(r.Samples)
			}
		}
	}
	if m.Named+m.Unattributed != nonCompanion || nonCompanion > m.TotalSamples-m.NonUserSamples {
		t.Errorf("named %d + unattributed %d, non-companion %d, total-nonuser %d", m.Named, m.Unattributed, nonCompanion, m.TotalSamples-m.NonUserSamples)
	}
	want := PhaseManifest{
		RequestedPeriod: 10, ActualPeriod: 10, PeriodVerified: true,
		RequestedWindow: 50 * time.Millisecond, Start: obs.Start, End: obs.End,
		RequestedBatches: 4, Batches: 1,
		FinalDrain: true, Complete: true,
		TotalSamples: 20, NonUserSamples: 3, LostSamples: 1, EligibleSamples: 16,
		Named: 14, Unattributed: 2, Companion: 4, Unreported: 0,
		RawSHA256: sha256.Sum256(obs.Raw),
		Cleanup:   CleanupReleased, Coverage: CoverageReduced,
	}
	if fmt.Sprint(m) != fmt.Sprint(want) {
		t.Errorf("manifest\n got %+v\nwant %+v", m, want)
	}
	again, err := SealPhase(manifestRequest, manifestReport(manifestBatches()), manifestObservation(t, manifestBatches()))
	if err != nil || again.RawSHA256 != m.RawSHA256 {
		t.Errorf("reseal digest %x, err %v; want stable %x", again.RawSHA256, err, m.RawSHA256)
	}
}

func TestSealPhaseMeasured(t *testing.T) {
	batches := manifestBatches()
	batches[0].PCs = batches[0].PCs[:1]
	batches[0].TotalSamples, batches[0].DroppedSamples = 17, 0 // 14 named + 3 non-user
	m, err := SealPhase(manifestRequest, manifestReport(batches), manifestObservation(t, batches))
	if err != nil {
		t.Fatal(err)
	}
	if m.Coverage != CoverageMeasured || m.Named != 14 || m.Unattributed != 0 || m.Unreported != 0 {
		t.Errorf("coverage %s named %d unattributed %d unreported %d; want measured 14 0 0", m.Coverage, m.Named, m.Unattributed, m.Unreported)
	}
}

func TestSealPhaseRefuses(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*PhaseReport, *PhaseObservation)
		want string
	}{
		{"companion counted as base", func(r *PhaseReport, o *PhaseObservation) {
			// 14 eligible holds the 12 base samples, not 12 plus 4 companions.
			r.Batches[0].TotalSamples, r.TotalSamples = 18, 18
			r.Batches[0].PCs[0].Reasons = []StallSamples{{1, 10}, {2, 4}}
			o.Raw = manifestRaw(t, r.Batches)
		}, ""},
		{"forged zero total", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].TotalSamples, r.Batches[0].NonUserSamples, r.Batches[0].DroppedSamples = 0, 0, 0
			r.TotalSamples, r.DroppedSamples = 0, 0
			o.Raw = manifestRaw(t, r.Batches)
		}, "exceed eligible"},
		{"report total differs from batches", func(r *PhaseReport, o *PhaseObservation) {
			r.TotalSamples = 0
		}, "report total"},
		{"non-user and dropped exceed total", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].NonUserSamples = 20
			o.Raw = manifestRaw(t, r.Batches)
		}, "exceed batch total"},
		{"truncated raw", func(r *PhaseReport, o *PhaseObservation) {
			o.Raw = o.Raw[:len(o.Raw)-1]
		}, "raw output"},
		{"truncated batch", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].PCs = r.Batches[0].PCs[:1]
		}, "raw output"},
		{"dropped batch", func(r *PhaseReport, o *PhaseObservation) {
			r.BatchesDrained++
		}, "batch count"},
		{"unknown reason", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].PCs[1].Reasons[0].Index = 9
			o.Raw = manifestRaw(t, r.Batches)
		}, "not in inventory"},
		{"inventory version", func(r *PhaseReport, o *PhaseObservation) {
			r.Info.Inventory.CUPTIVersion++
		}, "version"},
		{"window ends before start", func(r *PhaseReport, o *PhaseObservation) {
			o.End = o.Start.Add(-time.Nanosecond)
		}, "window"},
		{"lost below batch drops", func(r *PhaseReport, o *PhaseObservation) {
			r.DroppedSamples = 0
		}, "dropped"},
		{"batch dropped consistently", func(r *PhaseReport, o *PhaseObservation) {
			// The session read twice; one batch is removed from every
			// count derived from Batches, but not from Info.Reads.
			r.Info.Reads = 2
		}, "reads"},
		{"failed session with more batches than reads", func(r *PhaseReport, o *PhaseObservation) {
			r.Info.Failed = true
			r.Batches = append(r.Batches, r.Batches[0])
			r.BatchesDrained++
			r.TotalSamples *= 2
			r.DroppedSamples *= 2
			o.Raw = manifestRaw(t, r.Batches)
		}, "reads"},
		{"final batch claimed but absent", func(r *PhaseReport, o *PhaseObservation) {
			r.FinalBatch = true
		}, "reads"},
		{"requested period differs from session", func(r *PhaseReport, o *PhaseObservation) {
			r.Info.Config.Period = 7
		}, "period"},
		{"companion exceeds base", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].PCs[0].Reasons = []StallSamples{{1, 4}, {2, 5}, {3, 4}}
			o.Raw = manifestRaw(t, r.Batches)
		}, "companion"},
		{"companion without selected base", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].PCs[0].Reasons = []StallSamples{{2, 5}, {3, 4}}
			o.Raw = manifestRaw(t, r.Batches)
		}, "companion"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			batches := manifestBatches()
			r, o := manifestReport(batches), manifestObservation(t, batches)
			tt.edit(&r, &o)
			_, err := SealPhase(manifestRequest, r, o)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("SealPhase: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("SealPhase error %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestSealPhaseStates(t *testing.T) {
	for _, tt := range []struct {
		name  string
		edit  func(*PhaseReport, *PhaseObservation)
		check func(PhaseManifest) bool
	}{
		{"base", func(*PhaseReport, *PhaseObservation) {},
			func(m PhaseManifest) bool {
				return m.Complete && m.Coverage == CoverageMeasured && m.Cleanup == CleanupReleased
			}},
		{"no final drain", func(r *PhaseReport, o *PhaseObservation) { r.FinalDrained = false },
			func(m PhaseManifest) bool { return !m.Complete && !m.FinalDrain && m.Coverage == CoverageReduced }},
		{"drain limited", func(r *PhaseReport, o *PhaseObservation) { r.DrainLimited = true },
			func(m PhaseManifest) bool { return !m.Complete && m.DrainLimited && m.Coverage == CoverageReduced }},
		{"canceled", func(r *PhaseReport, o *PhaseObservation) { r.Canceled = true },
			func(m PhaseManifest) bool { return m.Coverage == CoverageReduced }},
		{"consumer failed", func(r *PhaseReport, o *PhaseObservation) { r.ConsumerErr = errors.New("sink full") },
			func(m PhaseManifest) bool { return m.Coverage == CoverageReduced }},
		{"session failed", func(r *PhaseReport, o *PhaseObservation) { r.Info.Failed = true },
			func(m PhaseManifest) bool { return m.SessionFailed && m.Coverage == CoverageReduced }},
		{"unreported eligible", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].TotalSamples, r.TotalSamples = 19, 19
			o.Raw = manifestRaw(t, r.Batches)
		}, func(m PhaseManifest) bool { return m.Unreported == 2 && m.Coverage == CoverageReduced }},
		{"rejected period readback", func(r *PhaseReport, o *PhaseObservation) { r.Info.Effective[0].AttributeStatus = 1 },
			func(m PhaseManifest) bool { return m.ActualPeriod == 0 && m.Coverage == CoverageReduced }},
		{"unverified period", func(r *PhaseReport, o *PhaseObservation) { r.Info.Verified = false },
			func(m PhaseManifest) bool {
				return m.ActualPeriod == 10 && !m.PeriodVerified && m.Coverage == CoverageReduced
			}},
		{"final batch", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches = append(r.Batches, Batch{CUPTIVersion: 130001, Valid: true})
			r.BatchesDrained++
			r.FinalBatch = true
			o.Raw = manifestRaw(t, r.Batches)
		}, func(m PhaseManifest) bool { return m.Batches == 2 && m.Coverage == CoverageMeasured }},
		{"failed read", func(r *PhaseReport, o *PhaseObservation) { r.Info.Reads++; r.Info.Failed = true },
			func(m PhaseManifest) bool { return m.Coverage == CoverageReduced }},
		{"remaining in final batch", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].RemainingPCs = 3
			o.Raw = manifestRaw(t, r.Batches)
		}, func(m PhaseManifest) bool { return !m.Complete && m.FinalDrain && m.Coverage == CoverageReduced }},
		{"remaining in report", func(r *PhaseReport, o *PhaseObservation) { r.RemainingPCs = 3 },
			func(m PhaseManifest) bool { return !m.Complete && m.RemainingPCs == 3 }},
		{"zero samples", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches[0].TotalSamples, r.Batches[0].NonUserSamples, r.Batches[0].DroppedSamples = 0, 0, 0
			r.Batches[0].PCs = nil
			r.TotalSamples, r.DroppedSamples = 0, 0
			o.Raw = manifestRaw(t, r.Batches)
		}, func(m PhaseManifest) bool { return m.Coverage == CoverageUnavailable && m.Complete }},
		{"no batches", func(r *PhaseReport, o *PhaseObservation) {
			*r = manifestReport(nil)
			o.Raw = nil
		}, func(m PhaseManifest) bool { return m.Coverage == CoverageUnavailable && m.Batches == 0 }},
		{"close pending", func(r *PhaseReport, o *PhaseObservation) { r.Info.Closed = false },
			func(m PhaseManifest) bool { return m.Cleanup == CleanupPending }},
		{"close failed", func(r *PhaseReport, o *PhaseObservation) { r.Info.Closed, r.Info.Failed = false, true },
			func(m PhaseManifest) bool { return m.Cleanup == CleanupFailed && m.SessionFailed }},
		{"released after failure", func(r *PhaseReport, o *PhaseObservation) { r.Info.Failed = true },
			func(m PhaseManifest) bool { return m.Cleanup == CleanupReleased && m.SessionFailed }},
		{"actual period differs", func(r *PhaseReport, o *PhaseObservation) {
			r.Info.Effective[0].AttributeData[0] = 12
			r.Info.Verified = false
		}, func(m PhaseManifest) bool {
			return m.RequestedPeriod == 10 && m.ActualPeriod == 12 && !m.PeriodVerified
		}},
		{"no period readback", func(r *PhaseReport, o *PhaseObservation) { r.Info.Effective = nil },
			func(m PhaseManifest) bool {
				return m.RequestedPeriod == 10 && m.ActualPeriod == 0 && m.Coverage == CoverageReduced
			}},
		{"hardware full", func(r *PhaseReport, o *PhaseObservation) { r.HardwareFull = true },
			func(m PhaseManifest) bool { return m.HardwareFull && m.Coverage == CoverageReduced }},
		{"invalid batch", func(r *PhaseReport, o *PhaseObservation) {
			r.Batches = append(r.Batches, Batch{CUPTIVersion: 130001, TotalSamples: 99, PCs: []PC{{Name: "kernel", Reasons: []StallSamples{{1, 99}}}}})
			r.BatchesDrained++
			r.Info.Reads++
			o.Raw = manifestRaw(t, r.Batches)
		}, func(m PhaseManifest) bool {
			return m.InvalidBatches == 1 && m.TotalSamples == 17 && m.Named == 14 && m.Coverage == CoverageReduced
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			batches := manifestBatches()
			r, o := manifestReport(batches), manifestObservation(t, batches)
			// Start from a measured manifest: no loss, no unattributed PC.
			r.Batches[0].PCs = r.Batches[0].PCs[:1]
			r.Batches[0].TotalSamples, r.TotalSamples = 17, 17
			r.Batches[0].DroppedSamples, r.DroppedSamples = 0, 0
			o.Raw = manifestRaw(t, r.Batches)
			tt.edit(&r, &o)
			m, err := SealPhase(manifestRequest, r, o)
			if err != nil {
				t.Fatalf("SealPhase: %v", err)
			}
			if !tt.check(m) {
				t.Fatalf("manifest %+v", m)
			}
		})
	}
}

// TestSealPhaseFromCapture seals reports produced by capturePhase itself, so
// the read and drain fields SealPhase cross-checks come from the session.
func TestSealPhaseFromCapture(t *testing.T) {
	seal := func(r PhaseReport) (PhaseManifest, error) {
		raw := manifestRaw(t, r.Batches)
		return SealPhase(PhaseRequest{Period: 10, MaxBatches: 2}, r, PhaseObservation{Raw: raw})
	}
	starter := func(ctx driver.Context, cfg Config) (*Session, error) {
		return startSession(ctx, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicOnePC)
	}
	r, err := capturePhase(context.Background(), 5, testPhaseConfig(), starter, func() error { return nil }, func(context.Context) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	m, err := seal(r)
	if err != nil || !m.Complete || m.Cleanup != CleanupReleased || m.ActualPeriod != 10 || !m.PeriodVerified {
		t.Errorf("drained capture: err %v manifest %+v", err, m)
	}

	var reads atomic.Int32
	polled := make(chan struct{})
	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches, cfg.Limits.PollInterval, cfg.Config.MaxReads = 1, time.Millisecond, 2
	r, _ = capturePhase(context.Background(), 5, cfg, stalePhaseStarter(&reads, polled), func() error { return nil },
		func(context.Context) error { <-polled; return nil })
	m, err = seal(r)
	if err != nil || m.Complete || !m.DrainLimited || m.Coverage == CoverageMeasured {
		t.Errorf("limited capture: err %v complete %v limited %v coverage %s", err, m.Complete, m.DrainLimited, m.Coverage)
	}

	if len(r.Batches) == 0 || r.Batches[0].TotalSamples != 5 {
		t.Fatalf("limited capture batches %+v, want the 5-sample poll batch first", r.Batches)
	}
	r.Batches, r.BatchesDrained = r.Batches[1:], r.BatchesDrained-1
	r.TotalSamples -= 5
	if _, err := seal(r); err == nil || !strings.Contains(err.Error(), "reads") {
		t.Errorf("capture with a batch removed consistently: err %v", err)
	}
}

func TestBatchAppendBinaryDistinct(t *testing.T) {
	base := manifestBatches()[0]
	enc := func(b Batch) string {
		raw, err := b.AppendBinary(nil)
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	edits := []func(*Batch){
		func(b *Batch) { b.Valid = false },
		func(b *Batch) { b.RemainingPCs++ },
		func(b *Batch) { b.PCs[0].Name = "kernel2" },
		func(b *Batch) { b.PCs[0].Reasons = b.PCs[0].Reasons[:2] },
		func(b *Batch) { b.PCs = b.PCs[:1] },
		func(b *Batch) { b.PCs[1].Offset++ },
	}
	seen := map[string]int{enc(base): -1}
	for i, edit := range edits {
		b := manifestBatches()[0]
		edit(&b)
		s := enc(b)
		if j, ok := seen[s]; ok {
			t.Errorf("edit %d encodes like %d", i, j)
		}
		seen[s] = i
	}
	if !slices.Equal([]byte(enc(base)), []byte(enc(manifestBatches()[0]))) {
		t.Error("encoding not deterministic")
	}
}

func ExampleSealPhase() {
	inv := ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{
		{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
		{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"},
	}}
	batch := Batch{CUPTIVersion: 130001, Valid: true, TotalSamples: 12, NonUserSamples: 2,
		PCs: []PC{{Name: "kernel", Reasons: []StallSamples{{1, 10}, {2, 3}}}}}
	raw, _ := batch.AppendBinary(nil)
	report := PhaseReport{
		Info: Info{
			CUPTIVersion: 130001, Inventory: inv, Closed: true, Reads: 1, Verified: true,
			Config: Config{Period: 10, ReasonIDs: []uint32{1, 2}},
			Effective: []cupti.PCSamplingConfigurationInfo{
				{AttributeType: cupti.PcSamplingConfigurationAttrTypeSamplingPeriod, AttributeData: [3]uint64{10}},
			},
		},
		Batches: []Batch{batch}, BatchesDrained: 1, TotalSamples: 12, FinalDrained: true,
	}
	m, err := SealPhase(PhaseRequest{Period: 10, MaxBatches: 1}, report, PhaseObservation{Raw: raw})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("eligible", m.EligibleSamples, "named", m.Named, "companion", m.Companion)
	fmt.Println("actual period", m.ActualPeriod, "complete", m.Complete)
	fmt.Println("coverage", m.Coverage, "cleanup", m.Cleanup)
	// Output:
	// eligible 10 named 10 companion 3
	// actual period 10 complete true
	// coverage measured cleanup released
}

func ExampleBatch_AppendBinary() {
	b := Batch{CUPTIVersion: 130001, Valid: true, TotalSamples: 5,
		PCs: []PC{{Name: "kernel", Reasons: []StallSamples{{Index: 1, Samples: 5}}}}}
	raw, err := b.AppendBinary(nil)
	fmt.Println(len(raw), err)
	// Output: 100 <nil>
}
