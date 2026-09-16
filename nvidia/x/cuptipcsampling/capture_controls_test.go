package cuptipcsampling

import (
	"errors"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// testPublicControlAPI accepts every native call and reports the qualified
// version. Individual controls replace one hook to exercise a single failure.
func testPublicControlAPI() samplingControlAPI {
	var saved []cupti.PCSamplingConfigurationInfo
	return samplingControlAPI{
		device:  func(cupti.Context) (uint32, error) { return 0, nil },
		version: func() (uint32, error) { return 130001, nil },
		set: func(p *cupti.PCSamplingConfigurationInfoParams) error {
			a := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
			for i := range a {
				a[i].AttributeStatus = 0
			}
			saved = append(saved, a...)
			return nil
		},
		get: func(p *cupti.PCSamplingConfigurationInfoParams) error {
			a := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
			for i := range a {
				for _, v := range saved {
					if a[i].AttributeType == v.AttributeType {
						a[i] = v
					}
				}
			}
			return nil
		},
		enable:  func(*cupti.PCSamplingEnableParams) error { return nil },
		start:   func(*cupti.PCSamplingStartParams) error { return nil },
		stop:    func(*cupti.PCSamplingStopParams) error { return nil },
		disable: func(*cupti.PCSamplingDisableParams) error { return nil },
	}
}

func testPublicInventory(cupti.Context, int) (ReasonInventory, error) {
	return ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{{Index: 7, Name: "reason"}}}, nil
}

func testPublicNames(entries, nameBytes, totalBytes int) (*samplingNames, error) {
	return newSamplingNames(entries, nameBytes, totalBytes, func(uintptr, int) (string, error) { return "kernel", nil }, func(uintptr) {})
}

func testPublicEmptyQuery(*cupti.PCSamplingGetDataParams) error { return nil }

// testPublicOnePC writes one complete PC record through the native output
// pointer, the way CUPTI fills a caller-supplied buffer. name selects the
// native name pointer, so successive drains can present distinct names.
func testPublicOnePC(p *cupti.PCSamplingGetDataParams) error { return testPublicOnePCNamed(p, 99) }

func testPublicOnePCNamed(p *cupti.PCSamplingGetDataParams, name uintptr) error {
	d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
	pcs := unsafe.Slice((*cupti.PCSamplingPCData)(nativePointer(d.PPcData)), int(d.CollectNumPcs))
	reasons := unsafe.Slice((*cupti.PCSamplingStallReason)(nativePointer(pcs[0].StallReason)), 1)
	reasons[0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 7, Samples: 2}
	pcs[0].CubinCrc, pcs[0].PcOffset = 0x1234, 0x40
	pcs[0].FunctionIndex, pcs[0].CorrelationId = 3, 11
	pcs[0].FunctionName, pcs[0].StallReasonCount = name, 1
	d.TotalNumPcs, d.TotalSamples = 1, 5
	return nil
}

func testPublicStart(t *testing.T, api samplingControlAPI, query func(*cupti.PCSamplingGetDataParams) error) *Session {
	t.Helper()
	s, err := startSession(5, testPublicConfig(), api, testPublicInventory, testPublicNames, query)
	if s == nil || err != nil {
		t.Fatalf("session=%v err=%v", s, err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// A rejected construction must leave no session, no retained error state and
// no package ownership: a later qualified start has to succeed.
func TestPublicStartConstructionFailure(t *testing.T) {
	versionFailure := errors.New("version unavailable")
	inventoryFailure := errors.New("inventory unavailable")
	nameFailure := errors.New("allocator unavailable")
	for _, tt := range []struct {
		name      string
		api       func(samplingControlAPI) samplingControlAPI
		inventory func(cupti.Context, int) (ReasonInventory, error)
		names     func(int, int, int) (*samplingNames, error)
		want      string
	}{
		{name: "version error", api: func(a samplingControlAPI) samplingControlAPI {
			a.version = func() (uint32, error) { return 0, versionFailure }
			return a
		}, want: versionFailure.Error()},
		{name: "unqualified version", api: func(a samplingControlAPI) samplingControlAPI {
			a.version = func() (uint32, error) { return 130000, nil }
			return a
		}, want: "unqualified sampling version 130000"},
		{name: "inventory error", inventory: func(cupti.Context, int) (ReasonInventory, error) {
			return ReasonInventory{}, inventoryFailure
		}, want: inventoryFailure.Error()},
		{name: "inventory version", inventory: func(cupti.Context, int) (ReasonInventory, error) {
			return ReasonInventory{CUPTIVersion: 27, Reasons: []StallReason{{Index: 7}}}, nil
		}, want: "reason inventory version changed"},
		{name: "unknown reason", inventory: func(cupti.Context, int) (ReasonInventory, error) {
			return ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{{Index: 8}}}, nil
		}, want: "unknown or duplicate selected reason"},
		{name: "name owner", names: func(int, int, int) (*samplingNames, error) { return nil, nameFailure }, want: nameFailure.Error()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			api := testPublicControlAPI()
			if tt.api != nil {
				api = tt.api(api)
			}
			inventory := testPublicInventory
			if tt.inventory != nil {
				inventory = tt.inventory
			}
			names := testPublicNames
			if tt.names != nil {
				names = tt.names
			}
			s, err := startSession(5, testPublicConfig(), api, inventory, names, testPublicEmptyQuery)
			if s != nil || err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("session=%v err=%v", s, err)
			}
			// Ownership was never taken, or was returned with the storage.
			next := testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery)
			if next.Info().Closed {
				t.Fatal("qualified start blocked after rejected construction")
			}
		})
	}
}

// Only one session of this package may own the collector at a time. This does
// not detect external registrants; the caller still coordinates those.
func TestPublicStartExclusive(t *testing.T) {
	testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery)
	s, err := startSession(5, testPublicConfig(), testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicEmptyQuery)
	if s != nil || err == nil || !strings.Contains(err.Error(), "another session owns PC sampling") {
		t.Fatalf("session=%v err=%v", s, err)
	}
}

// A returned Batch is owned by the caller: editing it must not reach the
// session, and a later batch must not observe the edit.
func TestPublicBatchOwnership(t *testing.T) {
	s := testPublicStart(t, testPublicControlAPI(), testPublicOnePC)
	first, err := s.Read()
	if err != nil || !first.Valid || len(first.PCs) != 1 || first.PCs[0].Name != "kernel" || len(first.PCs[0].Reasons) != 1 {
		t.Fatalf("batch=%+v err=%v", first, err)
	}
	first.PCs[0].Name = "edited"
	first.PCs[0].Reasons[0].Samples = 999
	first.TotalSamples = 999
	second, err := s.Read()
	if err != nil || len(second.PCs) != 1 || second.PCs[0].Name != "kernel" || second.PCs[0].Reasons[0].Samples != 2 || second.TotalSamples != 5 {
		t.Fatalf("batch=%+v err=%v", second, err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	final, ok := s.Final()
	if !ok {
		t.Fatal("final unavailable")
	}
	final.TotalSamples = 999
	if again, _ := s.Final(); again.TotalSamples == 999 {
		t.Fatal("final aliases caller")
	}
}

// A failed native disable retains cleanup ownership for an explicit retry.
// The retry may succeed, but the earlier failure stays an error.
func TestPublicCloseRetryAfterDisableFailure(t *testing.T) {
	failure := errors.New("disable failed")
	api := testPublicControlAPI()
	disable := api.disable
	calls := 0
	api.disable = func(p *cupti.PCSamplingDisableParams) error {
		if calls++; calls == 1 {
			return failure
		}
		return disable(p)
	}
	s := testPublicStart(t, api, testPublicEmptyQuery)
	if err := s.Close(); !errors.Is(err, failure) {
		t.Fatalf("close=%v", err)
	}
	if _, ok := s.Final(); ok {
		t.Fatal("final available after failed disable")
	}
	if info := s.Info(); info.Closed || !info.Failed {
		t.Fatalf("info=%+v", info)
	}
	err := s.Close()
	if err == nil || errors.Is(err, failure) || !strings.Contains(err.Error(), "previously failed") {
		t.Fatalf("retry=%v", err)
	}
	if _, ok := s.Final(); !ok {
		t.Fatal("final unavailable after successful retry")
	}
	// Sticky: a later Close repeats the recorded outcome, never a clean one.
	if got := s.Close(); got == nil || got.Error() != err.Error() {
		t.Fatalf("repeat=%v", got)
	}
	if info := s.Info(); !info.Closed || !info.Failed || calls != 2 {
		t.Fatalf("info=%+v calls=%d", info, calls)
	}
}

// Reads are budgeted, and exhausting the budget fails the session rather than
// silently dropping records.
func TestPublicReadBudget(t *testing.T) {
	s := testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery)
	for i := uint(0); i < testPublicConfig().MaxReads; i++ {
		if _, err := s.Read(); err != nil {
			t.Fatalf("read %d: %v", i, err)
		}
	}
	if _, err := s.Read(); err == nil || !strings.Contains(err.Error(), "read budget exhausted") {
		t.Fatalf("read=%v", err)
	}
	if info := s.Info(); info.Reads != testPublicConfig().MaxReads || !info.Failed {
		t.Fatalf("info=%+v", info)
	}
	if err := s.Close(); err == nil {
		t.Fatal("close hid a failed session")
	}
}

// Stop ends the range without ending the session or releasing storage.
func TestPublicStop(t *testing.T) {
	s := testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery)
	if err := s.Stop(); err != nil {
		t.Fatal(err)
	}
	if info := s.Info(); info.Starts != 1 || info.Stops != 1 || info.Closed {
		t.Fatalf("info=%+v", info)
	}
	if err := s.Stop(); err == nil {
		t.Fatal("second stop accepted")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

// Names accumulate for the session, so a config that Start accepts can still
// exhaust MaxNames on a later read. The budget must fail visibly, and it must
// never silently drop a name or return a partial batch as valid.
func TestPublicNameBudgetAcrossReads(t *testing.T) {
	c := testPublicConfig()
	c.MaxNames = 2 * c.PCsPerBuffer // the per-drain worst case Start enforces
	if err := validateConfig(5, c); err != nil {
		t.Fatalf("Start rejected the documented minimum: %v", err)
	}
	name := uintptr(100)
	s, err := startSession(5, c, testPublicControlAPI(), testPublicInventory, testPublicNames, func(p *cupti.PCSamplingGetDataParams) error {
		name++
		return testPublicOnePCNamed(p, name)
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if b, err := s.Read(); err != nil || len(b.PCs) != 1 {
		t.Fatalf("batch=%+v err=%v", b, err)
	}
	b, err := s.Read()
	if err == nil || !strings.Contains(err.Error(), "native name entry budget exhausted") {
		t.Fatalf("read=%v", err)
	}
	if b.Valid || len(b.PCs) != 0 {
		t.Fatalf("exhausted read returned data: %+v", b)
	}
	if !s.Info().Failed {
		t.Fatal("name exhaustion did not fail the session")
	}
}

// Loss seen by an earlier Read survives to Close even when the final buffer is
// itself clean. Close still reports an error, and Final stays available.
func TestPublicCloseWithPriorReadLoss(t *testing.T) {
	s := testPublicStart(t, testPublicControlAPI(), func(p *cupti.PCSamplingGetDataParams) error {
		d := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
		d.DroppedSamples, d.TotalSamples = 7, 12
		return nil
	})
	b, err := s.Read()
	if err != nil || b.DroppedSamples != 7 {
		t.Fatalf("batch=%+v err=%v", b, err)
	}
	err = s.Close()
	if err == nil || !strings.Contains(err.Error(), "final sampling output has loss or remaining PCs") {
		t.Fatalf("close=%v", err)
	}
	final, ok := s.Final()
	if !ok || final.DroppedSamples != 0 {
		t.Fatalf("final=%+v available=%v", final, ok)
	}
	if info := s.Info(); !info.Closed || !info.Failed {
		t.Fatalf("info=%+v", info)
	}
}

// A failed native stop during Close does not block release, because disable
// succeeded. Storage is freed, the error is reported, and no retry is owed.
func TestPublicCloseStopFailure(t *testing.T) {
	failure := errors.New("stop failed")
	api := testPublicControlAPI()
	api.stop = func(*cupti.PCSamplingStopParams) error { return failure }
	s := testPublicStart(t, api, testPublicEmptyQuery)
	err := s.Close()
	if !errors.Is(err, failure) {
		t.Fatalf("close=%v", err)
	}
	if _, ok := s.Final(); !ok {
		t.Fatal("final unavailable after a successful disable")
	}
	info := s.Info()
	if !info.Closed || !info.Failed || info.Stops != 0 {
		t.Fatalf("info=%+v", info)
	}
	// Nothing is owed: a repeat Close replays the recorded outcome only.
	if got := s.Close(); got == nil || got.Error() != err.Error() {
		t.Fatalf("repeat=%v", got)
	}
}
