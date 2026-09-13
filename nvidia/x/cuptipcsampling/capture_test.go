package cuptipcsampling

import (
	"errors"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func testPublicConfig() Config {
	return Config{Period: 10, WorkerSleepMS: 1, ReasonIDs: []uint32{7}, MaxReasons: 8, ScratchBytes: 1024, HardwareBytes: 1024, MaxNativeBytes: 2048, PCsPerBuffer: 1, MaxOutputBytes: 304, MaxReads: 2, MaxNames: 4, MaxNameBytes: 20, MaxTotalNameBytes: 80}
}

func TestPublicConfig(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*Config)
	}{
		{"period", func(c *Config) { c.Period = 32 }},
		{"duplicate", func(c *Config) { c.ReasonIDs = []uint32{7, 7} }},
		{"native overflow", func(c *Config) { c.ScratchBytes = ^uint64(0); c.MaxNativeBytes = ^uint64(0) }},
		{"output", func(c *Config) { c.MaxOutputBytes-- }},
		{"names", func(c *Config) { c.MaxNames = 1 }},
		{"serialization", func(c *Config) { c.CollectionMode = cupti.PcSamplingCollectionModeKernelSerialized }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := testPublicConfig()
			tt.edit(&c)
			// Nil dependencies prove rejection precedes native calls.
			if s, err := startSession(5, c, samplingControlAPI{}, nil, nil, nil); err == nil || s != nil {
				t.Fatalf("session=%v err=%v", s, err)
			}
		})
	}
}

// A start that fails after the collector has been enabled must hand back a
// session, because only the session can disable what was already turned on.
// The failure is injected into the configuration rather than into enable:
// enabling is now the first native call, so a failed enable leaves nothing to
// clean up. TestPublicStartEnableFailureTouchesNothing covers that case.
func TestPublicStartCleanup(t *testing.T) {
	for _, failConfig := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "partial configure"}[failConfig], func(t *testing.T) {
			var saved []cupti.PCSamplingConfigurationInfo
			failure := errors.New("configure failed")
			api := samplingControlAPI{
				version: func() (uint32, error) { return 130001, nil },
				set: func(p *cupti.PCSamplingConfigurationInfoParams) error {
					if failConfig {
						return failure
					}
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
			inventory := ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{{Index: 7, Name: "reason"}}}
			config := testPublicConfig()
			s, err := startSession(5, config, api, func(cupti.Context, int) (ReasonInventory, error) { return inventory, nil }, func(a, b, c int) (*samplingNames, error) {
				return newSamplingNames(a, b, c, func(uintptr, int) (string, error) { return "name", nil }, func(uintptr) {})
			}, func(*cupti.PCSamplingGetDataParams) error { return nil })
			if s == nil || (failConfig && !errors.Is(err, failure)) || (!failConfig && err != nil) {
				t.Fatalf("session=%v err=%v", s, err)
			}
			t.Cleanup(func() { _ = s.Close() })
			if _, ok := s.Final(); ok {
				t.Fatal("premature final")
			}
			config.ReasonIDs[0] = 99
			info := s.Info()
			// A configuration that was never applied was never read back, so
			// Verified reports the failure rather than the intent.
			if info.Config.ReasonIDs[0] != 7 || info.Verified == failConfig {
				t.Fatalf("info=%+v", info)
			}
			info.Config.ReasonIDs[0] = 98
			info.Inventory.Reasons[0].Name = "changed"
			if got := s.Info(); got.Config.ReasonIDs[0] != 7 || got.Inventory.Reasons[0].Name != "reason" {
				t.Fatal("metadata aliases caller")
			}
			if !failConfig {
				if b, err := s.Read(); err != nil || !b.Valid {
					t.Fatalf("batch=%+v err=%v", b, err)
				}
			}
			err = s.Close()
			if (err != nil) != failConfig {
				t.Fatalf("close=%v", err)
			}
			if b, ok := s.Final(); !ok || !b.Valid || b.CUPTIVersion != 130001 {
				t.Fatalf("final=%+v available=%v", b, ok)
			}
			if got := s.Info(); !got.Closed || got.Failed != failConfig {
				t.Fatalf("info=%+v", got)
			}
		})
	}
}

func TestPublicZeroSession(t *testing.T) {
	for _, s := range []*Session{nil, {}} {
		if _, err := s.Read(); err == nil {
			t.Fatal("read")
		}
		if s.Stop() == nil || s.Close() == nil {
			t.Fatal("cleanup")
		}
		if _, ok := s.Final(); ok {
			t.Fatal("final")
		}
		if s.Info().Closed {
			t.Fatal("info")
		}
	}
}

// CUPTI rejects the reason inventory on a context where sampling is not
// enabled, so the order of the two is a correctness property, not a detail.
// Getting it backwards cost a hardware run: cuptiPCSamplingGetNumStallReasons
// returned CUPTI_ERROR_INVALID_OPERATION, which the header does not document.
func TestPublicStartEnablesBeforeQueryingReasons(t *testing.T) {
	var order []string
	api := testPublicControlAPI()
	enable := api.enable
	api.enable = func(p *cupti.PCSamplingEnableParams) error {
		order = append(order, "enable")
		return enable(p)
	}
	inventory := func(ctx cupti.Context, n int) (ReasonInventory, error) {
		order = append(order, "reasons")
		return testPublicInventory(ctx, n)
	}
	s, err := startSession(5, testPublicConfig(), api, inventory, testPublicNames, testPublicEmptyQuery)
	if s == nil || err != nil {
		t.Fatalf("session=%v err=%v", s, err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if !slices.Equal(order, []string{"enable", "reasons"}) {
		t.Fatalf("call order = %v, want enable before reasons and each once", order)
	}
}

// A start that cannot enable sampling has changed nothing, so it returns no
// session to clean up, disables nothing, configures nothing, and leaves the
// collector free for the next caller.
func TestPublicStartEnableFailureTouchesNothing(t *testing.T) {
	failure := errors.New("enable failed")
	var disables, sets int
	api := testPublicControlAPI()
	api.enable = func(*cupti.PCSamplingEnableParams) error { return failure }
	api.disable = func(*cupti.PCSamplingDisableParams) error { disables++; return nil }
	api.set = func(*cupti.PCSamplingConfigurationInfoParams) error { sets++; return nil }
	s, err := startSession(5, testPublicConfig(), api, testPublicInventory, testPublicNames, testPublicEmptyQuery)
	if s != nil || !errors.Is(err, failure) {
		t.Fatalf("session=%v err=%v", s, err)
	}
	if disables != 0 || sets != 0 {
		t.Fatalf("disables=%d sets=%d, want no native state touched", disables, sets)
	}
	if next := testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery); next.Info().Closed {
		t.Fatal("collector still held after a start that enabled nothing")
	}
}

// An exclusivity rejection happens before any native call, so the session that
// already owns the collector cannot have its configuration or its enabled state
// disturbed by the one that lost.
func TestPublicStartExclusiveTouchesNothing(t *testing.T) {
	testPublicStart(t, testPublicControlAPI(), testPublicEmptyQuery)
	var calls int
	api := testPublicControlAPI()
	enable := api.enable
	api.enable = func(p *cupti.PCSamplingEnableParams) error {
		calls++
		return enable(p)
	}
	s, err := startSession(5, testPublicConfig(), api, testPublicInventory, testPublicNames, testPublicEmptyQuery)
	if s != nil || err == nil {
		t.Fatalf("session=%v err=%v", s, err)
	}
	if calls != 0 {
		t.Fatalf("enable calls = %d, want the losing start to touch nothing", calls)
	}
}
