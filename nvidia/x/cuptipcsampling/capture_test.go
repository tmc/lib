package cuptipcsampling

import (
	"errors"
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

func TestPublicStartCleanup(t *testing.T) {
	for _, failEnable := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "partial enable"}[failEnable], func(t *testing.T) {
			var saved []cupti.PCSamplingConfigurationInfo
			failure := errors.New("enable failed")
			api := samplingControlAPI{
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
				enable: func(*cupti.PCSamplingEnableParams) error {
					if failEnable {
						return failure
					}
					return nil
				},
				start:   func(*cupti.PCSamplingStartParams) error { return nil },
				stop:    func(*cupti.PCSamplingStopParams) error { return nil },
				disable: func(*cupti.PCSamplingDisableParams) error { return nil },
			}
			inventory := ReasonInventory{CUPTIVersion: 130001, Reasons: []StallReason{{Index: 7, Name: "reason"}}}
			config := testPublicConfig()
			s, err := startSession(5, config, api, func(cupti.Context, int) (ReasonInventory, error) { return inventory, nil }, func(a, b, c int) (*samplingNames, error) {
				return newSamplingNames(a, b, c, func(uintptr, int) (string, error) { return "name", nil }, func(uintptr) {})
			}, func(*cupti.PCSamplingGetDataParams) error { return nil })
			if s == nil || (failEnable && !errors.Is(err, failure)) || (!failEnable && err != nil) {
				t.Fatalf("session=%v err=%v", s, err)
			}
			t.Cleanup(func() { _ = s.Close() })
			if _, ok := s.Final(); ok {
				t.Fatal("premature final")
			}
			config.ReasonIDs[0] = 99
			info := s.Info()
			if info.Config.ReasonIDs[0] != 7 || !info.Verified {
				t.Fatalf("info=%+v", info)
			}
			info.Config.ReasonIDs[0] = 98
			info.Inventory.Reasons[0].Name = "changed"
			if got := s.Info(); got.Config.ReasonIDs[0] != 7 || got.Inventory.Reasons[0].Name != "reason" {
				t.Fatal("metadata aliases caller")
			}
			if !failEnable {
				if b, err := s.Read(); err != nil || !b.Valid {
					t.Fatalf("batch=%+v err=%v", b, err)
				}
			}
			err = s.Close()
			if (err != nil) != failEnable {
				t.Fatalf("close=%v", err)
			}
			if b, ok := s.Final(); !ok || !b.Valid || b.CUPTIVersion != 130001 {
				t.Fatalf("final=%+v available=%v", b, ok)
			}
			if got := s.Info(); !got.Closed || got.Failed != failEnable {
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
