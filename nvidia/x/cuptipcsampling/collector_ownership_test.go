package cuptipcsampling

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
)

func TestMetricCollectorExclusion(t *testing.T) {
	for _, name := range []string{"start", "enable"} {
		t.Run(name, func(t *testing.T) {
			api := testPublicControlAPI()
			api.device = func(cupti.Context) (uint32, error) { return 7, nil }
			api.enable = func(*cupti.PCSamplingEnableParams) error {
				t.Fatal("conflicting collector enabled sampling")
				return nil
			}
			lease, err := cuptiowner.AcquireDevice(7)
			if err != nil {
				t.Fatal(err)
			}
			defer lease.Release()
			if name == "start" {
				s, err := startSession(5, testPublicConfig(), api, testPublicInventory, testPublicNames, testPublicEmptyQuery)
				if err == nil || s != nil {
					t.Fatalf("session=%v err=%v", s, err)
				}
			} else {
				close, err := enableSampling(5, api)
				if err == nil || close != nil {
					t.Fatal("conflicting enable admitted")
				}
			}
		})
	}
}

func TestSamplingConstructionCleanupRetry(t *testing.T) {
	for _, name := range []string{"inventory", "enable output"} {
		t.Run(name, func(t *testing.T) {
			api := testPublicControlAPI()
			api.device = func(cupti.Context) (uint32, error) { return 7, nil }
			failure := errors.New("disable failed")
			fail := true
			api.disable = func(*cupti.PCSamplingDisableParams) error {
				if fail {
					return failure
				}
				return nil
			}
			inventory := testPublicInventory
			if name == "inventory" {
				inventory = func(cupti.Context, int) (ReasonInventory, error) {
					return ReasonInventory{}, errors.New("inventory failed")
				}
			} else {
				api.enable = func(p *cupti.PCSamplingEnableParams) error { p.Ctx++; return nil }
			}
			s, err := startSession(5, testPublicConfig(), api, inventory, testPublicNames, testPublicEmptyQuery)
			if s == nil || !errors.Is(err, failure) {
				t.Fatalf("session=%v err=%v", s, err)
			}
			defer func() { fail = false; _ = s.Close() }()
			if lease, err := cuptiowner.AcquireDevice(7); err == nil {
				lease.Release()
				t.Fatal("failed cleanup released device")
			}
			other, err := cuptiowner.AcquireDevice(8)
			if err != nil {
				t.Fatal(err)
			}
			other.Release()
			if !errors.Is(s.Close(), failure) {
				t.Fatal("cleanup retry lost failure")
			}
			fail = false
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			lease, err := cuptiowner.AcquireDevice(7)
			if err != nil {
				t.Fatal(err)
			}
			lease.Release()
		})
	}
}

func TestEnableCleanupRetry(t *testing.T) {
	api := testPublicControlAPI()
	api.enable = func(p *cupti.PCSamplingEnableParams) error { p.Ctx++; return nil }
	calls := 0
	api.disable = func(p *cupti.PCSamplingDisableParams) error {
		if p.Ctx != 5 {
			t.Error("cleanup used corrupted context")
		}
		calls++
		if calls == 1 {
			return errors.New("disable failed")
		}
		return nil
	}
	close, err := enableSampling(5, api)
	if close == nil || err == nil {
		t.Fatal("malformed enable lost cleanup owner")
	}
	defer close()
	if err := close(); err == nil {
		t.Fatal("missing disable failure")
	}
	if lease, err := cuptiowner.AcquireDevice(0); err == nil {
		lease.Release()
		t.Fatal("released after failure")
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if calls != 2 {
		t.Fatalf("disable calls=%d, want 2", calls)
	}
	lease, err := cuptiowner.AcquireDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}
