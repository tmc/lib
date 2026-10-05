package cudahotswap

import (
	"context"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

type settingsKernel struct {
	serviceKernel
	config cudalaunch.Config
}

func (k *settingsKernel) Launch(config cudalaunch.Config, _ ...any) error {
	k.config = config
	return nil
}

func TestServiceSettingsRevisionAndLaunch(t *testing.T) {
	k := &settingsKernel{serviceKernel: serviceKernel{raw: 1}}
	s, leases := testService(t, &serviceLoader{kernel: k})
	registerService(t, s)
	original, err := s.Status("kernel")
	if err != nil {
		t.Fatal(err)
	}
	flags := []string{"--fmad=false"}
	state, err := s.SetCompileFlags("kernel", flags, "compile")
	if err != nil || state.ConfigRevision != 1 || state.Generation != original.Generation || state.Version != original.Version || state.Label != "compile" {
		t.Fatalf("compile setting: %+v, %v", state, err)
	}
	flags[0] = "mutated"
	if s.settings["kernel"].flags[0] != "--fmad=false" {
		t.Fatal("compile flags alias caller")
	}
	g := LaunchGeometry{Grid: cudashape.Grid3D{X: 2, Y: 1, Z: 1}, Block: driver.Dims(32, 1, 1), SharedMem: 128}
	state, err = s.SetGeometry("kernel", &g, "geometry")
	if err != nil || state.ConfigRevision != 2 || state.Generation != original.Generation || state.Label != "geometry" {
		t.Fatalf("geometry setting: %+v, %v", state, err)
	}
	g.Grid.X = 99
	if len(*leases) != 1 || (*leases)[0].calls != 1 {
		t.Fatal("settings acquire or retain a GPU lease")
	}
	caller := cudalaunch.Config{Grid: cudashape.Grid3D{X: 3, Y: 1, Z: 1}, Block: driver.Dims(16, 1, 1)}
	p, err := s.Launch(context.Background(), "kernel", caller)
	if err != nil || k.config.Grid.X != 2 || k.config.Block.X != 32 || k.config.SharedMem != 128 || k.config.Stream != caller.Stream {
		t.Fatalf("actual launch override: %+v, %v", k.config, err)
	}
	if err := p.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetGeometry("kernel", nil, "clear"); err != nil {
		t.Fatal(err)
	}
	p, err = s.Launch(context.Background(), "kernel", caller)
	if err != nil || k.config != caller {
		t.Fatalf("clear override: %+v, %v", k.config, err)
	}
	if err := p.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	state, _ = s.Status("kernel")
	state.Name = "detached"
	if s.List()[0].Name != "kernel" {
		t.Fatal("status aliases internal state")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestServiceSettingsRefusalIsAtomic(t *testing.T) {
	s, leases := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	before, _ := s.Status("kernel")
	for _, flags := range [][]string{{""}, {"bad\x00flag"}, {strings.Repeat("x", (16<<10)+1)}, make([]string, 65)} {
		if _, err := s.SetCompileFlags("kernel", flags, "bad"); err == nil {
			t.Fatal("invalid flags admitted")
		}
	}
	for _, geometry := range []LaunchGeometry{{}, {Grid: cudashape.Grid3D{X: 1, Y: 1, Z: 1}, Block: driver.Dims(1025, 1, 1)}} {
		if _, err := s.SetGeometry("kernel", &geometry, "bad"); err == nil {
			t.Fatal("invalid geometry admitted")
		}
	}
	if _, err := s.SetCompileFlags("missing", nil, "bad"); err == nil {
		t.Fatal("unknown kernel admitted")
	}
	if after, _ := s.Status("kernel"); after != before || len(*leases) != 1 {
		t.Fatalf("refusal changed state or lease: %+v", after)
	}
	s.mu.Lock()
	state := s.states["kernel"]
	state.ConfigRevision = math.MaxUint64
	s.states["kernel"] = state
	s.mu.Unlock()
	if _, err := s.SetCompileFlags("kernel", nil, "overflow"); err == nil {
		t.Fatal("revision overflow admitted")
	}
	if s.settings["kernel"].flags != nil {
		t.Fatal("overflow changed flags")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCompileFlags("kernel", nil, "closed"); err == nil {
		t.Fatal("closed service changed settings")
	}
}

func TestServiceSettingsConcurrent(t *testing.T) {
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 16; j++ {
				if _, err := s.SetCompileFlags("kernel", []string{"--fmad=false"}, "concurrent"); err != nil {
					t.Error(err)
				}
				s.Status("kernel")
				s.List()
			}
		}()
	}
	wg.Wait()
	state, _ := s.Status("kernel")
	if state.ConfigRevision != 128 || state.Generation != 1 {
		t.Fatalf("lost revisions: %+v", state)
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
