package cuptipcsampling

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

type reannouncementCapture func(context.Context, driver.Context, PhaseConfig, func() error, func(context.Context) error) (PhaseReport, error)

func TestPhaseReannouncement(t *testing.T) {
	checkPhaseReannouncement(t, func(ctx context.Context, cu driver.Context, cfg PhaseConfig, sync func() error, work func(context.Context) error) (PhaseReport, error) {
		return capturePhase(ctx, cu, cfg, flushPhaseStarter(), sync, work)
	})
}

// The separate public-wrapper overlay control substitutes only Start in the
// actual CapturePhase wrapper, then runs this same lifecycle and SealPhase.
// This test's native boundary is explicitly controlled, not a GPU result.
func checkPhaseReannouncement(t *testing.T, capture reannouncementCapture) {
	t.Helper()
	modules := cudacapture.NewModuleTracker()
	imageA := fmt.Sprintf("%x", sha256.Sum256([]byte("__global__ void first() {}")))
	imageB := fmt.Sprintf("%x", sha256.Sum256([]byte("__global__ void replacement() {}")))
	if _, err := modules.Load("before attach", 100, imageA, cudacapture.CodeFiles{Sources: []string{"kernel.cu"}}); err != nil {
		t.Fatal(err)
	}
	cfg := testPhaseConfig()
	cfg.Limits.MaxBatches = 1
	cfg.Modules = modules
	cfg.MetadataLimits = cudacapture.ReannouncementLimits{MaxEntries: 8, MaxBytes: 4096}
	calls, works, syncs, delivered := 0, 0, 0, 0
	cfg.MetadataConsumer = func(r cudacapture.Reannouncement) error {
		calls++
		if calls == 1 && (len(r.Entries) != 1 || r.Entries[0].Generation != 1 || r.Entries[0].ImageDigest != imageA) {
			t.Fatalf("attach: %+v", r)
		}
		if len(r.Entries[0].Code.Sources) != 0 {
			r.Entries[0].Code.Sources[0] = "sink mutation"
		}
		return nil
	}
	cfg.Consumer = func(b Batch) error {
		delivered++
		// Consumer mutation must not corrupt retained raw/accounting evidence.
		if len(b.PCs) > 0 {
			b.PCs[0].Name = ""
			b.PCs[0].Reasons[0].Samples = 0
		}
		return nil
	}
	r, err := capture(context.Background(), 5, cfg, func() error { syncs++; return nil }, func(context.Context) error {
		works++
		if g, e := modules.Load("duplicate", 100, imageA, cudacapture.CodeFiles{}); e != nil || g != 1 {
			t.Fatalf("duplicate %d %v", g, e)
		}
		if e := modules.Unload(100); e != nil {
			t.Fatal(e)
		}
		if g, e := modules.Load("reload", 100, imageA, cudacapture.CodeFiles{}); e != nil || g != 2 {
			t.Fatalf("reload %d %v", g, e)
		}
		if _, e := modules.Load("JIT replacement", 100, imageB, cudacapture.CodeFiles{}); e != nil {
			t.Fatal(e)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 || works != 1 || syncs != 1 || delivered != 2 {
		t.Fatalf("calls metadata/work/sync/batches %d/%d/%d/%d", calls, works, syncs, delivered)
	}
	if r.MetadataStart == nil || r.MetadataEnd == nil {
		t.Fatal("missing actual snapshots")
	}
	if len(r.MetadataStart.Entries) != 1 || r.MetadataStart.Entries[0].Code.Sources[0] != "kernel.cu" {
		t.Fatal("attach snapshot changed")
	}
	end := r.MetadataEnd.Entries
	if len(end) != 3 || end[0].Active || end[1].Active || !end[2].Active || end[2].Generation != 3 || end[2].ImageDigest != imageB || !end[2].Code.Empty() {
		t.Fatalf("end: %+v", end)
	}
	if !r.Info.Closed || len(r.Batches) != 2 || r.TotalSamples != 30 {
		t.Fatalf("native lifecycle changed: %+v", r)
	}
	for _, b := range r.Batches {
		if len(b.PCs) != 1 || b.PCs[0].Name != "kernel" || len(b.PCs[0].Reasons) != 1 || b.PCs[0].Reasons[0].Samples != 2 {
			t.Fatalf("consumer corrupted retained PC evidence: %+v", b)
		}
	}
	var raw []byte
	for _, b := range r.Batches {
		raw, err = b.AppendBinary(raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	m, err := SealPhase(PhaseRequest{Period: cfg.Config.Period, MaxBatches: cfg.Limits.MaxBatches}, r, PhaseObservation{Raw: raw})
	if err != nil {
		t.Fatal(err)
	}
	if m.MetadataStart == nil || m.MetadataEnd == nil || len(m.MetadataEnd.Entries) != 3 || m.TotalSamples != 30 || m.Named != 4 || m.MetadataFailed {
		t.Fatalf("manifest: %+v", m)
	}
	r.MetadataStart.Entries[0].Code.Sources[0] = "report mutation"
	if m.MetadataStart.Entries[0].Code.Sources[0] != "kernel.cu" {
		t.Fatal("manifest aliases report")
	}
}

func TestPhaseReannouncementFailure(t *testing.T) {
	for _, terminal := range []bool{false, true} {
		t.Run(map[bool]string{false: "attach", true: "terminal"}[terminal], func(t *testing.T) {
			cfg := testPhaseConfig()
			cfg.Modules = cudacapture.NewModuleTracker()
			cfg.MetadataLimits = cudacapture.ReannouncementLimits{MaxEntries: 1, MaxBytes: 1}
			sentinel := errors.New("metadata sink failed")
			workErr := errors.New("workload failed")
			calls, starts, works := 0, 0, 0
			cfg.MetadataConsumer = func(cudacapture.Reannouncement) error {
				calls++
				if !terminal || calls == 2 {
					return sentinel
				}
				return nil
			}
			start := func(cu driver.Context, c Config) (*Session, error) {
				starts++
				return startSession(cu, c, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicEmptyQuery)
			}
			r, err := capturePhase(context.Background(), 5, cfg, start, func() error { return nil }, func(context.Context) error { works++; return workErr })
			if !errors.Is(err, sentinel) || !errors.Is(r.MetadataErr, sentinel) {
				t.Fatalf("lost replay error: %v %+v", err, r)
			}
			if terminal {
				if starts != 1 || works != 1 || !r.Info.Closed || r.MetadataEnd == nil || !errors.Is(err, workErr) {
					t.Fatalf("terminal: %+v %v", r, err)
				}
			} else if starts != 0 || works != 0 || r.MetadataStart == nil {
				t.Fatalf("attach started work: %d/%d", starts, works)
			}
		})
	}
}

func TestPhaseReannouncementPublicValidation(t *testing.T) {
	cfg := testPhaseConfig()
	cfg.Modules = cudacapture.NewModuleTracker()
	_, err := CapturePhase(context.Background(), 5, cfg, func() error { return nil }, func(context.Context) error { t.Fatal("workload ran"); return nil })
	if err == nil {
		t.Fatal("invalid metadata limits reached native Start")
	}
}

func TestPhaseReannouncementCancellationAndConsumer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cfg := testPhaseConfig()
	cfg.Modules = cudacapture.NewModuleTracker()
	cfg.MetadataLimits = cudacapture.ReannouncementLimits{MaxEntries: 1, MaxBytes: 1}
	sinkErr := errors.New("sample consumer failure")
	cfg.Consumer = func(Batch) error { return sinkErr }
	r, _ := capturePhase(ctx, 5, cfg, flushPhaseStarter(), func() error { return nil }, func(context.Context) error { cancel(); return ctx.Err() })
	if !r.Canceled || !errors.Is(r.ConsumerErr, sinkErr) || !r.Info.Closed || r.MetadataEnd == nil || len(r.Batches) == 0 {
		t.Fatalf("lost terminal evidence: %+v", r)
	}
	var raw []byte
	for _, b := range r.Batches {
		var err error
		raw, err = b.AppendBinary(raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	m, err := SealPhase(PhaseRequest{Period: cfg.Config.Period}, r, PhaseObservation{Raw: raw})
	if err != nil || m.Cleanup != CleanupReleased || m.MetadataEnd == nil {
		t.Fatalf("seal: %+v %v", m, err)
	}
}

func TestPhaseReannouncementPendingCleanup(t *testing.T) {
	cfg := testPhaseConfig()
	cfg.Modules = cudacapture.NewModuleTracker()
	cfg.MetadataLimits = cudacapture.ReannouncementLimits{MaxEntries: 1, MaxBytes: 1}
	closeErr := errors.New("native disable failed")
	fail := true
	var owned *Session
	starter := func(cu driver.Context, c Config) (*Session, error) {
		api := testPublicControlAPI()
		disable := api.disable
		api.disable = func(p *cupti.PCSamplingDisableParams) error {
			if fail {
				return closeErr
			}
			return disable(p)
		}
		var err error
		owned, err = startSession(cu, c, api, testPublicInventory, testPublicNames, testPublicEmptyQuery)
		return owned, err
	}
	r, err := capturePhase(context.Background(), 5, cfg, starter, func() error { return nil }, func(context.Context) error { return nil })
	if owned != nil {
		defer func() {
			fail = false
			_ = owned.Close()
			if !owned.Info().Closed {
				t.Error("test cleanup retained ownership")
			}
		}()
	}
	if !errors.Is(err, closeErr) || r.Info.Closed || r.MetadataEnd == nil {
		t.Fatalf("cleanup ownership erased: %+v %v", r, err)
	}
	var raw []byte
	for _, b := range r.Batches {
		raw, err = b.AppendBinary(raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	m, err := SealPhase(PhaseRequest{Period: cfg.Config.Period}, r, PhaseObservation{Raw: raw})
	if err != nil || m.Cleanup == CleanupReleased || m.MetadataEnd == nil {
		t.Fatalf("manifest: %+v %v", m, err)
	}
}
