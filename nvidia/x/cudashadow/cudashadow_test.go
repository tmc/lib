package cudashadow_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudashadow"
)

func validProvenance(seq uint64, ts int64, hook cudashadow.Hook) cudashadow.Provenance {
	return cudashadow.Provenance{
		DeviceID:      0,
		DriverVersion: "575.57.08",
		KernelModule:  "nvidia-uvm.ko",
		Hook:          hook,
		Sequence:      seq,
		TimestampNano: ts,
		TTL:           0,
	}
}

func TestSnapshot_Serialization(t *testing.T) {
	snap := cudashadow.Snapshot{
		ID:         "snap-00000001",
		Provenance: validProvenance(1, 1000000, cudashadow.HookBlockActivate),
		Policy: cudashadow.PolicyInfo{
			Name:    "gpu_ext-eviction-fifo",
			Version: "1.0.0",
			Domain:  cudashadow.DomainEviction,
			Seed:    42,
		},
		Input: cudashadow.Input{
			Provenance: validProvenance(1, 1000000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction: &cudashadow.EvictionData{
				PID:         1234,
				ChunkAddr:   0x7f000000,
				ChunkSize:   2 * 1024 * 1024,
				VABlockAddr: 0x7f000000,
				AccessCount: 1,
			},
		},
		Decision: cudashadow.Decision{
			Type:   cudashadow.DecisionBypass,
			Reason: "fifo activate default lru",
		},
		Status: cudashadow.StatusValid,
		ObservedOutcome: &cudashadow.ObservedOutcome{
			TimestampNano: 1005000,
			Action:        "default_activate",
			LatencyNano:   5000,
			Details:       map[string]string{"result": "success"},
		},
	}

	data, err := json.Marshal(snap)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	decoded, err := cudashadow.DecodeSnapshot(data)
	if err != nil {
		t.Fatalf("DecodeSnapshot failed: %v", err)
	}

	if decoded.ID != snap.ID {
		t.Errorf("decoded ID = %q, want %q", decoded.ID, snap.ID)
	}
	if decoded.Status != snap.Status {
		t.Errorf("decoded Status = %q, want %q", decoded.Status, snap.Status)
	}
	if decoded.Decision.Type != snap.Decision.Type {
		t.Errorf("decoded Decision.Type = %q, want %q", decoded.Decision.Type, snap.Decision.Type)
	}
	if decoded.ObservedOutcome == nil || decoded.ObservedOutcome.Action != "default_activate" {
		t.Errorf("decoded ObservedOutcome = %+v, want action default_activate", decoded.ObservedOutcome)
	}

	// Negative control 1: trailing data rejected
	trailing := append(data, []byte(" extra junk")...)
	if _, err := cudashadow.DecodeSnapshot(trailing); err == nil {
		t.Errorf("DecodeSnapshot with trailing data succeeded, want error")
	}

	// Negative control 2: unknown fields rejected
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal to map failed: %v", err)
	}
	raw["unknown_extra_field"] = "malicious_payload"
	corrupt, err := json.Marshal(raw)
	if err != nil {
		t.Fatalf("marshal corrupt failed: %v", err)
	}
	if _, err := cudashadow.DecodeSnapshot(corrupt); err == nil {
		t.Errorf("DecodeSnapshot with unknown fields succeeded, want error")
	}

	// Negative control 3: missing required field rejected
	delete(raw, "unknown_extra_field")
	delete(raw, "id")
	missingID, err := json.Marshal(raw)
	if err != nil {
		t.Fatalf("marshal missing ID failed: %v", err)
	}
	if _, err := cudashadow.DecodeSnapshot(missingID); err == nil {
		t.Errorf("DecodeSnapshot missing ID succeeded, want error")
	}
}

func TestBackendSpy_AssertsZeroMutations(t *testing.T) {
	spy := cudashadow.NewActiveBackendSpy()
	if err := spy.AssertZeroMutations(); err != nil {
		t.Fatalf("fresh spy has unexpected mutations: %v", err)
	}
	if n := spy.CallCount(); n != 0 {
		t.Fatalf("fresh spy has call count %d, want 0", n)
	}

	// Perform standard evaluation with evaluator
	eval, err := cudashadow.NewEvaluator(10)
	if err != nil {
		t.Fatalf("NewEvaluator failed: %v", err)
	}
	polInfo := cudashadow.PolicyInfo{
		Name:    "fifo-test",
		Version: "1.0",
		Domain:  cudashadow.DomainEviction,
	}
	pol, err := cudashadow.NewFIFOPolicy(polInfo)
	if err != nil {
		t.Fatalf("NewFIFOPolicy failed: %v", err)
	}

	in := cudashadow.Input{
		Provenance: validProvenance(1, 1000, cudashadow.HookBlockActivate),
		Domain:     cudashadow.DomainEviction,
		Eviction: &cudashadow.EvictionData{
			PID:       100,
			ChunkAddr: 0x1000,
		},
	}
	snap, err := eval.Evaluate(context.Background(), pol, in)
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}
	if snap.Status != cudashadow.StatusValid {
		t.Fatalf("snap.Status = %s, want valid", snap.Status)
	}

	// Verify spy still asserts zero mutations
	if err := spy.AssertZeroMutations(); err != nil {
		t.Fatalf("spy detected mutation after shadow evaluation: %v", err)
	}

	// Prove the spy bites: simulate an active mutation attempt
	if err := spy.MutatePTX("kernel_test", []byte("ptx code")); err != nil {
		t.Fatalf("MutatePTX failed: %v", err)
	}
	if err := spy.AssertZeroMutations(); err == nil {
		t.Fatalf("AssertZeroMutations did not catch simulated PTX mutation")
	}
	if len(spy.Calls()) != 1 {
		t.Fatalf("spy.Calls length = %d, want 1", len(spy.Calls()))
	}

	spy.Reset()
	if err := spy.AssertZeroMutations(); err != nil {
		t.Fatalf("spy.AssertZeroMutations after Reset failed: %v", err)
	}
}

func TestBackendSpy_ErrorPathsZeroMutations(t *testing.T) {
	spy := cudashadow.NewActiveBackendSpy()
	eval, err := cudashadow.NewEvaluator(10)
	if err != nil {
		t.Fatalf("NewEvaluator failed: %v", err)
	}
	polInfo := cudashadow.PolicyInfo{
		Name:    "fifo-test",
		Version: "1.0",
		Domain:  cudashadow.DomainEviction,
	}
	pol, err := cudashadow.NewFIFOPolicy(polInfo)
	if err != nil {
		t.Fatalf("NewFIFOPolicy failed: %v", err)
	}

	// 1. Canceled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	in := cudashadow.Input{
		Provenance: validProvenance(1, 1000, cudashadow.HookBlockActivate),
		Domain:     cudashadow.DomainEviction,
		Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000},
	}
	_, err = eval.Evaluate(ctx, pol, in)
	if err == nil {
		t.Errorf("Evaluate with canceled context succeeded, want error")
	}

	// 2. Unsupported domain
	inUnsupported := cudashadow.Input{
		Provenance: validProvenance(2, 2000, cudashadow.HookPagePrefetch),
		Domain:     cudashadow.DomainPrefetch,
		Prefetch:   &cudashadow.PrefetchData{PID: 1, FaultAddr: 0x2000},
	}
	_, err = eval.Evaluate(context.Background(), pol, inUnsupported)
	if err == nil || !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Errorf("Evaluate unsupported domain err = %v, want ErrUnsupported", err)
	}

	inValid := cudashadow.Input{
		Provenance: validProvenance(10, 1000, cudashadow.HookBlockActivate),
		Domain:     cudashadow.DomainEviction,
		Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000},
	}
	sValid, err := eval.Evaluate(context.Background(), pol, inValid)
	if err != nil || sValid.Status != cudashadow.StatusValid {
		t.Fatalf("Evaluate valid failed: %v", err)
	}

	// 3. Stale input: sequence rewind (5 <= 10)
	inStale := cudashadow.Input{
		Provenance: validProvenance(5, 1000, cudashadow.HookBlockActivate),
		Domain:     cudashadow.DomainEviction,
		Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000},
	}
	snapStale, err := eval.Evaluate(context.Background(), pol, inStale)
	if err != nil {
		t.Fatalf("Evaluate stale returned unexpected error: %v", err)
	}
	if snapStale.Status != cudashadow.StatusStale {
		t.Errorf("snapStale.Status = %s, want stale", snapStale.Status)
	}

	// Assert spy still shows zero mutations across all error/stale paths
	if err := spy.AssertZeroMutations(); err != nil {
		t.Errorf("Backend spy saw mutations on error paths: %v", err)
	}
}

func TestPolicyCorpus_Evaluation(t *testing.T) {
	eval, err := cudashadow.NewEvaluator(100)
	if err != nil {
		t.Fatalf("NewEvaluator failed: %v", err)
	}
	ctx := context.Background()

	// 1. FIFO Eviction Policy
	t.Run("FIFO", func(t *testing.T) {
		pol, err := cudashadow.NewFIFOPolicy(cudashadow.PolicyInfo{
			Name: "fifo", Version: "1.0", Domain: cudashadow.DomainEviction,
		})
		if err != nil {
			t.Fatalf("NewFIFOPolicy failed: %v", err)
		}

		// Activate chunk 0x1000, then 0x2000
		in1 := cudashadow.Input{
			Provenance: validProvenance(1, 100, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000},
		}
		s1, err := eval.Evaluate(ctx, pol, in1)
		if err != nil || s1.Decision.Type != cudashadow.DecisionBypass {
			t.Fatalf("s1 decision = %v, want bypass, err = %v", s1.Decision, err)
		}

		in2 := cudashadow.Input{
			Provenance: validProvenance(2, 200, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x2000},
		}
		s2, err := eval.Evaluate(ctx, pol, in2)
		if err != nil || s2.Decision.Type != cudashadow.DecisionBypass {
			t.Fatalf("s2 decision = %v, want bypass, err = %v", s2.Decision, err)
		}

		// Access chunk 0x1000 (should not promote in FIFO)
		inAccess := cudashadow.Input{
			Provenance: validProvenance(3, 300, cudashadow.HookBlockAccess),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000},
		}
		sAccess, err := eval.Evaluate(ctx, pol, inAccess)
		if err != nil || sAccess.Decision.Type != cudashadow.DecisionBypass {
			t.Fatalf("sAccess decision = %v, want bypass, err = %v", sAccess.Decision, err)
		}

		// Evict prepare: oldest (0x1000) must be evicted first
		inEvict := cudashadow.Input{
			Provenance: validProvenance(4, 400, cudashadow.HookEvictPrepare),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{},
		}
		sEvict, err := eval.Evaluate(ctx, pol, inEvict)
		if err != nil || sEvict.Decision.Type != cudashadow.DecisionEvict {
			t.Fatalf("sEvict decision = %v, want evict, err = %v", sEvict.Decision, err)
		}
		if sEvict.Decision.Parameters["target_chunk"] != 0x1000 {
			t.Errorf("evicted chunk = 0x%x, want 0x1000", sEvict.Decision.Parameters["target_chunk"])
		}
	})

	// 2. LFU Eviction Policy
	t.Run("LFU", func(t *testing.T) {
		pol, err := cudashadow.NewLFUPolicy(cudashadow.PolicyInfo{
			Name: "lfu", Version: "1.0", Domain: cudashadow.DomainEviction,
			Parameters: map[string]string{"threshold": "3", "priority_pid": "42"},
		})
		if err != nil {
			t.Fatalf("NewLFUPolicy failed: %v", err)
		}

		// Activate chunks 0xA and 0xB
		eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(10, 1000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{PID: 10, ChunkAddr: 0xA},
		})
		eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(11, 1100, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{PID: 10, ChunkAddr: 0xB},
		})

		// Access chunk 0xA multiple times to exceed threshold 3
		for i := uint64(0); i < 3; i++ {
			eval.Evaluate(ctx, pol, cudashadow.Input{
				Provenance: validProvenance(12+i, 1200+int64(i*10), cudashadow.HookBlockAccess),
				Domain:     cudashadow.DomainEviction,
				Eviction:   &cudashadow.EvictionData{PID: 10, ChunkAddr: 0xA},
			})
		}

		// Evict prepare: chunk 0xB (lowest frequency) should be evicted
		sEvict, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(20, 2000, cudashadow.HookEvictPrepare),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{},
		})
		if err != nil || sEvict.Decision.Type != cudashadow.DecisionEvict {
			t.Fatalf("sEvict decision = %v, want evict, err = %v", sEvict.Decision, err)
		}
		if sEvict.Decision.Parameters["target_chunk"] != 0xB {
			t.Errorf("evicted chunk = 0x%x, want 0xB", sEvict.Decision.Parameters["target_chunk"])
		}
	})

	// 3. MRU Eviction Policy
	t.Run("MRU", func(t *testing.T) {
		pol, err := cudashadow.NewMRUPolicy(cudashadow.PolicyInfo{
			Name: "mru", Version: "1.0", Domain: cudashadow.DomainEviction,
		})
		if err != nil {
			t.Fatalf("NewMRUPolicy failed: %v", err)
		}

		// Activate 0x1, 0x2
		eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(30, 3000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1},
		})
		eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(31, 3100, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x2},
		})
		// Access 0x1 -> 0x1 is now most recently used
		eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(32, 3200, cudashadow.HookBlockAccess),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1},
		})

		// Evict: MRU chunk (0x1) must be evicted
		sEvict, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(33, 3300, cudashadow.HookEvictPrepare),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{},
		})
		if err != nil || sEvict.Decision.Type != cudashadow.DecisionEvict {
			t.Fatalf("sEvict decision = %v, want evict, err = %v", sEvict.Decision, err)
		}
		if sEvict.Decision.Parameters["target_chunk"] != 0x1 {
			t.Errorf("evicted chunk = 0x%x, want 0x1", sEvict.Decision.Parameters["target_chunk"])
		}
	})

	// 4. Adaptive Prefetch Policy
	t.Run("AdaptivePrefetch", func(t *testing.T) {
		pol, err := cudashadow.NewAdaptivePrefetchPolicy(cudashadow.PolicyInfo{
			Name: "prefetch", Version: "1.0", Domain: cudashadow.DomainPrefetch,
		})
		if err != nil {
			t.Fatalf("NewAdaptivePrefetchPolicy failed: %v", err)
		}

		// Fault 1: cold start -> throttle
		s1, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(40, 4000, cudashadow.HookPagePrefetch),
			Domain:     cudashadow.DomainPrefetch,
			Prefetch:   &cudashadow.PrefetchData{PID: 1, FaultAddr: 0x10000, FirstPage: 0, OuterPage: 1},
		})
		if err != nil || s1.Decision.Type != cudashadow.DecisionThrottle {
			t.Fatalf("s1 decision = %v, want throttle, err = %v", s1.Decision, err)
		}

		// Fault 2: stride 4096 bytes -> irregular until second stride
		s2, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(41, 4100, cudashadow.HookPagePrefetch),
			Domain:     cudashadow.DomainPrefetch,
			Prefetch:   &cudashadow.PrefetchData{PID: 1, FaultAddr: 0x11000, FirstPage: 1, OuterPage: 2},
		})
		if err != nil || s2.Decision.Type != cudashadow.DecisionThrottle {
			t.Fatalf("s2 decision = %v, want throttle, err = %v", s2.Decision, err)
		}

		// Fault 3: stride 4096 bytes again -> steady sequential stride -> prefetch
		s3, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(42, 4200, cudashadow.HookPagePrefetch),
			Domain:     cudashadow.DomainPrefetch,
			Prefetch:   &cudashadow.PrefetchData{PID: 1, FaultAddr: 0x12000, FirstPage: 2, OuterPage: 3},
		})
		if err != nil || s3.Decision.Type != cudashadow.DecisionPrefetch {
			t.Fatalf("s3 decision = %v, want prefetch, err = %v", s3.Decision, err)
		}
		if s3.Decision.Parameters["stride_bytes"] != 4096 {
			t.Errorf("stride = %d, want 4096", s3.Decision.Parameters["stride_bytes"])
		}
	})

	// 5. Priority Timeslice Scheduling Policy
	t.Run("PriorityTimeslice", func(t *testing.T) {
		pol, err := cudashadow.NewPriorityTimeslicePolicy(cudashadow.PolicyInfo{
			Name: "sched", Version: "1.0", Domain: cudashadow.DomainScheduling,
		})
		if err != nil {
			t.Fatalf("NewPriorityTimeslicePolicy failed: %v", err)
		}

		sHigh, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(50, 5000, cudashadow.HookSchedTimeslice),
			Domain:     cudashadow.DomainScheduling,
			Scheduling: &cudashadow.SchedulingData{PID: 10, ChannelID: 1, Priority: 8, QueueLength: 1},
		})
		if err != nil || sHigh.Decision.Type != cudashadow.DecisionTimeslice {
			t.Fatalf("sHigh decision = %v, want timeslice, err = %v", sHigh.Decision, err)
		}
		if sHigh.Decision.Parameters["timeslice_us"] != 8000 {
			t.Errorf("sHigh timeslice = %d, want 8000", sHigh.Decision.Parameters["timeslice_us"])
		}

		// High contention (queue length 5) -> quantum halved
		sContended, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(51, 5100, cudashadow.HookSchedTimeslice),
			Domain:     cudashadow.DomainScheduling,
			Scheduling: &cudashadow.SchedulingData{PID: 10, ChannelID: 1, Priority: 8, QueueLength: 5},
		})
		if err != nil || sContended.Decision.Parameters["timeslice_us"] != 4000 {
			t.Errorf("sContended timeslice = %d, want 4000", sContended.Decision.Parameters["timeslice_us"])
		}
	})

	// 6. PTX Shadow Policy
	t.Run("PTXShadow", func(t *testing.T) {
		pol, err := cudashadow.NewPTXShadowPolicy(cudashadow.PolicyInfo{
			Name: "ptx", Version: "1.0", Domain: cudashadow.DomainPTXInstrument,
		})
		if err != nil {
			t.Fatalf("NewPTXShadowPolicy failed: %v", err)
		}

		sPTX, err := eval.Evaluate(ctx, pol, cudashadow.Input{
			Provenance: validProvenance(60, 6000, cudashadow.HookPTXInstrument),
			Domain:     cudashadow.DomainPTXInstrument,
			PTX: &cudashadow.PTXData{
				KernelName:     "_Z8myKernelPfi",
				TargetArch:     "sm_90",
				FunctionOffset: 0x400,
				CodeSize:       2048,
			},
		})
		if err != nil || sPTX.Decision.Type != cudashadow.DecisionSkip {
			t.Fatalf("sPTX decision = %v, want skip, err = %v", sPTX.Decision, err)
		}
		if !strings.Contains(sPTX.Decision.Reason, "shadow mode") {
			t.Errorf("sPTX reason = %q, want shadow mode note", sPTX.Decision.Reason)
		}
	})
}

func TestReplay_Determinism(t *testing.T) {
	eval, err := cudashadow.NewEvaluator(50)
	if err != nil {
		t.Fatalf("NewEvaluator failed: %v", err)
	}
	ctx := context.Background()

	pol, err := cudashadow.NewFIFOPolicy(cudashadow.PolicyInfo{
		Name: "fifo", Version: "1.0", Domain: cudashadow.DomainEviction, Seed: 1234,
	})
	if err != nil {
		t.Fatalf("NewFIFOPolicy failed: %v", err)
	}

	// Record a trace of 10 operations
	for i := uint64(1); i <= 8; i++ {
		in := cudashadow.Input{
			Provenance: validProvenance(i, int64(i*1000), cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x1000 * i},
		}
		if _, err := eval.Evaluate(ctx, pol, in); err != nil {
			t.Fatalf("Evaluate step %d failed: %v", i, err)
		}
	}
	// Add 2 evictions
	for i := uint64(9); i <= 10; i++ {
		in := cudashadow.Input{
			Provenance: validProvenance(i, int64(i*1000), cudashadow.HookEvictPrepare),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{},
		}
		if _, err := eval.Evaluate(ctx, pol, in); err != nil {
			t.Fatalf("Evaluate step %d failed: %v", i, err)
		}
	}

	recorded := eval.Snapshots()
	if len(recorded) != 10 {
		t.Fatalf("recorded snapshots length = %d, want 10", len(recorded))
	}

	// Replay through fresh evaluator and fresh policy instance
	replayPol, err := cudashadow.NewFIFOPolicy(cudashadow.PolicyInfo{
		Name: "fifo", Version: "1.0", Domain: cudashadow.DomainEviction, Seed: 1234,
	})
	if err != nil {
		t.Fatalf("NewFIFOPolicy replay failed: %v", err)
	}

	replayed, err := eval.Replay(ctx, replayPol, recorded)
	if err != nil {
		t.Fatalf("Replay failed: %v", err)
	}
	if len(replayed) != len(recorded) {
		t.Fatalf("replayed length = %d, want %d", len(replayed), len(recorded))
	}

	for i := range recorded {
		if replayed[i].Decision.Type != recorded[i].Decision.Type {
			t.Errorf("step %d: decision mismatch: replayed %s, recorded %s",
				i, replayed[i].Decision.Type, recorded[i].Decision.Type)
		}
		if replayed[i].Decision.Parameters["target_chunk"] != recorded[i].Decision.Parameters["target_chunk"] {
			t.Errorf("step %d: target_chunk mismatch", i)
		}
	}

	// Negative control: alter recorded decision and verify Replay detects mismatch
	corrupted := make([]cudashadow.Snapshot, len(recorded))
	copy(corrupted, recorded)
	corrupted[9].Decision.Type = cudashadow.DecisionProtect // Mismatch!
	if _, err := eval.Replay(ctx, replayPol, corrupted); err == nil {
		t.Errorf("Replay with corrupted decision succeeded, want error")
	}
}

func TestNegativeControls(t *testing.T) {
	eval, err := cudashadow.NewEvaluator(3) // Bounded small capacity = 3
	if err != nil {
		t.Fatalf("NewEvaluator failed: %v", err)
	}
	ctx := context.Background()

	pol, err := cudashadow.NewFIFOPolicy(cudashadow.PolicyInfo{
		Name: "fifo", Version: "1.0", Domain: cudashadow.DomainEviction,
	})
	if err != nil {
		t.Fatalf("NewFIFOPolicy failed: %v", err)
	}

	// 1. Missing mandatory provenance fields
	t.Run("MissingProvenance", func(t *testing.T) {
		badIn := cudashadow.Input{
			Provenance: cudashadow.Provenance{}, // Empty
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{},
		}
		if _, err := eval.Evaluate(ctx, pol, badIn); err == nil {
			t.Errorf("Evaluate with empty provenance succeeded, want error")
		}
	})

	// 2. TTL expiration
	t.Run("TTLExpiration", func(t *testing.T) {
		eval.SetClock(func() int64 { return 1000000000 }) // Current time = 1s
		staleIn := cudashadow.Input{
			Provenance: cudashadow.Provenance{
				DeviceID:      0,
				DriverVersion: "575.57.08",
				KernelModule:  "nvidia-uvm.ko",
				Hook:          cudashadow.HookBlockActivate,
				Sequence:      100,
				TimestampNano: 500000000,              // 0.5s ago
				TTL:           100 * time.Millisecond, // TTL is 100ms < 500ms
			},
			Domain:   cudashadow.DomainEviction,
			Eviction: &cudashadow.EvictionData{ChunkAddr: 0x500},
		}
		snap, err := eval.Evaluate(ctx, pol, staleIn)
		if err != nil {
			t.Fatalf("Evaluate stale returned err: %v", err)
		}
		if snap.Status != cudashadow.StatusStale {
			t.Errorf("snap.Status = %s, want %s", snap.Status, cudashadow.StatusStale)
		}
	})

	// 3. Out-of-order sequence
	t.Run("OutOfOrderSequence", func(t *testing.T) {
		eval.SetClock(func() int64 { return 2000000000 })
		inSeq200 := cudashadow.Input{
			Provenance: validProvenance(200, 2000000000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x600},
		}
		sValid, err := eval.Evaluate(ctx, pol, inSeq200)
		if err != nil || sValid.Status != cudashadow.StatusValid {
			t.Fatalf("inSeq200 failed: status=%s, err=%v", sValid.Status, err)
		}

		// Sequence 150 < 200 arrives out of order
		inSeq150 := cudashadow.Input{
			Provenance: validProvenance(150, 2000000100, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x700},
		}
		sStale, err := eval.Evaluate(ctx, pol, inSeq150)
		if err != nil {
			t.Fatalf("inSeq150 returned error: %v", err)
		}
		if sStale.Status != cudashadow.StatusStale {
			t.Errorf("sStale.Status = %s, want stale", sStale.Status)
		}
	})

	// 4. Bounded capacity ring buffer
	t.Run("BoundedCapacity", func(t *testing.T) {
		// Evaluator capacity is 3. We insert 5 items:
		for i := uint64(301); i <= 305; i++ {
			in := cudashadow.Input{
				Provenance: validProvenance(i, int64(i*10000), cudashadow.HookBlockActivate),
				Domain:     cudashadow.DomainEviction,
				Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x10 * i},
			}
			if _, err := eval.Evaluate(ctx, pol, in); err != nil {
				t.Fatalf("Evaluate step %d failed: %v", i, err)
			}
		}

		snaps := eval.Snapshots()
		if len(snaps) != 3 {
			t.Errorf("snapshots length = %d, want capacity limit 3", len(snaps))
		}
		if eval.TotalEvaluated() < 5 {
			t.Errorf("TotalEvaluated = %d, want >= 5", eval.TotalEvaluated())
		}
		// The oldest preserved snapshot should be sequence 303, not 301
		if snaps[0].Provenance.Sequence != 303 {
			t.Errorf("snaps[0] sequence = %d, want 303", snaps[0].Provenance.Sequence)
		}
		if snaps[2].Provenance.Sequence != 305 {
			t.Errorf("snaps[2] sequence = %d, want 305", snaps[2].Provenance.Sequence)
		}
	})

	// 5. Cancellation
	t.Run("ContextCancellation", func(t *testing.T) {
		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()
		in := cudashadow.Input{
			Provenance: validProvenance(400, 4000000000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x800},
		}
		if _, err := eval.Evaluate(canceledCtx, pol, in); err == nil {
			t.Errorf("Evaluate with canceled context succeeded, want error")
		}
	})

	// 6. Registered corpus catalog
	t.Run("CorpusCatalog", func(t *testing.T) {
		catalog := cudashadow.RegisteredCorpusPolicies()
		if len(catalog) < 6 {
			t.Errorf("catalog length = %d, want >= 6", len(catalog))
		}
		for _, pi := range catalog {
			if err := pi.Validate(); err != nil {
				t.Errorf("catalog policy %q failed validation: %v", pi.Name, err)
			}
		}
	})

	// 7. Record baseline outcome
	t.Run("RecordOutcome", func(t *testing.T) {
		in := cudashadow.Input{
			Provenance: validProvenance(500, 5000000000, cudashadow.HookBlockActivate),
			Domain:     cudashadow.DomainEviction,
			Eviction:   &cudashadow.EvictionData{ChunkAddr: 0x900},
		}
		snap, err := eval.Evaluate(ctx, pol, in)
		if err != nil {
			t.Fatalf("Evaluate failed: %v", err)
		}

		outcome := cudashadow.ObservedOutcome{
			TimestampNano: 5000005000,
			Action:        "evict_success",
			LatencyNano:   5000,
		}
		if err := eval.RecordOutcome(snap.ID, outcome); err != nil {
			t.Fatalf("RecordOutcome failed: %v", err)
		}

		// Find snapshot and verify outcome
		found := false
		for _, s := range eval.Snapshots() {
			if s.ID == snap.ID {
				found = true
				if s.ObservedOutcome == nil || s.ObservedOutcome.Action != "evict_success" {
					t.Errorf("recorded outcome mismatch: %+v", s.ObservedOutcome)
				}
			}
		}
		if !found {
			t.Errorf("snapshot %s not found in snapshots", snap.ID)
		}

		// Non-existent snapshot ID returns error
		if err := eval.RecordOutcome("non_existent_id", outcome); err == nil {
			t.Errorf("RecordOutcome non_existent_id succeeded, want error")
		}
	})
}
