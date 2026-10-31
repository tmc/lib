package autotune_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

var workerSpecFlag = flag.String("worker-spec", "", "internal test flag for SubprocessTuneRunner child worker")

func TestSubprocessTuneRunnerHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_AUTOTUNE_WORKER_HELPER") != "1" {
		return
	}
	raw := strings.TrimSpace(*workerSpecFlag)
	if raw == "" {
		fmt.Fprintln(os.Stderr, "missing -worker-spec")
		os.Exit(2)
	}
	var spec autotune.TuneJobSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		fmt.Fprintf(os.Stderr, "unmarshal spec: %v\n", err)
		os.Exit(2)
	}
	norm := spec.Norm()
	key := norm.PGOCacheKey()
	cubin := []byte(fmt.Sprintf("CUBIN-SUBPROC-M%d-PID%d", norm.BucketM, os.Getpid()))
	proof := autotune.PGOPromotionProof{
		Key:            key,
		CandidateName:  fmt.Sprintf("subproc_cross_m%d_mega", norm.BucketM),
		CubinSHA256:    sha256.Sum256(cubin),
		SourceSHA256:   sha256.Sum256([]byte("// source")),
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    1.14,
		HoldoutRatio:   1.12,
		AANullRatio:    1.001,
		OneSidedPValue: 0.015625,
		TokensMatched:  true,
		PromotedAt:     time.Now().UTC(),
	}
	var hit, stored bool
	if norm.CacheDir != "" {
		cache := autotune.NewVerifiedPGOCache(norm.CacheDir)
		if cachedProof, _, err := cache.LoadVerified(key); err == nil {
			hit = true
			proof = cachedProof
		} else {
			if err := cache.Store(key, proof, cubin); err != nil {
				fmt.Fprintf(os.Stderr, "cache store: %v\n", err)
				os.Exit(1)
			}
			stored = true
		}
	}
	res := autotune.TuneJobResult{
		Spec:            norm,
		CacheKey:        key,
		CacheHit:        hit,
		CacheStored:     stored,
		Promoted:        true,
		WinningSchedule: proof.CandidateName,
		KernelName:      fmt.Sprintf("egtune_cross_m%d", norm.BucketM),
		PTXASIterations: 2,
		RegistersPerTh:  96,
		TotalSpillBytes: 0,
		Proof:           proof,
		DurationMs:      1.25,
	}
	enc, err := json.Marshal(res)
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal result: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(enc))
	os.Exit(0)
}

func TestTuneJobSpecValidationAndKey(t *testing.T) {
	spec := autotune.TuneJobSpec{
		BucketM:    4,
		CrossLayer: true,
		PTXASLoop:  true,
	}
	if err := spec.Validate(); err != nil {
		t.Fatalf("Validate valid spec: %v", err)
	}
	norm := spec.Norm()
	if norm.Arch != "sm_121a" || norm.Device != "GB10" || norm.DeviceSMs != 48 {
		t.Fatalf("unexpected defaults in Norm: %+v", norm)
	}
	key := spec.PGOCacheKey()
	if err := key.Validate(); err != nil {
		t.Fatalf("PGOCacheKey.Validate: %v", err)
	}
	if key.ShapeBucket != 4 || key.Arch != "sm_121a" {
		t.Fatalf("unexpected PGOCacheKey: %+v", key)
	}

	bad := autotune.TuneJobSpec{BucketM: 0}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected error for BucketM=0")
	}
}

func TestSubprocessTuneRunnerInProcessAndEpochExclusivity(t *testing.T) {
	allow := true
	epochDaemon := autotune.NewEpochDaemon(autotune.ThrottleFunc(func() bool { return allow }))

	var inFlight atomic.Int32
	var maxInFlight atomic.Int32
	runner := &autotune.SubprocessTuneRunner{
		EpochDaemon: epochDaemon,
		WorkerFunc: func(_ context.Context, spec autotune.TuneJobSpec) (autotune.TuneJobResult, error) {
			cur := inFlight.Add(1)
			for {
				prev := maxInFlight.Load()
				if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			key := spec.PGOCacheKey()
			return autotune.TuneJobResult{
				Spec:            spec,
				CacheKey:        key,
				Promoted:        true,
				WinningSchedule: fmt.Sprintf("cross_m%d_winner", spec.BucketM),
				KernelName:      fmt.Sprintf("cross_m%d", spec.BucketM),
				PTXASIterations: 2,
				RegistersPerTh:  96,
				TotalSpillBytes: 0,
			}, nil
		},
	}

	base := autotune.TuneJobSpec{
		Arch:       "sm_121a",
		Device:     "GB10",
		DeviceSMs:  48,
		CrossLayer: true,
		PTXASLoop:  true,
	}
	buckets := []int{1, 2, 4, 8}
	results, err := runner.RunBuckets(t.Context(), base, buckets)
	if err != nil {
		t.Fatalf("RunBuckets: %v", err)
	}
	if len(results) != len(buckets) {
		t.Fatalf("len(results) = %d, want %d", len(results), len(buckets))
	}
	for i, m := range buckets {
		if results[i].Spec.BucketM != m || !results[i].Promoted {
			t.Fatalf("result[%d] = %+v, want BucketM=%d Promoted=true", i, results[i], m)
		}
	}
	if got := epochDaemon.Epochs(); got != uint64(len(buckets)) {
		t.Fatalf("EpochDaemon.Epochs() = %d, want %d", got, len(buckets))
	}

	// Verify concurrent RunJob calls are serialized by EpochDaemon (maxInFlight == 1).
	var wg sync.WaitGroup
	for _, m := range []int{1, 2, 4, 8} {
		wg.Add(1)
		go func(bucket int) {
			defer wg.Done()
			s := base
			s.BucketM = bucket
			if _, err := runner.RunJob(t.Context(), s); err != nil {
				t.Errorf("concurrent RunJob(M=%d): %v", bucket, err)
			}
		}(m)
	}
	wg.Wait()
	if got := maxInFlight.Load(); got != 1 {
		t.Fatalf("max concurrent workers in flight = %d, want 1", got)
	}

	// Verify ThrottleGate blocks execution when allow == false.
	allow = false
	s := base
	s.BucketM = 1
	if _, err := runner.RunJob(t.Context(), s); !errors.Is(err, autotune.ErrThrottled) {
		t.Fatalf("RunJob when throttled = %v, want ErrThrottled", err)
	}
}

func TestSubprocessTuneRunnerIsolatedChildProcess(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cacheDir := t.TempDir()
	runner := &autotune.SubprocessTuneRunner{
		Command: []string{exe, "-test.run=^TestSubprocessTuneRunnerHelperProcess$"},
		Env:     []string{"GO_WANT_AUTOTUNE_WORKER_HELPER=1"},
	}
	base := autotune.TuneJobSpec{
		Arch:          "sm_121a",
		Device:        "GB10",
		DeviceSMs:     48,
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		CrossLayer:    true,
		PTXASLoop:     true,
		CacheDir:      cacheDir,
	}
	buckets := []int{1, 2, 4, 8}

	// 1. Cold run across buckets 1,2,4,8 in isolated child processes.
	cold, err := runner.RunBuckets(t.Context(), base, buckets)
	if err != nil {
		t.Fatalf("RunBuckets subprocess cold: %v", err)
	}
	if len(cold) != len(buckets) {
		t.Fatalf("len(cold) = %d, want %d", len(cold), len(buckets))
	}
	for i, m := range buckets {
		r := cold[i]
		if r.CacheHit || !r.CacheStored || !r.Promoted || r.Spec.BucketM != m {
			t.Fatalf("cold[%d] (M=%d) = %+v, want CacheHit=false CacheStored=true Promoted=true", i, m, r)
		}
	}

	// 2. Warm run across buckets 1,2,4,8 hits VerifiedPGOCache.
	warm, err := runner.RunBuckets(t.Context(), base, buckets)
	if err != nil {
		t.Fatalf("RunBuckets subprocess warm: %v", err)
	}
	for i, m := range buckets {
		r := warm[i]
		if !r.CacheHit || r.CacheStored || !r.Promoted || r.Spec.BucketM != m {
			t.Fatalf("warm[%d] (M=%d) = %+v, want CacheHit=true CacheStored=false Promoted=true", i, m, r)
		}
		if r.Proof.CubinSHA256 != cold[i].Proof.CubinSHA256 {
			t.Fatalf("warm[%d] digest %x != cold %x", i, r.Proof.CubinSHA256, cold[i].Proof.CubinSHA256)
		}
	}
}
