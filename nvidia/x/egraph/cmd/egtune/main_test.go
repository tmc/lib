package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

var testWorkerSpecFlag = flag.String("worker-spec", "", "internal test flag for egtune subprocess helper")

func TestEGTuneSubprocessHelper(t *testing.T) {
	if os.Getenv("GO_WANT_EGTUNE_HELPER_PROCESS") != "1" {
		return
	}
	raw := strings.TrimSpace(*testWorkerSpecFlag)
	if raw == "" {
		os.Exit(2)
	}
	var out, errBuf bytes.Buffer
	code := run([]string{"-worker-spec", raw}, &out, &errBuf)
	if code != 0 {
		_, _ = os.Stderr.Write(errBuf.Bytes())
		os.Exit(code)
	}
	_, _ = os.Stdout.Write(out.Bytes())
	os.Exit(0)
}

func TestEGTuneBucketsColdAndWarmPGOCache(t *testing.T) {
	t.Setenv("EGTUNE_FORCE_STATIC_PTXAS", "1")
	cacheDir := filepath.Join(t.TempDir(), "pgo-cache")

	// 1. Cold run across -buckets 1,2,4,8 with -json and -pgo-cache-dir.
	var coldOut, coldErr bytes.Buffer
	code := run([]string{
		"-arch", "sm_121a",
		"-device", "GB10",
		"-sms", "48",
		"-buckets", "1,2,4,8",
		"-cross-layer",
		"-ptxas-loop",
		"-pgo-cache-dir", cacheDir,
		"-json",
	}, &coldOut, &coldErr)
	if code != 0 {
		t.Fatalf("cold run exit %d: %s", code, coldErr.String())
	}

	var coldResults []autotune.TuneJobResult
	if err := json.Unmarshal(coldOut.Bytes(), &coldResults); err != nil {
		t.Fatalf("unmarshal cold JSON: %v\n%s", err, coldOut.String())
	}
	wantBuckets := []int{1, 2, 4, 8}
	if len(coldResults) != len(wantBuckets) {
		t.Fatalf("len(coldResults) = %d, want %d", len(coldResults), len(wantBuckets))
	}
	for i, m := range wantBuckets {
		r := coldResults[i]
		if r.Spec.BucketM != m {
			t.Fatalf("coldResults[%d].Spec.BucketM = %d, want %d", i, r.Spec.BucketM, m)
		}
		if r.CacheHit {
			t.Fatalf("coldResults[%d] (M=%d) unexpected CacheHit=true", i, m)
		}
		if !r.CacheStored {
			t.Fatalf("coldResults[%d] (M=%d) expected CacheStored=true", i, m)
		}
		if !r.Promoted || r.WinningSchedule == "" {
			t.Fatalf("coldResults[%d] (M=%d) expected Promoted=true and WinningSchedule, got %+v", i, m, r)
		}
		if r.PTXASIterations < 2 || r.TotalSpillBytes != 0 {
			t.Fatalf("coldResults[%d] (M=%d) expected PTXASIterations >= 2 and 0 spills, got iters=%d spills=%d",
				i, m, r.PTXASIterations, r.TotalSpillBytes)
		}
		if !r.Proof.TokensMatched || r.Proof.HoldoutBlocks < 6 || r.Proof.HoldoutRatio <= 1.0 {
			t.Fatalf("coldResults[%d] (M=%d) invalid promotion proof: %+v", i, m, r.Proof)
		}
	}

	// Verify VerifiedPGOCache contains all 4 buckets on disk.
	cache := autotune.NewVerifiedPGOCache(cacheDir)
	keys, err := cache.List()
	if err != nil {
		t.Fatalf("cache.List: %v", err)
	}
	if len(keys) != len(wantBuckets) {
		t.Fatalf("len(cache.List()) = %d, want %d", len(keys), len(wantBuckets))
	}

	// 2. Warm run on the same -pgo-cache-dir: every bucket hits VerifiedPGOCache.
	var warmOut, warmErr bytes.Buffer
	code = run([]string{
		"-arch", "sm_121a",
		"-device", "GB10",
		"-sms", "48",
		"-buckets", "1,2,4,8",
		"-cross-layer",
		"-ptxas-loop",
		"-pgo-cache-dir", cacheDir,
		"-json",
	}, &warmOut, &warmErr)
	if code != 0 {
		t.Fatalf("warm run exit %d: %s", code, warmErr.String())
	}

	var warmResults []autotune.TuneJobResult
	if err := json.Unmarshal(warmOut.Bytes(), &warmResults); err != nil {
		t.Fatalf("unmarshal warm JSON: %v\n%s", err, warmOut.String())
	}
	for i, m := range wantBuckets {
		r := warmResults[i]
		if !r.CacheHit || r.CacheStored || !r.Promoted {
			t.Fatalf("warmResults[%d] (M=%d) = %+v, want CacheHit=true CacheStored=false Promoted=true", i, m, r)
		}
		if r.Proof.CubinSHA256 != coldResults[i].Proof.CubinSHA256 {
			t.Fatalf("warmResults[%d] CubinSHA256 %x != cold %x", i, r.Proof.CubinSHA256, coldResults[i].Proof.CubinSHA256)
		}
	}
}

func TestEGTuneSubprocessAndTextOutput(t *testing.T) {
	t.Setenv("EGTUNE_FORCE_STATIC_PTXAS", "1")
	cacheDir := filepath.Join(t.TempDir(), "subproc-cache")

	// Cold run with -subprocess and human-readable text output.
	var textOut, errBuf bytes.Buffer
	code := run([]string{
		"-buckets", "1,4",
		"-pgo-cache-dir", cacheDir,
		"-subprocess",
	}, &textOut, &errBuf)
	if code != 0 {
		t.Fatalf("subprocess text run exit %d: %s", code, errBuf.String())
	}
	outStr := textOut.String()
	for _, want := range []string{
		"bucket M=1 status=promoted",
		"bucket M=4 status=promoted",
		"cache_stored=true",
	} {
		if !strings.Contains(outStr, want) {
			t.Errorf("text output missing %q:\n%s", want, outStr)
		}
	}

	// Warm run with -subprocess and -json.
	var jsonOut bytes.Buffer
	errBuf.Reset()
	code = run([]string{
		"-buckets", "1,4",
		"-pgo-cache-dir", cacheDir,
		"-subprocess",
		"-json",
	}, &jsonOut, &errBuf)
	if code != 0 {
		t.Fatalf("subprocess warm json run exit %d: %s", code, errBuf.String())
	}
	var warm []autotune.TuneJobResult
	if err := json.Unmarshal(jsonOut.Bytes(), &warm); err != nil {
		t.Fatalf("unmarshal subprocess warm json: %v", err)
	}
	if len(warm) != 2 || !warm[0].CacheHit || !warm[1].CacheHit {
		t.Fatalf("unexpected subprocess warm results: %+v", warm)
	}
}

func TestEGTuneErrors(t *testing.T) {
	for _, args := range [][]string{
		{"-buckets", "bad"},
		{"-buckets", "0"},
		{"-beam", "0"},
		{"-hidden", "0"},
		{"-worker-spec", "{bad-json"},
		{"positional-arg"},
	} {
		var out, errBuf bytes.Buffer
		if code := run(args, &out, &errBuf); code == 0 {
			t.Errorf("args %v succeeded, want non-zero exit code", args)
		}
	}
}
