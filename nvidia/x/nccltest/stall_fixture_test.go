package nccltest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// withTwoCUDALease admits the native helper's first driver call only after
// checking the canonical inherited owner. Host controls supply a callback.
func withTwoCUDALease(getenv func(string) string, run func()) error {
	if _, err := cudalock.Inherited(getenv); err != nil {
		return err
	}
	run()
	return nil
}

func TestTwoCUDAInheritedAdmission(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvml.Loaded() || nccl.Loaded() {
			t.Errorf("native libraries loaded: driver=%v nvml=%v nccl=%v", driver.Loaded(), nvml.Loaded(), nccl.Loaded())
		}
	}
	checkUnloaded()
	t.Cleanup(checkUnloaded)
	dir := filepath.Join(t.TempDir(), "lock")
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	for _, tc := range []struct {
		name string
		held string
		want bool
	}{
		{"unset", "", false},
		{"presence_only", "1", false},
		{"malformed_pid", "bogus", false},
		{"mismatched_owner", strconv.Itoa(os.Getpid() + 1), false},
		{"canonical_live_owner", strconv.Itoa(os.Getpid()), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			err := withTwoCUDALease(func(key string) string {
				switch key {
				case "GPU_LOCK_HELD":
					return tc.held
				case "GPU_LOCKDIR":
					return dir
				}
				return ""
			}, func() { calls++ })
			if tc.want {
				if err != nil || calls != 1 {
					t.Fatalf("admission = %v, callback calls = %d; want one admitted callback", err, calls)
				}
			} else if !errors.Is(err, cudalock.ErrNotInherited) || calls != 0 {
				t.Fatalf("admission = %v, callback calls = %d; want refused without callback", err, calls)
			}
		})
	}
	if err := lease.Release(); err != nil {
		t.Fatal(err)
	}
	calls := 0
	err = withTwoCUDALease(func(key string) string {
		if key == "GPU_LOCK_HELD" {
			return strconv.Itoa(os.Getpid())
		}
		if key == "GPU_LOCKDIR" {
			return dir
		}
		return ""
	}, func() { calls++ })
	if !errors.Is(err, cudalock.ErrNotInherited) || calls != 0 {
		t.Fatalf("released owner admission = %v, callback calls = %d; want refused without callback", err, calls)
	}
}

func TestSimulatedTwoRankStallFixture(t *testing.T) {
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		t.Fatalf("os.Hostname() = %q, %v; want non-empty host identity", host, err)
	}

	const baseNs = int64(1_700_000_000_000_000_000)

	t.Run("PositiveSameHostSkew", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		const (
			peerDelay     = 50 * time.Millisecond
			uncertaintyNS = int64(time.Millisecond)
		)
		obs0, obs1 := simulatedTwoRankHostEnqueueObservations(host, baseNs, peerDelay, uncertaintyNS)
		if err := attr.AddObservation(obs0); err != nil {
			t.Fatalf("AddObservation(rank 0): %v", err)
		}
		if err := attr.AddObservation(obs1); err != nil {
			t.Fatalf("AddObservation(rank 1): %v", err)
		}
		report, err := attr.Attribute(simulatedStallCommHash, 1)
		if err != nil {
			t.Fatalf("Attribute: %v", err)
		}
		if !report.Admitted || report.Diagnosis != ncclinspector.DiagnosisHostEnqueueSkew || report.DelayedRank != 1 {
			t.Fatalf("Attribute report = %+v; want Admitted=true Diagnosis=%q DelayedRank=1", report, ncclinspector.DiagnosisHostEnqueueSkew)
		}
		if report.DelayNS != peerDelay.Nanoseconds() {
			t.Fatalf("DelayNS = %d; want %d", report.DelayNS, peerDelay.Nanoseconds())
		}
		if report.UncertaintyNS != 2*uncertaintyNS {
			t.Fatalf("UncertaintyNS = %d; want %d", report.UncertaintyNS, 2*uncertaintyNS)
		}
	})

	t.Run("MissingHostnameIncompatibleClocks", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		obs0, obs1 := simulatedTwoRankHostEnqueueObservations("", baseNs, 50*time.Millisecond, int64(time.Millisecond))
		if err := attr.AddObservation(obs0); err != nil {
			t.Fatalf("AddObservation(rank 0): %v", err)
		}
		if err := attr.AddObservation(obs1); err != nil {
			t.Fatalf("AddObservation(rank 1): %v", err)
		}
		report, err := attr.Attribute(simulatedStallCommHash, 1)
		if err != nil {
			t.Fatalf("Attribute: %v", err)
		}
		if report.Admitted || report.Diagnosis != ncclinspector.DiagnosisIncompatibleClocks || report.DelayedRank != -1 {
			t.Fatalf("Attribute report = %+v; want Admitted=false Diagnosis=%q DelayedRank=-1", report, ncclinspector.DiagnosisIncompatibleClocks)
		}
		if !strings.Contains(report.Reason, "source host identity missing") {
			t.Fatalf("Reason = %q; want substring %q", report.Reason, "source host identity missing")
		}
	})

	t.Run("UncertaintyOverlapWithholdsCausality", func(t *testing.T) {
		attr := ncclinspector.NewTwoRankAttributor()
		const (
			peerDelay     = 1500 * time.Nanosecond
			uncertaintyNS = int64(1000) // combined uncertainty = 2000ns >= 1500ns delay
		)
		obs0, obs1 := simulatedTwoRankHostEnqueueObservations(host, baseNs, peerDelay, uncertaintyNS)
		if err := attr.AddObservation(obs0); err != nil {
			t.Fatalf("AddObservation(rank 0): %v", err)
		}
		if err := attr.AddObservation(obs1); err != nil {
			t.Fatalf("AddObservation(rank 1): %v", err)
		}
		report, err := attr.Attribute(simulatedStallCommHash, 1)
		if err != nil {
			t.Fatalf("Attribute: %v", err)
		}
		if report.Admitted || report.Diagnosis != ncclinspector.DiagnosisOrderIndeterminate || report.DelayedRank != -1 {
			t.Fatalf("Attribute report = %+v; want Admitted=false Diagnosis=%q DelayedRank=-1", report, ncclinspector.DiagnosisOrderIndeterminate)
		}
		if report.Diagnosis == ncclinspector.DiagnosisHostEnqueueSkew {
			t.Fatalf("Attribute unexpectedly admitted %q under overlapping uncertainty", report.Diagnosis)
		}
	})
}
