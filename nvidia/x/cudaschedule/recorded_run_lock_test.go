//go:build darwin || linux

package cudaschedule

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func recordedScriptFixture(pid int, host, label, token, suffix string) []byte {
	return []byte(fmt.Sprintf("pid=%d host=%s label=%s token=%s epoch=1 at=2026-10-15T00:00:00+00:00 %s\n", pid, host, label, token, suffix))
}

func TestRecordedGuardScriptPin(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("..", "cudalock", "reference", "gpu-lock.sh"))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != recordedLockDigest {
		t.Fatalf("actual reviewed script digest=%s want=%s", got, recordedLockDigest)
	}
	if recordedLockDigest == "37e06f6b22d97458ca50ff8e3f2175da00e71e5838f42dd7f6bb9e0322633c67" {
		t.Fatal("historical script still admitted")
	}
}

func TestRecordedGuardScriptOwner(t *testing.T) {
	label := "work token=label-value epoch=2 pid=900 host=label-host"
	body := recordedScriptFixture(42, "dgx", label, "dgx-42-1-0012", "lane=short expect=3 limit=11 limit_src=explicit")
	for _, tc := range []struct {
		name, host, label string
		pid               int
		body              []byte
		want              bool
	}{
		{"short local host", "dgx", label, 42, body, true},
		{"full local hostname", "dgx.example", label, 42, body, true},
		{"wrong pid", "dgx", label, 43, body, false},
		{"wrong host", "other.example", label, 42, body, false},
		{"empty local host", "", label, 42, body, false},
		{"full owner hostname", "dgx.example", label, 42, []byte(strings.Replace(string(body), "host=dgx ", "host=dgx.example ", 1)), false},
		{"different label", "dgx", "work", 42, body, false},
		{"empty token", "dgx", label, 42, recordedScriptFixture(42, "dgx", label, "", "lane=short"), false},
		{"wrong token holder", "dgx", label, 42, recordedScriptFixture(42, "dgx", label, "dgx-43-1-12", "lane=short"), false},
		{"token text", "dgx", label, 42, recordedScriptFixture(42, "dgx", label, "dgx-42-1-random", "lane=short"), false},
		{"survivor", "dgx", label, 42, recordedScriptFixture(50, "dgx", label, "dgx-42-1-12", "orphan-of=42"), false},
		{"orphan before permit", "dgx", label, 42, recordedScriptFixture(42, "dgx", label, "dgx-42-1-12", "orphan-of=42"), false},
		{"duplicate lane", "dgx", label, 42, append(append([]byte(nil), body[:len(body)-1]...), []byte(" lane=batch\n")...), false},
		{"unknown metadata", "dgx", label, 42, append(append([]byte(nil), body[:len(body)-1]...), []byte(" future=1\n")...), false},
		{"multiple lines", "dgx", label, 42, append(append([]byte(nil), body...), []byte("pid=50\n")...), false},
		{"empty owner", "dgx", label, 42, nil, false},
		{"oversized owner", "dgx", label, 42, []byte(strings.Repeat("x", 8193)), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			owner, err := recordedScriptOwner(tc.body, tc.pid, tc.host, tc.label)
			if (err == nil) != tc.want {
				t.Fatalf("owner=%+v error=%v accepted=%v want=%v", owner, err, err == nil, tc.want)
			}
			if err == nil && (owner.token != "dgx-42-1-0012" || owner.label != label || owner.pid != 42) {
				t.Fatalf("label text supplied metadata: %+v", owner)
			}
		})
	}
}

func TestRecordedGuardLeaseMatches(t *testing.T) {
	owner, err := recordedScriptOwner(recordedScriptFixture(42, "dgx", "work", "dgx-42-1-12", "lane=short expect=3 limit=11 limit_src=explicit"), 42, "dgx", "work")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, token     string
		expected, limit int64
		source          string
		want            bool
	}{
		{"initial observation", "", 3, 11, "explicit", true},
		{"same retained token", owner.token, 3, 11, "explicit", true},
		{"changed token", "dgx-42-1-13", 3, 11, "explicit", false},
		{"changed estimate", owner.token, 4, 11, "explicit", false},
		{"changed limit", owner.token, 3, 12, "explicit", false},
		{"default limit", owner.token, 3, 11, "default", false},
		{"zero estimate", owner.token, 0, 11, "explicit", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			actual := owner
			actual.limitSource = tc.source
			err := recordedLeaseMatches(actual, tc.token, tc.expected, tc.limit)
			if (err == nil) != tc.want {
				t.Fatalf("lease error=%v want accepted=%v", err, tc.want)
			}
		})
	}
}

func TestRecordedGuardLeaseRetirement(t *testing.T) {
	const token = "dgx-42-1-12"
	for _, tc := range []struct {
		name string
		body []byte
		want bool
	}{
		{"own holder", recordedScriptFixture(42, "dgx", "work", token, "lane=short"), false},
		{"same token new pid", recordedScriptFixture(50, "dgx", "work", token, "lane=short"), false},
		{"survivor", recordedScriptFixture(50, "dgx", "work", token, "orphan-of=42"), false},
		{"orphan without observed token", recordedScriptFixture(50, "dgx", "work", "dgx-42-1-99", "orphan-of=42"), false},
		{"new shell owner", recordedScriptFixture(50, "dgx", "other", "dgx-50-1-12", "lane=batch"), true},
		{"new Go owner", []byte("pid=50 host=dgx label=other epoch=1 at=2026-10-15T00:00:00Z\n"), true},
		{"Go label contains old token", []byte("pid=50 host=dgx label=other token=" + token + " orphan-of=42 epoch=1 at=2026-10-15T00:00:00Z\n"), true},
		{"foreign newer owner", []byte("pid=50 host=other label=unrelated epoch=1 at=2026-10-15T00:00:00Z\n"), true},
		{"empty present owner", nil, false},
		{"malformed owner", []byte("not an owner\n"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := string(tc.body)
			err := recordedLeaseRetired(tc.body, 42, token)
			if (err == nil) != tc.want || string(tc.body) != before {
				t.Fatalf("retirement error=%v accepted=%v want=%v; changed=%v", err, err == nil, tc.want, string(tc.body) != before)
			}
		})
	}
}

func TestRecordedGuardClosedCustody(t *testing.T) {
	for _, tc := range []struct {
		name string
		c    RecordedCustody
		err  error
		want bool
	}{
		{"no holder started", RecordedCustody{Phase: "keeper-ready", SupervisorCreation: "not-started"}, nil, true},
		{"unknown supervisor", RecordedCustody{Phase: "holder-started", SupervisorCreation: "unknown", Holder: RecordedProcess{PID: 42}}, nil, false},
		{"observed groups retired", RecordedCustody{Holder: RecordedProcess{PID: 42}, SupervisorCreation: "observed", Supervisor: RecordedProcess{PID: 50, Group: 50}}, nil, true},
		{"cleanup error", RecordedCustody{Holder: RecordedProcess{PID: 42}, SupervisorCreation: "observed", Supervisor: RecordedProcess{PID: 50, Group: 50}}, errors.New("owned lease remains"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := recordedClosedCustody(tc.c, tc.err); got != tc.want {
				t.Fatalf("closed=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestRecordedGuardInheritedAdmission(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", "")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := h.Release(); err != nil {
			t.Error(err)
		}
	}()
	f, err := os.OpenFile(dir+".guard", os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Millisecond)
	start := time.Now()
	owner, err := recordedInherited(ctx, os.Getenv)
	cancel()
	if owner.PID != 0 || !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > time.Second {
		t.Fatalf("busy guard admission owner=%+v error=%v elapsed=%s", owner, err, time.Since(start))
	}
	admit, stop := context.WithTimeout(t.Context(), time.Second)
	defer stop()
	var attempts atomic.Int32
	retrying := make(chan struct{})
	type result struct {
		pid int
		err error
	}
	done := make(chan result, 1)
	go func() {
		owner, err := recordedInherited(admit, func(key string) string {
			if key == "GPU_LOCK_HELD" && attempts.Add(1) == 2 {
				close(retrying)
			}
			return os.Getenv(key)
		})
		done <- result{owner.PID, err}
	}()
	select {
	case <-retrying:
	case <-admit.Done():
		t.Fatal("inherited guard refusal did not retry within admission budget")
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_UN); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-done:
		if got.err != nil || got.pid != h.Owner().PID {
			t.Fatalf("fresh actual inherited admission pid=%d error=%v", got.pid, got.err)
		}
	case <-admit.Done():
		t.Fatal("inherited admission did not recover after guard unlock")
	}
	ctx, cancel = context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	start = time.Now()
	owner, err = recordedInherited(ctx, func(string) string { return "" })
	if owner.PID != 0 || !errors.Is(err, cudalock.ErrNotInherited) || ctx.Err() != nil || time.Since(start) > 200*time.Millisecond {
		t.Fatalf("nonbusy refusal retried: owner=%+v error=%v elapsed=%s", owner, err, time.Since(start))
	}
	canceled, stop := context.WithCancel(t.Context())
	stop()
	if _, err := recordedInherited(canceled, os.Getenv); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
}
