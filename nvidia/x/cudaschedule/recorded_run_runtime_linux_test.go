//go:build linux

package cudaschedule

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestRecordedEnvironmentNoDuplicateSettings(t *testing.T) {
	base := []string{"KEEP=yes", "GPU_LOCKDIR=old", "GPU_LOCK_HELD=123", "GPU_LOCKDIR=other", recordedConfigEnv + "=stale"}
	got := recordedEnvironment(base, "GPU_LOCKDIR=new", recordedConfigEnv+"=actual")
	want := []string{"KEEP=yes", "GPU_LOCK_HELD=123", "GPU_LOCKDIR=new", recordedConfigEnv + "=actual"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("environment=%q want%q", got, want)
	}
	if base[1] != "GPU_LOCKDIR=old" {
		t.Fatal("changed caller environment")
	}
}

func TestRecordedRegularFileRefusesBlockingPaths(t *testing.T) {
	dir := t.TempDir()
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	regular := filepath.Join(dir, "regular")
	if err := os.WriteFile(regular, nil, 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(regular, link); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{fifo, link} {
		if _, err := recordedReadRegular(path, 64); err == nil {
			t.Fatalf("accepted nonregular path %s", path)
		}
	}
	body, err := recordedReadOutput(regular, digestPTX(""), 64)
	if err != nil || len(body) != 0 {
		t.Fatalf("empty bounded output=%q/%v", body, err)
	}
	if _, err := recordedReadOutput(regular, digestPTX("wrong"), 64); err == nil {
		t.Fatal("accepted stale output digest")
	}
}

func TestRecordedLockSourceEnvironmentIntegerSettings(t *testing.T) {
	config := recordedConfig{LockDirectory: "/owned/lock", Label: "request-label", Expected: 3 * time.Second, Limits: recordedLimits{Queue: 7 * time.Second, Lease: 11 * time.Second}}
	env := recordedHolderEnvironment(config)
	want := map[string]string{"GPU_LOCK_POLL": "1", "GPU_LOCK_KILLWAIT": "1", "GPU_LOCKDIR": "/owned/lock", "GPU_LOCK_LABEL": "request-label", "GPU_LOCK_EXPECT": "3", "GPU_LOCK_MAXWAIT": "7", "GPU_LOCK_LIMIT": "11"}
	got := make(map[string]string)
	for _, entry := range env {
		key, value, _ := strings.Cut(entry, "=")
		if _, ok := want[key]; ok {
			if _, duplicate := got[key]; duplicate {
				t.Fatalf("duplicate emitted setting %s", key)
			}
			got[key] = value
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("actual holder environment=%v want=%v", got, want)
	}
}

func TestRecordedMonitorHolderLiteralStatus(t *testing.T) {
	for _, tc := range []struct {
		name      string
		args      []string
		interrupt bool
		wantError bool
	}{
		{"normal_zero", []string{"-c", "exit 0"}, false, false},
		{"literal_nonzero", []string{"-c", "exit 17"}, false, true},
		{"signal", []string{"-c", "exec sleep 2"}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			executable, args := "/bin/sh", tc.args
			if tc.interrupt {
				executable, args = "/usr/bin/sleep", []string{"2"}
			}
			child, err := recordedStartChild(executable, args, os.Environ(), t.TempDir(), 64)
			if child != nil {
				t.Cleanup(func() {
					ctx, cancel := context.WithTimeout(context.Background(), time.Second)
					defer cancel()
					if err := recordedRetire(ctx, []int{child.identity.Group}); err != nil {
						t.Errorf("holder cleanup: %v", err)
					}
					if err := child.poll(); err != nil {
						t.Errorf("holder wait/drain: %v", err)
					}
					if err := child.closePipes(); err != nil {
						t.Errorf("holder pipes: %v", err)
					}
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if tc.interrupt {
				// Bind the direct child to its stable post-exec image before signaling.
				current, err := recordedIdentify(child.identity.PID)
				if err != nil {
					t.Fatal(err)
				}
				child.identity = current
				if err := child.signal(syscall.SIGTERM); err != nil {
					t.Fatal(err)
				}
			}
			end := time.Now().Add(time.Second)
			for !child.reaped && time.Now().Before(end) {
				if err := child.poll(); err != nil {
					t.Fatal(err)
				}
				time.Sleep(time.Millisecond)
			}
			if !child.reaped {
				t.Fatal("missing actual holder wait status")
			}
			err = recordedMonitorHolder(child)
			if (err != nil) != tc.wantError {
				t.Fatalf("actual holder status=%v error=%v wantError=%v", child.status, err, tc.wantError)
			}
		})
	}
}
