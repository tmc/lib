package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type cliLoader struct{}

func (cliLoader) Compile(_ context.Context, source cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return &cudahotswap.Compilation{Entry: source.Entry, PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry replay() { ret; }"}, nil
}
func (cliLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	return cliKernel{}, nil
}

type cliKernel struct{}

func (cliKernel) Raw() driver.Function                     { return 1 }
func (cliKernel) Launch(cudalaunch.Config, ...any) error   { return nil }
func (cliKernel) Wait(context.Context, *cuda.Stream) error { return nil }
func (cliKernel) Close() error                             { return nil }

type cancelWriter struct {
	bytes.Buffer
	cancel context.CancelFunc
}

func (w *cancelWriter) Write(data []byte) (int, error) {
	n, err := w.Buffer.Write(data)
	w.cancel()
	return n, err
}

func TestRunWatchActualServiceReport(t *testing.T) {
	// Actual Service APIs, PTX parser and per-request cudalock owners are used
	// with explicit fake Loader/Kernel and an isolated host-only lock directory.
	// No CUDA device or native call is made.
	t.Setenv("GPU_LOCK_HELD", "")
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "lock"))
	for _, wrong := range []bool{false, true} {
		t.Run(strconv.FormatBool(wrong), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "candidate.cu")
			if err := os.WriteFile(path, []byte("candidate"), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			out := &cancelWriter{cancel: cancel}
			var stderr bytes.Buffer
			closed := false
			err := run(ctx, []string{"-watch", path, "-poll"}, out, &stderr, func(ctx context.Context, c config) (*cudahotswap.Service, func() error, error) {
				s, err := cudahotswap.NewService(cliLoader{})
				if err != nil {
					return nil, nil, err
				}
				cleanup := func() error { closed = true; return s.Close(context.Background()) }
				if _, err := s.Register(ctx, "kernel", cudahotswap.Source{Code: "baseline", Filename: c.path, Entry: c.entry}, "baseline"); err != nil {
					return nil, cleanup, err
				}
				_, err = s.SetCheck("kernel", cudahotswap.CheckConfig{Checker: func(_ context.Context, state cudahotswap.State) error {
					if state.SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte("candidate"))) {
						return errors.New("missing source identity")
					}
					if wrong {
						return errors.New("application check refused")
					}
					return nil
				}}, "host application check")
				return s, cleanup, err
			})
			if !errors.Is(err, context.Canceled) || !closed {
				t.Fatalf("watch CLI lifetime: closed=%t, err=%v", closed, err)
			}
			var report struct {
				Kernel string
				Result cudahotswap.PushResult
				Error  string
			}
			if err := json.Unmarshal(out.Bytes(), &report); err != nil {
				t.Fatalf("actual Push JSON report: %s: %v", out.String(), err)
			}
			if report.Kernel != "kernel" || !report.Result.Published || report.Result.RolledBack != wrong || (report.Error != "") != wrong {
				t.Fatalf("CLI changed actual Push verdict: %+v", report)
			}
		})
	}
}

type cliCleanupOwner struct {
	calls int
	fail  bool
}

func (o *cliCleanupOwner) Close() error {
	o.calls++
	if o.fail {
		return errors.New("lease release failed")
	}
	return nil
}

func TestSessionCleanupRetainsTeardownLeaseUntilDependentsClose(t *testing.T) {
	lease := new(cliCleanupOwner)
	acquires, serviceCloses, nativeCloses := 0, 0, 0
	failNative := true
	o := &sessionOwners{
		acquire:      func(context.Context, string) (io.Closer, error) { acquires++; return lease, nil },
		serviceClose: func() error { serviceCloses++; return nil },
		nativeClose: func() error {
			nativeCloses++
			if failNative {
				return errors.New("dependent still live")
			}
			return nil
		},
	}
	if err := o.close(); err == nil || lease.calls != 0 || serviceCloses != 1 || acquires != 1 {
		t.Fatal("failed dependent cleanup released its teardown lease")
	}
	failNative = false
	lease.fail = true
	if err := o.close(); err == nil || lease.calls != 1 || acquires != 1 || nativeCloses != 2 || serviceCloses != 1 {
		t.Fatal("dependent retry lost lease or repeated completed service cleanup")
	}
	lease.fail = false
	if err := o.close(); err != nil || lease.calls != 2 || nativeCloses != 2 || acquires != 1 || o.lease != nil {
		t.Fatalf("lease-only retry repeated completed dependent cleanup: %v", err)
	}
}

func TestRunRefusesInvalidArgumentsBeforeDevice(t *testing.T) {
	for _, args := range [][]string{nil, {"-watch", "x", "-n", "0"}, {"-watch", "x", "-n", "4097"}, {"-watch", "x", "-device", "-1"}, {"-watch", "x", "unexpected"}} {
		err := run(context.Background(), args, io.Discard, io.Discard, func(context.Context, config) (*cudahotswap.Service, func() error, error) {
			t.Fatal("invalid arguments reached CUDA initialization")
			return nil, nil, nil
		})
		if err == nil {
			t.Fatalf("invalid arguments admitted: %v", args)
		}
	}
}

func TestRunPartialSessionOwnerCleanup(t *testing.T) {
	original := errors.New("partial setup failure")
	cleanupErr := errors.New("owner close failed")
	closed := 0
	err := run(context.Background(), []string{"-watch", "x"}, io.Discard, io.Discard, func(context.Context, config) (*cudahotswap.Service, func() error, error) {
		return nil, func() error { closed++; return cleanupErr }, original
	})
	if closed != 1 || !errors.Is(err, original) || !errors.Is(err, cleanupErr) {
		t.Fatalf("partial session cleanup: closed=%d, err=%v", closed, err)
	}
}

func TestRunCancellationDoesNotHideCleanupFailure(t *testing.T) {
	cleanupErr := errors.New("native owner remains live")
	err := run(context.Background(), []string{"-watch", "x"}, io.Discard, io.Discard, func(context.Context, config) (*cudahotswap.Service, func() error, error) {
		return nil, func() error { return cleanupErr }, context.Canceled
	})
	if !errors.Is(err, cleanupErr) || onlyCancellation(err) {
		t.Fatal("CLI normal cancellation suppressed a joined cleanup failure")
	}
	if !onlyCancellation(errors.Join(context.Canceled, fmt.Errorf("request: %w", context.Canceled))) {
		t.Fatal("ordinary signal cancellation was treated as failure")
	}
}
