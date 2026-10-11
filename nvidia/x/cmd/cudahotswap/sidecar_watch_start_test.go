package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

func sidecarFirstWatchEdit(path, code string) error {
	temporary := path + ".first-edit"
	file, err := os.OpenFile(temporary, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer os.Remove(temporary)
	_, err = file.WriteString(code)
	err = errors.Join(err, file.Close())
	if err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func TestSidecarWatchInitialEditCPU(t *testing.T) {
	// Actual CLI and Service.Watch; the loader and application check are explicit
	// CPU fakes. No device, CUDA call or real gpu-lock is involved.
	t.Setenv("GPU_LOCK_HELD", "")
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "host-lock"))
	for _, poll := range []bool{false, true} {
		for _, edit := range []bool{false, true} {
			t.Run(fmt.Sprintf("poll_%t/edit_%t", poll, edit), func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "chain.cu")
				if err := os.WriteFile(path, []byte("baseline"), 0600); err != nil {
					t.Fatal(err)
				}
				duration := 2 * time.Second
				if !edit {
					duration = 150 * time.Millisecond
				}
				ctx, cancel := context.WithTimeout(context.Background(), duration)
				defer cancel()
				output := &cancelWriter{cancel: cancel}
				var stderr bytes.Buffer
				closed := false
				candidate := "// watcher first edit\nbaseline"
				args := []string{"-watch", path}
				if poll {
					args = append(args, "-poll")
				}
				err := run(ctx, args, output, &stderr, func(ctx context.Context, c config) (*cudahotswap.Service, func() error, error) {
					service, err := cudahotswap.NewService(cliLoader{})
					if err != nil {
						return nil, nil, err
					}
					close := func() error { closed = true; return service.Close(context.Background()) }
					if _, err = service.Register(ctx, "kernel", cudahotswap.Source{Code: "baseline", Filename: c.path, Entry: c.entry}, "baseline"); err != nil {
						return service, close, err
					}
					_, err = service.SetCheck("kernel", cudahotswap.CheckConfig{Checker: func(_ context.Context, state cudahotswap.State) error {
						if state.SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte(candidate))) {
							return errors.New("wrong candidate identity")
						}
						return nil
					}}, "CPU application check")
					if err == nil && edit {
						err = sidecarFirstWatchEdit(path, candidate)
					}
					return service, close, err
				})
				if !closed {
					t.Fatal("service not closed")
				}
				if !edit {
					if !errors.Is(err, context.DeadlineExceeded) || output.Len() != 0 {
						t.Fatalf("unchanged source emitted report: err=%v output=%s", err, output.String())
					}
					return
				}
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("first edit did not report: %v %s", err, stderr.String())
				}
				var report struct {
					Kernel string
					Result cudahotswap.PushResult
					Error  string
				}
				if err := json.Unmarshal(output.Bytes(), &report); err != nil {
					t.Fatal(err)
				}
				if !report.Result.Published || report.Error != "" || report.Result.State.Version != 2 {
					t.Fatalf("wrong first-edit report: %+v", report)
				}
			})
		}
	}
}
