package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func directoryBundle(t *testing.T, dir string) string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("directory contains %d entries, want one", len(entries))
	}
	return filepath.Join(dir, entries[0].Name())
}

func writeDirectoryBundle(t *testing.T, dir, value string) string {
	t.Helper()
	w, close, err := cudaflightrecorder.NewDirectoryWriterFactory(dir).CreateWriter("incident", "tenant")
	if err != nil {
		t.Fatal(err)
	}
	_, writeErr := io.WriteString(w, value)
	if err := errors.Join(writeErr, close()); err != nil {
		t.Fatal(err)
	}
	return directoryBundle(t, dir)
}

func TestDirectoryWriterPreservesExistingBundle(t *testing.T) {
	dir := t.TempDir()
	want := "retained incident evidence"
	path := writeDirectoryBundle(t, dir, want)
	// A new factory models another coordinator or process using the same IDs.
	_, close, err := cudaflightrecorder.NewDirectoryWriterFactory(dir).CreateWriter("incident", "tenant")
	if close != nil {
		if err := close(); err != nil {
			t.Error(err)
		}
	}
	if !errors.Is(err, fs.ErrExist) {
		t.Errorf("duplicate creation = %v, want fs.ErrExist", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("retained bundle changed: %q, want %q", got, want)
	}
}

func TestDirectoryWriterRefusesSymlink(t *testing.T) {
	for _, exists := range []bool{false, true} {
		t.Run(fmt.Sprint(exists), func(t *testing.T) {
			dir := t.TempDir()
			path := writeDirectoryBundle(t, dir, "placeholder")
			if err := os.Remove(path); err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(t.TempDir(), "retained")
			want := "separate retained evidence"
			if exists {
				if err := os.WriteFile(target, []byte(want), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.Symlink(target, path); err != nil {
				t.Fatal(err)
			}
			_, close, err := cudaflightrecorder.NewDirectoryWriterFactory(dir).CreateWriter("incident", "tenant")
			if close != nil {
				if err := close(); err != nil {
					t.Error(err)
				}
			}
			if !errors.Is(err, fs.ErrExist) {
				t.Errorf("symlink creation = %v, want fs.ErrExist", err)
			}
			info, err := os.Lstat(path)
			if err != nil || info.Mode()&os.ModeSymlink == 0 {
				t.Fatalf("existing symlink changed: %v, %v", info, err)
			}
			got, err := os.ReadFile(target)
			if exists {
				if err != nil || string(got) != want {
					t.Fatalf("symlink target changed: %q, %v", got, err)
				}
			} else if !errors.Is(err, fs.ErrNotExist) {
				t.Fatalf("dangling target created: %q, %v", got, err)
			}
		})
	}
}

func TestDirectoryWriterConcurrentCreation(t *testing.T) {
	dir := t.TempDir()
	type result struct {
		value string
		err   error
	}
	results := make(chan result, 16)
	start := make(chan struct{})
	var group sync.WaitGroup
	for i := 0; i < cap(results); i++ {
		group.Add(1)
		go func(i int) {
			defer group.Done()
			<-start
			w, close, err := cudaflightrecorder.NewDirectoryWriterFactory(dir).CreateWriter("incident", "tenant")
			if err != nil {
				results <- result{err: err}
				return
			}
			value := fmt.Sprintf("winner %d", i)
			_, err = io.WriteString(w, value)
			results <- result{value, errors.Join(err, close())}
		}(i)
	}
	close(start)
	group.Wait()
	close(results)
	var winner string
	successes := 0
	for r := range results {
		if r.err == nil {
			successes++
			winner = r.value
		} else if !errors.Is(r.err, fs.ErrExist) {
			t.Errorf("creation failed unexpectedly: %v", r.err)
		}
	}
	if successes != 1 {
		t.Fatalf("successful creators = %d, want one", successes)
	}
	got, err := os.ReadFile(directoryBundle(t, dir))
	if err != nil || string(got) != winner {
		t.Fatalf("bundle = %q, %v; want %q", got, err, winner)
	}
}

func TestEscalationDirectoryCollisionIsFailure(t *testing.T) {
	dir := t.TempDir()
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{Window: time.Second, Capacity: 8})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	defer rec.Stop()
	clock := cudaflightrecorder.NewFakeClock(time.Unix(1700000000, 0))
	trigger := func(reason string) (*cudaflightrecorder.IncidentReceipt, error) {
		budgets := cudaflightrecorder.QualificationBudgets()
		budgets.HysteresisCount = 1
		coord, err := cudaflightrecorder.NewEscalationCoordinator(rec, budgets,
			cudaflightrecorder.WithClock(clock),
			cudaflightrecorder.WithResourceSampler(&cudaflightrecorder.FakeResourceSampler{}),
			cudaflightrecorder.WithCUPTINegotiator(&cudaflightrecorder.FakeCUPTINegotiator{}),
			cudaflightrecorder.WithCaptureWriterFactory(cudaflightrecorder.NewDirectoryWriterFactory(dir)))
		if err != nil {
			t.Fatal(err)
		}
		defer coord.Stop()
		if err := coord.Arm(); err != nil {
			t.Fatal(err)
		}
		return coord.Trigger(context.Background(), cudaflightrecorder.TriggerRequest{TenantID: "tenant", Reason: reason})
	}
	first, err := trigger("first incident")
	if err != nil || first == nil || first.Outcome != "captured" {
		t.Fatalf("first capture = %+v, %v", first, err)
	}
	path := directoryBundle(t, dir)
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	second, err := trigger("second incident")
	if !errors.Is(err, fs.ErrExist) || second == nil || second.Outcome != "failed" || second.IncidentID != first.IncidentID {
		t.Errorf("colliding capture = %+v, %v; want failed existing ID %q", second, err, first.IncidentID)
	}
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("first capture changed on collision: equal=%v, error=%v", bytes.Equal(got, want), err)
	}
}
