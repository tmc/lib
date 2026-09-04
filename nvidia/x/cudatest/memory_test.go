package cudatest

import (
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeMemoryTelemetry struct {
	mu      sync.Mutex
	free    []uint64
	total   []uint64
	syncErr error
	memErr  error
	reads   int
}

func (f *fakeMemoryTelemetry) Synchronize() error { return f.syncErr }
func (f *fakeMemoryTelemetry) Memory() (uint64, uint64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.memErr != nil {
		return 0, 0, f.memErr
	}
	i := f.reads
	if i >= len(f.free) {
		i = len(f.free) - 1
	}
	f.reads++
	return f.free[i], f.total[i], nil
}

func TestDeviceMemoryCheck(t *testing.T) {
	tests := []struct {
		name      string
		free      []uint64
		total     []uint64
		tolerance uint64
		want      string
	}{
		{"clean", []uint64{900, 900}, []uint64{1000, 1000}, 0, ""},
		{"released", []uint64{800, 900}, []uint64{1000, 1000}, 0, ""},
		{"within tolerance", []uint64{900, 896}, []uint64{1000, 1000}, 4, ""},
		{"growth", []uint64{900, 895}, []uint64{1000, 1000}, 4, "grew by 5 bytes"},
		{"total changed", []uint64{900, 900}, []uint64{1000, 2000}, 0, "telemetry changed"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &fakeMemoryTelemetry{free: tt.free, total: tt.total}
			check, err := beginDeviceMemoryCheck(backend, new(sync.Mutex), tt.tolerance)
			if err != nil {
				t.Fatal(err)
			}
			err = check.Check()
			if (err == nil) != (tt.want == "") || err != nil && !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Check() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestDeviceMemoryCheckErrors(t *testing.T) {
	if _, err := BeginDeviceMemoryCheck(nil, 0); err == nil || !strings.Contains(err.Error(), "nil device") {
		t.Fatalf("BeginDeviceMemoryCheck(nil) error = %v", err)
	}
	wantErr := errors.New("telemetry unavailable")
	for _, tt := range []struct {
		name    string
		backend *fakeMemoryTelemetry
		want    string
	}{
		{"baseline synchronization", &fakeMemoryTelemetry{syncErr: wantErr}, "synchronize before"},
		{"baseline telemetry", &fakeMemoryTelemetry{memErr: wantErr}, "capture memory baseline"},
		{"invalid baseline", &fakeMemoryTelemetry{free: []uint64{2}, total: []uint64{1}}, "invalid memory baseline"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := beginDeviceMemoryCheck(tt.backend, new(sync.Mutex), 0)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("begin error = %v, want substring %q", err, tt.want)
			}
		})
	}
	check, err := beginDeviceMemoryCheck(&fakeMemoryTelemetry{free: []uint64{1, 1}, total: []uint64{2, 2}}, new(sync.Mutex), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := check.Check(); err != nil {
		t.Fatal(err)
	}
	if err := check.Check(); err == nil || !strings.Contains(err.Error(), "already completed") {
		t.Fatalf("second Check() error = %v", err)
	}
	var nilCheck *DeviceMemoryCheck
	if err := nilCheck.Check(); err == nil {
		t.Fatal("nil Check succeeded")
	}
}

func TestDeviceMemoryCheckFinalErrorsReleaseScope(t *testing.T) {
	backend := &fakeMemoryTelemetry{free: []uint64{900, 900}, total: []uint64{1000, 1000}}
	check, err := beginDeviceMemoryCheck(backend, new(sync.Mutex), 0)
	if err != nil {
		t.Fatal(err)
	}
	backend.syncErr = errors.New("sync failed")
	if err := check.Check(); err == nil || !strings.Contains(err.Error(), "synchronize after") {
		t.Fatalf("Check() error = %v", err)
	}
}

func TestDeviceMemoryChecksSerialize(t *testing.T) {
	lock := new(sync.Mutex)
	first, err := beginDeviceMemoryCheck(&fakeMemoryTelemetry{free: []uint64{1, 1}, total: []uint64{2, 2}}, lock, 0)
	if err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		close(started)
		second, err := beginDeviceMemoryCheck(&fakeMemoryTelemetry{free: []uint64{1, 1}, total: []uint64{2, 2}}, lock, 0)
		if err == nil {
			err = second.Check()
		}
		done <- err
	}()
	<-started
	select {
	case err := <-done:
		t.Fatalf("second scope completed before first: %v", err)
	case <-time.After(10 * time.Millisecond):
	}
	if err := first.Check(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
