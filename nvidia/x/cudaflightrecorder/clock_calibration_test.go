package cudaflightrecorder

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// fakeClock stands in for the CUPTI clock read. Tests install it on a single
// recorder; no test in this file reaches libcupti.
type fakeClock struct {
	mu                         sync.Mutex
	calls                      int
	hostNs, deviceNs, uncertNs int64
	err                        error
	hook                       func(call int) // runs outside mu, before returning
}

func (f *fakeClock) calibrate() (hostNs, deviceNs, uncertaintyNs int64, err error) {
	f.mu.Lock()
	f.calls++
	call := f.calls
	hostNs, deviceNs, uncertaintyNs, err = f.hostNs, f.deviceNs, f.uncertNs, f.err
	hook := f.hook
	f.mu.Unlock()
	if hook != nil {
		hook(call)
	}
	return hostNs, deviceNs, uncertaintyNs, err
}

func (f *fakeClock) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func newFakeClockRecorder(t *testing.T, cfg Config, f *fakeClock) *Recorder {
	t.Helper()
	r, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	r.calibrate = f.calibrate
	return r
}

// clockState is the recorder state that calibration may change.
type clockState struct {
	running, closed               bool
	calibrated                    bool
	hostNs, deviceNs, uncertainNs int64
	calibratedAt                  time.Time
}

func readClockState(r *Recorder) clockState {
	r.mu.Lock()
	defer r.mu.Unlock()
	return clockState{
		running:      r.running,
		closed:       r.closed,
		calibrated:   r.activityCalibrated,
		hostNs:       r.activityHostNs,
		deviceNs:     r.activityDeviceNs,
		uncertainNs:  r.activityUncertaintyNs,
		calibratedAt: r.activityCalibratedAt,
	}
}

func snapshotClock(t *testing.T, r *Recorder) (int64, string) {
	t.Helper()
	m, err := r.Snapshot(context.Background(), io.Discard, "clock")
	if err != nil {
		t.Fatal(err)
	}
	return m.ClockUncertaintyNs, m.ClockUncertaintySource
}

func lastRecord(t *testing.T, r *Recorder) Record {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.count == 0 {
		t.Fatal("no records")
	}
	return r.records[(r.head-1+len(r.records))%len(r.records)]
}

// TestNewWiresCUPTIClockReader checks that New installs the CUPTI clock
// reader. Every other test replaces the reader with a fake, so a recorder whose
// default reader was nil or a fake would pass them all. The reader is compared
// by function identity and never called: calling it would load libcupti.
func TestNewWiresCUPTIClockReader(t *testing.T) {
	r, err := New(Config{})
	if err != nil {
		t.Fatal(err)
	}
	if r.calibrate == nil {
		t.Fatal("New installed no clock reader")
	}
	got := reflect.ValueOf(r.calibrate).Pointer()
	want := reflect.ValueOf(calibrateCUPTITimestamp).Pointer()
	if got != want {
		t.Fatalf("clock reader = %s, want %s", runtime.FuncForPC(got).Name(), runtime.FuncForPC(want).Name())
	}
}

func TestStartDoesNotCalibrateClock(t *testing.T) {
	for _, tt := range []struct {
		name       string
		configured int64
		wantNs     int64
		wantSource string
	}{
		{"unavailable", 0, -1, "unavailable"},
		{"configured", 2500, 2500, "configured"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 3}
			r := newFakeClockRecorder(t, Config{Capacity: 4, ClockUncertaintyNs: tt.configured}, f)
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			defer r.Stop()
			if n := f.callCount(); n != 0 {
				t.Fatalf("Start made %d calibration calls, want 0", n)
			}
			if readClockState(r).calibrated {
				t.Fatal("calibrated after Start")
			}
			r.PushActivityRecord(cuptiactivity.Record{Name: "k", Start: 1<<20 + 100, End: 1<<20 + 200, Complete: true})
			rec := lastRecord(t, r)
			if rec.TimestampNs != 0 || rec.Labels["activity.uncalibrated"] != "true" {
				t.Errorf("record = %d %v, want timestamp 0 labeled activity.uncalibrated", rec.TimestampNs, rec.Labels)
			}
			if ns, src := snapshotClock(t, r); ns != tt.wantNs || src != tt.wantSource {
				t.Errorf("snapshot clock = (%d, %q), want (%d, %q)", ns, src, tt.wantNs, tt.wantSource)
			}
		})
	}
}

func TestCalibrateClockEnablesMapping(t *testing.T) {
	f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
	r := newFakeClockRecorder(t, Config{Capacity: 4, ClockUncertaintyNs: 2500}, f)
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	before := time.Now()
	if err := r.CalibrateClock(); err != nil {
		t.Fatalf("CalibrateClock: %v", err)
	}
	if n := f.callCount(); n != 1 {
		t.Fatalf("calibration calls = %d, want 1", n)
	}
	s := readClockState(r)
	if !s.calibrated || s.hostNs != f.hostNs || s.deviceNs != f.deviceNs || s.uncertainNs != f.uncertNs {
		t.Fatalf("state = %+v, want calibrated with fake values", s)
	}
	if s.calibratedAt.Before(before) {
		t.Errorf("calibratedAt %v precedes the call at %v", s.calibratedAt, before)
	}
	if !s.running || s.closed {
		t.Errorf("running=%v closed=%v, want running and open", s.running, s.closed)
	}
	r.PushActivityRecord(cuptiactivity.Record{Name: "k", Start: 1<<20 + 100, End: 1<<20 + 200, Complete: true})
	rec := lastRecord(t, r)
	if rec.TimestampNs != 1<<40+100 {
		t.Errorf("timestamp = %d, want %d", rec.TimestampNs, int64(1<<40+100))
	}
	if _, ok := rec.Labels["activity.uncalibrated"]; ok {
		t.Errorf("labels = %v, want no activity.uncalibrated", rec.Labels)
	}
	if ns, src := snapshotClock(t, r); src != "estimated" || ns < f.uncertNs {
		t.Errorf("snapshot clock = (%d, %q), want estimated >= %d", ns, src, f.uncertNs)
	}
}

// TestSnapshotCalibratedClockIsEstimated reads the manifest of a recorder
// calibrated 10s ago against a fake clock with a 7ns bracket, both as Snapshot
// returns it and as the bundle stores it. Only the bracket was measured; the
// rest is the assumed 11 ppm drift, so the source must be "estimated".
func TestSnapshotCalibratedClockIsEstimated(t *testing.T) {
	f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
	r := newFakeClockRecorder(t, Config{Capacity: 4}, f)
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	if err := r.CalibrateClock(); err != nil {
		t.Fatal(err)
	}
	const age = 10 * time.Second
	r.mu.Lock()
	r.activityCalibratedAt = r.activityCalibratedAt.Add(-age)
	r.mu.Unlock()

	var buf bytes.Buffer
	m, err := r.Snapshot(context.Background(), &buf, "estimated")
	if err != nil {
		t.Fatal(err)
	}
	b, err := ReadBundle(&buf)
	if err != nil {
		t.Fatal(err)
	}
	// 7ns bracket plus 11 ppm of at least 10s; allow a minute of scheduling
	// delay between the calibration and the snapshot.
	const minNs, maxNs = 7 + 110000, 7 + 11*70_000
	for _, got := range []*Manifest{m, &b.Manifest} {
		if got.ClockUncertaintySource != "estimated" || got.ClockUncertaintyNs < minNs || got.ClockUncertaintyNs > maxNs {
			t.Errorf("manifest clock = (%d, %q), want estimated in [%d, %d]", got.ClockUncertaintyNs, got.ClockUncertaintySource, minNs, maxNs)
		}
	}
}

func TestCalibrateClockErrorLeavesStateUnchanged(t *testing.T) {
	boom := errors.New("boom")
	for _, tt := range []struct {
		name  string
		prior bool // a successful calibration precedes the failing one
	}{
		{"uncalibrated", false},
		{"calibrated", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
			r := newFakeClockRecorder(t, Config{Capacity: 4}, f)
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			defer r.Stop()
			if tt.prior {
				if err := r.CalibrateClock(); err != nil {
					t.Fatal(err)
				}
			}
			want := readClockState(r)
			f.mu.Lock()
			f.hostNs, f.deviceNs, f.uncertNs, f.err = 99, 98, 97, boom
			f.mu.Unlock()
			err := r.CalibrateClock()
			if !errors.Is(err, boom) {
				t.Fatalf("CalibrateClock = %v, want %v", err, boom)
			}
			if got := readClockState(r); got != want {
				t.Errorf("state = %+v, want unchanged %+v", got, want)
			}
		})
	}
}

// TestCalibrateClockErrorPrefix checks that a failed clock read is reported
// with the package prefix exactly once, including for the errors the CUPTI
// reader itself returns.
func TestCalibrateClockErrorPrefix(t *testing.T) {
	for _, readErr := range []error{
		errors.New("boom"),
		errCUPTITimestampUnavailable,
		errHostClockBackwards,
	} {
		t.Run(readErr.Error(), func(t *testing.T) {
			r := newFakeClockRecorder(t, Config{Capacity: 4}, &fakeClock{err: readErr})
			if err := r.Start(); err != nil {
				t.Fatal(err)
			}
			defer r.Stop()
			err := r.CalibrateClock()
			if !errors.Is(err, readErr) {
				t.Fatalf("CalibrateClock = %v, want %v", err, readErr)
			}
			got, want := err.Error(), "cudaflightrecorder: calibrate clock: "+readErr.Error()
			if got != want || strings.Count(got, "cudaflightrecorder:") != 1 {
				t.Errorf("CalibrateClock = %q, want %q with one package prefix", err, want)
			}
		})
	}
}

func TestCalibrateClockRefusedWhenNotRunning(t *testing.T) {
	for _, tt := range []struct {
		name  string
		setup func(*Recorder) error
	}{
		{"never started", func(*Recorder) error { return nil }},
		{"stopped", func(r *Recorder) error {
			if err := r.Start(); err != nil {
				return err
			}
			r.Stop()
			return nil
		}},
		{"stopped before start", func(r *Recorder) error { r.Stop(); return nil }},
		{"closed", func(r *Recorder) error {
			if err := r.Start(); err != nil {
				return err
			}
			return r.Close()
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
			r := newFakeClockRecorder(t, Config{Capacity: 4}, f)
			if err := tt.setup(r); err != nil {
				t.Fatal(err)
			}
			want := readClockState(r)
			if err := r.CalibrateClock(); err == nil {
				t.Fatal("CalibrateClock succeeded, want error")
			}
			if n := f.callCount(); n != 0 {
				t.Errorf("calibration calls = %d, want 0", n)
			}
			if got := readClockState(r); got != want {
				t.Errorf("state = %+v, want unchanged %+v", got, want)
			}
		})
	}
}

// TestCalibrateClockStopDuringCalibration stops the recorder while a
// calibration is in flight. The result must not be published, and the
// recorder must stay stopped.
func TestCalibrateClockStopDuringCalibration(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
	r := newFakeClockRecorder(t, Config{Capacity: 4}, f)
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	f.mu.Lock()
	f.hook = func(int) {
		close(entered)
		<-release
	}
	f.mu.Unlock()
	done := make(chan error, 1)
	go func() { done <- r.CalibrateClock() }()
	select {
	case <-entered:
	case err := <-done:
		t.Fatalf("CalibrateClock returned %v without reading the recorder's clock", err)
	}
	r.Stop()
	close(release)
	if err := <-done; err == nil {
		t.Fatal("CalibrateClock succeeded after Stop, want error")
	}
	s := readClockState(r)
	if s.calibrated || s.running || !s.closed {
		t.Errorf("state = %+v, want stopped and uncalibrated", s)
	}
	if ns, src := snapshotClock(t, r); ns != -1 || src != "unavailable" {
		t.Errorf("snapshot clock = (%d, %q), want (-1, unavailable)", ns, src)
	}
	if err := r.Start(); err == nil {
		t.Error("Start after Stop succeeded")
	}
}

// TestCalibrateClockConcurrentStop races Stop against CalibrateClock. A
// calibration either completes before Stop and is kept, or fails and
// publishes nothing; the recorder never runs again.
func TestCalibrateClockConcurrentStop(t *testing.T) {
	for i := 0; i < 200; i++ {
		f := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
		r := newFakeClockRecorder(t, Config{Capacity: 4}, f)
		if err := r.Start(); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		var calErr error
		wg.Add(2)
		go func() { defer wg.Done(); calErr = r.CalibrateClock() }()
		go func() { defer wg.Done(); r.Stop() }()
		wg.Wait()
		s := readClockState(r)
		if s.running || !s.closed {
			t.Fatalf("iteration %d: running=%v closed=%v, want stopped", i, s.running, s.closed)
		}
		if (calErr == nil) != s.calibrated {
			t.Fatalf("iteration %d: CalibrateClock = %v but calibrated = %v", i, calErr, s.calibrated)
		}
	}
}

// TestCalibrateClockKeepsNewest overlaps two calibrations. The first is held
// inside its clock read until the second enters one or a timeout passes. If
// the two could overlap, the older first read would be published last.
func TestCalibrateClockKeepsNewest(t *testing.T) {
	firstEntered := make(chan struct{})
	secondEntered := make(chan struct{})
	var mu sync.Mutex
	calls := 0
	r, err := New(Config{Capacity: 4})
	if err != nil {
		t.Fatal(err)
	}
	// Each read reports its call number as the device timestamp.
	r.calibrate = func() (int64, int64, int64, error) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()
		switch n {
		case 1:
			close(firstEntered)
			select {
			case <-secondEntered:
			case <-time.After(100 * time.Millisecond):
			}
		case 2:
			close(secondEntered)
		}
		return 1 << 40, int64(n), 7, nil
	}
	if err := r.Start(); err != nil {
		t.Fatal(err)
	}
	defer r.Stop()
	errs := make(chan error, 2)
	go func() { errs <- r.CalibrateClock() }()
	select {
	case <-firstEntered:
	case err := <-errs:
		t.Fatalf("CalibrateClock returned %v without reading the recorder's clock", err)
	}
	go func() { errs <- r.CalibrateClock() }()
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if s := readClockState(r); s.deviceNs != 2 {
		t.Errorf("published device read %d, want 2 (the newer calibration)", s.deviceNs)
	}
}

// TestCalibrateClockSeamIsPerRecorder calibrates two recorders at once with
// different clocks. Each must use only its own.
func TestCalibrateClockSeamIsPerRecorder(t *testing.T) {
	boom := errors.New("boom")
	fa := &fakeClock{hostNs: 1 << 40, deviceNs: 1 << 20, uncertNs: 7}
	fb := &fakeClock{err: boom}
	a := newFakeClockRecorder(t, Config{Capacity: 4}, fa)
	b := newFakeClockRecorder(t, Config{Capacity: 4}, fb)
	for _, r := range []*Recorder{a, b} {
		if err := r.Start(); err != nil {
			t.Fatal(err)
		}
		defer r.Stop()
	}
	var wg sync.WaitGroup
	var errA, errB error
	wg.Add(2)
	go func() { defer wg.Done(); errA = a.CalibrateClock() }()
	go func() { defer wg.Done(); errB = b.CalibrateClock() }()
	wg.Wait()
	if errA != nil || !errors.Is(errB, boom) {
		t.Fatalf("errors = (%v, %v), want (nil, %v)", errA, errB, boom)
	}
	if fa.callCount() != 1 || fb.callCount() != 1 {
		t.Errorf("calls = (%d, %d), want (1, 1)", fa.callCount(), fb.callCount())
	}
	if sa, sb := readClockState(a), readClockState(b); !sa.calibrated || sa.deviceNs != fa.deviceNs || sb.calibrated {
		t.Errorf("states = %+v, %+v, want only the first calibrated", sa, sb)
	}
}

// ExampleRecorder_CalibrateClock shows CalibrateClock refusing a recorder
// that is not running. A refused call reads no clock and changes nothing, so
// snapshots still report the clock uncertainty as unavailable.
//
// On a running recorder CalibrateClock loads libcupti and reads the CUPTI
// device clock, so this example does not make that call. A program that wants
// calibrated activity timestamps calls it after Start:
//
//	if err := rec.CalibrateClock(); err != nil {
//		log.Printf("activity timestamps stay uncalibrated: %v", err)
//	}
func ExampleRecorder_CalibrateClock() {
	rec, err := New(Config{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(rec.CalibrateClock())

	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	rec.Stop()
	fmt.Println(rec.CalibrateClock())

	m, err := rec.Snapshot(context.Background(), io.Discard, "refused")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(m.ClockUncertaintyNs, m.ClockUncertaintySource)

	// Output:
	// cudaflightrecorder: recorder not running
	// cudaflightrecorder: recorder is closed
	// -1 unavailable
}
