//go:build linux

package cuda_test

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestConcurrentWithContextAndClose(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for range 100 {
				if err := dev.WithContext(func() error { return nil }); err != nil {
					return
				}
			}
		}()
	}
	close(start)
	if err := dev.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	if err := dev.WithContext(func() error { return nil }); err == nil {
		t.Fatal("WithContext succeeded after Close")
	}
	if err := dev.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestStreamOperationSerializedWithClose(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	stream.AddCleanup(func() {
		close(entered)
		<-release
	})
	destroyDone := make(chan error, 1)
	go func() { destroyDone <- stream.Close() }()
	<-entered
	queryDone := make(chan error, 1)
	go func() {
		_, err := stream.Query()
		queryDone <- err
	}()
	select {
	case err := <-queryDone:
		if err == nil {
			t.Fatal("Query crossed concurrent Close")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Query remained blocked during cleanup callback")
	}
	close(release)
	if err := <-destroyDone; err != nil {
		t.Fatal(err)
	}
}

func TestWithHandleExcludesClose(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	submitDone := make(chan error, 1)
	go func() {
		submitDone <- stream.WithHandle(func(driver.Stream) error {
			close(entered)
			<-release
			return nil
		})
	}()
	<-entered
	destroyDone := make(chan error, 1)
	go func() { destroyDone <- stream.Close() }()
	select {
	case err := <-destroyDone:
		t.Fatalf("Close crossed live-handle submission: %v", err)
	default:
	}
	close(release)
	if err := <-submitDone; err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-destroyDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close remained blocked after submission returned")
	}
	if err := stream.WithHandle(func(driver.Stream) error { return nil }); err == nil {
		t.Fatal("WithHandle succeeded after Close")
	}
}

func TestEventOperationsAfterCloseFail(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Close(); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		do   func() error
	}{
		{"record", func() error { return event.Record(stream) }},
		{"query", func() error { _, err := event.Query(); return err }},
		{"synchronize", func() error { return event.Synchronize(context.Background()) }},
		{"elapsed", func() error { _, err := event.ElapsedSince(event); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.do(); err == nil {
				t.Fatal("operation after Close succeeded")
			}
		})
	}
}

func TestEventQuerySerializedWithClose(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-start
		for {
			if _, err := event.Query(); err != nil {
				return
			}
		}
	}()
	close(start)
	if err := event.Close(); err != nil {
		t.Fatal(err)
	}
	<-done
}

func TestContextRunRestoresCurrent(t *testing.T) {
	// The current context is per thread, so the checks below read the one
	// Run restored only while this goroutine stays on the same thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Release()

	defer driver.Context(0).SetCurrent()
	if err := driver.Context(0).SetCurrent(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.Run(func() error {
		current, err := driver.CtxGetCurrent()
		if err != nil {
			return err
		}
		if current != ctx.Raw() {
			t.Errorf("current context = %v, want %v", current, ctx.Raw())
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	current, err := driver.CtxGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	if current != 0 {
		t.Fatalf("current context after Run = %v, want none", current)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Run callback did not panic")
			}
		}()
		_ = ctx.Run(func() error { panic("sentinel") })
	}()
	current, err = driver.CtxGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	if current != 0 {
		t.Fatalf("current context after panic = %v, want none", current)
	}
}

func TestContextRunRestoresChangedCurrent(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	dev := cudatest.RequireDevice(t)
	defer dev.Close()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Release()
	previous, err := driver.CtxGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	defer previous.SetCurrent()
	sentinel := errors.New("callback error")
	for _, panicCallback := range []bool{false, true} {
		if err := ctx.Raw().SetCurrent(); err != nil {
			t.Fatal(err)
		}
		func() {
			if panicCallback {
				defer func() {
					if got := recover(); got != sentinel {
						t.Errorf("panic = %v, want %v", got, sentinel)
					}
				}()
			}
			err := ctx.Run(func() error {
				if err := driver.Context(0).SetCurrent(); err != nil {
					return err
				}
				if panicCallback {
					panic(sentinel)
				}
				return sentinel
			})
			if !errors.Is(err, sentinel) {
				t.Errorf("Run error = %v, want %v", err, sentinel)
			}
		}()
		current, err := driver.CtxGetCurrent()
		if err != nil {
			t.Fatal(err)
		}
		if current != ctx.Raw() {
			t.Fatalf("current context = %v, want restored %v", current, ctx.Raw())
		}
	}
}

func TestRepeatedRelease(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		release func() error
	}{
		{"event", event.Close},
		{"stream", stream.Close},
		{"context", ctx.Release},
		{"device", dev.Close},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.release(); err != nil {
				t.Fatalf("first release: %v", err)
			}
			if err := tt.release(); err != nil {
				t.Fatalf("second release: %v", err)
			}
		})
	}
}
