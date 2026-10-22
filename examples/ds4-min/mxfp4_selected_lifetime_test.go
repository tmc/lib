package main

import (
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestMXFP4ModuleCleanupRetry(t *testing.T) {
	var events []string
	syncs, releases := 0, 0
	s := mxfp4ModuleLifetime{
		synchronize: func() error {
			events = append(events, "sync")
			syncs++
			if syncs == 1 {
				return io.ErrUnexpectedEOF
			}
			return nil
		},
		unload: func() error {
			events = append(events, "unload")
			releases++
			if releases == 1 {
				return io.ErrShortWrite
			}
			return nil
		},
	}
	if err := s.close(); !errors.Is(err, io.ErrUnexpectedEOF) || !s.closed || s.unload == nil || releases != 0 {
		t.Fatalf("sync failure lost owner: %v", err)
	}
	if err := s.close(); !errors.Is(err, io.ErrShortWrite) || s.unload == nil {
		t.Fatalf("unload failure lost owner: %v", err)
	}
	if err := s.close(); err != nil {
		t.Fatal(err)
	}
	if err := s.close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"sync", "sync", "unload", "sync", "unload"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("release events=%v want=%v", events, want)
	}
}

func TestMXFP4UnloadContextRestoreFailure(t *testing.T) {
	handle, unloads := 1, 0
	context := func(call func() error) error {
		if err := call(); err != nil {
			return err
		}
		return io.ErrUnexpectedEOF
	}
	s := mxfp4ModuleLifetime{synchronize: func() error { return nil }}
	s.unload = func() error {
		if handle == 0 {
			return nil
		}
		return mxfp4UnloadInContext(context, func() error { unloads++; return nil }, func() { handle = 0 })
	}
	if err := s.close(); !errors.Is(err, io.ErrUnexpectedEOF) || handle != 0 {
		t.Fatalf("post-unload restore failure = %v handle=%d", err, handle)
	}
	if err := s.close(); err != nil || unloads != 1 {
		t.Fatalf("retry unloads=%d err=%v", unloads, err)
	}
}

func TestMXFP4DeviceRanges(t *testing.T) {
	valid := []mxfp4DeviceRange{{1, 4}, {5, 36}, {41, 17}, {58, 4}}
	if err := checkMXFP4DeviceRanges(valid...); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []mxfp4DeviceRange{{0, 4}, {3, 4}, {4, 1}, {^uint64(0), 2}, {100, 0}} {
		ranges := append(append([]mxfp4DeviceRange(nil), valid...), bad)
		if err := checkMXFP4DeviceRanges(ranges...); err == nil {
			t.Fatalf("accepted device range %+v", bad)
		}
	}
}
