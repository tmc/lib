package main

import (
	"context"
	"errors"
	"slices"
	"testing"
)

// closeLog records the calls closeStepTimer makes, in order, and returns
// the error fail assigns to each.
type closeLog struct {
	calls []string
	fail  map[string]error
}

func (l *closeLog) call(name string) error {
	l.calls = append(l.calls, name)
	return l.fail[name]
}

type fakeStream struct{ log *closeLog }

func (s *fakeStream) Synchronize(context.Context) error { return s.log.call("sync") }
func (s *fakeStream) Close() error                      { return s.log.call("stream") }

type fakeEvent struct {
	name string
	log  *closeLog
}

// errNilEventClosed is returned by Close on a nil *fakeEvent, which
// closeStepTimer must skip.
var errNilEventClosed = errors.New("nil event closed")

func (e *fakeEvent) Close() error {
	if e == nil {
		return errNilEventClosed
	}
	return e.log.call(e.name)
}

// TestCloseStepTimer checks that closeStepTimer drains the stream before
// closing anything, closes every non-nil event and then the stream once
// each even after earlier steps fail, and joins every failure in order with
// its identity kept.
func TestCloseStepTimer(t *testing.T) {
	errSync := errors.New("sync failed")
	errAnchor := errors.New("anchor failed")
	errStart := errors.New("start failed")
	errEnd := errors.New("end failed")
	errStream := errors.New("stream failed")
	for _, tt := range []struct {
		name      string
		noStream  bool
		events    []string // "" is a nil event
		fail      map[string]error
		wantCalls []string
		wantErr   string
		wantIs    []error
	}{
		{
			name:      "all succeed",
			events:    []string{"anchor", "start", "end"},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
		},
		{
			name:      "every step fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"sync": errSync, "anchor": errAnchor, "start": errStart, "end": errEnd, "stream": errStream},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr: "synchronize stream: sync failed\n" +
				"close event: anchor failed\n" +
				"close event: start failed\n" +
				"close event: end failed\n" +
				"close stream: stream failed",
			wantIs: []error{errSync, errAnchor, errStart, errEnd, errStream},
		},
		{
			name:      "sync fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"sync": errSync},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr:   "synchronize stream: sync failed",
			wantIs:    []error{errSync},
		},
		{
			name:      "first event fails",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"anchor": errAnchor},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr:   "close event: anchor failed",
			wantIs:    []error{errAnchor},
		},
		{
			name:      "last event and stream fail",
			events:    []string{"anchor", "start", "end"},
			fail:      map[string]error{"end": errEnd, "stream": errStream},
			wantCalls: []string{"sync", "anchor", "start", "end", "stream"},
			wantErr:   "close event: end failed\nclose stream: stream failed",
			wantIs:    []error{errEnd, errStream},
		},
		{
			name:      "nil events skipped",
			events:    []string{"anchor", "", ""},
			fail:      map[string]error{"anchor": errAnchor},
			wantCalls: []string{"sync", "anchor", "stream"},
			wantErr:   "close event: anchor failed",
			wantIs:    []error{errAnchor},
		},
		{
			name:      "nil stream",
			noStream:  true,
			events:    []string{"anchor", "start", "end"},
			wantCalls: []string{"anchor", "start", "end"},
		},
		{
			name:     "nothing built",
			noStream: true,
			events:   []string{"", "", ""},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			log := &closeLog{fail: tt.fail}
			stream := &fakeStream{log: log}
			if tt.noStream {
				stream = nil
			}
			var events []*fakeEvent
			for _, name := range tt.events {
				var e *fakeEvent
				if name != "" {
					e = &fakeEvent{name: name, log: log}
				}
				events = append(events, e)
			}
			err := closeStepTimer(stream, events)
			if !slices.Equal(log.calls, tt.wantCalls) {
				t.Errorf("calls = %q, want %q", log.calls, tt.wantCalls)
			}
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("err = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("err = nil, want %q", tt.wantErr)
			}
			if err.Error() != tt.wantErr {
				t.Errorf("err = %q, want %q", err, tt.wantErr)
			}
			for _, want := range tt.wantIs {
				if !errors.Is(err, want) {
					t.Errorf("errors.Is(err, %v) = false", want)
				}
			}
		})
	}
}

// TestStepTimerCloseZero checks that close on a timer with nothing built
// makes no calls through nil pointers and reports no error.
func TestStepTimerCloseZero(t *testing.T) {
	if err := (&stepTimer{}).close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}
