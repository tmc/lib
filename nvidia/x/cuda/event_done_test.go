package cuda

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
)

func TestEventDoneOnQueryError(t *testing.T) {
	for _, tt := range []struct {
		name  string
		event *Event
	}{
		{"destroyed", &Event{destroyed: true}},
		{"closed device", &Event{device: &Device{}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e := tt.event
			if done, err := e.Query(); done || err == nil {
				t.Fatalf("Query = %v, %v; want false and an error", done, err)
			}
			channels := make([]<-chan struct{}, 16)
			for i := range channels {
				channels[i] = e.Done()
				for _, prior := range channels[:i] {
					if channels[i] == prior {
						t.Fatal("Done reused a channel")
					}
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			for _, ch := range channels {
				select {
				case _, ok := <-ch:
					if ok {
						t.Fatal("Done sent a value instead of closing")
					}
				case <-ctx.Done():
					t.Fatal("Done did not close after Query failed")
				}
			}
			if err := e.Synchronize(ctx); err == nil {
				t.Fatal("Synchronize lost the error after Done closed")
			}
		})
	}
}

func TestEventDoneAcrossRecordings(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	dev, err := NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
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
	defer event.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	var previous <-chan struct{}
	for range 2 {
		if err := event.Record(stream); err != nil {
			t.Fatal(err)
		}
		ch := event.Done()
		if previous != nil {
			if ch == previous {
				t.Fatal("Done reused the previous recording's channel")
			}
			select {
			case <-previous:
			default:
				t.Fatal("previous channel is no longer closed")
			}
		}
		select {
		case <-ch:
		case <-ctx.Done():
			t.Fatal("Done did not close for a recorded event")
		}
		if done, err := event.Query(); !done || err != nil {
			t.Fatalf("Query = %v, %v; want true, nil", done, err)
		}
		previous = ch
	}
}
