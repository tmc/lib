package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type hostLoader struct{}

func (hostLoader) Compile(_ context.Context, s cudahotswap.Source) (*cudahotswap.Compilation, error) {
	return &cudahotswap.Compilation{PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }", Entry: "entry", Key: s.Code}, nil
}
func (hostLoader) Load(context.Context, *cudahotswap.Compilation) (cudahotswap.Kernel, error) {
	return hostKernel{}, nil
}

type hostKernel struct{}

func (hostKernel) Raw() driver.Function                     { return 1 }
func (hostKernel) Launch(cudalaunch.Config, ...any) error   { return nil }
func (hostKernel) Wait(context.Context, *cuda.Stream) error { return nil }
func (hostKernel) Close() error                             { return nil }

func TestEventsOrderedAndDisconnect(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s, err := cudahotswap.NewService(hostLoader{})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close(context.Background())
	if _, err := s.Register(ctx, "kernel", cudahotswap.Source{Code: "baseline", Entry: "entry"}, "original"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetCheck("kernel", cudahotswap.CheckConfig{Checker: func(context.Context, cudahotswap.State) error { return nil }}, "host check"); err != nil {
		t.Fatal(err)
	}
	h, _ := New(s, Resources{Arguments: map[string][]string{"kernel": {}}, Streams: map[string]*cuda.Stream{"main": {}}})
	server := httptest.NewServer(h)
	defer server.Close()
	c := &Client{URL: server.URL, HTTP: server.Client()}
	pin, err := c.Call(ctx, "pin", Request{Name: "kernel", Label: "transport pin"})
	if err != nil || pin.State == nil || pin.State.Label == "transport pin" || pin.RequestLabel != "transport pin" {
		t.Fatalf("pin snapshot %v %v", pin, err)
	}
	before := *pin.State
	updated, err := s.SetCompileFlags("kernel", []string{"flag"}, "producer config")
	if err != nil {
		t.Fatal(err)
	}
	if before.ConfigRevision == updated.ConfigRevision {
		t.Fatal("control did not advance configuration")
	}
	released, err := c.Call(ctx, "release_pin", Request{Handle: pin.Handle, Label: "transport release"})
	if err != nil || released.State == nil || *released.State != before {
		t.Fatalf("pin snapshot changed %v %v", released, err)
	}
	launched, err := c.Call(ctx, "launch", Request{Name: "kernel", Label: "transport launch", Stream: "main", Geometry: &cudahotswap.LaunchGeometry{}})
	if err != nil || launched.State == nil || launched.State.ConfigRevision != updated.ConfigRevision || launched.State.Label != "producer config" {
		t.Fatalf("launch snapshot %v %v", launched, err)
	}
	waited, err := c.Call(ctx, "wait", Request{Handle: launched.Handle, Label: "transport wait"})
	if err != nil || waited.State == nil || *waited.State != *launched.State {
		t.Fatalf("wait snapshot %v %v", waited, err)
	}
	stream, err := c.Events(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		result, err := c.Call(ctx, "push", Request{Name: "kernel", Label: fmt.Sprintf("push%d", i), Source: cudahotswap.Source{Code: fmt.Sprintf("source%d", i), Entry: "entry"}})
		if err != nil || result.Failure != nil {
			t.Fatalf("push %v %v", result, err)
		}
		frame, err := stream.Read()
		if err != nil {
			t.Fatal(err)
		}
		if frame.Event.Sequence != uint64(i) || frame.Event.Label != fmt.Sprintf("push%d", i) {
			t.Fatalf("order %v", frame.Event)
		}
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		h.mu.Lock()
		active := h.streams
		h.mu.Unlock()
		if active == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("disconnect retained stream")
		}
		time.Sleep(time.Millisecond)
	}
	// A released service subscription admits the full 32-slot budget again.
	var owners []*cudahotswap.Subscription
	defer func() {
		for _, owner := range owners {
			owner.Close()
		}
	}()
	for i := 0; i < 32; i++ {
		owner, err := s.Subscribe(ctx, 3)
		if err != nil {
			t.Fatalf("slot %d: %v", i, err)
		}
		owners = append(owners, owner)
	}
}

func TestBoundedEventFrames(t *testing.T) {
	for _, body := range []string{`{}`, `{"event":{},"result":{}}`, strings.Repeat("x", (512<<10)+1)} {
		stream := newEventStream(io.NopCloser(strings.NewReader(body + "\n")))
		if _, err := stream.Read(); err == nil {
			t.Fatal("accepted malformed frame")
		}
		stream.Close()
	}
}

func TestEscapedRetainedReportFrame(t *testing.T) {
	frame := Frame{Event: &cudahotswap.Event{Sequence: 1, Report: &cudahotswap.Report{Reason: strings.Repeat("\x00", 30000)}}}
	b, err := json.Marshal(frame)
	if err != nil {
		t.Fatal(err)
	}
	stream := newEventStream(io.NopCloser(strings.NewReader(string(b) + "\n")))
	defer stream.Close()
	got, err := stream.Read()
	if err != nil || got.Event.Report.Reason != frame.Event.Report.Reason {
		t.Fatalf("escaped bounded report: %v", err)
	}
}

func ExampleClient_Call() {
	h, _ := New(new(fakeService), Resources{})
	server := httptest.NewServer(h)
	defer server.Close()
	client := &Client{URL: server.URL, HTTP: server.Client()}
	response, err := client.Call(context.Background(), "status", Request{Name: "kernel"})
	fmt.Println(response.State.Generation, response.State.ConfigRevision, err)
	// Output: 9 7 <nil>
}
