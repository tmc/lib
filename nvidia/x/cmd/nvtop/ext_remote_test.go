package main

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
)

// A pipeDialer dials in-memory streams, one per host, whose writers
// the test holds.
type pipeDialer struct {
	mu sync.Mutex
	w  map[string]*io.PipeWriter
}

func (p *pipeDialer) dial(host string) (io.ReadCloser, error) {
	if host == "down" {
		return nil, errors.New("connection refused")
	}
	r, w := io.Pipe()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.w == nil {
		p.w = make(map[string]*io.PipeWriter)
	}
	p.w[host] = w
	return r, nil
}

// writer waits for host to be dialed and returns its stream's writer.
func (p *pipeDialer) writer(t *testing.T, host string) *io.PipeWriter {
	var w *io.PipeWriter
	eventually(t, "dial "+host, func() bool {
		p.mu.Lock()
		defer p.mu.Unlock()
		w = p.w[host]
		return w != nil
	})
	return w
}

// frameLine returns a recorded frame of devs at t with evs.
func frameLine(t *testing.T, at time.Time, devs []*device, evs []event) []byte {
	var b bytes.Buffer
	if err := newRecorder(&b).write(at, devs, evs); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// eventually waits for cond, polled, to hold.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); !cond(); {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func TestRemote(t *testing.T) {
	p := new(pipeDialer)
	type result struct {
		s   *remoteSource
		err error
	}
	opened := make(chan result)
	go func() {
		s, err := openRemote([]string{"alpha", "beta"}, p.dial)
		opened <- result{s, err}
	}()
	// alpha has two devices, beta one.
	alpha, beta := p.writer(t, "alpha"), p.writer(t, "beta")
	beta1, alpha1 := frameLine(t, testNow, testDevices()[1:], nil), frameLine(t, testNow, testDevices(), nil)
	go beta.Write(beta1)
	go alpha.Write(alpha1)
	r := <-opened
	if r.err != nil {
		t.Fatal(r.err)
	}
	s := r.s
	devs := s.devices()
	var titles []string
	for i, d := range devs {
		if d.index != i {
			t.Errorf("device %d has index %d", i, d.index)
		}
		titles = append(titles, d.title())
	}
	want := "alpha: NVIDIA GeForce RTX 4090,alpha: NVIDIA GB10,beta: NVIDIA GB10"
	if got := strings.Join(titles, ","); got != want {
		t.Errorf("titles = %s, want %s", got, want)
	}

	// A new frame from beta, with an Xid, shows on the next refresh.
	b := testDevices()[1:]
	b[0].gpuUtil = some[uint32](64)
	xid := event{at: testNow.Add(time.Second), dev: 0, kind: nvml.EventTypeXidCriticalError, data: 79}
	if _, err := beta.Write(frameLine(t, testNow.Add(time.Second), b, []event{xid})); err != nil {
		t.Fatal(err)
	}
	eventually(t, "beta's second frame", func() bool {
		gather(s, nil, devs, true)
		return devs[2].gpuUtil == some[uint32](64)
	})
	if devs[2].host != "beta" || len(devs[0].procs) != 3 {
		t.Errorf("after refresh: %+v, %d processes on alpha's first", devs[2], len(devs[0].procs))
	}
	evs := s.events()
	if len(evs) != 1 || evs[0].dev != 2 || evs[0].data != 79 {
		t.Errorf("events = %+v, want beta's Xid on device 2", evs)
	}

	// When alpha's stream ends, its devices report nothing and its
	// first device a lost connection.
	alpha.CloseWithError(errors.New("broken pipe"))
	eventually(t, "alpha lost", func() bool {
		gather(s, nil, devs, true)
		return !devs[0].gpuUtil.ok
	})
	if !devs[0].name.ok || devs[0].procs != nil || !devs[2].gpuUtil.ok {
		t.Errorf("after alpha is lost: %+v, %+v", devs[0], devs[2])
	}
	evs = s.events()
	if len(evs) != 1 || evs[0].dev != 0 || evs[0].String() != "connection to host lost: broken pipe" || !evs[0].fault() {
		t.Errorf("events = %+v, want alpha lost on device 0", evs)
	}
	if err := s.close(); err != nil {
		t.Errorf("close = %v", err)
	}
}

func TestRemoteErrors(t *testing.T) {
	p := new(pipeDialer)
	if _, err := openRemote([]string{"down"}, p.dial); err == nil || err.Error() != "down: connection refused" {
		t.Errorf("dial error = %v", err)
	}

	opened := make(chan error)
	go func() {
		_, err := openRemote([]string{"gamma"}, p.dial)
		opened <- err
	}()
	p.writer(t, "gamma").Write([]byte("nvtop: command not found\n"))
	if err := <-opened; err == nil || !strings.HasPrefix(err.Error(), "gamma: bad frame: ") {
		t.Errorf("bad first frame error = %v", err)
	}
}

func TestRenderRemote(t *testing.T) {
	devs := testDevices()
	devs[0].host, devs[1].host = "alpha", "beta"
	u := newUI(devs, options{ext: true, encodeHide: 30 * time.Second}, 1234, func() time.Time { return testNow })
	if u.pid != 0 {
		t.Errorf("pid = %d, want 0 to list every remote process", u.pid)
	}
	u.handle(keyF9)
	if u.overlay != overlayNone {
		t.Errorf("F9 opened the kill overlay on a remote process")
	}
	s := newScreen(120, 30, true)
	u.render(s)
	golden(t, "render/remote.golden", []byte(s.String()))
}
