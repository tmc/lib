package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"time"
)

// A remoteSource reads the devices of other machines, each from an nvtop
// recording to its standard output there: nvtop -x -record -.
type remoteSource struct {
	hosts []*remoteHost
	devs  []*device
	of    []*remoteHost // the host of each device
}

// A remoteHost is one machine's stream of frames.
type remoteHost struct {
	name   string
	first  int // index of its first device
	stream io.ReadCloser
	ready  chan error // receives once the first frame is read, or not

	mu      sync.Mutex
	cur     *frame
	pending []wireEvent // device indexes are the host's
	closing bool
}

// A dialer starts the stream of frames from host.
type dialer func(host string) (io.ReadCloser, error)

// eventHostLost is the event a remote host's devices report when its
// stream ends. It is not an NVML event type.
const eventHostLost = 1 << 63

// firstFrameTimeout bounds the wait for each host's first frame, which
// takes an ssh connection and an NVML start.
const firstFrameTimeout = 30 * time.Second

// openRemote starts streams from hosts and waits for the first frame of
// each, which gives its devices.
func openRemote(hosts []string, dial dialer) (*remoteSource, error) {
	s := new(remoteSource)
	for _, name := range hosts {
		stream, err := dial(name)
		if err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		h := &remoteHost{name: name, stream: stream, ready: make(chan error, 1)}
		s.hosts = append(s.hosts, h)
		go h.read()
	}
	timeout := time.After(firstFrameTimeout)
	for _, h := range s.hosts {
		var err error
		select {
		case err = <-h.ready:
		case <-timeout:
			err = errors.New("no data")
		}
		if err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %w", h.name, err)
		}
		h.first = len(s.devs)
		h.mu.Lock()
		f := h.cur
		h.mu.Unlock()
		for i := range f.Devices {
			d := fromWire(&f.Devices[i], len(s.devs))
			d.host = h.name
			s.devs = append(s.devs, &d)
			s.of = append(s.of, h)
		}
	}
	return s, nil
}

// read reads frames until the stream ends.
func (h *remoteHost) read() {
	lines := bufio.NewScanner(h.stream)
	lines.Buffer(nil, 64<<20)
	first := true
	var err error
	for lines.Scan() {
		var f frame
		if err = json.Unmarshal(lines.Bytes(), &f); err != nil {
			err = fmt.Errorf("bad frame: %w", err)
			break
		}
		h.mu.Lock()
		h.cur = &f
		h.pending = append(h.pending, f.Events...)
		h.mu.Unlock()
		if first {
			h.ready <- nil
			first = false
		}
	}
	if err == nil {
		err = lines.Err()
	}
	if err == nil {
		err = errors.New("connection closed")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.closing {
		return
	}
	h.cur = nil
	if first {
		h.ready <- err
		return
	}
	h.pending = append(h.pending, wireEvent{Time: time.Now(), Kind: eventHostLost, Msg: err.Error()})
}

func (s *remoteSource) devices() []*device { return s.devs }

// refresh sets d as its host's latest frame has it, or, if the host is
// lost, reporting nothing.
func (s *remoteSource) refresh(d *device) {
	h := s.of[d.index]
	h.mu.Lock()
	defer h.mu.Unlock()
	if i := d.index - h.first; h.cur != nil && i < len(h.cur.Devices) {
		*d = fromWire(&h.cur.Devices[i], d.index)
		d.host = h.name
		return
	}
	d.resetDynamic()
	d.procs = nil
}

// processes does nothing: refresh sets the processes too.
func (s *remoteSource) processes(d *device) {}

func (s *remoteSource) events() []event {
	var evs []event
	for _, h := range s.hosts {
		h.mu.Lock()
		for _, e := range h.pending {
			evs = append(evs, event{at: e.Time, dev: h.first + e.Dev, kind: e.Kind, data: e.Data, msg: e.Msg})
		}
		h.pending = nil
		h.mu.Unlock()
	}
	slices.SortStableFunc(evs, func(a, b event) int { return a.at.Compare(b.at) })
	return evs
}

// close ends the streams. A host lost before has been reported as an
// event, so it is not an error here.
func (s *remoteSource) close() error {
	for _, h := range s.hosts {
		h.mu.Lock()
		h.closing = true
		h.mu.Unlock()
		h.stream.Close()
	}
	return nil
}

// sshDialer returns a dialer that runs cmd (by default nvtop) on each
// host with ssh, asking it to record every delay.
func sshDialer(cmd string, delay time.Duration) dialer {
	if cmd == "" {
		cmd = "nvtop"
	}
	cmd += " -x -record -"
	if delay > 0 {
		cmd += fmt.Sprintf(" -d %d", delay/(100*time.Millisecond))
	}
	return func(host string) (io.ReadCloser, error) {
		c := exec.Command("ssh", "-T", "-o", "BatchMode=yes", host, cmd)
		c.WaitDelay = time.Second
		s := &sshStream{cmd: c}
		c.Stderr = &s.stderr
		out, err := c.StdoutPipe()
		if err != nil {
			return nil, err
		}
		s.out = out
		if err := c.Start(); err != nil {
			return nil, err
		}
		return s, nil
	}
}

// An sshStream is the standard output of an ssh command. At its end it
// reports how the command failed, with the last line it wrote to
// standard error.
type sshStream struct {
	cmd    *exec.Cmd
	out    io.Reader
	stderr tail
	once   sync.Once
	err    error
}

func (s *sshStream) Read(p []byte) (int, error) {
	n, err := s.out.Read(p)
	if err == io.EOF {
		if werr := s.wait(); werr != nil {
			err = werr
		}
	}
	return n, err
}

func (s *sshStream) wait() error {
	s.once.Do(func() {
		if err := s.cmd.Wait(); err != nil {
			msg := strings.TrimSpace(string(s.stderr.b))
			if i := strings.LastIndexByte(msg, '\n'); i >= 0 {
				msg = msg[i+1:]
			}
			if msg != "" {
				err = fmt.Errorf("%v: %s", err, msg)
			}
			s.err = err
		}
	})
	return s.err
}

func (s *sshStream) Close() error {
	s.cmd.Process.Kill()
	s.wait()
	return nil
}

// A tail keeps the last few kilobytes written to it.
type tail struct {
	mu sync.Mutex
	b  []byte
}

func (t *tail) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.b = append(t.b, p...)
	if n := len(t.b) - 4096; n > 0 {
		t.b = t.b[n:]
	}
	return len(p), nil
}
