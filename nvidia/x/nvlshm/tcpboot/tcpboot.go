// Package tcpboot provides a minimal TCP implementation of nvlshm.Bootstrap
// for single-host multi-process jobs and tests. Rank 0 runs a hub that all
// ranks (including rank 0 itself) connect to; every collective operation is
// one lockstep round: each rank sends one framed message, the hub combines
// them, and every rank receives the result.
//
// It is intentionally simple — no reconnects, no elasticity. Production
// deployments should implement nvlshm.Bootstrap over their launcher's
// rendezvous (MPI, a collective library, etc.).
//
// # Failure isolation
//
// A collective blocks until every rank contributes, so one rank that dies or
// stalls would otherwise hang the whole world. Set RoundTimeout before the
// first collective to bound each round: a rank whose round expires returns an
// error and, once it closes, releases the ranks still waiting on it.
// Each peer must set its own timeout; a timeout does not close the connection.
package tcpboot

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"
)

const dialTimeout = 30 * time.Second

// A Bootstrap coordinates world ranks through a hub on rank 0.
// It implements nvlshm.Bootstrap.
// Collective calls and Close must not run concurrently.
type Bootstrap struct {
	// RoundTimeout bounds each collective round. Zero means unbounded, in
	// which case a dead or stalled rank blocks its peers indefinitely. Set it
	// before the first collective and do not modify it afterwards.
	RoundTimeout time.Duration

	rank  int
	world int
	seq   int
	conn  net.Conn
	r     *bufio.Reader
	hub   *hub // non-nil on rank 0
	// failed records the error that broke lockstep. A stream that has lost a
	// round is unusable: the peer may still deliver the missed reply, so a
	// later round would read it as its own.
	failed error
}

// New connects rank rank of world ranks through a hub at addr
// (e.g. "127.0.0.1:29500"). Rank 0 starts the hub and also connects to it.
// All ranks must call New; it returns once this rank's connection is
// established.
func New(rank, world int, addr string) (*Bootstrap, error) {
	if world <= 0 || rank < 0 || rank >= world {
		return nil, fmt.Errorf("tcpboot: rank %d out of range [0,%d)", rank, world)
	}
	b := &Bootstrap{rank: rank, world: world}
	if rank == 0 {
		h, err := newHub(addr, world)
		if err != nil {
			return nil, err
		}
		b.hub = h
	}
	deadline := time.Now().Add(dialTimeout)
	var conn net.Conn
	var err error
	for {
		conn, err = net.DialTimeout("tcp", addr, time.Until(deadline))
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("tcpboot: dial %s: %w", addr, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
	if _, err := fmt.Fprintf(conn, "hello %d\n", rank); err != nil {
		conn.Close()
		return nil, fmt.Errorf("tcpboot: hello: %w", err)
	}
	b.conn = conn
	b.r = bufio.NewReader(conn)
	return b, nil
}

// Rank returns this process's rank.
func (b *Bootstrap) Rank() int { return b.rank }

// WorldSize returns the number of ranks.
func (b *Bootstrap) WorldSize() int { return b.world }

// round runs one lockstep collective round, contributing payload and
// returning the hub's combined result (the unique non-empty payload, or "").
func (b *Bootstrap) round(payload string) (string, error) {
	if b.failed != nil {
		return "", b.failed
	}
	b.seq++
	if b.RoundTimeout > 0 {
		if err := b.conn.SetDeadline(time.Now().Add(b.RoundTimeout)); err != nil {
			return "", b.fail(fmt.Errorf("tcpboot: set round %d deadline: %w", b.seq, err))
		}
	}
	enc := base64.StdEncoding.EncodeToString([]byte(payload))
	if _, err := fmt.Fprintf(b.conn, "%d %s\n", b.seq, enc); err != nil {
		return "", b.fail(fmt.Errorf("tcpboot: send round %d: %w", b.seq, err))
	}
	line, err := b.r.ReadString('\n')
	if err != nil {
		return "", b.fail(fmt.Errorf("tcpboot: receive round %d: %w", b.seq, err))
	}
	seqStr, encResp, _ := strings.Cut(strings.TrimSpace(line), " ")
	seq, err := strconv.Atoi(seqStr)
	if err != nil || seq != b.seq {
		return "", b.fail(fmt.Errorf("tcpboot: round mismatch: sent %d, got %q", b.seq, line))
	}
	resp, err := base64.StdEncoding.DecodeString(encResp)
	if err != nil {
		return "", b.fail(fmt.Errorf("tcpboot: decode round %d: %w", b.seq, err))
	}
	if b.RoundTimeout > 0 {
		if err := b.conn.SetDeadline(time.Time{}); err != nil {
			return "", b.fail(fmt.Errorf("tcpboot: clear round %d deadline: %w", b.seq, err))
		}
	}
	return string(resp), nil
}

// fail records err as the reason lockstep broke and returns it.
func (b *Bootstrap) fail(err error) error {
	b.failed = err
	return err
}

// Barrier blocks until every rank has called Barrier.
func (b *Bootstrap) Barrier() error {
	_, err := b.round("")
	return err
}

// BroadcastString sends s from root to all ranks and returns root's value.
func (b *Bootstrap) BroadcastString(s string, root int) (string, error) {
	if b.rank == root {
		return b.round(s)
	}
	return b.round("")
}

// Close closes this rank's connection and, on rank 0, shuts down the hub.
func (b *Bootstrap) Close() error {
	err := b.conn.Close()
	if b.hub != nil {
		b.hub.close()
	}
	return err
}

// hub is rank 0's coordination server: one goroutine that accepts world
// connections and then relays lockstep rounds.
type hub struct {
	ln    net.Listener
	done  chan struct{}
	conns []net.Conn
}

func newHub(addr string, world int) (*hub, error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("tcpboot: listen %s: %w", addr, err)
	}
	h := &hub{ln: ln, done: make(chan struct{}), conns: make([]net.Conn, world)}
	go h.run(world)
	return h, nil
}

func (h *hub) run(world int) {
	defer func() {
		for _, c := range h.conns {
			if c != nil {
				c.Close()
			}
		}
	}()

	readers := make([]*bufio.Reader, world)
	for range world {
		conn, err := h.ln.Accept()
		if err != nil {
			return // closed
		}
		r := bufio.NewReader(conn)
		line, err := r.ReadString('\n')
		var rank int
		if err != nil || len(line) < 7 || line[:6] != "hello " {
			conn.Close()
			return
		}
		rank, err = strconv.Atoi(strings.TrimSpace(line[6:]))
		if err != nil || rank < 0 || rank >= world || h.conns[rank] != nil {
			conn.Close()
			return
		}
		h.conns[rank] = conn
		readers[rank] = r
	}

	// Lockstep rounds: read one line from every rank, reply the combined
	// payload (the unique non-empty one) to every rank.
	for {
		var seq, payload string
		for rank, r := range readers {
			line, err := r.ReadString('\n')
			if err != nil {
				return // a rank left; the others will see EOF
			}
			s, p, _ := strings.Cut(strings.TrimSpace(line), " ")
			if rank == 0 {
				seq = s
			} else if s != seq {
				return // ranks out of lockstep; abort
			}
			if dec, err := base64.StdEncoding.DecodeString(p); err == nil && len(dec) > 0 {
				payload = p
			}
		}
		for _, c := range h.conns {
			if _, err := fmt.Fprintf(c, "%s %s\n", seq, payload); err != nil {
				return
			}
		}
		select {
		case <-h.done:
			return
		default:
		}
	}
}

func (h *hub) close() {
	close(h.done)
	h.ln.Close()
}
