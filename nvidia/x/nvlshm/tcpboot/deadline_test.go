package tcpboot

import (
	"bufio"
	"errors"
	"net"
	"strings"
	"testing"
	"time"
)

type deadlineConn struct {
	net.Conn
	failAt int
	calls  int
	writes int
	err    error
}

func (c *deadlineConn) SetDeadline(time.Time) error {
	c.calls++
	if c.calls == c.failAt {
		return c.err
	}
	return nil
}

func (c *deadlineConn) Write(p []byte) (int, error) {
	c.writes++
	return len(p), nil
}

func TestDeadlineFailureIsTerminal(t *testing.T) {
	for _, tt := range []struct {
		name   string
		failAt int
		writes int
	}{
		{"set", 1, 0},
		{"clear", 2, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			want := errors.New("deadline failure")
			conn := &deadlineConn{failAt: tt.failAt, err: want}
			b := &Bootstrap{RoundTimeout: time.Second, conn: conn, r: bufio.NewReader(strings.NewReader("1 \n"))}
			err := b.Barrier()
			if !errors.Is(err, want) {
				t.Fatalf("Barrier = %v, want %v", err, want)
			}
			if got := b.Barrier(); got != err {
				t.Fatalf("second Barrier = %v, want original %v", got, err)
			}
			if conn.calls != tt.failAt || conn.writes != tt.writes || b.seq != 1 {
				t.Fatalf("after failure: deadlines=%d writes=%d seq=%d", conn.calls, conn.writes, b.seq)
			}
		})
	}
}
