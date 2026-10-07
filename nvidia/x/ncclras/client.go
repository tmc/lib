package ncclras

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strconv"
	"strings"
	"time"
)

// Client queries an explicit host:port. Timeout bounds the entire dial,
// handshake and response and must be positive and at most one minute.
// Limits bounds the response independently of the timeout. The zero Client
// is not usable. Status issues no CONTROL commands.
type Client struct {
	Address string
	Timeout time.Duration
	Limits  Limits
}

// Status exchanges client protocol 2, configures JSON output and requests one
// status response, which the server terminates with EOF. Cancellation closes
// the connection; it does not affect the NCCL job.
func (c Client) Status(ctx context.Context) (*Status, error) {
	if ctx == nil || c.Address == "" || strings.TrimSpace(c.Address) != c.Address || c.Timeout <= 0 || c.Timeout > time.Minute {
		return nil, errors.New("ras client: address, context and bounded timeout are required")
	}
	host, port, err := net.SplitHostPort(c.Address)
	if err != nil {
		return nil, fmt.Errorf("ras address: %w", err)
	}
	n, err := strconv.Atoi(port)
	if host == "" || n < 1 || n > 65535 || err != nil || strings.ContainsAny(host, "\r\n\t ") {
		return nil, errors.New("ras address: explicit host and numeric port are required")
	}
	if _, err := c.Limits.checked(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	conn, err := (&net.Dialer{Timeout: c.Timeout}).DialContext(ctx, "tcp", c.Address)
	if err != nil {
		return nil, fmt.Errorf("ras dial: %w", err)
	}
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	deadline, _ := ctx.Deadline()
	if err := conn.SetDeadline(deadline); err != nil {
		return nil, err
	}
	r := bufio.NewReaderSize(conn, 4096)
	commands := []struct{ command, reply string }{
		{"CLIENT PROTOCOL 2\n", "SERVER PROTOCOL 2\n"},
		{"TIMEOUT " + strconv.FormatFloat(c.Timeout.Seconds(), 'g', -1, 64) + "\n", "OK\n"},
		{"SET FORMAT json\n", "OK\n"},
	}
	for _, x := range commands {
		if _, err := io.WriteString(conn, x.command); err != nil {
			return nil, clientError(ctx, err)
		}
		line, err := r.ReadSlice('\n')
		if err != nil {
			return nil, clientError(ctx, err)
		}
		if string(line) != x.reply {
			return nil, fmt.Errorf("ras protocol: unexpected response %q", line)
		}
	}
	if _, err := io.WriteString(conn, "STATUS\n"); err != nil {
		return nil, clientError(ctx, err)
	}
	s, err := Read(r, c.Limits)
	if err != nil {
		return nil, clientError(ctx, err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s, nil
}

func clientError(ctx context.Context, err error) error {
	if e := ctx.Err(); e != nil {
		return e
	}
	return fmt.Errorf("ras exchange: %w", err)
}
