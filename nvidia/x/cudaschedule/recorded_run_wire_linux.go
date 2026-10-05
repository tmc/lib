//go:build linux

package cudaschedule

import (
	"context"
	"fmt"
	"net"
	"syscall"
	"time"
)

func recordedWireDeadline(ctx context.Context) (time.Time, error) {
	if err := ctx.Err(); err != nil {
		return time.Time{}, err
	}
	end, ok := ctx.Deadline()
	if !ok {
		return time.Time{}, fmt.Errorf("recorded helper: wire requires deadline")
	}
	next := time.Now().Add(25 * time.Millisecond)
	if end.Before(next) {
		next = end
	}
	return next, nil
}

func recordedReceive(ctx context.Context, conn *net.UnixConn, nonce string) (recordedMessage, error) {
	var buffer [recordedMessageLimit + 1]byte
	for {
		deadline, err := recordedWireDeadline(ctx)
		if err != nil {
			return recordedMessage{}, err
		}
		if err := conn.SetReadDeadline(deadline); err != nil {
			return recordedMessage{}, err
		}
		n, _, flags, _, err := conn.ReadMsgUnix(buffer[:], nil)
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			return recordedMessage{}, fmt.Errorf("recorded helper: receive: %w", err)
		}
		if flags&syscall.MSG_TRUNC != 0 || n > recordedMessageLimit {
			return recordedMessage{}, fmt.Errorf("recorded helper: truncated or oversized packet")
		}
		return readRecordedMessage(buffer[:n], nonce)
	}
}

func recordedSend(ctx context.Context, conn *net.UnixConn, msg recordedMessage) error {
	packet, err := writeRecordedMessage(msg)
	if err != nil {
		return err
	}
	for {
		deadline, err := recordedWireDeadline(ctx)
		if err != nil {
			return err
		}
		if err := conn.SetWriteDeadline(deadline); err != nil {
			return err
		}
		n, err := conn.Write(packet)
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() && n == 0 {
				continue
			}
			return fmt.Errorf("recorded helper: send: %w", err)
		}
		if n != len(packet) {
			return fmt.Errorf("recorded helper: partial packet write")
		}
		return nil
	}
}

func recordedAccept(ctx context.Context, listener *net.UnixListener) (*net.UnixConn, error) {
	for {
		deadline, err := recordedWireDeadline(ctx)
		if err != nil {
			return nil, err
		}
		if err := listener.SetDeadline(deadline); err != nil {
			return nil, err
		}
		conn, err := listener.AcceptUnix()
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			return nil, fmt.Errorf("recorded helper: accept: %w", err)
		}
		return conn, nil
	}
}
