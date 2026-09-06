//go:build !windows

package nvlshm

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// fdExchangeTimeout bounds each receive during the exchange.
const fdExchangeTimeout = 120 // seconds

// exchangeFDs passes one file descriptor from every rank in senderRanks to
// every rank, over AF_UNIX datagram sockets in a rank-0-owned temporary
// directory. It is a port of MoonEP's _exchange_ipc_fds.
//
// localFD is this rank's descriptor to send, or -1 if this rank is not a
// sender. The returned map holds one received descriptor per sender rank
// (including this rank's own duplicate, when it is a sender); the caller owns
// them and must close each after import. The caller may close localFD as soon
// as exchangeFDs returns: the trailing barrier guarantees all peers have
// received their copies.
//
// Sender identity travels in the 4-byte little-endian rank payload of each
// datagram, not in the socket address.
func exchangeFDs(b Bootstrap, localFD int, senderRanks []int) (map[int]int, error) {
	rank, world := b.Rank(), b.WorldSize()

	sending := localFD >= 0
	for _, s := range senderRanks {
		if s < 0 || s >= world {
			return nil, fmt.Errorf("nvlshm: sender rank %d out of range [0,%d)", s, world)
		}
		if s == rank && !sending {
			return nil, fmt.Errorf("nvlshm: rank %d listed as sender but localFD < 0", rank)
		}
	}

	// Rank 0 creates the socket directory; everyone learns its path.
	var dir string
	var err error
	if rank == 0 {
		dir, err = os.MkdirTemp("", "nvlshm_ipc_")
		if err != nil {
			return nil, fmt.Errorf("nvlshm: create ipc dir: %w", err)
		}
	}
	dir, err = b.BroadcastString(dir, 0)
	if err != nil {
		return nil, fmt.Errorf("nvlshm: broadcast ipc dir: %w", err)
	}
	defer func() {
		if rank == 0 {
			os.RemoveAll(dir)
		}
	}()

	sock, err := unix.Socket(unix.AF_UNIX, unix.SOCK_DGRAM, 0)
	if err != nil {
		return nil, fmt.Errorf("nvlshm: socket: %w", err)
	}
	defer unix.Close(sock)

	tv := unix.Timeval{Sec: fdExchangeTimeout}
	if err := unix.SetsockoptTimeval(sock, unix.SOL_SOCKET, unix.SO_RCVTIMEO, &tv); err != nil {
		return nil, fmt.Errorf("nvlshm: set receive timeout: %w", err)
	}

	sockPath := filepath.Join(dir, fmt.Sprintf("rank_%d", rank))
	if err := unix.Bind(sock, &unix.SockaddrUnix{Name: sockPath}); err != nil {
		return nil, fmt.Errorf("nvlshm: bind %s: %w", sockPath, err)
	}

	// All sockets must be bound before anyone sends.
	if err := b.Barrier(); err != nil {
		return nil, fmt.Errorf("nvlshm: pre-send barrier: %w", err)
	}

	received := make(map[int]int, len(senderRanks))
	closeReceived := func() {
		for _, fd := range received {
			unix.Close(fd)
		}
	}

	if sending {
		var payload [4]byte
		binary.LittleEndian.PutUint32(payload[:], uint32(rank))
		rights := unix.UnixRights(localFD)
		for r := 0; r < world; r++ {
			to := &unix.SockaddrUnix{Name: filepath.Join(dir, fmt.Sprintf("rank_%d", r))}
			if err := unix.Sendmsg(sock, payload[:], rights, to, 0); err != nil {
				closeReceived()
				return nil, fmt.Errorf("nvlshm: send fd to rank %d: %w", r, err)
			}
		}
	}

	buf := make([]byte, 16)
	oob := make([]byte, unix.CmsgSpace(4))
	for len(received) < len(senderRanks) {
		n, oobn, _, _, err := unix.Recvmsg(sock, buf, oob, 0)
		if err != nil {
			closeReceived()
			return nil, fmt.Errorf("nvlshm: receive fd (%d/%d received): %w",
				len(received), len(senderRanks), err)
		}
		if n < 4 {
			closeReceived()
			return nil, fmt.Errorf("nvlshm: short fd datagram: %d bytes", n)
		}
		sender := int(int32(binary.LittleEndian.Uint32(buf[:4])))

		msgs, err := unix.ParseSocketControlMessage(oob[:oobn])
		if err != nil {
			closeReceived()
			return nil, fmt.Errorf("nvlshm: parse control message from rank %d: %w", sender, err)
		}
		var fds []int
		for _, m := range msgs {
			got, err := unix.ParseUnixRights(&m)
			if err != nil {
				closeReceived()
				return nil, fmt.Errorf("nvlshm: parse rights from rank %d: %w", sender, err)
			}
			fds = append(fds, got...)
		}
		if len(fds) != 1 {
			for _, fd := range fds {
				unix.Close(fd)
			}
			closeReceived()
			return nil, fmt.Errorf("nvlshm: expected 1 fd from rank %d, got %d", sender, len(fds))
		}
		if _, dup := received[sender]; dup {
			unix.Close(fds[0])
			closeReceived()
			return nil, fmt.Errorf("nvlshm: duplicate fd from rank %d", sender)
		}
		received[sender] = fds[0]
	}

	// All ranks have their fds before senders close originals and rank 0
	// removes the directory.
	if err := b.Barrier(); err != nil {
		closeReceived()
		return nil, fmt.Errorf("nvlshm: post-receive barrier: %w", err)
	}
	return received, nil
}
