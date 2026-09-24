package cuptisass

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// A Channel is the host end of NVBit's device-to-host channel (ChannelHost in
// core/utils/channel.hpp). Device threads append records to a buffer, and when
// it fills or is flushed the device stores the number of bytes written in a
// doorbell word and waits for the host to set it back to zero. The host never
// takes a lock: the doorbell is the only word both sides write, and each side
// writes it only when the other is waiting for it.
//
// A Channel has one receiver; it is not safe for concurrent use.
type Channel struct {
	doorbell *int32
	buf      []byte
	head     int // read offset into buf of the batch being received
}

// NewChannel returns the host end of a channel whose device end writes records
// into buf and batch sizes into doorbell. Both must stay valid, at the same
// addresses, for as long as the Channel is used. For a live device they are
// page-locked host memory mapped into the device's address space; see
// [CheckDeviceChannel].
func NewChannel(doorbell *int32, buf []byte) (*Channel, error) {
	if doorbell == nil {
		return nil, fmt.Errorf("cuptisass: channel: nil doorbell")
	}
	if len(buf) == 0 || len(buf) > math.MaxInt32 {
		return nil, fmt.Errorf("cuptisass: channel: buffer of %d bytes, want 1 to %d", len(buf), math.MaxInt32)
	}
	return &Channel{doorbell: doorbell, buf: buf}, nil
}

// Recv copies up to len(p) bytes of the batch the device has signalled and
// returns how many it copied, or 0 if the doorbell is clear. As in NVBit's
// ChannelHost::recv, it then stores the number of bytes still unread in the
// doorbell, so the device stays waiting until the batch is read in full, and
// the next Recv continues where this one stopped.
//
// A doorbell that is negative or larger than the buffer left to read is an
// error: the device end is not following the protocol, and nothing is copied.
func (c *Channel) Recv(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, fmt.Errorf("cuptisass: channel: empty receive buffer")
	}
	pending := atomic.LoadInt32(c.doorbell)
	if pending == 0 {
		return 0, nil
	}
	if pending < 0 || int(pending) > len(c.buf)-c.head {
		return 0, fmt.Errorf("cuptisass: channel: doorbell %d with %d bytes left to read", pending, len(c.buf)-c.head)
	}
	n := copy(p, c.buf[c.head:c.head+int(pending)])
	left := int(pending) - n
	if left > 0 {
		c.head += n
	} else {
		c.head = 0
	}
	atomic.StoreInt32(c.doorbell, int32(left))
	return n, nil
}

// A RecordLayout describes the fixed-size records a channel carries, as far as
// [Channel.Receive] needs to know them.
type RecordLayout struct {
	Size       int // bytes per record
	CTAXOffset int // byte offset of the record's little-endian int32 cta_id_x
}

// RegValsLayout is the layout of NVBit's record_reg_vals reg_info_t: six int32
// fields beginning with cta_id_x, then 32 lanes of 8 uint32 register values.
var RegValsLayout = RecordLayout{Size: 1048, CTAXOffset: 0}

// SentinelCTA is the cta_id_x of the record NVBit's record_reg_vals flush
// kernel pushes to mark the end of a context's records.
const SentinelCTA = -1

// ChannelStats counts what [Channel.Receive] delivered.
type ChannelStats struct {
	Records int64 // records passed to the callback
	Batches int64 // doorbell batches received
	Bytes   int64 // bytes received, including the sentinel record

	// Terminated reports that a sentinel record ended the stream. It is
	// false when Receive stopped for any other reason, including
	// cancellation: a stream that was cut off is not a complete one.
	Terminated bool
}

const (
	minPoll = 10 * time.Microsecond
	maxPoll = time.Millisecond
)

// Receive delivers the records of the channel to fn, in order, until a record
// whose cta_id_x is [SentinelCTA] ends the stream. The record slice is valid
// only during the call. While the doorbell is clear, Receive sleeps, backing
// off from 10µs to 1ms; it never spins on the doorbell.
//
// Each batch is copied out and the doorbell released before fn sees any of it.
// Every record before the sentinel is delivered. Bytes after the sentinel in
// the same batch, a batch that is not a whole number of records, and a
// negative cta_id_x other than the sentinel are errors. (NVBit's
// record_reg_vals receiver drops the bytes after a sentinel without a word.)
// Records without a sentinel field, such as mem_trace's, never terminate; the
// caller ends those with ctx.
//
// Receive returns what it delivered so far with the first error from Recv or
// fn, or with ctx.Err() if ctx is done first.
func (c *Channel) Receive(ctx context.Context, layout RecordLayout, fn func(record []byte) error) (ChannelStats, error) {
	var st ChannelStats
	if layout.Size <= 0 || layout.Size > len(c.buf) || layout.CTAXOffset < 0 || layout.CTAXOffset+4 > layout.Size {
		return st, fmt.Errorf("cuptisass: channel: invalid record layout %+v for a %d-byte buffer", layout, len(c.buf))
	}
	batch := make([]byte, len(c.buf))
	poll := minPoll
	timer := time.NewTimer(poll)
	defer timer.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		n, err := c.Recv(batch)
		if err != nil {
			return st, err
		}
		if n == 0 {
			timer.Reset(poll)
			select {
			case <-ctx.Done():
				return st, ctx.Err()
			case <-timer.C:
			}
			poll = min(2*poll, maxPoll)
			continue
		}
		poll = minPoll
		st.Batches++
		st.Bytes += int64(n)
		if n%layout.Size != 0 {
			return st, fmt.Errorf("cuptisass: channel: batch of %d bytes is not a whole number of %d-byte records", n, layout.Size)
		}
		for off := 0; off < n; off += layout.Size {
			rec := batch[off : off+layout.Size]
			switch cta := int32(binary.LittleEndian.Uint32(rec[layout.CTAXOffset:])); {
			case cta == SentinelCTA:
				if rest := n - off - layout.Size; rest > 0 {
					return st, fmt.Errorf("cuptisass: channel: %d bytes follow the sentinel record", rest)
				}
				st.Terminated = true
				return st, nil
			case cta < 0:
				return st, fmt.Errorf("cuptisass: channel: record %d has cta_id_x %d, which is neither a CTA nor the sentinel", st.Records, cta)
			}
			if err := fn(rec); err != nil {
				return st, err
			}
			st.Records++
		}
	}
}

// CheckDeviceChannel reports whether a device channel on target, an
// architecture such as "sm_121a", has been qualified by a native run.
//
// No native run has qualified one on any target yet, so for the targets NVBit
// supports (compute capability 3.5 through 12.1, whatever the suffix) it
// returns an error wrapping [cudaerrors.ErrHardwarePending], whatever hardware
// this host has. For any other target it returns an error wrapping
// [cudaerrors.ErrUnsupported].
func CheckDeviceChannel(target string) error {
	cc, ok := smCapability(target)
	if !ok || cc < 35 || cc > 121 {
		return fmt.Errorf("cuptisass: device channel on %q: NVBit supports sm_35 to sm_121: %w", target, cudaerrors.ErrUnsupported)
	}
	return fmt.Errorf("cuptisass: device channel on %s: no native run has qualified it: %w", target, cudaerrors.ErrHardwarePending)
}

// smCapability parses "sm_" followed by two or three digits and an optional
// "a" or "f" suffix, returning the digits as a number.
func smCapability(target string) (int, bool) {
	s, ok := strings.CutPrefix(target, "sm_")
	if !ok {
		return 0, false
	}
	s = strings.TrimRight(s, "af")
	if len(target)-len("sm_")-len(s) > 1 || len(s) < 2 || len(s) > 3 {
		return 0, false
	}
	n, err := strconv.Atoi(s)
	if err != nil || s[0] == '-' || s[0] == '+' {
		return 0, false
	}
	return n, true
}
