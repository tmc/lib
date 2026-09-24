package cuptisass_test

import (
	"context"
	"encoding/binary"
	"errors"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaliveness"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// devChannel is a Go model of NVBit's ChannelDev (core/utils/channel.hpp:60-124),
// with goroutines for warps. It is the device end the tests drive the host
// end with; the real one is CUDA code, exercised only by the native test.
type devChannel struct {
	buf        []byte
	doorbell   *int32
	head, tail atomic.Uint64 // offsets of buff_write_head_ptr and buff_write_tail_ptr
}

// push is ChannelDev::push: claim space with an atomic add on head; the first
// warp to overrun the buffer waits for the writers before it and flushes, the
// rest wait for the flush to reset head, and all of them retry.
func (d *devChannel) push(rec []byte) {
	n := uint64(len(rec))
	end := uint64(len(d.buf))
	for {
		curr := d.head.Add(n) - n
		if curr+n > end {
			if curr <= end {
				for d.tail.Load() != curr {
					runtime.Gosched()
				}
				d.flush()
			} else {
				for d.head.Load() > end {
					runtime.Gosched()
				}
			}
			continue
		}
		copy(d.buf[curr:], rec)
		d.tail.Add(n)
		return
	}
}

// flush is ChannelDev::flush: ring the doorbell with the bytes written, wait
// for the host to clear it, then reset tail and head.
func (d *devChannel) flush() {
	n := d.tail.Load()
	if n == 0 {
		return
	}
	atomic.StoreInt32(d.doorbell, int32(n))
	for atomic.LoadInt32(d.doorbell) != 0 {
		runtime.Gosched()
	}
	d.tail.Store(0)
	d.head.Store(0)
}

// layout16 is a 16-byte record: int32 cta_id_x, uint32 id, 8 bytes of padding.
var layout16 = cuptisass.RecordLayout{Size: 16, CTAXOffset: 0}

func record16(cta int32, id uint32) []byte {
	b := make([]byte, 16)
	binary.LittleEndian.PutUint32(b[0:], uint32(cta))
	binary.LittleEndian.PutUint32(b[4:], id)
	return b
}

func newPair(t *testing.T, size int) (*devChannel, *cuptisass.Channel) {
	t.Helper()
	d := &devChannel{buf: make([]byte, size), doorbell: new(int32)}
	c, err := cuptisass.NewChannel(d.doorbell, d.buf)
	if err != nil {
		t.Fatal(err)
	}
	return d, c
}

// TestTelemetryChannel_RingBufferPushPop pushes records from concurrent warps
// through a buffer that holds six of them, so the channel flushes many times,
// and checks that every record arrives once.
func TestTelemetryChannel_RingBufferPushPop(t *testing.T) {
	const warps, perWarp = 8, 200
	// 100 bytes: six 16-byte records and 4 bytes no record fits in, which is
	// the overrun case push must handle rather than a buffer that divides
	// evenly.
	dev, host := newPair(t, 100)

	var wg sync.WaitGroup
	for w := range warps {
		wg.Go(func() {
			for i := range perWarp {
				dev.push(record16(int32(w), uint32(w*perWarp+i)))
			}
		})
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		// record_reg_vals' flush_channel kernel: push the sentinel, then flush.
		dev.push(record16(cuptisass.SentinelCTA, 0))
		dev.flush()
		close(done)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	var got []uint32
	st, err := host.Receive(ctx, layout16, func(rec []byte) error {
		got = append(got, binary.LittleEndian.Uint32(rec[4:]))
		return nil
	})
	if err != nil {
		t.Fatalf("Receive: %v (stats %+v)", err, st)
	}
	<-done
	slices.Sort(got)
	want := make([]uint32, warps*perWarp)
	for i := range want {
		want[i] = uint32(i)
	}
	if !slices.Equal(got, want) {
		t.Errorf("received %d records, want each of %d exactly once", len(got), len(want))
	}
	if !st.Terminated || st.Records != warps*perWarp {
		t.Errorf("stats %+v, want %d records and Terminated", st, warps*perWarp)
	}
	// 1601 records at no more than six per batch.
	if min := int64((warps*perWarp + 1 + 5) / 6); st.Batches < min {
		t.Errorf("%d batches, want at least %d for a six-record buffer", st.Batches, min)
	}
	if st.Bytes != 16*(warps*perWarp+1) {
		t.Errorf("%d bytes, want %d", st.Bytes, 16*(warps*perWarp+1))
	}
}

// TestChannelRecvPartial checks ChannelHost::recv's partial read
// (channel.hpp:234-265): the doorbell is left holding the bytes still unread,
// which keeps the device waiting, and the read offset advances until the
// batch is done and then returns to the start.
func TestChannelRecvPartial(t *testing.T) {
	doorbell := new(int32)
	buf := make([]byte, 64)
	for i := range buf {
		buf[i] = byte(i)
	}
	c, err := cuptisass.NewChannel(doorbell, buf)
	if err != nil {
		t.Fatal(err)
	}
	p := make([]byte, 20)
	if n, err := c.Recv(p); n != 0 || err != nil {
		t.Fatalf("Recv with a clear doorbell = %d, %v; want 0, nil", n, err)
	}
	atomic.StoreInt32(doorbell, 48)
	for _, step := range []struct {
		n, first, left int
	}{
		{20, 0, 28},
		{20, 20, 8},
		{8, 40, 0},
	} {
		n, err := c.Recv(p)
		if err != nil || n != step.n || p[0] != byte(step.first) {
			t.Fatalf("Recv = %d, %v, first byte %d; want %d bytes from offset %d", n, err, p[0], step.n, step.first)
		}
		if got := atomic.LoadInt32(doorbell); int(got) != step.left {
			t.Fatalf("doorbell %d after reading %d bytes, want %d left", got, n, step.left)
		}
	}
	atomic.StoreInt32(doorbell, 16)
	if n, err := c.Recv(p); n != 16 || err != nil || p[0] != 0 {
		t.Fatalf("next batch: Recv = %d, %v, first byte %d; want 16 from offset 0", n, err, p[0])
	}

	for _, bad := range []int32{-1, 65} {
		atomic.StoreInt32(doorbell, bad)
		if n, err := c.Recv(p); err == nil {
			t.Errorf("doorbell %d: Recv = %d, nil; want an error", bad, n)
		}
	}
	if _, err := cuptisass.NewChannel(nil, buf); err == nil {
		t.Error("NewChannel(nil, buf) succeeded")
	}
	if _, err := cuptisass.NewChannel(doorbell, nil); err == nil {
		t.Error("NewChannel(doorbell, nil) succeeded")
	}
}

// ring plays a device that has already written each batch: it copies one into
// the buffer, rings the doorbell and waits for it to clear before the next.
func ring(d *devChannel, batches ...[]byte) {
	for _, b := range batches {
		copy(d.buf, b)
		atomic.StoreInt32(d.doorbell, int32(len(b)))
		for atomic.LoadInt32(d.doorbell) != 0 {
			time.Sleep(time.Microsecond)
		}
	}
}

func records(recs ...[]byte) []byte { return slices.Concat(recs...) }

type result struct {
	st  cuptisass.ChannelStats
	ids []uint32
	err error
}

func receive(t *testing.T, timeout time.Duration, layout cuptisass.RecordLayout, size int, batches ...[]byte) result {
	t.Helper()
	dev, host := newPair(t, size)
	go ring(dev, batches...)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	var r result
	r.st, r.err = host.Receive(ctx, layout, func(rec []byte) error {
		r.ids = append(r.ids, binary.LittleEndian.Uint32(rec[4:]))
		return nil
	})
	return r
}

// TestTelemetryChannel_FlushSentinelDrain checks the end-of-stream protocol of
// NVBit's record_reg_vals: its flush kernel pushes a record with cta_id_x -1
// and flushes (tool_func/flush_channel.cu:5-14), and the receiver stops there.
func TestTelemetryChannel_FlushSentinelDrain(t *testing.T) {
	const size = 256
	sentinel := record16(cuptisass.SentinelCTA, 99)

	// Planted: five records and the sentinel in one batch. The five are
	// delivered, the sentinel is not, and the stream is complete.
	planted := receive(t, time.Minute, layout16, size,
		records(record16(0, 0), record16(1, 1), record16(0, 2), record16(2, 3), record16(1, 4), sentinel))
	if planted.err != nil {
		t.Fatalf("planted sentinel: %v", planted.err)
	}
	if !slices.Equal(planted.ids, []uint32{0, 1, 2, 3, 4}) {
		t.Errorf("planted sentinel: delivered %v, want 0-4 and not the sentinel's 99", planted.ids)
	}

	// Across batches: the records of an earlier flush are delivered before
	// the batch holding the sentinel.
	split := receive(t, time.Minute, layout16, size,
		records(record16(0, 0), record16(1, 1), record16(2, 2)),
		records(record16(3, 3), sentinel))
	if split.err != nil || !split.st.Terminated || !slices.Equal(split.ids, []uint32{0, 1, 2, 3}) || split.st.Batches != 2 {
		t.Errorf("sentinel in second batch: %v, stats %+v, ids %v; want 4 records in 2 batches, terminated", split.err, split.st, split.ids)
	}

	// Adjacent: cta_id_x -2 is negative but is not the sentinel. It must
	// neither end the stream nor pass as a CTA. It is last in its batch, so
	// taking it for the sentinel would end the stream cleanly.
	adjacent := receive(t, 5*time.Second, layout16, size,
		records(record16(0, 0), record16(-2, 1)), sentinel)
	if adjacent.err == nil || adjacent.st.Terminated || slices.Contains(adjacent.ids, 1) {
		t.Errorf("cta_id_x -2: %v, stats %+v, ids %v; want an error, not terminated, record 1 not delivered", adjacent.err, adjacent.st, adjacent.ids)
	}

	// Bytes after the sentinel in the same batch are reported, not dropped
	// as NVBit's receiver drops them; the records before it still arrive.
	trailing := receive(t, 5*time.Second, layout16, size,
		records(record16(0, 0), sentinel, record16(0, 1)))
	if trailing.err == nil || trailing.st.Terminated || !slices.Equal(trailing.ids, []uint32{0}) {
		t.Errorf("record after sentinel: %v, stats %+v, ids %v; want an error after delivering record 0", trailing.err, trailing.st, trailing.ids)
	}

	// No sentinel: the stream is cut off by the deadline, and that is not
	// a completed collection.
	cut := receive(t, 50*time.Millisecond, layout16, size, records(record16(0, 0), record16(0, 1)))
	if !errors.Is(cut.err, context.DeadlineExceeded) || cut.st.Terminated || cut.st.Records != 2 {
		t.Errorf("no sentinel: %v, stats %+v; want DeadlineExceeded after 2 records, not terminated", cut.err, cut.st)
	}

	// The sentinel field is where the layout says. mem_trace's mem_access_t
	// begins with a uint64 grid_launch_id, whose low word here is 0xffffffff,
	// and has cta_id_x at offset 8.
	memTrace := cuptisass.RecordLayout{Size: 16, CTAXOffset: 8}
	rec := func(cta int32, id uint32) []byte {
		b := make([]byte, 16)
		binary.LittleEndian.PutUint32(b[0:], 0xffffffff)
		binary.LittleEndian.PutUint32(b[4:], id)
		binary.LittleEndian.PutUint32(b[8:], uint32(cta))
		return b
	}
	offset := receive(t, time.Minute, memTrace, size, records(rec(3, 0), rec(4, 1), rec(cuptisass.SentinelCTA, 2)))
	if offset.err != nil || !slices.Equal(offset.ids, []uint32{0, 1}) {
		t.Errorf("cta_id_x at offset 8: %v, ids %v; want records 0 and 1", offset.err, offset.ids)
	}

	// RegValsLayout is reg_info_t: 6 int32 and 32x8 uint32.
	if want := 6*4 + 32*8*4; cuptisass.RegValsLayout.Size != want || cuptisass.RegValsLayout.CTAXOffset != 0 {
		t.Errorf("RegValsLayout = %+v, want size %d, cta_id_x at 0", cuptisass.RegValsLayout, want)
	}

	err := cudaliveness.Assert(
		cudaliveness.Planted{Site: "sentinel after five records", Reported: terminated(planted), Ran: true},
		cudaliveness.Counted{Site: "records before the sentinel", Want: 5, Got: planted.st.Records,
			Why: "five records precede the sentinel in the planted batch", Ran: true},
		cudaliveness.Adjacent{Site: "cta_id_x -2", Reported: terminated(adjacent), Ran: true},
		cudaliveness.Adjacent{Site: "no sentinel", Reported: terminated(cut), Ran: true},
	)
	if err != nil {
		t.Error(err)
	}
}

func terminated(r result) int {
	if r.st.Terminated {
		return 1
	}
	return 0
}

func TestReceiveRejectsLayout(t *testing.T) {
	_, host := newPair(t, 64)
	for _, l := range []cuptisass.RecordLayout{{Size: 0}, {Size: 128}, {Size: 16, CTAXOffset: 13}, {Size: 16, CTAXOffset: -1}} {
		if _, err := host.Receive(context.Background(), l, func([]byte) error { return nil }); err == nil {
			t.Errorf("Receive accepted layout %+v for a 64-byte buffer", l)
		}
	}
}

// TestTelemetryChannel_HardwarePendingReport checks that no target reports a
// working device channel before a native run has shown one: targets NVBit
// supports are pending, whatever this host has, and others are refused.
func TestTelemetryChannel_HardwarePendingReport(t *testing.T) {
	tests := []struct {
		target string
		want   error
	}{
		{"sm_121a", cudaerrors.ErrHardwarePending},
		{"sm_121", cudaerrors.ErrHardwarePending},
		{"sm_90a", cudaerrors.ErrHardwarePending},
		{"sm_100f", cudaerrors.ErrHardwarePending},
		{"sm_35", cudaerrors.ErrHardwarePending},
		{"sm_30", cudaerrors.ErrUnsupported},
		{"sm_122", cudaerrors.ErrUnsupported},
		{"sm_1210", cudaerrors.ErrUnsupported},
		{"sm_90af", cudaerrors.ErrUnsupported},
		{"gfx90a", cudaerrors.ErrUnsupported},
		{"", cudaerrors.ErrUnsupported},
	}
	for _, tt := range tests {
		err := cuptisass.CheckDeviceChannel(tt.target)
		if !errors.Is(err, tt.want) {
			t.Errorf("CheckDeviceChannel(%q) = %v, want %v", tt.target, err, tt.want)
		}
		other := cudaerrors.ErrUnsupported
		if tt.want == other {
			other = cudaerrors.ErrHardwarePending
		}
		if errors.Is(err, other) {
			t.Errorf("CheckDeviceChannel(%q) = %v, which also matches %v", tt.target, err, other)
		}
	}
}
