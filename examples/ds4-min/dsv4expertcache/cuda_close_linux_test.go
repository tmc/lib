//go:build linux

package dsv4expertcache

import (
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

type strictCUDAUploadClose struct {
	*fakeCUDAUpload
	trace        []string
	destroyed    bool
	restoreFault bool
}

func (b *strictCUDAUploadClose) drain() error {
	if b.destroyed {
		return errors.New("synchronize destroyed stream")
	}
	b.trace = append(b.trace, "drain")
	return b.fakeCUDAUpload.drain()
}

func (b *strictCUDAUploadClose) freeSlab(i int) error {
	b.trace = append(b.trace, []string{"slab0", "slab1", "slab2"}[i])
	return b.fakeCUDAUpload.freeSlab(i)
}

func (b *strictCUDAUploadClose) freeStage() error {
	b.trace = append(b.trace, "host")
	return b.fakeCUDAUpload.freeStage()
}

func (b *strictCUDAUploadClose) destroyStream() error {
	b.trace = append(b.trace, "destroy")
	if b.destroyed {
		return nil
	}
	if err := b.fakeCUDAUpload.destroyStream(); err != nil {
		return err
	}
	b.destroyed = true
	if b.restoreFault {
		b.restoreFault = false
		return errors.New("destroy context restore")
	}
	return nil
}

func TestCUDAUploadOrderedClose(t *testing.T) {
	b := &strictCUDAUploadClose{fakeCUDAUpload: fakeUpload(), restoreFault: true}
	u, err := newCUDAUpload(b, 2, [3]int{2, 2, 2}, 12)
	if err != nil {
		t.Fatal(err)
	}
	b.fail["free-slab1"] = 1
	if err = u.Close(); err == nil || !u.closing || !u.drained || u.closed {
		t.Fatal("partial cleanup admission", err)
	}
	if !reflect.DeepEqual(b.trace, []string{"drain", "slab0", "slab1"}) || b.live["slab0"] || !b.live["slab1"] || !b.live["slab2"] || !b.live["host"] || b.destroyed {
		t.Fatal("cleanup crossed failed owner", b.trace, b.live)
	}
	if _, err = u.Submit(Upload{Slot: 0}, 0, b.buf); err == nil {
		t.Fatal("submit while closing")
	}
	l, _ := cache(t).Acquire([]Key{key(0)})
	if err = u.Stage(source(t), l, false); err == nil {
		t.Fatal("stage while closing")
	}
	if err = l.Abort(); err != nil {
		t.Fatal(err)
	}
	b.trace = nil
	if err = u.Close(); err == nil || !b.destroyed || u.closed {
		t.Fatal("destroy restoration error lost", err)
	}
	if !reflect.DeepEqual(b.trace, []string{"slab0", "slab1", "slab2", "host", "destroy"}) {
		t.Fatal("retry re-drained or reordered", b.trace)
	}
	b.trace = nil
	if err = u.Close(); err != nil || !u.closed {
		t.Fatal("terminal retry failed", err)
	}
	if !reflect.DeepEqual(b.trace, []string{"slab0", "slab1", "slab2", "host", "destroy"}) || b.calls["drain"] != 1 || b.calls["free-stream"] != 1 {
		t.Fatal("destroyed stream reused or double freed", b.trace, b.calls)
	}
	before := len(b.trace)
	if err = u.Close(); err != nil || len(b.trace) != before {
		t.Fatal("closed owner ran operations")
	}
}

func TestCUDAUploadDeviceDestroyOwnership(t *testing.T) {
	for _, consumed := range []bool{false, true} {
		t.Run(map[bool]string{false: "raw-failure", true: "restore-failure"}[consumed], func(t *testing.T) {
			stream := new(cuda.Stream)
			b := &cudaUploadDevice{stream: stream}
			cause := errors.New("destroy failure")
			destroys := 0
			destroy := func(got *cuda.Stream) error {
				if got != stream {
					t.Fatal("wrong stream")
				}
				destroys++
				return cause
			}
			live := func(got *cuda.Stream) bool {
				if got != stream {
					t.Fatal("wrong lifetime probe")
				}
				return !consumed
			}
			if err := b.destroyStreamWithOperations(destroy, live); !errors.Is(err, cause) || (b.stream == nil) != consumed {
				t.Fatal("stream status lost", err, b.stream)
			}
			if consumed {
				if err := b.destroyStreamWithOperations(destroy, live); err != nil || destroys != 1 {
					t.Fatal("consumed stream destroyed twice", err, destroys)
				}
			} else {
				if err := b.destroyStreamWithOperations(func(*cuda.Stream) error { destroys++; return nil }, live); err != nil || b.stream != nil || destroys != 2 {
					t.Fatal("raw failure could not retry", err, destroys)
				}
			}
		})
	}
}
