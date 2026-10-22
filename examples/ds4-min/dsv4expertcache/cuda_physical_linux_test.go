//go:build linux

package dsv4expertcache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// countedCUDAUpload forwards every operation to the real backend. These counts
// observe native calls, not the stream's private cleanup callback list.
type countedCUDAUpload struct {
	cudaUploadBackend
	copies, drains, destroys int
	retainedCopies           int
}

func (b *countedCUDAUpload) copy(component, slot int, src []byte) error {
	if err := b.cudaUploadBackend.copy(component, slot, src); err != nil {
		return err
	}
	b.copies++
	b.retainedCopies++
	return nil
}
func (b *countedCUDAUpload) drain() error {
	if err := b.cudaUploadBackend.drain(); err != nil {
		return err
	}
	b.drains++
	b.retainedCopies = 0
	return nil
}
func (b *countedCUDAUpload) destroyStream() error {
	if err := b.cudaUploadBackend.destroyStream(); err != nil {
		return err
	}
	b.destroys++
	return nil
}

type nativeConsumer struct {
	event    *cuda.Event
	failOnce bool
}

var errConsumerControl = errors.New("injected consumer completion error")

func (c *nativeConsumer) Wait() error {
	if c.failOnce {
		c.failOnce = false
		return errConsumerControl
	}
	return c.event.Synchronize(context.Background())
}

type drainedConsumer struct{ backend cudaUploadBackend }

func (c drainedConsumer) Wait() error { return c.backend.drain() }

func TestCUDAUploadPhysicalTriplets(t *testing.T) {
	gate := os.Getenv("DS4_CUDA_UPLOAD_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_UPLOAD_GATE is not set")
	}
	if gate != "1" {
		t.Fatal("DS4_CUDA_UPLOAD_GATE must be 1")
	}
	pid, err := strconv.Atoi(os.Getenv("GPU_LOCK_HELD"))
	if err != nil || pid <= 1 || syscall.Kill(pid, 0) != nil {
		t.Fatal("live gpu-lock holder is required")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	exeBytes, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("build binary_sha256=%x go=%s lock_pid=%d", sha256.Sum256(exeBytes), runtime.Version(), pid)

	device, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	var upload *CUDAUpload
	var consumers []*cuda.Event
	t.Cleanup(func() {
		if upload != nil {
			if upload.lease != nil && !upload.lease.done {
				if err := upload.lease.Release(drainedConsumer{upload.backend}); err != nil {
					t.Errorf("terminal consumer drain: %v", err)
					return // Keep the owner rather than freeing active allocations.
				}
			}
			if err := upload.Close(); err != nil {
				t.Errorf("terminal uploader close: %v", err)
				return
			}
		}
		for _, event := range consumers {
			if err := event.Close(); err != nil {
				t.Errorf("terminal consumer event destroy: %v", err)
				return
			}
		}
		if err := device.Close(); err != nil {
			t.Errorf("terminal device close: %v", err)
		}
	})
	name, err := device.Name()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := device.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		t.Fatalf("requires admitted DGX SM 12.1, got %d.%d", major, minor)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	var free, total uint
	if err := device.WithContext(func() error { var e error; free, total, e = driver.MemGetInfo(); return e }); err != nil {
		t.Fatal(err)
	}
	if free < 64<<20 {
		t.Fatalf("memory admission: %d available, need 64 MiB", free)
	}
	t.Logf("device ordinal=0 name=%q sm=%d.%d driver_api=%d free=%d total=%d", name, major, minor, version, free, total)

	strides := [3]int{17, 31, 47}
	offsets := [3]int{0, 68, 192}
	var payload []byte
	for component, n := range strides {
		for expert := 0; expert < 4; expert++ {
			for i := 0; i < n; i++ {
				payload = append(payload, byte((component*73+expert*41+i*i*7+i*13+11)%251))
			}
		}
	}
	if len(payload) != 380 {
		t.Fatal("fixture extent mismatch")
	}
	id := Identity{sha256.Sum256(payload), sha256.Sum256([]byte("raw-byte-fixture-v1")), sha256.Sum256([]byte("strides=17,31,47;experts=4;slots=2"))}
	source, err := NewSource(bytes.NewReader(payload), uint64(len(payload)), id, map[int]Triplet{0: {
		Gate: Tensor{0, 68, 17, 4}, Up: Tensor{68, 124, 31, 4}, Down: Tensor{192, 188, 47, 4},
	}})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := New(id, 2, 95, 190)
	if err != nil {
		t.Fatal(err)
	}
	upload, err = NewCUDAUpload(device, 2, strides, 190)
	if err != nil {
		t.Fatal(err)
	}
	actual, ok := upload.backend.(*cudaUploadDevice)
	if !ok {
		t.Fatal("physical gate requires real CUDA backend")
	}
	if actual.host.Len() != 47 {
		t.Fatalf("pinned stage length %d, want 47", actual.host.Len())
	}
	counter := &countedCUDAUpload{cudaUploadBackend: actual}
	upload.backend = counter
	pointers := [4]uintptr{actual.host.Ptr(), uintptr(actual.slabs[0].Ptr()), uintptr(actual.slabs[1].Ptr()), uintptr(actual.slabs[2].Ptr())}
	for _, ptr := range pointers {
		if ptr == 0 {
			t.Fatal("missing allocation owner")
		}
	}
	requests := [][]int{{3, 1, 3}, {0, 1}, {2, 3}}
	wantSlots := [][]int{{0, 1, 0}, {0, 1}, {0, 1}}
	wantExperts := [][2]int{{3, 1}, {0, 1}, {2, 3}}
	wantMisses := []int{2, 1, 2}
	bytesCompared, dtohCalls := 0, 0
	for round, experts := range requests {
		keys := make([]Key, len(experts))
		for i, expert := range experts {
			keys[i] = Key{id, 0, expert}
		}
		lease, err := cache.Acquire(keys)
		if err != nil {
			t.Fatal(err)
		}
		if len(lease.Uploads()) != wantMisses[round] {
			t.Fatalf("round %d miss count %d", round, len(lease.Uploads()))
		}
		before := counter.copies
		if err := upload.Stage(source, lease, false); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(lease.Slots(), wantSlots[round]) {
			t.Fatalf("round %d slots %v, want %v", round, lease.Slots(), wantSlots[round])
		}
		if counter.copies-before != 3*wantMisses[round] || counter.retainedCopies != 3*wantMisses[round] || len(upload.events) != 3*wantMisses[round] {
			t.Fatalf("round %d copy/retention/event counts %d/%d/%d", round, counter.copies-before, counter.retainedCopies, len(upload.events))
		}
		var got [3][2][]byte
		for component, n := range strides {
			for slot := 0; slot < 2; slot++ {
				got[component][slot] = bytes.Repeat([]byte{0xff}, n)
				if err := actual.slabs[component].Slice(slot*n, (slot+1)*n).CopyToHostAsync(got[component][slot], actual.stream); err != nil {
					t.Fatal(err)
				}
				dtohCalls++
			}
		}
		event, err := device.NewEventWithFlags(driver.EventDisableTiming)
		if event != nil {
			consumers = append(consumers, event)
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := event.Record(actual.stream); err != nil {
			t.Fatal(err)
		}
		completion := &nativeConsumer{event: event, failOnce: round == 1}
		if round == 1 {
			if err := lease.Release(completion); !errors.Is(err, errConsumerControl) {
				t.Fatalf("failed-release control: %v", err)
			}
			if lease.done || cache.active != lease {
				t.Fatal("failed release discarded lease")
			}
		}
		if round < 2 {
			drains, copies := counter.drains, counter.copies
			if upload.Close() == nil {
				t.Fatal("Close accepted active consumer")
			}
			if _, err := cache.Acquire(keys); err == nil {
				t.Fatal("Acquire accepted active consumer")
			}
			if err := upload.Stage(source, lease, false); err == nil {
				t.Fatal("Stage accepted active consumer")
			}
			if counter.drains != drains || counter.copies != copies || len(upload.events) != 3*wantMisses[round] {
				t.Fatal("active consumer negative changed resources")
			}
			current := [4]uintptr{actual.host.Ptr(), uintptr(actual.slabs[0].Ptr()), uintptr(actual.slabs[1].Ptr()), uintptr(actual.slabs[2].Ptr())}
			if current != pointers {
				t.Fatal("active consumer negative changed allocation owners")
			}
		}
		if err := lease.Release(completion); err != nil {
			t.Fatal(err)
		}
		for component, n := range strides {
			for slot, expert := range wantExperts[round] {
				want := payload[offsets[component]+expert*n : offsets[component]+(expert+1)*n]
				if !bytes.Equal(got[component][slot], want) {
					for i := range want {
						if got[component][slot][i] != want[i] {
							t.Fatalf("round=%d component=%d slot=%d index=%d got=%02x want=%02x", round, component, slot, i, got[component][slot][i], want[i])
						}
					}
				}
				bytesCompared += n
			}
		}
		t.Logf("round=%d unique_uploads=%d h2d=%d dtoh=6 bytes_compared=190 slots=%v", round, wantMisses[round], counter.copies-before, lease.Slots())
	}
	if counter.copies != 15 || dtohCalls != 18 || len(consumers) != 3 || bytesCompared != 570 {
		t.Fatalf("denominators HtoD=%d DtoH=%d consumer_events=%d bytes=%d", counter.copies, dtohCalls, len(consumers), bytesCompared)
	}
	if err := upload.Close(); err != nil {
		t.Fatal(err)
	}
	if !upload.closed || len(upload.events) != 0 || counter.retainedCopies != 0 || actual.host.Ptr() != 0 || actual.stream != nil {
		t.Fatal("terminal cleanup retained owner")
	}
	for _, slab := range actual.slabs {
		if slab.Ptr() != 0 {
			t.Fatal("terminal slab retained")
		}
	}
	drains, destroys := counter.drains, counter.destroys
	if err := upload.Close(); err != nil || counter.drains != drains || counter.destroys != destroys {
		t.Fatal("Close was not idempotent")
	}
	t.Log("scope=model-free-adapter leases=3 unique_uploads=5 HtoD=15 DtoH=18 consumer_events=3 exact_bytes=570 active_controls=2 failed_release_controls=1")
}
