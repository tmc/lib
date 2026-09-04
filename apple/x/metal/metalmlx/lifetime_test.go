package metalmlx

import (
	"sync"
	"testing"
)

func TestCommandEncoderClose(t *testing.T) {
	d, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err == nil {
		t.Fatal("Device.Close with live encoder succeeded")
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("second CommandEncoder.Close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("second Device.Close: %v", err)
	}
}

func TestDeviceCloseRejectsLiveAllocation(t *testing.T) {
	d, err := NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	b, err := d.Allocator().Malloc(64)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err == nil {
		t.Fatal("Device.Close with live allocation succeeded")
	}
	d.Allocator().Free(b)
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestEventDestroy(t *testing.T) {
	d := testDevice(t)
	e, err := NewEvent(d)
	if err != nil {
		t.Fatal(err)
	}
	e.Destroy()
	e.Destroy()
	if _, err := e.Value(); err == nil {
		t.Fatal("Value after Destroy succeeded")
	}
	if err := e.Signal(1); err == nil {
		t.Fatal("Signal after Destroy succeeded")
	}
	if err := e.Wait(1); err == nil {
		t.Fatal("Wait after Destroy succeeded")
	}
}

func TestFenceDestroy(t *testing.T) {
	d := testDevice(t)
	f, err := NewFence(d)
	if err != nil {
		t.Fatal(err)
	}
	f.Destroy()
	f.Destroy()
	if _, err := f.Count(); err == nil {
		t.Fatal("Count after Destroy succeeded")
	}
	if err := f.UpdateCPU(); err == nil {
		t.Fatal("UpdateCPU after Destroy succeeded")
	}
	if err := f.WaitCPU(); err == nil {
		t.Fatal("WaitCPU after Destroy succeeded")
	}
}

func TestEventConcurrentDestroy(t *testing.T) {
	d := testDevice(t)
	e, err := NewEvent(d)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := uint64(0); i < 100; i++ {
				if err := e.Signal(i); err != nil {
					return
				}
				if _, err := e.Value(); err != nil {
					return
				}
			}
		}()
	}
	close(start)
	e.Destroy()
	wg.Wait()
}

func TestFenceConcurrentDestroy(t *testing.T) {
	d := testDevice(t)
	f, err := NewFence(d)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for range 100 {
				if err := f.UpdateCPU(); err != nil {
					return
				}
				if _, err := f.Count(); err != nil {
					return
				}
			}
		}()
	}
	close(start)
	f.Destroy()
	wg.Wait()
}
