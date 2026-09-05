//go:build linux

package cuptiactivity_test

import (
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestNativeCollection(t *testing.T) {
	if os.Getenv("CUPTI_ACTIVITY_TEST") != "1" {
		t.Skip("CUPTI_ACTIVITY_TEST=1 required")
	}
	config := cuptiactivity.Config{
		Kinds:       []cupti.ActivityKind{cupti.ActivityKindDriver, cupti.ActivityKindMemcpy},
		BufferBytes: 1 << 20, MaxBuffers: 16, MaxRecords: 100000,
	}
	session, err := cuptiactivity.Start(config)
	if session != nil {
		defer func() {
			if err := session.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	} // An opted-in unsupported library is not a pass.
	if other, err := cuptiactivity.Start(config); err == nil || other != nil {
		t.Fatal("second collector accepted")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	memory, err := cudamem.Alloc[uint32](dev, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := memory.Free(); err != nil {
			t.Error(err)
		}
	}()
	want, got := make([]uint32, memory.Len()), make([]uint32, memory.Len())
	for i := range want {
		want[i] = uint32(i*7 + 3)
	}
	if err := memory.CopyFromHost(want); err != nil {
		t.Fatal(err)
	}
	if err := memory.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(want, got) {
		t.Fatal("copy round trip differs")
	}
	if err := dev.WithContext(driver.CtxSynchronize); err != nil {
		t.Fatal(err)
	}
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
	r := session.Result()
	if !r.Closed || r.PendingBuffers != 0 || r.RefusedBuffers != 0 || r.IncompleteRecords != 0 || r.DiscardedRecords != 0 || r.DroppedRecords != 0 || r.CallbackErrors != 0 || r.FirstError != nil {
		t.Fatalf("unclean collection: closed=%v pending=%d refused=%d incomplete=%d discarded=%d dropped=%d errors=%d first=%v", r.Closed, r.PendingBuffers, r.RefusedBuffers, r.IncompleteRecords, r.DiscardedRecords, r.DroppedRecords, r.CallbackErrors, r.FirstError)
	}
	api, copies := 0, 0
	var copyBytes uint64
	for _, record := range r.Records {
		if !record.Complete {
			t.Fatal("incomplete record in clean result")
		}
		switch record.Kind {
		case cupti.ActivityKindDriver:
			api++
		case cupti.ActivityKindMemcpy:
			copies++
			copyBytes += record.Bytes
		}
	}
	if api == 0 || copies < 2 || copyBytes < uint64(len(want)*8) {
		t.Fatalf("insufficient activity: APIs=%d copies=%d bytes=%d", api, copies, copyBytes)
	}
	t.Logf("collected APIs=%d copies=%d bytes=%d; no timing comparison", api, copies, copyBytes)
}
