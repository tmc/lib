package cuptipcsampling

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestSamplingDrain(t *testing.T) {
	b := testSamplingBuffer(t)
	got, err := b.drain(5, 130001, func(p *cupti.PCSamplingGetDataParams) error {
		if p.Size != 32 || p.Ctx != 5 || p.PPriv != 0 || p.PcSamplingData != uintptr(unsafe.Pointer(&b.data)) {
			t.Fatalf("params=%+v", p)
		}
		if b.data.TotalNumPcs != 0 || b.data.DroppedSamples != 0 {
			t.Fatal("previous output reused")
		}
		b.data.TotalNumPcs = 1
		b.data.RemainingNumPcs = 20
		b.data.TotalSamples = 1<<63 + 2
		b.pcs[0].FunctionName = 99
		b.pcs[0].StallReasonCount = 1
		b.reasons[0][0].Samples = 3
		return nil
	}, func(uintptr) (string, error) { return "kernel", nil })
	if err != nil || len(got.pcs) != 1 || got.remaining != 20 || got.total != 1<<63+2 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestSamplingDrainOwnsNamesAfterCopyFailure(t *testing.T) {
	b := testSamplingBuffer(t)
	var freed []uintptr
	failure := errors.New("name too long")
	names, err := newSamplingNames(2, 10, 20, func(ptr uintptr, _ int) (string, error) {
		if ptr == 11 {
			return "", failure
		}
		return "kernel", nil
	}, func(ptr uintptr) { freed = append(freed, ptr) })
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.drainOwned(5, 130001, func(*cupti.PCSamplingGetDataParams) error {
		b.data.TotalNumPcs = 2
		b.data.DroppedSamples = 9
		b.pcs[0].FunctionName, b.pcs[1].FunctionName = 11, 22
		return nil
	}, names)
	if !errors.Is(err, failure) || got.pcs != nil || got.dropped != 9 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	names.release()
	if !reflect.DeepEqual(freed, []uintptr{11, 22}) {
		t.Fatalf("lost native names: %v", freed)
	}
}

func TestSamplingDrainNameBudgetPrecedesNative(t *testing.T) {
	b := testSamplingBuffer(t)
	names, err := newSamplingNames(1, 10, 20, func(uintptr, int) (string, error) { return "kernel", nil }, func(uintptr) {})
	if err != nil {
		t.Fatal(err)
	}
	defer names.release()
	_, err = b.drainOwned(5, 130001, func(*cupti.PCSamplingGetDataParams) error { t.Fatal("native called without name capacity"); return nil }, names)
	if err == nil || b.data.TotalNumPcs != 1 {
		t.Fatal("budget gate changed output or accepted drain")
	}
}

func TestSamplingDrainValidatesBeforeReadingNames(t *testing.T) {
	b := testSamplingBuffer(t)
	names, err := newSamplingNames(2, 10, 20, func(uintptr, int) (string, error) {
		t.Fatal("read name from malformed output")
		return "", nil
	}, func(uintptr) { t.Fatal("freed unvalidated name") })
	if err != nil {
		t.Fatal(err)
	}
	defer names.release()
	_, err = b.drainOwned(5, 130001, func(*cupti.PCSamplingGetDataParams) error {
		b.data.TotalNumPcs = 2
		b.pcs[0].FunctionName = 11
		b.pcs[1].Size = 1
		return nil
	}, names)
	if err == nil {
		t.Fatal("accepted malformed PC output")
	}
}

func TestSamplingDrainFailure(t *testing.T) {
	failure := errors.New("hardware full")
	for _, nativeFailure := range []bool{false, true} {
		b := testSamplingBuffer(t)
		got, err := b.drain(5, 130001, func(p *cupti.PCSamplingGetDataParams) error {
			b.data.DroppedSamples = 1<<63 + 1
			b.data.HardwareBufferFull = 1
			p.PcSamplingData = 1
			b.data.TotalNumPcs = ^uint(0)
			if nativeFailure {
				return failure
			}
			return nil
		}, func(uintptr) (string, error) { t.Fatal("dereferenced failed native output"); return "", nil })
		if err == nil || got.pcs != nil || got.dropped != 1<<63+1 || got.hardwareFull != 1 {
			t.Fatalf("got=%+v err=%v", got, err)
		}
		if nativeFailure && !errors.Is(err, failure) {
			t.Fatal("native error lost")
		}
	}
}

func TestSamplingDrainGate(t *testing.T) {
	for _, version := range []uint32{0, 27, 130200} {
		b := testSamplingBuffer(t)
		_, err := b.drain(5, version, func(*cupti.PCSamplingGetDataParams) error {
			t.Fatal("called native for unqualified layout")
			return nil
		}, nil)
		if err == nil || b.data.TotalNumPcs != 1 {
			t.Fatal("gate reset buffer or accepted version")
		}
	}
}

func TestSamplingDrainDoesNotPromoteFullBuffer(t *testing.T) {
	b := testSamplingBuffer(t)
	got, err := b.drain(5, 130001, func(*cupti.PCSamplingGetDataParams) error {
		b.data.HardwareBufferFull = 1
		return nil
	}, nil)
	if err == nil || got.hardwareFull != 1 || got.pcs != nil {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
