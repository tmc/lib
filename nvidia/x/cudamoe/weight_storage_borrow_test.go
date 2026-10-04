package cudamoe

import (
	"bytes"
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestMoEWeightBorrowIndependent(t *testing.T) {
	s, _ := weightStorageHost(t)
	w, scales := weightStoragePayload(s)
	if err := s.Load(2, 1, bytes.NewReader(w), bytes.NewReader(scales)); err != nil {
		t.Fatal(err)
	}
	v, err := s.View(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, release, err := v.Borrow(s.device)
	if err != nil {
		t.Fatal(err)
	}
	d := b.Descriptor
	if d.Generation != 1 || d.Layer != 2 || d.Expert != 1 {
		t.Fatal("wrong borrowed range")
	}
	if b.WeightBytes != 3264 || b.ScaleBytes != 768 || b.Shape.Layers != 3 || b.Shape.Experts != 2 {
		t.Fatal("wrong admitted slab metadata")
	}
	b.Shape.ActiveLayers[0] = 1
	other, second, err := v.Borrow(s.device)
	if err != nil {
		t.Fatal(err)
	}
	second()
	if other.Shape.ActiveLayers[0] != 0 {
		t.Fatal("borrow metadata aliases owner")
	}
	if _, _, err := v.Borrow(&cuda.Device{}); err == nil {
		t.Fatal("different device admitted")
	}
	v.Close()
	if _, _, err := v.Borrow(s.device); err == nil {
		t.Fatal("closed view admitted")
	}
	if !errors.Is(s.Close(), cudamem.ErrInUse) {
		t.Fatal("borrow lost after public close")
	}
	if !errors.Is(s.Load(2, 1, bytes.NewReader(w), bytes.NewReader(scales)), cudamem.ErrInUse) {
		t.Fatal("borrowed range rewritten")
	}
	release()
	release()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMoEWeightBorrowConcurrentPublicClose(t *testing.T) {
	s, _ := weightStorageHost(t)
	w, sc := weightStoragePayload(s)
	if err := s.Load(0, 0, bytes.NewReader(w), bytes.NewReader(sc)); err != nil {
		t.Fatal(err)
	}
	v, err := s.View(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, release, err := v.Borrow(s.device)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); v.Close() }()
	go func() {
		defer wg.Done()
		if !errors.Is(s.Close(), cudamem.ErrInUse) {
			t.Error("owner close lost independent read pin")
		}
	}()
	wg.Wait()
	if !errors.Is(s.Close(), cudamem.ErrInUse) {
		t.Fatal("pin released by public close")
	}
	release()
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}
