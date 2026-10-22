package dsv41

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

var _ dsv4.Backend = (*Backend)(nil)

func TestNativeBackendOwnership(t *testing.T) {
	fixture := payloadFixture(t, 1)
	b, err := NewBackend(fixture.model)
	if err != nil {
		t.Fatal(err)
	}
	xi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	yi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	x, y := xi.(*request), yi.(*request)
	x.core = fixture
	snapshot, err := x.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if err := y.Restore(snapshot); err != nil {
		t.Fatal(err)
	}
	z, err := b.NewState(131)
	if err != nil {
		t.Fatal(err)
	}
	if z.Restore(snapshot) == nil {
		t.Fatal("wrong context snapshot accepted")
	}
	x.core.kv.raw[0][0] = 700
	if y.core.kv.raw[0][0] != 0 {
		t.Fatal("request storage aliased")
	}
	y.core.kv.raw[0][0] = 800
	if err := y.Restore(snapshot); err != nil || y.core.kv.raw[0][0] != 0 {
		t.Fatal("snapshot storage aliased", err)
	}
	foreign, err := NewBackend(fixture.model)
	if err != nil {
		t.Fatal(err)
	}
	f, err := foreign.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	if f.Restore(snapshot) == nil || foreign.Eval(x, 0, 1) == nil || foreign.Logits(x) != nil {
		t.Fatal("foreign request accepted")
	}
	if b.Eval(x, -1, 1) == nil || b.Eval(x, 0, 2) == nil {
		t.Fatal("invalid input reached forward")
	}
	if snapshot.Close() != nil || y.Restore(snapshot) == nil {
		t.Fatal("closed snapshot accepted")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(y.Reset(), dsv4.ErrClosed) || !errors.Is(b.Eval(x, 0, 1), dsv4.ErrClosed) || b.Logits(x) != nil {
		t.Fatal("closed backend remains usable")
	}
	if x.Close() != nil || y.Close() != nil || b.Close() != nil || foreign.Close() != nil {
		t.Fatal("close failed")
	}
}

func TestNativePayloadAtomicity(t *testing.T) {
	s := payloadFixture(t, 1)
	b, err := NewBackend(s.model)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	})
	xi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	yi, err := b.NewState(132)
	if err != nil {
		t.Fatal(err)
	}
	x, y := xi.(*request), yi.(*request)
	x.core = s
	var wire bytes.Buffer
	if err := x.Save(&wire); err != nil {
		t.Fatal(err)
	}
	if err := y.Load(bytes.NewReader(wire.Bytes())); err != nil {
		t.Fatal(err)
	}
	var before bytes.Buffer
	if err := y.Save(&before); err != nil || !bytes.Equal(before.Bytes(), wire.Bytes()) {
		t.Fatal("native round trip", err)
	}
	for _, offset := range []int{0, 8, 40, 72, 76, 84, 84 + 52 + 4} {
		bad := append([]byte(nil), wire.Bytes()...)
		bad[offset] ^= 1
		if y.Load(bytes.NewReader(bad)) == nil {
			t.Fatalf("corrupt offset%d accepted", offset)
		}
		var after bytes.Buffer
		if err := y.Save(&after); err != nil || !bytes.Equal(after.Bytes(), before.Bytes()) {
			t.Fatal("failed load published state", err)
		}
	}
	for _, n := range []int{0, 83, 84, len(wire.Bytes()) - 1} {
		if y.Load(bytes.NewReader(wire.Bytes()[:n])) == nil {
			t.Fatal("short native stream accepted")
		}
	}
	s.model.assetSHA = [32]byte{}
	var empty bytes.Buffer
	if x.Save(&empty) == nil || empty.Len() != 0 || y.Load(bytes.NewReader(wire.Bytes())) == nil {
		t.Fatal("unverified model admitted persistence")
	}
}

func ExampleBackend_NewState() {
	var backend *Backend
	_, err := backend.NewState(4)
	fmt.Println(errors.Is(err, dsv4.ErrClosed))
	// Output: true
}

func ExampleNewBackend() {
	_, err := NewBackend(nil)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleBackend_Eval() {
	var backend *Backend
	fmt.Println(errors.Is(backend.Eval(nil, 0, 0), dsv4.ErrClosed))
	// Output: true
}
func ExampleBackend_Logits() {
	var backend *Backend
	fmt.Println(backend.Logits(nil) == nil)
	// Output: true
}
func ExampleBackend_Close() {
	var backend Backend
	fmt.Println(backend.Close() == nil)
	// Output: true
}
