//go:build linux && (amd64 || arm64)

package bpfload

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestClosedMap(t *testing.T) {
	var o Object
	if _, err := o.MapInfo(); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := o.Lookup(nil, nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if err := o.Update(nil, nil, 0); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestCloseRetainsError(t *testing.T) {
	o := newObject(-1)
	first := o.Close()
	if first == nil {
		t.Fatal("invalid descriptor close succeeded")
	}
	if again := o.Close(); again != first {
		t.Fatalf("close error lost: %v versus %v", first, again)
	}
}

func TestRejectNonMap(t *testing.T) {
	f, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := mapInfo(int(f.Fd())); err == nil {
		t.Fatal("accepted non-map descriptor")
	}
}

// This opt-in check loads a socket filter returning zero and then closes it.
// It does not attach anything or claim GPU extension support. A refused load
// fails the opted-in admission rather than turning missing privilege into PASS.
func TestNativeLoad(t *testing.T) {
	if os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("set EBPFHOST_NATIVE=1 for kernel program load admission")
	}
	btf, log, err := LoadBTF(tinyBTF(binary.LittleEndian), 0)
	if err != nil {
		t.Fatalf("btf load refused: %v; log=%q", err, log)
	}
	if err := btf.Close(); err != nil {
		t.Fatal(err)
	}
	if invalid, _, err := LoadBTF([]byte{1, 2, 3}, 0); err == nil {
		invalid.Close()
		t.Fatal("kernel admitted malformed btf")
	}
	instructions := []byte{0xb7, 0, 0, 0, 0, 0, 0, 0, 0x95, 0, 0, 0, 0, 0, 0, 0}
	o, log, err := Load(ProgramSpec{Type: 1, Instructions: instructions})
	if err != nil {
		t.Fatalf("kernel load refused: %v; verifier=%q", err, log)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("upstream socket filter load and close passed; verifier=%q", log)

	// An invalid opcode must be rejected by the same kernel verifier.
	bad := []byte{0xff, 0, 0, 0, 0, 0, 0, 0}
	invalid, _, err := Load(ProgramSpec{Type: 1, Instructions: bad})
	if err == nil {
		invalid.Close()
		t.Fatal("verifier accepted invalid opcode")
	}
	t.Logf("invalid instruction refused: %v", err)

	m, err := CreateMap(MapSpec{Type: 2, KeySize: 4, ValueSize: 8, MaxEntries: 1, Name: "lib_bpf_test"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	key := make([]byte, 4)
	want := make([]byte, 8)
	binary.LittleEndian.PutUint64(want, 0x123456789abcdef0)
	if err := m.Update(key, want, 0); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 8)
	if err := m.Lookup(key, got); err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint64(got) != binary.LittleEndian.Uint64(want) {
		t.Fatalf("map result: %x", got)
	}
	info, err := m.MapInfo()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("map info: %+v", info)
	if root := os.Getenv("EBPFHOST_PIN_ROOT"); root != "" {
		dir, err := os.MkdirTemp(root, "lib-bpf-test-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.Remove(dir)
		path := filepath.Join(dir, "map")
		if err := m.Pin(path); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(path)
		if err := m.Pin(path); err == nil {
			t.Fatal("pin replaced existing object")
		}
		reopened, reInfo, err := OpenMap(path, true)
		if err != nil {
			t.Fatal(err)
		}
		defer reopened.Close()
		if reInfo.ID != info.ID {
			t.Fatalf("pin identity: %+v versus %+v", reInfo, info)
		}
		if err := reopened.Lookup(key, got); err != nil {
			t.Fatal(err)
		}
		if binary.LittleEndian.Uint64(got) != binary.LittleEndian.Uint64(want) {
			t.Fatalf("pinned map result: %x", got)
		}
		if err := reopened.Update(key, want, 0); err == nil {
			t.Fatal("readonly map permitted update")
		}
		t.Logf("pinned map open/info/read passed; existing pin replacement and readonly update refused")
	}
}
