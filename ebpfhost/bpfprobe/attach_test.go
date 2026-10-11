package bpfprobe

import (
	"context"
	"encoding/binary"
	"strings"
	"testing"
)

func TestDecodeEvent(t *testing.T) {
	raw := make([]byte, 168)
	for i := 0; i < 17; i++ {
		binary.LittleEndian.PutUint64(raw[i*8:], uint64(i+1))
	}
	binary.LittleEndian.PutUint64(raw[136:], 0x7ff)
	binary.LittleEndian.PutUint64(raw[144:], 123)
	e, err := DecodeEvent(raw)
	if err != nil {
		t.Fatal(err)
	}
	if e.Timestamp != 1 || e.PIDTGID != 2 || e.Tag != 3 || e.Arguments[10] != 14 || e.ReturnValue != 15 || e.StackPointer != 16 || e.InstructionPointer != 17 || e.ArgumentsMask != 0x7ff || e.ReturnKnown {
		t.Fatalf("event: %+v", e)
	}
	for _, flags := range []uint64{1 << 63, 0x107ff} {
		binary.LittleEndian.PutUint64(raw[136:], flags)
		if _, err := DecodeEvent(raw); err == nil {
			t.Fatalf("accepted flags %x", flags)
		}
	}
	binary.LittleEndian.PutUint64(raw[136:], 1<<16)
	if e, err := DecodeEvent(raw); err != nil || !e.ReturnKnown || e.ArgumentsMask != 0 {
		t.Fatalf("return: %+v %v", e, err)
	}
	if _, err := DecodeEvent(raw[:167]); err == nil {
		t.Fatal("accepted truncated record")
	}
}
func TestProbeZero(t *testing.T) {
	var p Probe
	if _, err := p.Read(context.Background()); err == nil {
		t.Fatal("zero probe read")
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRefuseThreadInheritance(t *testing.T) {
	_, err := validateConfig(Config{InheritThreads: true, TGID: 1})
	if err == nil || !strings.Contains(err.Error(), "inheritance unsupported") {
		t.Fatalf("validateConfig: got %v, want inheritance refusal", err)
	}
}

func TestAttachRefusesInheritance(t *testing.T) {
	cfg := Config{InheritThreads: true, PID: 1, TGID: 1, Path: "/does/not/exist"}
	if _, err := Attach(cfg); err == nil || !strings.Contains(err.Error(), "inheritance unsupported") {
		t.Fatalf("Attach: %v", err)
	}
	if _, err := AttachGroup([]Config{cfg}); err == nil || !strings.Contains(err.Error(), "inheritance unsupported") {
		t.Fatalf("AttachGroup: %v", err)
	}
}
