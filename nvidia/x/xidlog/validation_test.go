package xidlog

import (
	"errors"
	"io"
	"strings"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
)

func TestValidationKernelControlFraming(t *testing.T) {
	for _, suffix := range []string{"\r", "\t", "\x01"} {
		raw := "6,1,2,-;NVRM: Xid (PCI:0000:65:00): 31, text" + suffix + "\n"
		v, err := ParseKmsg(raw, fixtureOptions())
		if err != nil || v.Kind != Malformed || v.Raw != raw {
			t.Fatalf("unescaped control became fact: %+v %v", v, err)
		}
	}
}

func TestValidationKernelReadBoundaries(t *testing.T) {
	raw := "6,18446744073709551615,18446744073709551615,z,new;NVRM: Xid (PCI:0000:65:00): 31, text\n DEVICE=+pci:0000:65:00.0\n"
	r, err := NewReader(&recordInput{steps: []readStep{{raw: raw}}}, KernelRecords, fixtureOptions())
	if err != nil {
		t.Fatal(err)
	}
	v, err := r.Next()
	if err != nil || v.Kind != XID || v.Header.Sequence != ^uint64(0) || v.Header.MonotonicMicroseconds != ^uint64(0) || v.Header.Flags != "z" || v.Header.Extensions[0] != "new" || v.Raw != raw {
		t.Fatalf("record lost %+v %v", v, err)
	}
	for _, bad := range []string{strings.Replace(raw, "18446744073709551615", "18446744073709551616", 1), strings.Replace(raw, "6,", "-1,", 1), strings.TrimSuffix(raw, "\n")} {
		v, err := ParseKmsg(bad, fixtureOptions())
		if err != nil || v.Kind != Malformed || v.Raw != bad {
			t.Fatalf("invalid kernel extent %+v %v", v, err)
		}
	}
	r, _ = NewReader(&recordInput{steps: []readStep{{raw: "6,1,2,-;part"}, {raw: " remainder\n"}}}, KernelRecords, fixtureOptions())
	v, _ = r.Next()
	if !v.Partial || v.Kind != Malformed {
		t.Fatal("split record accepted")
	}
	v, _ = r.Next()
	if v.Kind != Malformed {
		t.Fatal("split remainder accepted")
	}
}

func TestValidationLossAndReadErrors(t *testing.T) {
	r, _ := NewReader(&recordInput{steps: []readStep{{raw: "unexpected", err: syscall.EPIPE}, {raw: "6,2,3,-;next\n"}}}, KernelRecords, fixtureOptions())
	v, err := r.Next()
	if err != nil || v.Raw != "unexpected" || v.Kind != Malformed {
		t.Fatalf("loss data lost %+v %v", v, err)
	}
	v, err = r.Next()
	if err != nil || v.Kind != Gap || v.Header != nil || v.Code != 0 {
		t.Fatalf("gap invented facts %+v %v", v, err)
	}
	v, err = r.Next()
	if err != nil || v.Header.Sequence != 2 {
		t.Fatalf("recovery %+v %v", v, err)
	}
	r, _ = NewReader(&recordInput{steps: []readStep{{err: syscall.EAGAIN}, {raw: "6,3,4,-;next\n"}}}, KernelRecords, fixtureOptions())
	if _, err = r.Next(); !errors.Is(err, syscall.EAGAIN) {
		t.Fatal("EAGAIN concealed")
	}
	if v, err = r.Next(); err != nil || v.Header.Sequence != 3 {
		t.Fatal("EAGAIN made terminal")
	}
	r, _ = NewReader(strings.NewReader("a\nb\n"), Text, Options{Source: fixtureOptions().Source, Product: xidcatalog.H100, Driver: "580", MaxBytes: 2})
	if _, err = r.Next(); err != nil {
		t.Fatal(err)
	}
	if _, err = r.Next(); err == nil {
		t.Fatal("total byte limit bypassed")
	}
	r, _ = NewReader(&recordInput{}, KernelRecords, fixtureOptions())
	if _, err = r.Next(); err != io.EOF {
		t.Fatal(err)
	}
}

func TestValidationRecoveryAndProduct(t *testing.T) {
	raw := "NVRM: Xid (PCI:0000:65:00): 154, pid=7, name=worker, GPU recovery action changed from 0x0 (None) to 0x2 (GPU Reset Required)"
	v, err := ParseText(raw, fixtureOptions())
	if err != nil || v.Recovery == nil || v.Recovery.From.Value != 0 || v.Recovery.To.Value != 2 || v.PID == nil || *v.PID != 7 || v.ProcessName != "worker" {
		t.Fatalf("actual recovery tokens lost %+v %v", v, err)
	}
	v, err = ParseText(strings.Replace(raw, "to 0x2", "to 0x100000000", 1), fixtureOptions())
	if err != nil || v.Recovery != nil || v.Problem == "" || v.Raw == "" {
		t.Fatalf("invalid recovery invented %+v %v", v, err)
	}
	for _, product := range []xidcatalog.Architecture{"L4", "GB10", "future"} {
		o := fixtureOptions()
		o.Product = product
		v, err := ParseText("NVRM: Xid (PCI:0000:65:00): 31, x", o)
		if err != nil || v.Product != product || v.Applicability != xidcatalog.UncoveredProduct {
			t.Fatalf("uncovered product inferred %+v %v", v, err)
		}
	}
}
