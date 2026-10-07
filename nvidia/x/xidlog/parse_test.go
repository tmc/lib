package xidlog

import (
	"errors"
	"io"
	"os"
	"strings"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/x/xidcatalog"
)

func fixtureOptions() Options {
	return Options{Source: Source{BootID: "caller-boot", InputID: "constructed fixture", PreviousBoot: true}, Product: xidcatalog.H100, Driver: "580.0"}
}
func TestConstructedFacts(t *testing.T) {
	for _, driver := range []string{"535.1", "550.1", "570.1", "580.1"} {
		t.Run(driver, func(t *testing.T) {
			data, err := os.ReadFile("testdata/synthetic.txt")
			if err != nil {
				t.Fatal(err)
			}
			o := fixtureOptions()
			o.Driver = driver
			r, err := NewReader(strings.NewReader(string(data)), Text, o)
			if err != nil {
				t.Fatal(err)
			}
			want := []Kind{GPU, XID, XID, XID}
			for i, k := range want {
				v, err := r.Next()
				if err != nil || v.Kind != k {
					t.Fatalf("row%d: %+v %v", i, v, err)
				}
				if v.Driver != driver || !v.Source.PreviousBoot {
					t.Fatal("lost provenance")
				}
				if i == 1 && (v.PID == nil || *v.PID != 12 || v.ProcessName != "worker") {
					t.Fatal("process lost")
				}
				if i == 2 && v.Recovery == nil {
					t.Fatal("recovery lost")
				}
				if i == 3 && v.Applicability != xidcatalog.UnknownID {
					t.Fatal("future code invented")
				}
			}
			if _, err := r.Next(); err != io.EOF {
				t.Fatalf("end %v", err)
			}
		})
	}
}
func TestExplicitUncoveredProduct(t *testing.T) {
	for _, product := range []xidcatalog.Architecture{"L4", "GB10"} {
		o := fixtureOptions()
		o.Product = product
		v, err := ParseText("NVRM: Xid (PCI:0000:65:00): 31, constructed", o)
		if err != nil || v.Applicability != xidcatalog.UncoveredProduct {
			t.Fatalf("%+v %v", v, err)
		}
	}
}
func TestKmsgHeaderAndFragment(t *testing.T) {
	o := fixtureOptions()
	raw := "6,339,5140900,-,future;NVRM: Xid (PCI:0000:65:00): 31, constructed\n DEVICE=+pci:0000:65:00.0\n"
	v, err := ParseKmsg(raw, o)
	if err != nil || v.Kind != XID || v.Header.Sequence != 339 || v.Header.MonotonicMicroseconds != 5140900 || len(v.Header.Extensions) != 1 || len(v.Header.Metadata) != 1 {
		t.Fatalf("%+v %v", v, err)
	}
	v, err = ParseKmsg(strings.Replace(raw, ",-,", ",c,", 1), o)
	if err != nil || v.Kind != Fragment || !v.Partial || v.Code != 0 {
		t.Fatalf("fragment %+v %v", v, err)
	}
}

type readStep struct {
	raw string
	err error
}
type recordInput struct{ steps []readStep }

func (r *recordInput) Read(p []byte) (int, error) {
	if len(r.steps) == 0 {
		return 0, io.EOF
	}
	s := r.steps[0]
	r.steps = r.steps[1:]
	return copy(p, s.raw), s.err
}
func TestKmsgEPIPEIsExplicit(t *testing.T) {
	r, err := NewReader(&recordInput{steps: []readStep{{err: syscall.EPIPE}, {raw: "6,4,100,-;unrelated\n"}}}, KernelRecords, fixtureOptions())
	if err != nil {
		t.Fatal(err)
	}
	v, err := r.Next()
	if err != nil || v.Kind != Gap {
		t.Fatalf("gap %+v %v", v, err)
	}
	v, err = r.Next()
	if err != nil || v.Kind != Other || v.Header.Sequence != 4 {
		t.Fatalf("next %+v %v", v, err)
	}
}
func TestRefusalsAndPartial(t *testing.T) {
	for _, raw := range []string{"NVRM: Xid (PCI:0000:65:ff): 31, x", "NVRM: Xid (PCI:0000:65:00): 31, pid=no, x", "NVRM: Xid (PCI:0000:65:00): 31", "NVRM: Xid (PCI:0000:65:00): 31, x\nsecond"} {
		v, err := ParseText(raw, fixtureOptions())
		if err != nil || v.Kind != Malformed {
			t.Fatalf("%q %+v %v", raw, v, err)
		}
	}
	r, _ := NewReader(strings.NewReader("NVRM: Xid (PCI:0000:65:00): 31, x"), Text, fixtureOptions())
	v, err := r.Next()
	if err != nil || !v.Partial || v.Kind != Malformed {
		t.Fatalf("partial %+v %v", v, err)
	}
	o := fixtureOptions()
	o.Source.BootID = ""
	if _, err := ParseText("x", o); err == nil {
		t.Fatal("missing boot accepted")
	}
	o = fixtureOptions()
	o.MaxRecordBytes = 4
	r, _ = NewReader(strings.NewReader("12345\n"), Text, o)
	if _, err := r.Next(); err == nil {
		t.Fatal("extent accepted")
	}
	o = fixtureOptions()
	o.MaxRecords = 1
	r, _ = NewReader(strings.NewReader("a\nb\n"), Text, o)
	r.Next()
	if _, err := r.Next(); err == nil {
		t.Fatal("population accepted")
	}
}
func TestUnknownRecoveryRetained(t *testing.T) {
	v, err := ParseText("NVRM: Xid (PCI:0000:65:00): 154, GPU recovery action changed from 0x0 (None) to 0xff (Future)", fixtureOptions())
	if err != nil || v.Recovery == nil || v.Recovery.To.Recognized || v.Recovery.To.Text != "Future" {
		t.Fatalf("%+v %v", v, err)
	}
}
func TestReadErrorRetained(t *testing.T) {
	fail := errors.New("caller read failure")
	r, _ := NewReader(&recordInput{steps: []readStep{{raw: "partial", err: fail}}}, KernelRecords, fixtureOptions())
	v, err := r.Next()
	if !errors.Is(err, fail) || v.Raw != "partial" || v.Kind != Malformed {
		t.Fatalf("%+v %v", v, err)
	}
}
func FuzzParseKmsg(f *testing.F) {
	f.Add("6,1,2,-;unrelated\n")
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > MaxRecordBytes {
			return
		}
		ParseKmsg(s, fixtureOptions())
	})
}
