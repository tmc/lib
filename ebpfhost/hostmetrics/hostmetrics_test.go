package hostmetrics

import (
	"fmt"
	"strings"
	"testing"
)

func fixture() Context { return Context{"node", "proc", "linux6.8", "mono", "boot", 10, 20} }
func ExampleReadPSI() {
	s, _ := ReadPSI(strings.NewReader("some avg10=1.25 avg60=0 avg300=0 total=42"), fixture(), "cpu", 512)
	fmt.Println(s.Metrics[0].Value, s.Metrics[3].UintValue)
	// Output: 1.25 42
}
func ExampleReadMeminfo() {
	s, _ := ReadMeminfo(strings.NewReader("MemTotal: 2 kB"), fixture(), 128)
	fmt.Println(s.Metrics[0].UintValue)
	// Output: 2048
}
func ExampleReadCounter() {
	s, _ := ReadCounter(strings.NewReader("9007199254740993"), fixture(), "node.net.rx_bytes", "eth0", 128)
	fmt.Println(s.Metrics[0].UintValue)
	// Output: 9007199254740993
}
func ExampleReadBlockStat() {
	s, _ := ReadBlockStat(strings.NewReader("1 0 2 0 3 0 4 0 0 0 0"), fixture(), "sda", 128)
	fmt.Println(s.Metrics[1].Unit)
	// Output: 512-byte_sectors
}
func ExampleUnit() {
	u, ok := Unit("node.rdma.port_rcv_data")
	fmt.Println(u, ok)
	// Output: 4-byte_words true
}
func ExampleSnapshot_Validate() {
	s, _ := ReadCounter(strings.NewReader("1"), fixture(), "node.net.rx_bytes", "eth0", 128)
	fmt.Println(s.Validate())
	// Output: <nil>
}
func TestInvalid(t *testing.T) {
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"oversize", func() error { _, e := ReadMeminfo(strings.NewReader("MemTotal: 2 kB"), fixture(), 2); return e }},
		{"missing", func() error { _, e := ReadMeminfo(strings.NewReader("MemFree: 2 kB"), fixture(), 128); return e }},
		{"unit", func() error { _, e := ReadMeminfo(strings.NewReader("MemTotal: 2 MB"), fixture(), 128); return e }},
		{"overflow", func() error {
			_, e := ReadMeminfo(strings.NewReader("MemTotal: 18446744073709551615 kB"), fixture(), 128)
			return e
		}},
		{"duplicate", func() error {
			_, e := ReadMeminfo(strings.NewReader("MemTotal: 2 kB\nMemTotal: 2 kB"), fixture(), 128)
			return e
		}},
		{"nan", func() error {
			_, e := ReadPSI(strings.NewReader("some avg10=NaN avg60=0 avg300=0 total=0"), fixture(), "cpu", 128)
			return e
		}},
		{"missingfull", func() error {
			_, e := ReadPSI(strings.NewReader("some avg10=0 avg60=0 avg300=0 total=0"), fixture(), "io", 128)
			return e
		}},
		{"identity", func() error {
			c := fixture()
			c.NodeID = ""
			_, e := ReadCounter(strings.NewReader("1"), c, "node.net.rx_bytes", "eth0", 128)
			return e
		}},
		{"negative", func() error {
			_, e := ReadCounter(strings.NewReader("-1"), fixture(), "node.net.rx_bytes", "eth0", 128)
			return e
		}},
		{"block", func() error { _, e := ReadBlockStat(strings.NewReader("1 2"), fixture(), "sda", 128); return e }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.run() == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
func TestCPUFull(t *testing.T) {
	s, e := ReadPSI(strings.NewReader("some avg10=0 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0"), fixture(), "cpu", 512)
	if e != nil || len(s.Metrics) != 4 {
		t.Fatalf("%+v %v", s, e)
	}
}

func ExampleContext() {
	c := fixture()
	fmt.Println(c.NodeID, c.QueryStartNanos, c.QueryEndNanos)
	// Output: node 10 20
}
func FuzzReaders(f *testing.F) {
	f.Add("MemTotal: 1 kB")
	f.Add("some avg10=0 avg60=0 avg300=0 total=0")
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 8192 {
			return
		}
		for _, run := range []func() (Snapshot, error){func() (Snapshot, error) { return ReadPSI(strings.NewReader(s), fixture(), "memory", 4096) }, func() (Snapshot, error) { return ReadMeminfo(strings.NewReader(s), fixture(), 4096) }, func() (Snapshot, error) { return ReadBlockStat(strings.NewReader(s), fixture(), "sda", 4096) }} {
			v, e := run()
			if e == nil {
				if e = v.Validate(); e != nil {
					t.Fatal(e)
				}
			}
		}
	})
}

func TestMalformedIdentity(t *testing.T) {
	c := fixture()
	c.SourceID = "bad\xff"
	if _, e := ReadCounter(strings.NewReader("1"), c, "node.net.rx_bytes", "eth0", 32); e == nil {
		t.Fatal("accepted invalid UTF-8 source")
	}
}
