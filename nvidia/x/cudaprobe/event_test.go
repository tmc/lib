package cudaprobe

import (
	"encoding/binary"
	"strings"
	"testing"
)

func capture(symbol string, sequence uint64, args ...uint64) Capture {
	c := Capture{Identity: Identity{Process: Process{PID: 7, Generation: 99}, Library: "fixture.so", BuildID: "synthetic-build", Version: "fixture-v1", Symbol: symbol, ABI: "linux-amd64-sysv-v1"}}
	u := binary.LittleEndian
	u.PutUint16(c.Data[:], 1)
	u.PutUint16(c.Data[2:], 7)
	u.PutUint64(c.Data[8:], sequence)
	u.PutUint64(c.Data[16:], 100)
	for i, arg := range args {
		u.PutUint64(c.Data[32+8*i:], arg)
	}
	return c
}

func TestLaunchGeometryDecode(t *testing.T) {
	for _, tt := range []struct {
		name string
		c    Capture
	}{
		{"runtime", capture("cudaLaunchKernel", 1, 99, 2|uint64(3)<<32, 4, 5|uint64(6)<<32, 7, 0xdead, 8, 9)},
		{"driver", capture("cuLaunchKernel", 1, 99, 2, 3, 4, 5, 6, 7, 8, 9, 0xdead, 0xbeef)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e, err := Decode(tt.c)
			if err != nil {
				t.Fatal(err)
			}
			want := Launch{Function: 99, Grid: [3]uint32{2, 3, 4}, Block: [3]uint32{5, 6, 7}, SharedBytes: 8, Stream: 9}
			if !e.Decoded || e.Launch == nil || *e.Launch != want {
				t.Fatalf("event=%+v launch=%+v", e, e.Launch)
			}
			if e.Identity != tt.c.Identity {
				t.Fatal("identity lost")
			}
		})
	}
}

func TestDecodeRefusesUnknownOrIncomplete(t *testing.T) {
	for _, tt := range []struct {
		name      string
		change    func(*Capture)
		reason    string
		malformed bool
	}{
		{"version", func(c *Capture) { c.Data[0] = 2 }, "version", false},
		{"abi", func(c *Capture) { c.Identity.ABI = "aarch64" }, "ABI", false},
		{"symbol", func(c *Capture) { c.Identity.Symbol = "cuMemAllocFromPoolAsync" }, "symbol", false},
		{"build", func(c *Capture) { c.Identity.BuildID = "" }, "identity", false},
		{"arguments", func(c *Capture) { c.Data[2] = 0 }, "arguments", false},
		{"return", func(c *Capture) { c.Data[2] = 1 }, "return", false},
		{"output", func(c *Capture) { c.Data[2] = 3 }, "output", false},
		{"reserved", func(c *Capture) { c.Data[4] = 1 }, "", true},
		{"process", func(c *Capture) { c.Identity.Process.Generation = 0 }, "", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := capture("cudaMalloc", 1, 0x1000, 64)
			tt.change(&c)
			e, err := Decode(c)
			if (err != nil) != tt.malformed {
				t.Fatalf("error=%v", err)
			}
			if e.Decoded || !strings.Contains(e.Reason, tt.reason) {
				t.Fatalf("event=%+v", e)
			}
			if e.Identity != c.Identity {
				t.Fatal("undecoded identity lost")
			}
		})
	}
}

func TestAllocationReadsReturnedValue(t *testing.T) {
	c := capture("cuMemAlloc_v2", 1, 0x1234, 4096)
	binary.LittleEndian.PutUint64(c.Data[120:], 0x8000)
	e, err := Decode(c)
	if err != nil || !e.Decoded || e.Address != 0x8000 || e.Bytes != 4096 {
		t.Fatalf("event=%+v err=%v", e, err)
	}
	// Failed allocation output is undefined, even if the capture has stale data.
	binary.LittleEndian.PutUint32(c.Data[24:], 2)
	e, err = Decode(c)
	if err != nil || !e.Decoded || e.Address != 0 || e.Status != 2 {
		t.Fatalf("event=%+v err=%v", e, err)
	}
}

func FuzzDecode(f *testing.F) {
	c := capture("cudaLaunchKernel", 1, 99, 2, 3)
	f.Add(c.Data[:])
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) != len(c.Data) {
			return
		}
		x := c
		copy(x.Data[:], b)
		e, _ := Decode(x)
		if e.Identity != x.Identity {
			t.Fatal("identity changed")
		}
		if e.Decoded && e.Reason != "" {
			t.Fatal("decoded event has refusal reason")
		}
	})
}

func TestAAPCS64NormalizedReplay(t *testing.T) {
	c := capture("cuLaunchKernel", 1, 99, 2, 3, 4, 5, 6, 7, 8, 9, 0xdead, 0xbeef)
	c.Identity.ABI = "linux-arm64-aapcs64-v1"
	e, err := Decode(c)
	if err != nil || !e.Decoded || e.Launch == nil || e.Launch.Grid != [3]uint32{2, 3, 4} || e.Launch.Stream != 9 {
		t.Fatalf("%+v %v", e, err)
	}
	// Explicit source ABI governs replay even on a different architecture.
	c.Identity.ABI = "linux-amd64-sysv-v1"
	old, err := Decode(c)
	if err != nil || old.Launch == nil || *old.Launch != *e.Launch {
		t.Fatalf("legacy replay: %+v %v", old, err)
	}
}
